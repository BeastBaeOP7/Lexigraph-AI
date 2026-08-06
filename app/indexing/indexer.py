import os
import pickle
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.config.settings import settings
from app.config.logging import logger
from app.models.document import DocumentChunk

class Indexer:
    """Handles storage, embedding generation, and lexical indexing for document chunks."""
    
    def __init__(self) -> None:
        # Ensure data directory exists
        chroma_path = Path(settings.chroma_db_path)
        chroma_path.parent.mkdir(parents=True, exist_ok=True)
        
        bm25_path = Path(settings.bm25_index_path)
        bm25_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize persistent Chroma client
        self.chroma_client = chromadb.PersistentClient(path=str(chroma_path))
        self.collection = self.chroma_client.get_or_create_collection(
            name="legal_contract_chunks"
        )
        
        # Lazy loaded embedding model
        self._embedding_model: Optional[SentenceTransformer] = None

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy loads the embedding model to optimize startup time."""
        if self._embedding_model is None:
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            self._embedding_model = SentenceTransformer(settings.embedding_model)
        return self._embedding_model

    def is_document_indexed(self, document_name: str) -> bool:
        """Checks if the document has already been indexed in ChromaDB."""
        try:
            results = self.collection.get(
                where={"document_name": document_name},
                limit=1
            )
            return len(results.get("ids", [])) > 0
        except Exception as e:
            logger.error(f"Error checking if document {document_name} is indexed: {e}")
            return False

    def remove_document_index(self, document_name: str) -> None:
        """Deletes any existing chunks for the given document name from Chroma (for re-indexing)."""
        try:
            self.collection.delete(where={"document_name": document_name})
            logger.info(f"Removed existing ChromaDB chunks for document: {document_name}")
        except Exception as e:
            logger.error(f"Error removing document {document_name} from Chroma: {e}")

    def add_chunks_to_chroma(self, chunks: List[DocumentChunk]) -> None:
        """Generates embeddings and inserts chunks into ChromaDB."""
        if not chunks:
            return

        texts = [chunk.text for chunk in chunks]
        logger.info(f"Generating dense embeddings for {len(chunks)} chunks...")
        embeddings = self.embedding_model.encode(texts, normalize_embeddings=True).tolist()
        
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        
        metadatas = []
        for chunk in chunks:
            # Pydantic dict handles serialization
            meta = {
                "document_name": chunk.document_name,
                "page_number": chunk.page_number,
                "section_number": chunk.section_number or "",
                "section_title": chunk.section_title or "",
                "clause_number": chunk.clause_number or ""
            }
            # Merge general document metadata
            for k, v in chunk.metadata.items():
                if isinstance(v, list):
                    meta[k] = ",".join(map(str, v))
                elif isinstance(v, (str, int, float, bool)):
                    meta[k] = v
            metadatas.append(meta)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"Successfully added {len(chunks)} chunks to ChromaDB.")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple tokenizer for BM25 preprocessing."""
        return re.findall(r'\w+', text.lower())

    def update_bm25_index(self, new_chunks: List[DocumentChunk]) -> None:
        """
        Incrementally appends chunks and rebuilds/updates the BM25 index on the disk.
        """
        bm25_path = Path(settings.bm25_index_path)
        existing_chunks: List[DocumentChunk] = []

        if bm25_path.exists():
            try:
                with open(bm25_path, "rb") as f:
                    data = pickle.load(f)
                    existing_chunks = data.get("chunks", [])
                logger.info(f"Loaded {len(existing_chunks)} existing chunks from BM25 storage.")
            except Exception as e:
                logger.warning(f"Could not load existing BM25 index, rebuilding: {e}")

        # Filter out existing chunks of the same document to avoid duplicate indexing
        new_doc_names = {chunk.document_name for chunk in new_chunks}
        existing_chunks = [c for c in existing_chunks if c.document_name not in new_doc_names]

        # Combine corpus
        all_chunks = existing_chunks + new_chunks
        
        if not all_chunks:
            logger.info("No chunks to index for BM25.")
            return

        logger.info(f"Rebuilding BM25 index with a total of {len(all_chunks)} chunks...")
        tokenized_corpus = [self._tokenize(chunk.text) for chunk in all_chunks]
        bm25 = BM25Okapi(tokenized_corpus)

        with open(bm25_path, "wb") as f:
            pickle.dump({"bm25": bm25, "chunks": all_chunks}, f)
        
        logger.info("BM25 index saved successfully.")

    def delete_document_from_bm25(self, document_name: str) -> None:
        """Removes all chunks of the document and rebuilds BM25 index."""
        bm25_path = Path(settings.bm25_index_path)
        if not bm25_path.exists():
            return
            
        try:
            with open(bm25_path, "rb") as f:
                data = pickle.load(f)
                existing_chunks = data.get("chunks", [])
        except Exception as e:
            logger.error(f"Failed to load BM25 for deletion: {e}")
            return
            
        remaining_chunks = [c for c in existing_chunks if c.document_name != document_name]
        
        if not remaining_chunks:
            if bm25_path.exists():
                bm25_path.unlink()
            logger.info("All documents removed. BM25 index file deleted.")
            return

        logger.info(f"Rebuilding BM25 index after deleting {document_name} ({len(remaining_chunks)} chunks remaining)...")
        tokenized_corpus = [self._tokenize(chunk.text) for chunk in remaining_chunks]
        bm25 = BM25Okapi(tokenized_corpus)

        with open(bm25_path, "wb") as f:
            pickle.dump({"bm25": bm25, "chunks": remaining_chunks}, f)
        logger.info(f"Successfully deleted {document_name} from BM25 index.")
