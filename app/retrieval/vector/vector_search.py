from typing import List, Tuple, Dict, Any, Optional

from app.config.settings import settings
from app.config.logging import logger
from app.indexing.indexer import Indexer
from app.models.document import DocumentChunk

class VectorSearcher:
    """Performs semantic search on the persistent ChromaDB database."""
    
    def __init__(self, indexer: Optional[Indexer] = None) -> None:
        self.indexer = indexer or Indexer()

    def search(
        self,
        query: str,
        top_k: int = settings.vector_k,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieves top semantic matches from ChromaDB.
        
        Returns:
            List[Tuple[DocumentChunk, float]]: Chunks with similarity score.
        """
        # Ensure we have active indexer and collection
        if not self.indexer.collection:
            logger.error("ChromaDB collection is not initialized.")
            return []

        try:
            # Generate query embedding
            logger.info("Generating embedding for search query...")
            query_embedding = self.indexer.embedding_model.encode(
                query,
                normalize_embeddings=True
            ).tolist()
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return []

        # Convert simple dictionary metadata filters into Chroma format
        where_filter = {}
        if metadata_filter:
            for k, v in metadata_filter.items():
                where_filter[k] = v

        try:
            results = self.indexer.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter if where_filter else None
            )
        except Exception as e:
            logger.error(f"ChromaDB query call failed: {e}")
            return []

        scored_chunks: List[Tuple[DocumentChunk, float]] = []
        if not results or "ids" not in results or not results["ids"] or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        distances = results["distances"][0] if "distances" in results and results["distances"] else [1.0] * len(ids)
        documents = results["documents"][0] if "documents" in results else [""] * len(ids)
        metadatas = results["metadatas"][0] if "metadatas" in results else [{}] * len(ids)

        for idx in range(len(ids)):
            meta = metadatas[idx]
            
            # Reconstruct DocumentChunk
            chunk = DocumentChunk(
                chunk_id=ids[idx],
                document_name=meta.get("document_name", ""),
                page_number=meta.get("page_number", 1),
                section_number=meta.get("section_number") or None,
                section_title=meta.get("section_title") or None,
                clause_number=meta.get("clause_number") or None,
                text=documents[idx],
                # Reconstruct original metadata if any extra fields
                metadata={
                    k: v for k, v in meta.items()
                    if k not in ["document_name", "page_number", "section_number", "section_title", "clause_number"]
                }
            )
            # Similarity = 1.0 - distance
            similarity = float(1.0 - distances[idx])
            scored_chunks.append((chunk, similarity))

        return scored_chunks
