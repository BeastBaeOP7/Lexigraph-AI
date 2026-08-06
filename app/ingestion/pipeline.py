from pathlib import Path
from typing import List, Optional

from app.config.logging import logger
from app.ingestion.loader import PDFLoader
from app.chunking.structure_chunker import StructureChunker
from app.indexing.indexer import Indexer
from app.models.document import DocumentChunk

class IngestionPipeline:
    """End-to-end ingestion and indexing pipeline for legal documents."""
    
    def __init__(self, indexer: Optional[Indexer] = None) -> None:
        self.indexer = indexer or Indexer()

    def process_document(self, file_path: Path, force_reindex: bool = False) -> List[DocumentChunk]:
        """
        Executes loader, chunker, and indexing for a single document.
        
        Args:
            file_path: Path to the target PDF document.
            force_reindex: If True, re-indexes the document even if already present.
            
        Returns:
            List[DocumentChunk]: The created document chunks.
        """
        doc_name = file_path.name
        
        if not force_reindex and self.indexer.is_document_indexed(doc_name):
            logger.info(f"Document {doc_name} is already indexed. Skipping.")
            return []
            
        logger.info(f"Starting ingestion pipeline for: {file_path}")
        
        # 1. Load document
        pages_text, metadata = PDFLoader.load(file_path)
        logger.info(f"Loaded {len(pages_text)} pages. Extracted metadata: {metadata}")
        
        # 2. Chunk document
        chunks = StructureChunker.chunk_document(pages_text, doc_name, metadata)
        logger.info(f"Segmented document into {len(chunks)} chunks.")
        
        if force_reindex:
            self.indexer.remove_document_index(doc_name)
            
        # 3. Add to indices
        self.indexer.add_chunks_to_chroma(chunks)
        self.indexer.update_bm25_index(chunks)
        
        logger.info(f"Finished pipeline for: {doc_name}")
        return chunks
