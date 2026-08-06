from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class DocumentChunk(BaseModel):
    """Standardized representation of a legal document chunk."""
    chunk_id: str = Field(..., description="Unique deterministic identifier (e.g. SHA-256 hash or UUID) of the chunk")
    document_name: str = Field(..., description="Name of the source document file")
    page_number: int = Field(..., description="1-based page number where the chunk is located")
    section_number: Optional[str] = Field(None, description="Number/identifier of the section (e.g. 'I', '1.1')")
    section_title: Optional[str] = Field(None, description="Title of the section (e.g. 'DEFINITIONS')")
    clause_number: Optional[str] = Field(None, description="Clause number/identifier (e.g. '1.1(a)')")
    text: str = Field(..., description="Text content of the chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional dynamic metadata extracted from the document")

class RetrievedChunk(DocumentChunk):
    """Chunk enriched with search, rerank, and overall confidence scores."""
    retrieval_score: float = Field(0.0, description="Reciprocal Rank Fusion or search similarity score")
    reranker_score: float = Field(0.0, description="Cross-encoder relevance score (scaled to 0-1)")
    confidence_score: float = Field(0.0, description="Combined, normalized confidence score")
