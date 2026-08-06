from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class DocumentMetadataResponse(BaseModel):
    """Response schema representing indexed contract metadata."""
    document_name: str = Field(..., description="File name of the contract")
    title: Optional[str] = Field(None, description="Extracted document title")
    parties: List[str] = Field(default_factory=list, description="Extracted contract parties")
    effective_date: Optional[str] = Field(None, description="Extracted effective date")
    page_count: int = Field(..., description="Total pages in document")

class AskResponse(BaseModel):
    """Response schema containing query completion response, citations, and source chunks."""
    answer: str = Field(..., description="The generated grounded answer text")
    confidence_score: float = Field(..., description="Combined normalized retrieval confidence metric")
    citations: List[Dict[str, Any]] = Field(..., description="Verified citation objects")
    retrieved_chunks: List[Dict[str, Any]] = Field(..., description="Source text chunks matched during search")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution logs and error info")

class HealthResponse(BaseModel):
    """Health check status response."""
    status: str = Field(..., description="Server health status indicator")

class DetailedHealthResponse(BaseModel):
    """Detailed health check response with connectivity status checks."""
    status: str = Field(..., description="Overall health status")
    chromadb_connected: bool = Field(..., description="Heartbeat test to ChromaDB")
    openai_configured: bool = Field(..., description="Availability check on OpenAI credentials")
    embedding_model_loaded: bool = Field(..., description="Loading test for SentenceTransformer models")

class ReadinessResponse(BaseModel):
    """Readiness status check response."""
    status: str = Field(..., description="Status indicator confirming readiness to serve traffic")

class ChatResponse(BaseModel):
    """Unified response schema for natural conversational chat."""
    intent: str = Field(..., description="The classified intent of the user message")
    answer: str = Field(..., description="The generated grounded answer text")
    confidence_score: float = Field(..., description="Combined normalized retrieval confidence metric")
    citations: List[Dict[str, Any]] = Field(..., description="Verified citation objects")
    retrieved_chunks: List[Dict[str, Any]] = Field(..., description="Source text chunks matched during search")
    validation_status: str = Field(..., description="Grounded retrieval validation check status (PASSED/FAILED)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution logs and error info")
