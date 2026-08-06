from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    """Request schema for asking a question to the contract agent."""
    question: str = Field(..., description="The query to ask the contract agent")
    metadata_filter: Optional[Dict[str, Any]] = Field(None, description="Metadata key-value filters to restrict retrieval")

class SearchRequest(BaseModel):
    """Request schema for performing semantic search only."""
    query: str = Field(..., description="The retrieval query")
    top_k: Optional[int] = Field(None, description="Number of results to retrieve")
    metadata_filter: Optional[Dict[str, Any]] = Field(None, description="Metadata key-value filters to restrict retrieval")

class CompareRequest(BaseModel):
    """Request schema for comparing multiple contracts."""
    question: str = Field(..., description="Comparison query to run against documents")
    document_names: List[str] = Field(..., description="List of document names to compare")

class ExtractRequest(BaseModel):
    """Request schema for extracting specific clause properties."""
    document_name: str = Field(..., description="Target document file name")
    extraction_type: str = Field(..., description="Type of info to extract (obligations, payment_terms, termination_conditions, renewal_clauses)")

class SummarizeRequest(BaseModel):
    """Request schema for summarizing a contract."""
    document_name: str = Field(..., description="Target document file name")

class ChatRequest(BaseModel):
    """Unified request schema for natural conversational chat."""
    message: str = Field(..., description="The user message or query")
