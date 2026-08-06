from typing import TypedDict, List, Dict, Any
from app.models.document import RetrievedChunk

class GraphState(TypedDict):
    """The structured state object maintained across nodes in the LangGraph workflow."""
    user_query: str
    retrieved_chunks: List[RetrievedChunk]
    validated_chunks: List[RetrievedChunk]
    generated_answer: str
    citations: List[Dict[str, Any]]
    confidence_score: float
    errors: List[str]
    metadata: Dict[str, Any]
    intent: str
