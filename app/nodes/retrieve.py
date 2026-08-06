from typing import Dict, Any
from app.graph.state import GraphState
from app.retrieval.hybrid.hybrid_search import HybridRetrievalEngine

def retrieve_node(state: GraphState) -> Dict[str, Any]:
    """Retrieves document chunks matching the query using the HybridRetrievalEngine."""
    query = state.get("user_query", "")
    metadata = state.get("metadata", {})
    metadata_filter = metadata.get("filter") if metadata else None
    
    engine = HybridRetrievalEngine()
    try:
        retrieved = engine.search(query, metadata_filter=metadata_filter)
        return {
            "retrieved_chunks": retrieved,
            "errors": state.get("errors", [])
        }
    except Exception as e:
        err_msg = f"Retrieval failed: {str(e)}"
        current_errors = list(state.get("errors", []))
        current_errors.append(err_msg)
        return {
            "retrieved_chunks": [],
            "errors": current_errors
        }
