from typing import Dict, Any
from app.graph.state import GraphState

def format_response_node(state: GraphState) -> Dict[str, Any]:
    """Formats the final output from GraphState into a structured dictionary response."""
    # Convert Pydantic chunk models to dicts for downstream consumers
    retrieved_serializable = []
    for chunk in state.get("retrieved_chunks", []):
        try:
            retrieved_serializable.append(chunk.model_dump())
        except AttributeError:
            # Fallback for Pydantic v1
            retrieved_serializable.append(chunk.dict())

    # Build the structured response object
    formatted = {
        "answer": state.get("generated_answer", ""),
        "confidence_score": float(state.get("confidence_score", 0.0)),
        "citations": state.get("citations", []),
        "retrieved_chunks": retrieved_serializable,
        "metadata": {
            "errors": state.get("errors", []),
            **state.get("metadata", {})
        }
    }
    return {"metadata": formatted}  # Returning final output inside metadata or state key as needed
