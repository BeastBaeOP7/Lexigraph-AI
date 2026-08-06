from typing import Dict, Any
from app.config.settings import settings
from app.config.logging import logger
from app.graph.state import GraphState

def validate_retrieval_node(state: GraphState) -> Dict[str, Any]:
    """Validates that retrieved chunks meet quantity and quality thresholds."""
    chunks = state.get("retrieved_chunks", [])
    errors = list(state.get("errors", []))
    
    num_chunks = len(chunks)
    max_conf = max(c.confidence_score for c in chunks) if chunks else 0.0
    
    # Extract retrieval and reranker scores for logging
    max_retrieval = max(c.retrieval_score for c in chunks) if chunks else 0.0
    max_reranker = max(c.reranker_score for c in chunks) if chunks else 0.0
    
    threshold = settings.min_confidence_threshold
    decision = "PASSED"
    
    if num_chunks < settings.min_chunks_required:
        decision = f"FAILED: Insufficient chunks ({num_chunks} < {settings.min_chunks_required})"
    elif max_conf < threshold:
        decision = f"FAILED: Confidence below threshold ({max_conf:.3f} < {threshold})"
    elif sum(len(c.text.strip()) for c in chunks) == 0:
        decision = "FAILED: Empty retrieved context"
        
    logger.info(
        f"Retrieval Validation Details:\n"
        f"- Number of retrieved chunks: {num_chunks}\n"
        f"- Retrieval confidence (RRF): {max_retrieval:.4f}\n"
        f"- Reranker confidence (CrossEncoder): {max_reranker:.4f}\n"
        f"- Combined confidence: {max_conf:.4f}\n"
        f"- Validation threshold: {threshold}\n"
        f"- Validation decision: {decision}"
    )
                
    if "FAILED" in decision:
        errors.append(decision)
        return {
            "validated_chunks": [],
            "generated_answer": "I couldn't find enough information in the uploaded documents.",
            "confidence_score": max_conf,
            "errors": errors
        }
        
    return {
        "validated_chunks": chunks,
        "confidence_score": max_conf,
        "errors": errors
    }
