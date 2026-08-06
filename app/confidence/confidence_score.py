from typing import List

def calculate_confidence_scores(
    reranker_scores: List[float],
    hybrid_scores: List[float]
) -> List[float]:
    """
    Computes a combined confidence score using normalized scores.
    
    Formula:
        confidence = 0.70 * norm_reranker_score + 0.30 * norm_hybrid_score
    """
    if not reranker_scores or not hybrid_scores or len(reranker_scores) != len(hybrid_scores):
        return [0.0] * len(reranker_scores)
        
    def normalize(vals: List[float]) -> List[float]:
        if not vals:
            return []
        min_v = min(vals)
        max_v = max(vals)
        diff = max_v - min_v
        if diff < 1e-9:
            # If all values are identical, treat them as maximum relevance (1.0)
            return [1.0] * len(vals)
        return [(v - min_v) / diff for v in vals]

    norm_rerank = normalize(reranker_scores)
    norm_hybrid = normalize(hybrid_scores)
    
    confidence_scores = []
    for nr, nh in zip(norm_rerank, norm_hybrid):
        score = 0.70 * nr + 0.30 * nh
        confidence_scores.append(float(score))
        
    return confidence_scores
