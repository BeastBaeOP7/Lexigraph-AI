import numpy as np
from typing import List, Tuple, Optional
from sentence_transformers import CrossEncoder

from app.config.settings import settings
from app.config.logging import logger
from app.models.document import DocumentChunk

class Reranker:
    """Reranks retrieved candidate document chunks using a Cross-Encoder model."""
    
    def __init__(self, model_name: str = settings.reranker_model) -> None:
        self.model_name = model_name
        self._model: Optional[CrossEncoder] = None

    @property
    def model(self) -> CrossEncoder:
        """Lazy load the CrossEncoder model."""
        if self._model is None:
            logger.info(f"Loading CrossEncoder model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
        return self._model

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Compute sigmoid function to map logit outputs to probability ranges [0, 1]."""
        return 1.0 / (1.0 + np.exp(-x))

    def rerank(
        self,
        query: str,
        chunks: List[DocumentChunk]
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Computes relevance scores for candidate chunks against the query.
        
        Returns:
            List[Tuple[DocumentChunk, float]]: Sorted chunks and their reranked scores in [0, 1].
        """
        if not chunks:
            return []

        pairs = [[query, chunk.text] for chunk in chunks]
        
        try:
            # Generate logits
            scores = self.model.predict(pairs)
        except Exception as e:
            logger.error(f"CrossEncoder prediction failed: {e}")
            # Return uniform fallback scores
            return [(chunk, 0.0) for chunk in chunks]

        # Convert scalar output to list if only one pair was predicted
        if isinstance(scores, (float, np.float32, np.float64)):
            scores = [scores]

        scored_chunks = []
        for chunk, score in zip(chunks, scores):
            # Scale score to [0, 1] range using sigmoid
            scaled_score = float(self._sigmoid(score))
            scored_chunks.append((chunk, scaled_score))

        # Sort descending by score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks
