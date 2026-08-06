import pickle
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from app.config.settings import settings
from app.config.logging import logger
from app.models.document import DocumentChunk

class BM25Searcher:
    """Performs lexical search on the saved BM25 index."""
    
    def __init__(self) -> None:
        self.bm25_path = Path(settings.bm25_index_path)
        self.bm25 = None
        self.chunks: List[DocumentChunk] = []
        self.load_index()

    def load_index(self) -> None:
        """Loads the BM25 index from disk if it exists."""
        if self.bm25_path.exists():
            try:
                with open(self.bm25_path, "rb") as f:
                    data = pickle.load(f)
                    self.bm25 = data.get("bm25")
                    self.chunks = data.get("chunks", [])
                logger.info(f"Loaded BM25 index with {len(self.chunks)} chunks.")
            except Exception as e:
                logger.error(f"Failed to load BM25 index: {e}")
                self.bm25 = None
                self.chunks = []
        else:
            logger.warning("BM25 index file not found. Search will return empty results.")

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text into lowercase words."""
        return re.findall(r'\w+', text.lower())

    def search(
        self,
        query: str,
        top_k: int = settings.bm25_k,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieves top lexical matches from BM25.
        
        Returns:
            List[Tuple[DocumentChunk, float]]: Chunks with BM25 score.
        """
        if not self.bm25 or not self.chunks:
            # Try reloading in case index was recently built
            self.load_index()
            if not self.bm25 or not self.chunks:
                return []

        tokenized_query = self._tokenize(query)
        try:
            scores = self.bm25.get_scores(tokenized_query)
        except Exception as e:
            logger.error(f"BM25 score calculation failed: {e}")
            return []

        scored_chunks = []
        for idx, score in enumerate(scores):
            chunk = self.chunks[idx]
            
            # Apply metadata filters
            if metadata_filter:
                matched = True
                for k, v in metadata_filter.items():
                    val = getattr(chunk, k, None) or chunk.metadata.get(k)
                    if isinstance(v, list):
                        if val not in v:
                            matched = False
                            break
                    else:
                        if val != v:
                            matched = False
                            break
                if not matched:
                    continue
                    
            scored_chunks.append((chunk, float(score)))

        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]
