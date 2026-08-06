import re
from typing import List, Tuple, Dict, Any, Optional

from app.config.settings import settings
from app.config.logging import logger
from app.models.document import DocumentChunk, RetrievedChunk
from app.retrieval.bm25.bm25_search import BM25Searcher
from app.retrieval.vector.vector_search import VectorSearcher
from app.retrieval.reranker.reranker import Reranker
from app.confidence.confidence_score import calculate_confidence_scores

class HybridRetrievalEngine:
    """Orchestrates direct clause lookup, BM25/Vector retrieval, RRF fusion, reranking, and scoring."""
    
    def __init__(
        self,
        bm25_searcher: Optional[BM25Searcher] = None,
        vector_searcher: Optional[VectorSearcher] = None,
        reranker: Optional[Reranker] = None
    ) -> None:
        self.bm25_searcher = bm25_searcher or BM25Searcher()
        self.vector_searcher = vector_searcher or VectorSearcher()
        self.reranker = reranker or Reranker()

    def parse_clause_reference(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """Parses clause or section numbers from query. Returns (clause_number, section_number)."""
        # Clause pattern: e.g. "Clause 5.2", "cl 4.1"
        clause_match = re.search(r'(?i)\b(?:clause|cl\.?)\s*(\d+(?:\.\d+)*|\([a-z0-9]\))\b', query)
        # Section pattern: e.g. "Section 8", "sec II"
        section_match = re.search(r'(?i)\b(?:section|sec\.?)\s*([IVXLCDM\d]+(?:\.\d+)*)\b', query)
        
        clause_num = clause_match.group(1) if clause_match else None
        section_num = section_match.group(1) if section_match else None
        
        # Loose number match if nothing else matched, e.g. "4.1"
        if not clause_num and not section_num:
            loose_match = re.search(r'\b(\d+\.\d+(?:\.\d+)*)\b', query)
            if loose_match:
                clause_num = loose_match.group(1)
                
        return clause_num, section_num

    def retrieve_direct_matches(
        self,
        clause_num: Optional[str],
        section_num: Optional[str],
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """Queries ChromaDB directly for specific clause or section numbers."""
        matches = []
        if not clause_num and not section_num:
            return matches
            
        where_filter: Dict[str, Any] = {}
        if metadata_filter:
            where_filter.update(metadata_filter)
            
        collection = self.vector_searcher.indexer.collection
        if not collection:
            return matches
            
        try:
            if clause_num:
                where_clause = {**where_filter, "clause_number": clause_num}
                res = collection.get(where=where_clause)
                matches.extend(self._parse_chroma_get_results(res))
            if section_num:
                where_section = {**where_filter, "section_number": section_num}
                res = collection.get(where=where_section)
                matches.extend(self._parse_chroma_get_results(res))
        except Exception as e:
            logger.error(f"Failed to retrieve direct clause matches: {e}")
            
        return matches

    def _parse_chroma_get_results(self, res: dict) -> List[DocumentChunk]:
        chunks = []
        if not res or "ids" not in res or not res["ids"]:
            return chunks
        ids = res["ids"]
        documents = res["documents"]
        metadatas = res["metadatas"]
        for idx in range(len(ids)):
            meta = metadatas[idx]
            chunk = DocumentChunk(
                chunk_id=ids[idx],
                document_name=meta.get("document_name", ""),
                page_number=meta.get("page_number", 1),
                section_number=meta.get("section_number") or None,
                section_title=meta.get("section_title") or None,
                clause_number=meta.get("clause_number") or None,
                text=documents[idx],
                metadata={
                    k: v for k, v in meta.items()
                    if k not in ["document_name", "page_number", "section_number", "section_title", "clause_number"]
                }
            )
            chunks.append(chunk)
        return chunks

    def reciprocal_rank_fusion(
        self,
        bm25_results: List[Tuple[DocumentChunk, float]],
        vector_results: List[Tuple[DocumentChunk, float]],
        rrf_k: int = settings.rrf_k
    ) -> List[Tuple[DocumentChunk, float]]:
        """Combines BM25 and Vector search rankings using RRF."""
        scores: Dict[str, float] = {}
        chunk_map: Dict[str, DocumentChunk] = {}
        
        # Process BM25
        for rank, (chunk, _) in enumerate(bm25_results):
            chunk_map[chunk.chunk_id] = chunk
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + (1.0 / (rrf_k + rank + 1))
            
        # Process Vector
        for rank, (chunk, _) in enumerate(vector_results):
            chunk_map[chunk.chunk_id] = chunk
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + (1.0 / (rrf_k + rank + 1))
            
        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(chunk_map[cid], score) for cid, score in sorted_ids]

    def search(
        self,
        query: str,
        top_k: int = settings.top_k,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        """
        Executes hybrid retrieval flow: direct lookup -> BM25 + Vector -> RRF -> Cross-Encoder -> confidence score.
        """
        # 1. Direct clause lookup check
        clause_num, section_num = self.parse_clause_reference(query)
        direct_chunks = []
        if clause_num or section_num:
            logger.info(f"Clause reference detected in query: clause={clause_num}, section={section_num}")
            direct_chunks = self.retrieve_direct_matches(clause_num, section_num, metadata_filter)
            logger.info(f"Retrieved {len(direct_chunks)} direct matches.")

        # 2. Retrieve lexical and semantic matches
        bm25_res = self.bm25_searcher.search(query, top_k=settings.bm25_k, metadata_filter=metadata_filter)
        vector_res = self.vector_searcher.search(query, top_k=settings.vector_k, metadata_filter=metadata_filter)
        
        # 3. Hybrid Reciprocal Rank Fusion (RRF)
        fused_results = self.reciprocal_rank_fusion(bm25_res, vector_res)
        
        # Deduplicate and merge direct matches at the front of candidate list
        candidates: List[Tuple[DocumentChunk, float]] = []
        seen_ids = set()
        
        for chunk in direct_chunks:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                # Give direct matches maximum RRF priority score
                candidates.append((chunk, 2.0))
                
        for chunk, score in fused_results:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                candidates.append((chunk, score))

        if not candidates:
            return []

        # Slice candidates for reranking to avoid performance bottlenecks
        rerank_candidates = [c[0] for c in candidates[:settings.bm25_k + settings.vector_k]]
        hybrid_scores_map = {c[0].chunk_id: c[1] for c in candidates}

        # 4. Cross-Encoder Reranking
        logger.info(f"Reranking {len(rerank_candidates)} candidates...")
        reranked = self.reranker.rerank(query, rerank_candidates)
        
        # Limit to final top_k
        top_reranked = reranked[:top_k]
        
        # 5. Confidence Score Calculation
        reranker_scores = [r[1] for r in top_reranked]
        hybrid_scores = [hybrid_scores_map[r[0].chunk_id] for r in top_reranked]
        
        confidence_scores = calculate_confidence_scores(reranker_scores, hybrid_scores)
        
        # 6. Map to RetrievedChunk list
        retrieved_chunks: List[RetrievedChunk] = []
        for idx, (chunk, rerank_score) in enumerate(top_reranked):
            retrieved_chunks.append(RetrievedChunk(
                chunk_id=chunk.chunk_id,
                document_name=chunk.document_name,
                page_number=chunk.page_number,
                section_number=chunk.section_number,
                section_title=chunk.section_title,
                clause_number=chunk.clause_number,
                text=chunk.text,
                metadata=chunk.metadata,
                retrieval_score=hybrid_scores_map[chunk.chunk_id],
                reranker_score=rerank_score,
                confidence_score=confidence_scores[idx]
            ))
            
        # Bug 3: Filter out low-value chunks (e.g. signature pages, acknowledgments)
        low_value_patterns = [
            r"(?i)\bend\s+of\s+document\b",
            r"(?i)\backnowledgement\b",
            r"(?i)\bsignature\b",
            r"(?i)\btable\s+of\s+contents\b",
            r"(?i)\bindex\b",
            r"(?i)\bblank\s+page\b"
        ]
        filtered_chunks = []
        for c in retrieved_chunks:
            is_low_val = False
            for pat in low_value_patterns:
                if re.search(pat, c.text):
                    is_low_val = True
                    break
            if not is_low_val:
                filtered_chunks.append(c)
                
        if filtered_chunks:
            return filtered_chunks
            
        return retrieved_chunks
