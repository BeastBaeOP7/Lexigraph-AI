from unittest.mock import MagicMock, patch
import pytest

from app.models.document import DocumentChunk
from app.retrieval.bm25.bm25_search import BM25Searcher
from app.retrieval.vector.vector_search import VectorSearcher
from app.retrieval.reranker.reranker import Reranker
from app.confidence.confidence_score import calculate_confidence_scores
from app.retrieval.hybrid.hybrid_search import HybridRetrievalEngine

@pytest.fixture
def sample_chunks():
    return [
        DocumentChunk(
            chunk_id="chunk_1",
            document_name="agreement.pdf",
            page_number=1,
            section_number="1",
            section_title="DEFINITIONS",
            clause_number="1.1",
            text="Affiliate means any corporate entity under common control.",
            metadata={}
        ),
        DocumentChunk(
            chunk_id="chunk_2",
            document_name="agreement.pdf",
            page_number=2,
            section_number="2",
            section_title="OBLIGATIONS",
            clause_number="2.1",
            text="The party shall safeguard all intellectual property.",
            metadata={}
        )
    ]

def test_confidence_score_normalization():
    scores = calculate_confidence_scores([0.8, 0.4], [0.05, 0.01])
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)

    equal_scores = calculate_confidence_scores([0.5, 0.5], [0.02, 0.02])
    assert equal_scores == [1.0, 1.0]

def test_rrf_rank_fusion(sample_chunks):
    engine = HybridRetrievalEngine(
        bm25_searcher=MagicMock(),
        vector_searcher=MagicMock(),
        reranker=MagicMock()
    )
    bm25_res = [(sample_chunks[0], 1.5), (sample_chunks[1], 0.5)]
    vector_res = [(sample_chunks[1], 0.9), (sample_chunks[0], 0.1)]
    
    fused = engine.reciprocal_rank_fusion(bm25_res, vector_res, rrf_k=60)
    assert len(fused) == 2
    assert fused[0][0].chunk_id in ["chunk_1", "chunk_2"]

def test_parse_clause_reference():
    engine = HybridRetrievalEngine(
        bm25_searcher=MagicMock(),
        vector_searcher=MagicMock(),
        reranker=MagicMock()
    )
    
    # Matches "Clause 5.2"
    cl, sec = engine.parse_clause_reference("Refer to Clause 5.2 of the contract.")
    assert cl == "5.2"
    assert sec is None
    
    # Matches "Section 8"
    cl, sec = engine.parse_clause_reference("Under Section 8, obligations are defined.")
    assert cl is None
    assert sec == "8"

    # Matches loose decimal "4.1.2"
    cl, sec = engine.parse_clause_reference("Refer to section 4.1.2")
    # Matches section_num because it starts with "section"
    assert cl is None
    assert sec == "4.1.2"

    # Loose match "4.1.2" (no section prefix)
    cl, sec = engine.parse_clause_reference("Refer to 4.1.2")
    assert cl == "4.1.2"
    assert sec is None

@patch("app.retrieval.reranker.reranker.CrossEncoder")
def test_reranker_sigmoid(mock_cross_encoder_cls, sample_chunks):
    mock_model = MagicMock()
    mock_model.predict.return_value = [2.0, -2.0]
    mock_cross_encoder_cls.return_value = mock_model

    reranker = Reranker()
    results = reranker.rerank("test query", sample_chunks)
    
    assert len(results) == 2
    assert results[0][1] > 0.8
    assert results[1][1] < 0.2
    assert results[0][0].chunk_id == "chunk_1"

@patch("app.retrieval.bm25.bm25_search.BM25Searcher.search")
@patch("app.retrieval.vector.vector_search.VectorSearcher.search")
@patch("app.retrieval.reranker.reranker.Reranker.rerank")
def test_hybrid_engine_orchestration(mock_rerank, mock_vector_search, mock_bm25_search, sample_chunks):
    mock_bm25_search.return_value = [(sample_chunks[0], 1.2)]
    mock_vector_search.return_value = [(sample_chunks[1], 0.8)]
    mock_rerank.return_value = [(sample_chunks[0], 0.9), (sample_chunks[1], 0.4)]
    
    engine = HybridRetrievalEngine()
    results = engine.search("Find general obligations", top_k=2)
    
    assert len(results) == 2
    assert results[0].retrieval_score > 0.0
    assert results[0].reranker_score == 0.9
    assert results[0].confidence_score == pytest.approx(1.0)
    assert results[1].confidence_score == pytest.approx(0.3)
