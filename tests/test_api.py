import io
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import pytest

from app.main import app
from app.models.document import RetrievedChunk

client = TestClient(app)

def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

@patch("app.main.IngestionPipeline")
def test_upload_pdf_success(mock_pipeline_cls):
    mock_pipeline = MagicMock()
    # Mock processes returning a valid indexed DocumentChunk
    mock_chunk = MagicMock()
    mock_chunk.document_name = "test.pdf"
    mock_chunk.metadata = {"title": "Test Title", "parties": ["Party A"], "effective_date": "2026-08-06"}
    mock_chunk.page_number = 1
    mock_pipeline.process_document.return_value = [mock_chunk]
    mock_pipeline_cls.return_value = mock_pipeline

    pdf_content = b"%PDF-1.4 dummy contents"
    res = client.post(
        "/upload",
        files={"file": ("test.pdf", pdf_content, "application/pdf")}
    )
    
    assert res.status_code == 200
    data = res.json()
    assert data["document_name"] == "test.pdf"
    assert data["title"] == "Test Title"
    assert data["parties"] == ["Party A"]

def test_upload_invalid_file_format():
    res = client.post(
        "/upload",
        files={"file": ("test.txt", b"plain text", "text/plain")}
    )
    assert res.status_code == 400
    assert "PDF" in res.json()["detail"]

@patch("app.api.endpoints.ask.app_graph.invoke")
def test_ask_endpoint(mock_invoke):
    # Mock graph output state formatter metadata
    mock_invoke.return_value = {
        "metadata": {
            "answer": "Obligations are defined in Section 2.",
            "confidence_score": 0.9,
            "citations": [{"document_name": "test.pdf", "page_number": 2, "clause_number": "2.1"}],
            "retrieved_chunks": [],
            "metadata": {}
        }
    }

    res = client.post("/ask", json={"question": "What are my obligations?"})
    assert res.status_code == 200
    data = res.json()
    assert data["answer"] == "Obligations are defined in Section 2."
    assert data["confidence_score"] == 0.9

@patch("app.api.endpoints.search.HybridRetrievalEngine.search")
def test_search_endpoint(mock_search):
    # Mock retrieval search result
    chunk = RetrievedChunk(
        chunk_id="chunk_1",
        document_name="test.pdf",
        page_number=1,
        section_number=None,
        section_title=None,
        clause_number=None,
        text="Target search text match.",
        metadata={},
        retrieval_score=0.9,
        reranker_score=0.95,
        confidence_score=0.92
    )
    mock_search.return_value = [chunk]

    res = client.post("/search", json={"query": "Search query text"})
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 1
    assert data[0]["chunk_id"] == "chunk_1"
    assert data[0]["text"] == "Target search text match."

@patch("app.api.endpoints.ask.app_graph.invoke")
def test_compare_endpoint(mock_invoke):
    mock_invoke.return_value = {
        "metadata": {
            "answer": "Comparing doc A and doc B...",
            "confidence_score": 0.8,
            "citations": [],
            "retrieved_chunks": [],
            "metadata": {}
        }
    }

    res = client.post(
        "/compare", 
        json={"question": "Compare termination terms", "document_names": ["doc_a.pdf", "doc_b.pdf"]}
    )
    assert res.status_code == 200
    data = res.json()
    assert "Comparing doc" in data["answer"]

@patch("app.api.endpoints.extract.app_graph.invoke")
def test_extract_endpoint(mock_invoke):
    mock_invoke.return_value = {
        "metadata": {
            "answer": "Obligation clauses detail...",
            "confidence_score": 0.85,
            "citations": [],
            "retrieved_chunks": [],
            "metadata": {}
        }
    }

    # Test valid extract
    res = client.post(
        "/extract",
        json={"document_name": "test.pdf", "extraction_type": "obligations"}
    )
    assert res.status_code == 200
    assert res.json()["answer"] == "Obligation clauses detail..."

    # Test invalid extract type
    res = client.post(
        "/extract",
        json={"document_name": "test.pdf", "extraction_type": "invalid_type"}
    )
    assert res.status_code == 400

@patch("app.api.endpoints.extract.app_graph.invoke")
def test_summarize_endpoint(mock_invoke):
    mock_invoke.return_value = {
        "metadata": {
            "answer": "Summary description.",
            "confidence_score": 0.9,
            "citations": [],
            "retrieved_chunks": [],
            "metadata": {}
        }
    }

    res = client.post("/summarize", json={"document_name": "test.pdf"})
    assert res.status_code == 200
    assert res.json()["answer"] == "Summary description."

@patch("app.main.IngestionPipeline")
def test_delete_document_endpoints(mock_pipeline_cls):
    mock_pipeline = MagicMock()
    mock_pipeline_cls.return_value = mock_pipeline

    # Test single delete
    res = client.delete("/documents/test_contract.pdf")
    assert res.status_code == 200
    assert res.json() == {"status": "success", "message": "Successfully deleted test_contract.pdf"}

    # Test clear all
    res = client.delete("/documents")
    assert res.status_code == 200
    assert res.json() == {"status": "success", "message": "Successfully cleared all documents"}
