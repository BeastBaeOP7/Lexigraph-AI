from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pytest

from app.main import app
from app.graph.graph import check_intent_route

client = TestClient(app)

def test_check_intent_route() -> None:
    """Verifies that check_intent_route correctly forwards standard intents."""
    assert check_intent_route({"intent": "QUESTION"}) == "QUESTION"
    assert check_intent_route({"intent": "COMPARE"}) == "COMPARE"
    assert check_intent_route({"intent": "EXTRACT"}) == "EXTRACT"
    assert check_intent_route({"intent": "SUMMARIZE"}) == "SUMMARIZE"
    assert check_intent_route({"intent": "UNKNOWN"}) == "UNKNOWN"
    assert check_intent_route({}) == "UNKNOWN"

@patch("app.nodes.classify.get_openai_client")
@patch("app.main.app_graph.invoke")
def test_post_chat_endpoint_success(mock_invoke, mock_openai) -> None:
    """Verifies the unified POST /chat endpoint success flow and schema mapping."""
    mock_invoke.return_value = {
        "intent": "SUMMARIZE",
        "generated_answer": "Summary text.",
        "confidence_score": 0.95,
        "citations": [{"document_name": "NDA.pdf", "page_number": 1}],
        "validated_chunks": [{"document_name": "NDA.pdf", "page_number": 1, "text": "Confidential info..."}],
        "metadata": {}
    }

    res = client.post("/chat", json={"message": "Summarize the NDA"})
    assert res.status_code == 200
    
    data = res.json()
    assert data["intent"] == "SUMMARIZE"
    assert data["answer"] == "Summary text."
    assert data["confidence_score"] == 0.95
    assert len(data["citations"]) == 1
    assert data["citations"][0]["document_name"] == "NDA.pdf"
    assert len(data["retrieved_chunks"]) == 1
    assert data["retrieved_chunks"][0]["text"] == "Confidential info..."

from app.chunking.structure_chunker import StructureChunker
from app.nodes.confidence import validate_retrieval_node

def test_structure_chunker_dominant_clause():
    pages = [(1, "Some text describing responsibilities. 2.2 Standard working schedule is Monday to Friday.")]
    chunks = StructureChunker.chunk_document(pages, "Handbook.pdf", {})
    assert len(chunks) == 1
    assert chunks[0].clause_number == "2.2"

def test_validate_retrieval_node_passed():
    class MockChunk:
        def __init__(self):
            self.confidence_score = 0.95
            self.retrieval_score = 0.8
            self.reranker_score = 0.9
            self.text = "Valid text info."
            self.document_name = "doc.pdf"
            self.page_number = 1
            self.clause_number = "1.1"
            self.section_number = "1"
            self.section_title = "Sec"
            self.metadata = {}
            
    state = {
        "retrieved_chunks": [MockChunk()],
        "errors": []
    }
    res = validate_retrieval_node(state)
    assert len(res["validated_chunks"]) == 1
    assert res["confidence_score"] == 0.95
