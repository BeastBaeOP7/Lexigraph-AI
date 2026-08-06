from unittest.mock import MagicMock, patch
import pytest

from app.models.document import RetrievedChunk
from app.graph.graph import app_graph
from app.graph.state import GraphState

@pytest.fixture
def base_state():
    return {
        "user_query": "What is the security deposit limit?",
        "retrieved_chunks": [],
        "validated_chunks": [],
        "generated_answer": "",
        "citations": [],
        "confidence_score": 0.0,
        "errors": [],
        "metadata": {}
    }

@pytest.fixture
def sample_retrieved_chunks():
    return [
        RetrievedChunk(
            chunk_id="chunk_a",
            document_name="agreement.pdf",
            page_number=1,
            section_number="3",
            section_title="DEPOSIT",
            clause_number="3.2",
            text="The tenant agrees to pay a security deposit of $1000.",
            metadata={},
            retrieval_score=0.8,
            reranker_score=0.9,
            confidence_score=0.85
        )
    ]

@patch("app.nodes.retrieve.HybridRetrievalEngine.search")
def test_graph_validation_fallback(mock_search, base_state):
    # Mocking retrieval to return empty lists to force validation failure
    mock_search.return_value = []

    final_state = app_graph.invoke(base_state)
    
    # Should route validate -> format directly, outputting fallback answer
    assert final_state["generated_answer"] == "I couldn't find enough information in the uploaded documents."
    assert final_state["citations"] == []
    
    # The compiled output from format node is saved under metadata
    metadata = final_state["metadata"]
    assert metadata["answer"] == "I couldn't find enough information in the uploaded documents."
    assert metadata["confidence_score"] == 0.0

@patch("app.nodes.retrieve.HybridRetrievalEngine.search")
@patch("app.nodes.generate.get_openai_client")
def test_graph_successful_flow(mock_openai_client_fn, mock_search, base_state, sample_retrieved_chunks):
    # Mock search to return valid chunks
    mock_search.return_value = sample_retrieved_chunks

    # Mock OpenAI completions call
    mock_choice = MagicMock()
    mock_choice.message.content = "The deposit is $1000. [agreement.pdf, Page 1, Clause 3.2]"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_client_fn.return_value = mock_client

    final_state = app_graph.invoke(base_state)

    # Generated answer should contain citation
    assert "The deposit is $1000." in final_state["generated_answer"]
    assert len(final_state["citations"]) == 1
    assert final_state["citations"][0]["document_name"] == "agreement.pdf"
    assert final_state["citations"][0]["clause_number"] == "3.2"

    metadata = final_state["metadata"]
    assert metadata["confidence_score"] == 0.85
    assert len(metadata["retrieved_chunks"]) == 1

@patch("app.nodes.retrieve.HybridRetrievalEngine.search")
@patch("app.nodes.generate.get_openai_client")
def test_graph_citation_cleanup(mock_openai_client_fn, mock_search, base_state, sample_retrieved_chunks):
    mock_search.return_value = sample_retrieved_chunks

    # Mock OpenAI to output one verified and one fake/hallucinated citation
    mock_choice = MagicMock()
    mock_choice.message.content = (
        "The deposit is $1000 [agreement.pdf, Page 1, Clause 3.2]. "
        "Also rent is due [lease.pdf, Page 4, Clause 1.2]."
    )
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_client_fn.return_value = mock_client

    final_state = app_graph.invoke(base_state)

    # Fake citation [lease.pdf...] should be stripped out from final answer
    answer = final_state["generated_answer"]
    assert "[agreement.pdf, Page 1, Clause 3.2]" in answer
    assert "[lease.pdf" not in answer
    
    # Check that verified citation list only contains the valid one
    assert len(final_state["citations"]) == 1
    assert final_state["citations"][0]["document_name"] == "agreement.pdf"
