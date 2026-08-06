from pathlib import Path
from unittest.mock import MagicMock, patch
import json
import pytest

from app.evaluation.testset import GoldenDatasetLoader
from app.evaluation.judge_eval import LLMJudgeEvaluator
from scripts.evaluate import run_evaluation

@pytest.fixture
def mock_golden_json(tmp_path):
    dataset_file = tmp_path / "golden_dataset.json"
    content = [
        {
            "question": "What is the deposit?",
            "expected_answer": "The deposit is $1000.",
            "expected_context": "Tenant agrees to pay $1000 deposit.",
            "expected_document": "lease.pdf",
            "expected_clause": "3"
        }
    ]
    dataset_file.write_text(json.dumps(content))
    return dataset_file

def test_golden_dataset_loader(mock_golden_json):
    data = GoldenDatasetLoader.load_dataset(mock_golden_json)
    assert len(data) == 1
    assert data[0]["question"] == "What is the deposit?"

def test_golden_dataset_loader_missing_field(tmp_path):
    dataset_file = tmp_path / "bad_golden.json"
    content = [{"question": "Missing context"}]
    dataset_file.write_text(json.dumps(content))
    
    with pytest.raises(ValueError, match="missing required fields"):
        GoldenDatasetLoader.load_dataset(dataset_file)

@patch("app.evaluation.judge_eval.get_openai_client")
def test_llm_judge_evaluator_faithfulness(mock_client_fn):
    # Mock LLM response for faithfulness
    mock_choice = MagicMock()
    mock_choice.message.content = '[{"statement": "statement A", "supported": true}]'
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_client_fn.return_value = mock_client

    evaluator = LLMJudgeEvaluator()
    score = evaluator.evaluate_faithfulness("Answer text", "Context text")
    assert score == 1.0

@patch("app.evaluation.judge_eval.get_openai_client")
def test_llm_judge_evaluator_context_precision(mock_client_fn):
    # Mock LLM response returning relevance booleans
    mock_choice = MagicMock()
    mock_choice.message.content = '[true, false]'
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_client_fn.return_value = mock_client

    evaluator = LLMJudgeEvaluator()
    score = evaluator.evaluate_context_precision("Question text", ["chunk A", "chunk B"])
    assert score == 1.0
