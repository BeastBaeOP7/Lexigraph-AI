from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from app.main import app

client = TestClient(app)

def test_ready_endpoint() -> None:
    """Verifies ready endpoint returns ready response status."""
    res = client.get("/ready")
    assert res.status_code == 200
    assert res.json() == {"status": "ready"}

@patch("app.api.endpoints.health.Indexer")
def test_detailed_health_endpoint_healthy(mock_indexer_cls) -> None:
    """Verifies detailed health connectivity statuses for DB and model."""
    mock_indexer = MagicMock()
    mock_indexer.chroma_client.heartbeat.return_value = 12345
    mock_indexer.embedding_model = MagicMock()
    mock_indexer_cls.return_value = mock_indexer

    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["chromadb_connected"] is True
    assert data["embedding_model_loaded"] is True
