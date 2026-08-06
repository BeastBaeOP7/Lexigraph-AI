import os
from fastapi import APIRouter
from app.config.settings import settings
from app.indexing.indexer import Indexer
from app.schemas.response import DetailedHealthResponse, ReadinessResponse

router = APIRouter()

@router.get("/health", response_model=DetailedHealthResponse)
async def health_check() -> dict:
    """Verifies connectivity to ChromaDB, OpenAI setup, and embedding model loading."""
    chroma_ok = False
    try:
        indexer = Indexer()
        indexer.chroma_client.heartbeat()
        chroma_ok = True
    except Exception:
        pass
        
    openai_ok = bool(settings.openai_api_key or os.environ.get("OPENAI_API_KEY"))
    
    embedding_ok = False
    try:
        indexer = Indexer()
        _ = indexer.embedding_model
        embedding_ok = True
    except Exception:
        pass
        
    overall_status = "healthy" if (chroma_ok and embedding_ok) else "unhealthy"
    
    return {
        "status": overall_status,
        "chromadb_connected": chroma_ok,
        "openai_configured": openai_ok,
        "embedding_model_loaded": embedding_ok
    }

@router.get("/ready", response_model=ReadinessResponse)
async def readiness_check() -> dict:
    """Verifies that the application server is fully active and ready to handle traffic."""
    return {"status": "ready"}
