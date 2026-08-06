from fastapi import APIRouter, HTTPException
from typing import List
from app.schemas.request import SearchRequest
from app.models.document import RetrievedChunk
from app.retrieval.hybrid.hybrid_search import HybridRetrievalEngine
from app.config.logging import logger
from app.config.settings import settings

router = APIRouter()

@router.post("/search", response_model=List[RetrievedChunk])
async def search_retrieval_only(request: SearchRequest) -> List[RetrievedChunk]:
    """Performs retrieval only, returning ranked chunks and scores without calling the LLM."""
    logger.info(f"Received search-only query: {request.query}")
    
    top_k = request.top_k or settings.top_k
    
    try:
        engine = HybridRetrievalEngine()
        results = engine.search(request.query, top_k=top_k, metadata_filter=request.metadata_filter)
        return results
    except Exception as e:
        logger.error(f"Search retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
