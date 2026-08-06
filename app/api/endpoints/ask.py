from fastapi import APIRouter, HTTPException
from app.schemas.request import AskRequest, CompareRequest
from app.schemas.response import AskResponse
from app.graph.graph import app_graph
from app.config.logging import logger

router = APIRouter()

@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest) -> dict:
    """Asks a question to the contract agent using the LangGraph workflow."""
    logger.info(f"Received QA query: {request.question}")
    
    initial_state = {
        "user_query": request.question,
        "retrieved_chunks": [],
        "validated_chunks": [],
        "generated_answer": "",
        "citations": [],
        "confidence_score": 0.0,
        "errors": [],
        "metadata": {
            "filter": request.metadata_filter
        }
    }
    
    try:
        final_state = app_graph.invoke(initial_state)
        formatted_output = final_state.get("metadata", {})
        if not formatted_output:
            raise HTTPException(status_code=500, detail="Malformed graph execution result.")
        return formatted_output
    except Exception as e:
        logger.error(f"Error executing agent query: {e}")
        raise HTTPException(status_code=500, detail=f"Graph execution failed: {str(e)}")

@router.post("/compare", response_model=AskResponse)
async def compare_contracts(request: CompareRequest) -> dict:
    """Restricts search to selected documents and returns a comparative analysis answer."""
    if len(request.document_names) < 2:
        raise HTTPException(status_code=400, detail="Comparison requires at least two document names.")
        
    logger.info(f"Comparing documents {request.document_names} for query: {request.question}")
    
    # Filter only on selected documents
    metadata_filter = {"document_name": request.document_names}
    
    initial_state = {
        "user_query": request.question,
        "retrieved_chunks": [],
        "validated_chunks": [],
        "generated_answer": "",
        "citations": [],
        "confidence_score": 0.0,
        "errors": [],
        "metadata": {
            "filter": metadata_filter
        }
    }
    
    try:
        final_state = app_graph.invoke(initial_state)
        formatted_output = final_state.get("metadata", {})
        if not formatted_output:
            raise HTTPException(status_code=500, detail="Malformed comparison result.")
        return formatted_output
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison workflow failed: {str(e)}")
