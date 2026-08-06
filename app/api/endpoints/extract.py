from fastapi import APIRouter, HTTPException
from app.schemas.request import ExtractRequest, SummarizeRequest
from app.schemas.response import AskResponse
from app.graph.graph import app_graph
from app.config.logging import logger

router = APIRouter()

EXTRACTION_PROMPTS = {
    "obligations": "What are the primary obligations and responsibilities of the parties in this contract?",
    "payment_terms": "What are the payment terms, invoicing schedules, and pricing details?",
    "termination_conditions": "What are the termination conditions, notice periods, and breach consequences?",
    "renewal_clauses": "What are the renewal terms, automatic extension clauses, and options?"
}

@router.post("/extract", response_model=AskResponse)
async def extract_information(request: ExtractRequest) -> dict:
    """Extracts key information properties (obligations, payments, etc.) from a single contract."""
    ext_type = request.extraction_type.lower().strip()
    if ext_type not in EXTRACTION_PROMPTS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported extraction type. Must be one of: {list(EXTRACTION_PROMPTS.keys())}"
        )
        
    question = EXTRACTION_PROMPTS[ext_type]
    logger.info(f"Extracting {ext_type} from {request.document_name}")
    
    metadata_filter = {"document_name": request.document_name}
    
    initial_state = {
        "user_query": question,
        "retrieved_chunks": [],
        "validated_chunks": [],
        "generated_answer": "",
        "citations": [],
        "confidence_score": 0.0,
        "errors": [],
        "metadata": {
            "filter": metadata_filter,
            "extraction_type": ext_type
        }
    }
    
    try:
        final_state = app_graph.invoke(initial_state)
        formatted_output = final_state.get("metadata", {})
        if not formatted_output:
            raise HTTPException(status_code=500, detail="Malformed extraction result.")
        return formatted_output
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Extraction workflow failed: {str(e)}")

@router.post("/summarize", response_model=AskResponse)
async def summarize_contract(request: SummarizeRequest) -> dict:
    """Generates a structured summarization of a contract document."""
    logger.info(f"Summarizing document: {request.document_name}")
    
    question = "Provide a comprehensive summary of this document highlighting the key terms, dates, parties, and core purpose."
    metadata_filter = {"document_name": request.document_name}
    
    initial_state = {
        "user_query": question,
        "retrieved_chunks": [],
        "validated_chunks": [],
        "generated_answer": "",
        "citations": [],
        "confidence_score": 0.0,
        "errors": [],
        "metadata": {
            "filter": metadata_filter,
            "task": "summarization"
        }
    }
    
    try:
        final_state = app_graph.invoke(initial_state)
        formatted_output = final_state.get("metadata", {})
        if not formatted_output:
            raise HTTPException(status_code=500, detail="Malformed summarization result.")
        return formatted_output
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization workflow failed: {str(e)}")
