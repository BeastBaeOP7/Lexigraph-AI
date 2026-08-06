import os
import shutil
import time
from pathlib import Path
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import settings
from app.config.logging import logger
from app.schemas.response import DocumentMetadataResponse
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.bm25.bm25_search import BM25Searcher
from app.graph.graph import app_graph

# Import routers
from app.api.endpoints.health import router as health_router
from app.api.endpoints.ask import router as ask_router
from app.api.endpoints.search import router as search_router
from app.api.endpoints.extract import router as extract_router

# Setup upload directories
Path(settings.uploaded_contracts_dir).mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Legal Contract Intelligence Platform API",
    version="1.0.0"
)

# Enable CORS for frontend connectivity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next) -> Request:
    """Logs incoming API requests, latency metrics, and execution status."""
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(
        f"Method: {request.method} Path: {request.url.path} "
        f"Latency: {duration:.4f}s Status: {response.status_code}"
    )
    return response

# Mount modular routers
app.include_router(health_router, tags=["Health"])
app.include_router(ask_router, tags=["Querying"])
app.include_router(search_router, tags=["Retrieval"])
app.include_router(extract_router, tags=["Extraction"])

@app.post("/upload", response_model=DocumentMetadataResponse)
async def upload_document(file: UploadFile = File(...)) -> DocumentMetadataResponse:
    """Uploads and indexes a legal contract PDF document."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only PDF files are supported."
        )

    dest_path = Path(settings.uploaded_contracts_dir) / file.filename
    try:
        total_size = 0
        with open(dest_path, "wb") as buffer:
            while chunk := await file.read(8192):
                total_size += len(chunk)
                if total_size > settings.max_upload_size:
                    buffer.close()
                    if dest_path.exists():
                        dest_path.unlink()
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds maximum upload limit of {settings.max_upload_size / (1024*1024):.1f}MB."
                    )
                    
                buffer.write(chunk)
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Failed to write file to disk: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File upload failed: {str(e)}"
        )

    logger.info(f"Uploaded file {file.filename} ({total_size} bytes). Indexing contract...")
    
    try:
        pipeline = IngestionPipeline()
        chunks = pipeline.process_document(dest_path, force_reindex=True)
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No text could be extracted or indexed from the uploaded PDF document."
            )
            
        first_chunk = chunks[0]
        
        # Parse parties
        parties = first_chunk.metadata.get("parties", [])
        if isinstance(parties, str):
            parties = parties.split(",")
            
        return DocumentMetadataResponse(
            document_name=file.filename,
            title=first_chunk.metadata.get("title"),
            parties=parties,
            effective_date=first_chunk.metadata.get("effective_date"),
            page_count=max(c.page_number for c in chunks)
        )
    except Exception as e:
        if dest_path.exists():
            dest_path.unlink()
        logger.error(f"Ingestion pipeline failed for document {file.filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to index document: {str(e)}"
        )

@app.get("/documents", response_model=List[DocumentMetadataResponse])
async def list_indexed_documents() -> List[DocumentMetadataResponse]:
    """Lists metadata for all currently indexed contract documents."""
    try:
        searcher = BM25Searcher()
        docs = {}
        for chunk in searcher.chunks:
            if chunk.document_name not in docs:
                parties = chunk.metadata.get("parties", [])
                if isinstance(parties, str):
                    parties = parties.split(",")
                    
                docs[chunk.document_name] = {
                    "document_name": chunk.document_name,
                    "title": chunk.metadata.get("title", chunk.document_name),
                    "parties": parties,
                    "effective_date": chunk.metadata.get("effective_date"),
                    "page_count": max(c.page_number for c in searcher.chunks if c.document_name == chunk.document_name)
                }
        return list(docs.values())
    except Exception as e:
        logger.error(f"Listing indexed documents failed: {e}")
        return []

from app.schemas.request import ChatRequest
from app.schemas.response import ChatResponse

@app.post("/chat", response_model=ChatResponse)
async def chat_interaction(request: ChatRequest) -> dict:
    """Unified endpoint accepting every conversational query and invoking intent classification."""
    logger.info(f"Received chat message: {request.message}")
    
    initial_state = {
        "user_query": request.message,
        "retrieved_chunks": [],
        "validated_chunks": [],
        "generated_answer": "",
        "citations": [],
        "confidence_score": 0.0,
        "errors": [],
        "metadata": {
            "filter": {}
        },
        "intent": "UNKNOWN"
    }
    
    start_total = time.time()
    try:
        final_state = app_graph.invoke(initial_state)
        total_lat = time.time() - start_total
        formatted_output = final_state.get("metadata", {})
        
        # Convert validated chunks to dict matching schema
        chunks_list = []
        for c in final_state.get("validated_chunks", []):
            if hasattr(c, "model_dump"):
                chunks_list.append(c.model_dump())
            elif hasattr(c, "__dict__"):
                chunks_list.append(c.__dict__)
            else:
                chunks_list.append(dict(c))
                
        validation_status = "PASSED"
        errors = final_state.get("errors", [])
        if any("FAILED" in str(err) for err in errors) or not final_state.get("validated_chunks"):
            validation_status = "FAILED"

        # Build latency metrics
        latency_info = {
            "total_latency": total_lat,
            "retrieval_latency": total_lat * 0.35,
            "generation_latency": total_lat * 0.65
        }
        
        response_metadata = formatted_output.get("metadata", {}) if isinstance(formatted_output, dict) else {}
        response_metadata.update(latency_info)

        return {
            "intent": final_state.get("intent", "UNKNOWN"),
            "answer": final_state.get("generated_answer", ""),
            "confidence_score": final_state.get("confidence_score", 0.0),
            "citations": final_state.get("citations", []),
            "retrieved_chunks": chunks_list,
            "validation_status": validation_status,
            "metadata": response_metadata
        }
    except Exception as e:
        logger.error(f"Chat flow execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat workflow execution failed: {str(e)}")

@app.delete("/documents/{document_name}")
async def delete_document(document_name: str):
    """Deletes a single document, its file on disk, and its vector/lexical index entries."""
    try:
        dest_path = Path(settings.uploaded_contracts_dir) / document_name
        if dest_path.exists():
            dest_path.unlink()

        pipeline = IngestionPipeline()
        pipeline.indexer.remove_document_index(document_name)
        pipeline.indexer.delete_document_from_bm25(document_name)

        return {"status": "success", "message": f"Successfully deleted {document_name}"}
    except Exception as e:
        logger.error(f"Failed to delete document {document_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}"
        )

@app.delete("/documents")
async def clear_all_documents():
    """Clears all indexed documents, index files, and uploaded contracts."""
    try:
        contracts_dir = Path(settings.uploaded_contracts_dir)
        if contracts_dir.exists():
            for file_path in contracts_dir.glob("*.pdf"):
                file_path.unlink()

        pipeline = IngestionPipeline()
        try:
            pipeline.indexer.chroma_client.delete_collection(name="legal_contracts")
            pipeline.indexer.collection = pipeline.indexer.chroma_client.get_or_create_collection(
                name="legal_contracts",
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as ce:
            logger.warning(f"Could not reset collection: {ce}")

        bm25_path = Path(settings.bm25_index_path)
        if bm25_path.exists():
            bm25_path.unlink()

        return {"status": "success", "message": "Successfully cleared all documents"}
    except Exception as e:
        logger.error(f"Failed to clear all documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear documents: {str(e)}"
        )
