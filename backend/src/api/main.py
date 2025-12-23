"""
FastAPI Application for Construction RAG System
================================================

This module provides the REST API for the RAG system.
It exposes endpoints for querying, document management, and system info.

API Design Principles:
---------------------
1. Clear endpoint naming (/api/v1/...)
2. Consistent response format
3. Proper error handling
4. Comprehensive documentation (OpenAPI/Swagger)
5. CORS support for frontend integration
"""

import os
import time
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger

from .config import settings
from .pipeline import RAGPipeline, AdvancedRAGPipeline, create_rag_pipeline


# ============================================
# Pydantic Models for API
# ============================================

class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    question: str = Field(..., description="The question to ask", min_length=1)
    top_k: int = Field(default=5, description="Number of chunks to retrieve", ge=1, le=20)
    use_agent: Optional[bool] = Field(default=None, description="Force agent mode")
    filter_source: Optional[str] = Field(default=None, description="Filter by source document")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What PPE is required on construction sites?",
                "top_k": 5,
                "use_agent": None
            }
        }


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    mode_used: str
    processing_time_ms: float
    token_usage: Dict[str, int]
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Required PPE includes hard hats, safety glasses, and steel-toed boots.",
                "sources": [{"source": "safety_manual.txt", "score": 0.92}],
                "confidence": 0.85,
                "mode_used": "simple",
                "processing_time_ms": 1250.5,
                "token_usage": {"prompt": 500, "completion": 150, "total": 650}
            }
        }


class IndexRequest(BaseModel):
    """Request model for document indexing."""
    source_path: str = Field(..., description="Path to documents")
    chunk_size: int = Field(default=512, ge=64, le=2048)
    chunk_overlap: int = Field(default=50, ge=0, le=256)
    clear_existing: bool = Field(default=False)


class IndexResponse(BaseModel):
    """Response model for indexing operation."""
    status: str
    documents_processed: int
    chunks_created: int
    total_vectors: int
    processing_time_seconds: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str
    vector_count: int


class StatsResponse(BaseModel):
    """System statistics response."""
    total_vectors: int
    queries_processed: int
    embedding_model: str
    embedding_dimension: int
    top_k: int
    similarity_threshold: float


# ============================================
# Application Lifecycle
# ============================================

# Global pipeline instance
rag_pipeline: Optional[AdvancedRAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    
    Initializes the RAG pipeline on startup.
    """
    global rag_pipeline
    
    logger.info("Starting Construction RAG API...")
    
    try:
        # Initialize the pipeline
        rag_pipeline = create_rag_pipeline(advanced=True)
        logger.info("RAG pipeline initialized successfully")
        
        # Check if we have indexed data
        stats = rag_pipeline.get_stats()
        if stats["total_vectors"] == 0:
            logger.warning("Vector store is empty. Run /api/v1/index to add documents.")
    
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        # Continue anyway - indexing endpoint can be used to set up
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down Construction RAG API...")


# ============================================
# FastAPI Application
# ============================================

app = FastAPI(
    title="Construction RAG API",
    description="""
    A production-grade Retrieval Augmented Generation (RAG) system
    for construction industry documents.
    
    ## Features
    
    - **Document Search**: Semantic search across construction documents
    - **Q&A**: Ask questions and get grounded answers with citations
    - **Agent Mode**: Complex multi-step reasoning for advanced queries
    - **Document Indexing**: Add new documents to the knowledge base
    
    ## Document Types Supported
    
    - Safety manuals
    - Contracts and agreements
    - Inspection reports
    - Specifications
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# API Endpoints
# ============================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Construction RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.
    
    Returns system status and basic metrics.
    """
    global rag_pipeline
    
    vector_count = 0
    if rag_pipeline:
        try:
            stats = rag_pipeline.get_stats()
            vector_count = stats.get("total_vectors", 0)
        except:
            pass
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        vector_count=vector_count
    )


@app.get("/api/v1/stats", response_model=StatsResponse, tags=["System"])
async def get_stats():
    """
    Get system statistics.
    
    Returns detailed system metrics and configuration.
    """
    global rag_pipeline
    
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    stats = rag_pipeline.get_stats()
    
    return StatsResponse(**stats)


@app.post("/api/v1/query", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest):
    """
    Query the RAG system.
    
    Ask a question and get an answer grounded in construction documents.
    
    The system will automatically choose between simple RAG and agent mode
    based on query complexity, unless `use_agent` is explicitly set.
    """
    global rag_pipeline
    
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    # Check if we have indexed data
    stats = rag_pipeline.get_stats()
    if stats["total_vectors"] == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed. Please run /api/v1/index first."
        )
    
    try:
        # Build filter if source specified
        filter_dict = None
        if request.filter_source:
            filter_dict = {"source": {"$contains": request.filter_source}}
        
        # Determine mode
        force_mode = None
        if request.use_agent is not None:
            force_mode = "agent" if request.use_agent else "simple"
        
        # Execute query
        response = rag_pipeline.query(
            question=request.question,
            top_k=request.top_k,
            filter_dict=filter_dict,
            force_mode=force_mode
        )
        
        return QueryResponse(
            answer=response.answer,
            sources=response.sources,
            confidence=response.confidence,
            mode_used=response.mode_used,
            processing_time_ms=response.processing_time_ms,
            token_usage=response.token_usage
        )
        
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/query/stream", tags=["Query"])
async def query_stream(request: QueryRequest):
    """
    Stream a RAG query response.
    
    Returns a streaming response for real-time UI updates.
    """
    global rag_pipeline
    
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    async def generate():
        try:
            # Get retrieval results
            retrieval_results = rag_pipeline.retriever.retrieve(
                query=request.question,
                top_k=request.top_k
            )
            
            # Stream generation
            for token in rag_pipeline.generator.generate_stream(
                question=request.question,
                retrieval_results=retrieval_results
            ):
                yield f"data: {token}\n\n"
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


@app.post("/api/v1/index", response_model=IndexResponse, tags=["Documents"])
async def index_documents(request: IndexRequest, background_tasks: BackgroundTasks):
    """
    Index documents into the RAG system.
    
    Loads documents from the specified path, chunks them, generates embeddings,
    and stores them in the vector database.
    
    This operation can take time for large document sets.
    """
    global rag_pipeline
    
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    # Validate path exists
    from pathlib import Path
    if not Path(request.source_path).exists():
        raise HTTPException(status_code=400, detail=f"Path not found: {request.source_path}")
    
    try:
        stats = rag_pipeline.index_documents(
            source_path=request.source_path,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            clear_existing=request.clear_existing
        )
        
        return IndexResponse(
            status="completed",
            documents_processed=stats["documents_processed"],
            chunks_created=stats["chunks_created"],
            total_vectors=stats["total_vectors"],
            processing_time_seconds=stats["processing_time_seconds"]
        )
        
    except Exception as e:
        logger.error(f"Indexing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/sources", tags=["Documents"])
async def list_sources():
    """
    List all indexed document sources.
    
    Returns a list of unique source documents in the index.
    """
    global rag_pipeline
    
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    # This would require iterating the vector store
    # For now, return placeholder
    return {
        "sources": [
            "construction_safety_manual.txt",
            "subcontractor_agreement.txt",
            "site_inspection_report.txt"
        ],
        "total_vectors": rag_pipeline.get_stats()["total_vectors"]
    }


@app.get("/api/v1/query/history", tags=["Query"])
async def query_history(limit: int = Query(default=10, le=100)):
    """
    Get recent query history.
    
    Returns the most recent queries processed by the system.
    """
    global rag_pipeline
    
    if not rag_pipeline:
        return {"history": []}
    
    history = rag_pipeline.query_history[-limit:]
    
    return {"history": history}


@app.delete("/api/v1/index", tags=["Documents"])
async def clear_index():
    """
    Clear all indexed documents.
    
    This will delete all vectors from the database.
    Use with caution!
    """
    global rag_pipeline
    
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        rag_pipeline.vector_store.clear()
        return {"status": "cleared", "vectors_remaining": 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Main Entry Point
# ============================================

def start_server():
    """Start the FastAPI server."""
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    start_server()
