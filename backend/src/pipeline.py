"""
Main RAG Pipeline for Construction RAG System
==============================================

This module brings together all components into a unified pipeline.
It's the main entry point for running the RAG system.

Pipeline Architecture:
---------------------

User Query
    │
    ▼
┌───────────────┐
│ Query Router  │ ──────────────────────────────┐
└───────────────┘                                │
    │                                           │
    │ (simple)                          (complex/agent)
    ▼                                           │
┌───────────────┐                               ▼
│  Embeddings   │                      ┌───────────────┐
└───────────────┘                      │    Agent      │
    │                                  └───────────────┘
    ▼                                           │
┌───────────────┐                               │
│ Vector Search │                               │
└───────────────┘                               │
    │                                           │
    ▼                                           ▼
┌───────────────┐                      ┌───────────────┐
│   Retrieval   │                      │ Tool Execution│
└───────────────┘                      └───────────────┘
    │                                           │
    ▼                                           │
┌───────────────┐                               │
│   Generator   │ ◄─────────────────────────────┘
└───────────────┘
    │
    ▼
Response

What This Module Provides:
-------------------------
1. RAGPipeline: Basic RAG (retrieve -> generate)
2. AdvancedRAGPipeline: With query routing and agent mode
3. Document indexing utilities
4. Query history and caching
"""

import time
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from .config import settings
from .ingestion import (
    Document, Chunk, load_documents,
    ConstructionDocumentCleaner, HybridChunker, ChunkingPipeline
)
from .embeddings import get_embedding_model, BaseEmbeddingModel
from .vectordb import get_vector_store, BaseVectorStore, SearchResult
from .retrieval import Retriever, RetrievalResult, create_retriever
from .llm import RAGGenerator, LLMResponse, OpenAILLM
from .agents import ConstructionRAGAgent, QueryRouter, AgentResponse
from .evaluation import RAGEvaluator


@dataclass
class RAGResponse:
    """
    Complete RAG response with all metadata.
    
    This is the final output format returned to users.
    """
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    mode_used: str  # "simple", "agent"
    retrieval_results: List[Dict[str, Any]]
    processing_time_ms: float
    token_usage: Dict[str, int]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to API-friendly dictionary."""
        return {
            "answer": self.answer,
            "sources": self.sources,
            "confidence": self.confidence,
            "mode_used": self.mode_used,
            "retrieval_results": self.retrieval_results,
            "processing_time_ms": self.processing_time_ms,
            "token_usage": self.token_usage,
            "metadata": self.metadata
        }


class RAGPipeline:
    """
    Core RAG Pipeline.
    
    This is the standard retrieve-then-generate pipeline
    that forms the foundation of the system.
    """
    
    def __init__(
        self,
        embedding_model: Optional[BaseEmbeddingModel] = None,
        vector_store: Optional[BaseVectorStore] = None,
        retriever: Optional[Retriever] = None,
        generator: Optional[RAGGenerator] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            embedding_model: Model for embeddings
            vector_store: Vector database
            retriever: Retrieval component
            generator: LLM generator
            top_k: Number of chunks to retrieve
            similarity_threshold: Minimum similarity score
        """
        logger.info("Initializing RAG Pipeline...")
        
        self.embedding_model = embedding_model or get_embedding_model()
        self.vector_store = vector_store or get_vector_store()
        
        self.retriever = retriever or Retriever(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
        self.generator = generator or RAGGenerator()
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        # Query history for analytics
        self.query_history: List[Dict[str, Any]] = []
        
        logger.info("RAG Pipeline initialized successfully")
    
    def index_documents(
        self,
        source_path: str,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        clear_existing: bool = False
    ) -> Dict[str, Any]:
        """
        Index documents from a path.
        
        This is the data ingestion step that populates the vector store.
        
        Args:
            source_path: Path to documents (file or directory)
            chunk_size: Size of chunks in tokens
            chunk_overlap: Overlap between chunks
            clear_existing: Whether to clear existing data
        
        Returns:
            Indexing statistics
        """
        logger.info(f"Starting document indexing from: {source_path}")
        start_time = time.time()
        
        # Clear existing if requested
        if clear_existing:
            logger.warning("Clearing existing vector store data")
            self.vector_store.clear()
        
        # Load documents
        documents = load_documents(source_path)
        logger.info(f"Loaded {len(documents)} documents")
        
        # Clean documents
        cleaner = ConstructionDocumentCleaner()
        for doc in documents:
            doc.content = cleaner.clean(doc.content)
        
        # Chunk documents
        chunker = HybridChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        pipeline = ChunkingPipeline(chunker=chunker)
        
        all_chunks = []
        for doc in documents:
            chunks = pipeline.process_document(doc)
            all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks")
        
        # Generate embeddings
        chunk_texts = [chunk.content for chunk in all_chunks]
        embeddings = self.embedding_model.embed(chunk_texts)
        
        logger.info(f"Generated embeddings: {embeddings.shape}")
        
        # Add to vector store
        self.vector_store.add_chunks(all_chunks, embeddings)
        
        elapsed_time = time.time() - start_time
        
        stats = {
            "documents_processed": len(documents),
            "chunks_created": len(all_chunks),
            "total_vectors": self.vector_store.count(),
            "processing_time_seconds": elapsed_time,
            "avg_chunk_tokens": sum(c.token_count for c in all_chunks) / len(all_chunks) if all_chunks else 0
        }
        
        logger.info(f"Indexing complete: {stats}")
        
        return stats
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> RAGResponse:
        """
        Execute a RAG query.
        
        This is the main entry point for asking questions.
        
        Args:
            question: User's question
            top_k: Override default top_k
            filter_dict: Metadata filters
        
        Returns:
            Complete RAG response
        """
        start_time = time.time()
        
        logger.info(f"Processing query: {question[:50]}...")
        
        # Step 1: Retrieve relevant chunks
        retrieval_results = self.retriever.retrieve(
            query=question,
            top_k=top_k or self.top_k,
            filter_dict=filter_dict
        )
        
        logger.debug(f"Retrieved {len(retrieval_results)} chunks")
        
        # Step 2: Generate answer
        llm_response = self.generator.generate(
            question=question,
            retrieval_results=retrieval_results
        )
        
        # Build response
        sources = []
        for result in retrieval_results:
            sources.append({
                "source": result.source,
                "score": result.score,
                "preview": result.content[:200]
            })
        
        processing_time = (time.time() - start_time) * 1000
        
        response = RAGResponse(
            answer=llm_response.answer,
            sources=sources,
            confidence=llm_response.confidence,
            mode_used="simple",
            retrieval_results=[r.to_dict() for r in retrieval_results],
            processing_time_ms=processing_time,
            token_usage={
                "prompt": llm_response.prompt_tokens,
                "completion": llm_response.completion_tokens,
                "total": llm_response.total_tokens
            }
        )
        
        # Track query history
        self.query_history.append({
            "question": question,
            "answer": llm_response.answer,
            "sources": [r.source for r in retrieval_results],
            "timestamp": time.time(),
            "processing_time_ms": processing_time
        })
        
        logger.info(f"Query processed in {processing_time:.0f}ms")
        
        return response
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "total_vectors": self.vector_store.count(),
            "queries_processed": len(self.query_history),
            "embedding_model": self.embedding_model.model_name,
            "embedding_dimension": self.embedding_model.dimension,
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold
        }


class AdvancedRAGPipeline(RAGPipeline):
    """
    Advanced RAG Pipeline with agentic capabilities.
    
    Extends the basic pipeline with:
    - Query routing (simple vs. agent mode)
    - Multi-step reasoning
    - Dynamic tool usage
    """
    
    def __init__(
        self,
        enable_agent_mode: bool = True,
        **kwargs
    ):
        """
        Initialize advanced RAG pipeline.
        
        Args:
            enable_agent_mode: Whether to enable agent for complex queries
            **kwargs: Arguments for base RAGPipeline
        """
        super().__init__(**kwargs)
        
        self.enable_agent_mode = enable_agent_mode
        self.router = QueryRouter() if enable_agent_mode else None
        self.agent = ConstructionRAGAgent(
            retriever=self.retriever,
            llm=OpenAILLM(),
            max_iterations=settings.max_agent_iterations
        ) if enable_agent_mode else None
        
        logger.info(f"Advanced RAG Pipeline initialized (agent mode: {enable_agent_mode})")
    
    def query(
        self,
        question: str,
        force_mode: Optional[str] = None,
        **kwargs
    ) -> RAGResponse:
        """
        Execute query with automatic mode selection.
        
        Args:
            question: User's question
            force_mode: Force "simple" or "agent" mode
            **kwargs: Additional arguments
        
        Returns:
            Complete RAG response
        """
        start_time = time.time()
        
        # Determine mode
        if force_mode:
            mode = force_mode
        elif self.enable_agent_mode and self.router:
            mode = self.router.route(question)
        else:
            mode = "simple"
        
        logger.info(f"Query mode: {mode}")
        
        if mode == "agent" and self.agent:
            # Use agent mode
            agent_response = self.agent.run(question)
            
            processing_time = (time.time() - start_time) * 1000
            
            return RAGResponse(
                answer=agent_response.answer,
                sources=[{"source": s} for s in agent_response.sources],
                confidence=0.8,  # Agent doesn't provide confidence yet
                mode_used="agent",
                retrieval_results=[],  # Agent manages its own retrieval
                processing_time_ms=processing_time,
                token_usage={"total": 0},  # TODO: Track agent tokens
                metadata={
                    "agent_steps": agent_response.total_steps,
                    "retrieval_used": agent_response.retrieval_used
                }
            )
        else:
            # Use simple RAG mode
            return super().query(question, **kwargs)


# Convenience function for quick setup
def create_rag_pipeline(
    advanced: bool = True,
    **kwargs
) -> RAGPipeline:
    """
    Create a RAG pipeline.
    
    Args:
        advanced: Use AdvancedRAGPipeline with agent mode
        **kwargs: Pipeline configuration
    
    Returns:
        Configured RAG pipeline
    """
    if advanced:
        return AdvancedRAGPipeline(**kwargs)
    else:
        return RAGPipeline(**kwargs)


if __name__ == "__main__":
    print("=" * 60)
    print("RAG Pipeline Demo")
    print("=" * 60)
    
    print("\nThis module provides the main RAG pipeline.")
    print("\nUsage example:")
    print("""
    from pipeline import create_rag_pipeline
    
    # Create pipeline
    rag = create_rag_pipeline(advanced=True)
    
    # Index documents
    stats = rag.index_documents("./data/sample_docs/")
    print(f"Indexed {stats['documents_processed']} documents")
    
    # Query
    response = rag.query("What PPE is required on site?")
    print(response.answer)
    print(f"Sources: {response.sources}")
    """)
    
    # Show stats if vector store has data
    try:
        rag = RAGPipeline()
        stats = rag.get_stats()
        print(f"\nCurrent Pipeline Stats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"\nCouldn't load pipeline: {e}")
