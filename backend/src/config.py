"""
Configuration Management for Construction RAG System
=====================================================

This module handles all configuration settings using Pydantic Settings.
It provides type-safe access to environment variables with defaults.

Design Decision:
- Using Pydantic Settings for automatic environment variable parsing
- Centralized configuration prevents scattered hardcoded values
- Type hints enable IDE autocomplete and catch errors early

ML Fundamentals Connection:
- Configuration management is similar to hyperparameter management in ML
- Just as we tune learning_rate, batch_size, here we tune chunk_size, top_k
- This is a form of "system hyperparameters" that affect RAG performance
"""

import os
from pathlib import Path
from typing import Literal, Optional
from functools import lru_cache

from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Settings(BaseSettings):
    """
    Centralized settings for the Construction RAG System.
    
    These settings control every aspect of the RAG pipeline:
    - Embedding model selection
    - Vector database configuration
    - LLM inference parameters
    - Retrieval strategy parameters
    """
    
    # ==========================================
    # OpenAI Configuration
    # ==========================================
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key for LLM and embeddings"
    )
    openai_model: str = Field(
        default="gpt-4-turbo-preview",
        description="OpenAI model for response generation"
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model (if using OpenAI embeddings)"
    )
    
    # ==========================================
    # Vector Database Configuration
    # ==========================================
    vector_db_type: Literal["chroma", "faiss", "pinecone"] = Field(
        default="chroma",
        description="Which vector database to use"
    )
    
    # Pinecone Settings
    pinecone_api_key: Optional[str] = Field(default=None)
    pinecone_environment: Optional[str] = Field(default=None)
    pinecone_index_name: str = Field(default="construction-rag")
    
    # ChromaDB Settings
    chroma_persist_directory: str = Field(
        default="./data/chroma_db",
        description="Directory to persist ChromaDB data"
    )
    
    # FAISS Settings
    faiss_index_path: str = Field(
        default="./data/faiss_index",
        description="Path to save/load FAISS index"
    )
    
    # ==========================================
    # Embedding Configuration
    # ==========================================
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="""
        Embedding model selection - CRITICAL DESIGN DECISION:
        
        Options considered:
        1. all-MiniLM-L6-v2 (384 dim): Fast, good quality, low memory
        2. all-mpnet-base-v2 (768 dim): Better quality, slower
        3. OpenAI text-embedding-3-small: Best quality, API cost
        
        Trade-off Analysis:
        - MiniLM: 5x faster inference, good enough for most use cases
        - For 10K documents: MiniLM saves ~2GB memory vs mpnet
        - OpenAI: Best semantic understanding but adds latency + cost
        
        We use MiniLM as default for balance of speed and quality.
        """
    )
    embedding_dimension: int = Field(
        default=384,
        description="Dimension of embedding vectors (must match model)"
    )
    use_openai_embeddings: bool = Field(
        default=False,
        description="Whether to use OpenAI embeddings instead of local"
    )
    
    # ==========================================
    # Chunking Configuration
    # ==========================================
    chunk_size: int = Field(
        default=512,
        description="""
        Chunk size in tokens - affects retrieval granularity.
        
        ML Intuition (Bias-Variance Tradeoff):
        - Small chunks (128-256): Low bias, high variance
          * Pro: More precise retrieval
          * Con: Loses context, more chunks to search
        
        - Large chunks (1024+): High bias, low variance
          * Pro: More context per chunk
          * Con: May include irrelevant info, fewer chunks
        
        512 is our sweet spot for construction documents.
        """
    )
    chunk_overlap: int = Field(
        default=50,
        description="""
        Overlap between chunks to preserve context at boundaries.
        
        Design Rationale:
        - 0 overlap: Information at chunk boundaries may be lost
        - 50-100 overlap: Captures cross-boundary concepts
        - >100 overlap: Diminishing returns, increases storage
        
        10% overlap (50 tokens for 512 chunk) is industry standard.
        """
    )
    
    # ==========================================
    # Retrieval Configuration
    # ==========================================
    top_k_results: int = Field(
        default=5,
        description="""
        Number of chunks to retrieve for context.
        
        Trade-off:
        - More chunks = more context but risk of noise
        - Fewer chunks = cleaner context but may miss info
        
        5 chunks * 512 tokens = 2560 tokens base context
        Leaves room for prompt + response in 4K context window
        """
    )
    similarity_threshold: float = Field(
        default=0.7,
        description="""
        Minimum similarity score to include chunk.
        
        Interpretation:
        - 0.9+: Very high confidence, strict matching
        - 0.7-0.9: Good relevance, typical threshold
        - 0.5-0.7: Tangentially related
        - <0.5: Likely noise
        
        0.7 balances precision and recall for RAG.
        """
    )
    
    # ==========================================
    # LLM Configuration
    # ==========================================
    max_tokens: int = Field(
        default=2048,
        description="Maximum tokens for LLM response"
    )
    temperature: float = Field(
        default=0.1,
        description="""
        LLM temperature for response generation.
        
        For RAG systems, we want deterministic, factual responses:
        - 0.0: Fully deterministic (may cause repetition)
        - 0.1: Slight variation, mostly consistent (our choice)
        - 0.7+: Creative responses (bad for factual QA)
        """
    )
    context_window: int = Field(
        default=4096,
        description="Maximum context window size"
    )
    
    # ==========================================
    # Logging Configuration
    # ==========================================
    log_level: str = Field(default="INFO")
    log_file: str = Field(default="./logs/rag_system.log")
    
    # ==========================================
    # API Configuration
    # ==========================================
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        description="Comma-separated list of allowed CORS origins"
    )
    
    # ==========================================
    # Agentic Features
    # ==========================================
    enable_agent_mode: bool = Field(
        default=True,
        description="Enable multi-step reasoning agent"
    )
    max_agent_iterations: int = Field(
        default=5,
        description="Maximum reasoning steps for agent"
    )
    
    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins string into list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    @validator("embedding_dimension")
    def validate_embedding_dimension(cls, v, values):
        """Ensure embedding dimension matches model."""
        model = values.get("embedding_model", "")
        expected = {
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-mpnet-base-v2": 768,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        if model in expected and v != expected[model]:
            raise ValueError(
                f"Embedding dimension {v} doesn't match model {model}. "
                f"Expected {expected[model]}"
            )
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Using lru_cache ensures we only parse environment variables once.
    This is a common pattern for configuration in FastAPI applications.
    """
    return Settings()


# Export a default instance for convenience
settings = get_settings()
