"""
Embeddings Module - __init__.py
================================

This module provides embedding generation functionality:
- Sentence Transformer embeddings (local)
- OpenAI embeddings (API-based)
- Embedding analysis utilities
"""

from .embedding_model import (
    BaseEmbeddingModel,
    SentenceTransformerEmbedding,
    OpenAIEmbedding,
    EmbeddingModelFactory,
    EmbeddingAnalyzer,
    get_embedding_model
)

__all__ = [
    "BaseEmbeddingModel",
    "SentenceTransformerEmbedding",
    "OpenAIEmbedding",
    "EmbeddingModelFactory",
    "EmbeddingAnalyzer",
    "get_embedding_model"
]
