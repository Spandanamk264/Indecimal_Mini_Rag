"""
VectorDB Module - __init__.py
==============================

This module provides vector database functionality:
- ChromaDB (default, local)
- FAISS (high-performance)
- Pinecone (cloud-based)
"""

from .vector_store import (
    SearchResult,
    BaseVectorStore,
    ChromaVectorStore,
    FAISSVectorStore,
    PineconeVectorStore,
    VectorStoreFactory,
    get_vector_store
)

__all__ = [
    "SearchResult",
    "BaseVectorStore",
    "ChromaVectorStore",
    "FAISSVectorStore",
    "PineconeVectorStore",
    "VectorStoreFactory",
    "get_vector_store"
]
