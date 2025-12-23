"""
Ingestion Module - __init__.py
===============================

This module provides all document ingestion functionality:
- Document loading (PDF, DOCX, TXT)
- Text cleaning and normalization
- Semantic chunking with overlap
"""

from .document_loader import (
    Document,
    BaseDocumentLoader,
    TextDocumentLoader,
    PDFDocumentLoader,
    DOCXDocumentLoader,
    DocumentLoaderFactory,
    DirectoryLoader,
    load_documents
)

from .text_cleaner import (
    TextCleaner,
    ConstructionDocumentCleaner,
    CleaningStats,
    clean_text
)

from .chunker import (
    Chunk,
    BaseChunker,
    TokenBasedChunker,
    SemanticChunker,
    HybridChunker,
    ChunkingPipeline,
    create_chunker
)

__all__ = [
    # Document Loading
    "Document",
    "BaseDocumentLoader",
    "TextDocumentLoader",
    "PDFDocumentLoader",
    "DOCXDocumentLoader",
    "DocumentLoaderFactory",
    "DirectoryLoader",
    "load_documents",
    
    # Text Cleaning
    "TextCleaner",
    "ConstructionDocumentCleaner",
    "CleaningStats",
    "clean_text",
    
    # Chunking
    "Chunk",
    "BaseChunker",
    "TokenBasedChunker",
    "SemanticChunker",
    "HybridChunker",
    "ChunkingPipeline",
    "create_chunker"
]
