"""
Embedding Model Module for Construction RAG System
===================================================

This module handles text embedding generation for semantic search.
Embeddings transform text into dense vector representations that capture
semantic meaning.

What are Embeddings?
-------------------
Embeddings are numerical representations of text in high-dimensional space.
Similar texts have similar vectors (close in vector space).

Example:
- "fall protection" and "safety harness" -> close vectors
- "fall protection" and "concrete pour" -> distant vectors

This enables semantic search: finding relevant documents based on meaning,
not just keyword matching.

ML Fundamentals Connection:
---------------------------
Embeddings are learned representations, like features in traditional ML:

1. Tokenization: Similar to feature extraction
   - Text -> Tokens (like image -> pixels)
   
2. Embedding Layer: Like a learned lookup table
   - Token ID -> Dense vector (learned during training)
   
3. Pooling: Aggregating token embeddings
   - Mean pooling: Average of all token vectors
   - [CLS] pooling: Use special classification token
   - This is like global average pooling in CNNs

4. Similarity Computation:
   - Cosine similarity: Most common, scale-invariant
   - Dot product: Faster, but magnitude-sensitive
   - Euclidean distance: Less common for text

Model Selection Rationale:
-------------------------
We evaluate embedding models on:
1. Domain fit: Does it understand construction terminology?
2. Speed: Inference time for thousands of chunks
3. Dimension: Trade-off between expressiveness and storage
4. Quality: Retrieval accuracy on construction queries

Our default (all-MiniLM-L6-v2):
- 384 dimensions
- 22M parameters
- 14k words/sec on CPU
- Good general-purpose performance
"""

import os
from abc import ABC, abstractmethod
from typing import List, Union, Optional
from functools import lru_cache
import hashlib

import numpy as np
from loguru import logger

from ..config import settings


class BaseEmbeddingModel(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text(s)."""
        pass
    
    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query (may differ from documents)."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return model name for logging."""
        pass


class SentenceTransformerEmbedding(BaseEmbeddingModel):
    """
    Embedding model using Sentence Transformers library.
    
    Sentence Transformers provides state-of-the-art embeddings
    trained specifically for semantic similarity tasks.
    
    How it works:
    1. Text -> Tokens (using model's tokenizer)
    2. Tokens -> Contextualized embeddings (transformer layers)
    3. Token embeddings -> Sentence embedding (pooling)
    
    Models available:
    - all-MiniLM-L6-v2: Fast, 384 dim (default)
    - all-mpnet-base-v2: Better quality, 768 dim
    - multi-qa-MiniLM-L6-cos-v1: Optimized for QA
    - all-distilroberta-v1: Good for diverse domains
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache_embeddings: bool = True
    ):
        """
        Initialize Sentence Transformer embedding model.
        
        Args:
            model_name: HuggingFace model name or path
            device: 'cpu', 'cuda', or None for auto-detect
            cache_embeddings: Cache embeddings for repeated texts
        """
        self._model_name = model_name
        self.cache_embeddings = cache_embeddings
        self._cache = {}
        
        logger.info(f"Loading embedding model: {model_name}")
        
        try:
            from sentence_transformers import SentenceTransformer
            
            # Auto-detect device if not specified
            if device is None:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.model = SentenceTransformer(model_name, device=device)
            self._dimension = self.model.get_sentence_embedding_dimension()
            self.device = device
            
            logger.info(
                f"Embedding model loaded: dim={self._dimension}, device={device}"
            )
            
        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
            raise
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts
        
        Returns:
            numpy array of shape (n_texts, dimension)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        # Check cache for each text
        uncached_texts = []
        uncached_indices = []
        cached_embeddings = {}
        
        if self.cache_embeddings:
            for i, text in enumerate(texts):
                key = self._get_cache_key(text)
                if key in self._cache:
                    cached_embeddings[i] = self._cache[key]
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # Embed uncached texts
        if uncached_texts:
            new_embeddings = self.model.encode(
                uncached_texts,
                show_progress_bar=len(uncached_texts) > 100,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            
            # Update cache
            if self.cache_embeddings:
                for i, text in enumerate(uncached_texts):
                    key = self._get_cache_key(text)
                    self._cache[key] = new_embeddings[i]
                    cached_embeddings[uncached_indices[i]] = new_embeddings[i]
            else:
                for i, idx in enumerate(uncached_indices):
                    cached_embeddings[idx] = new_embeddings[i]
        
        # Combine results in correct order
        result = np.array([cached_embeddings[i] for i in range(len(texts))])
        
        return result
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query.
        
        For symmetric models (like all-MiniLM), this is the same as embed().
        For asymmetric models, query embedding may differ from document embedding.
        """
        embedding = self.embed(query)
        return embedding[0] if embedding.ndim > 1 else embedding
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache = {}
        logger.info("Embedding cache cleared")


class OpenAIEmbedding(BaseEmbeddingModel):
    """
    OpenAI Embedding API wrapper.
    
    OpenAI's embedding models are state-of-the-art but require API calls.
    
    Models:
    - text-embedding-3-small: 1536 dim, $0.00002/1K tokens
    - text-embedding-3-large: 3072 dim, $0.00013/1K tokens
    - text-embedding-ada-002: 1536 dim (legacy)
    
    Trade-offs:
    + Best quality for general text
    + Handles diverse vocabulary well
    - Requires API key and internet
    - Costs money at scale
    - Adds latency
    """
    
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        batch_size: int = 100
    ):
        """
        Initialize OpenAI embedding model.
        
        Args:
            model_name: OpenAI embedding model name
            api_key: OpenAI API key (or from env OPENAI_API_KEY)
            batch_size: Number of texts to embed per API call
        """
        self._model_name = model_name
        self.batch_size = batch_size
        
        try:
            from openai import OpenAI
            
            self.client = OpenAI(api_key=api_key or settings.openai_api_key)
            
            # Dimension maps for OpenAI models
            self._dimensions = {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536
            }
            self._dimension = self._dimensions.get(model_name, 1536)
            
            logger.info(f"OpenAI embedding initialized: {model_name}")
            
        except ImportError:
            logger.error("openai not installed. Run: pip install openai")
            raise
    
    @property
    def dimension(self) -> int:
        return self._dimension
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings using OpenAI API."""
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            
            response = self.client.embeddings.create(
                model=self._model_name,
                input=batch
            )
            
            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return np.array(all_embeddings)
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query."""
        embedding = self.embed(query)
        return embedding[0]


class EmbeddingModelFactory:
    """
    Factory for creating embedding models.
    
    Centralizes model creation and configuration.
    """
    
    @staticmethod
    def create(
        model_type: str = "sentence_transformer",
        model_name: Optional[str] = None,
        **kwargs
    ) -> BaseEmbeddingModel:
        """
        Create an embedding model.
        
        Args:
            model_type: "sentence_transformer" or "openai"
            model_name: Specific model name
            **kwargs: Additional arguments for model
        
        Returns:
            Configured embedding model
        """
        if model_type == "sentence_transformer":
            name = model_name or settings.embedding_model
            return SentenceTransformerEmbedding(model_name=name, **kwargs)
        
        elif model_type == "openai":
            name = model_name or settings.openai_embedding_model
            return OpenAIEmbedding(model_name=name, **kwargs)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")


# Cached embedding model instance
@lru_cache(maxsize=1)
def get_embedding_model() -> BaseEmbeddingModel:
    """
    Get cached embedding model instance.
    
    Uses lru_cache to ensure model is loaded only once.
    This is important because loading models is expensive.
    """
    if settings.use_openai_embeddings:
        return EmbeddingModelFactory.create("openai")
    else:
        return EmbeddingModelFactory.create("sentence_transformer")


class EmbeddingAnalyzer:
    """
    Utility class for analyzing embeddings.
    
    Useful for:
    - Understanding embedding space
    - Debugging retrieval issues
    - Evaluating embedding quality
    """
    
    def __init__(self, model: BaseEmbeddingModel):
        self.model = model
    
    def compute_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = self.model.embed_query(text1)
        emb2 = self.model.embed_query(text2)
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)
    
    def find_nearest(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find nearest texts to a query.
        
        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of nearest to return
        
        Returns:
            List of (text, similarity) tuples
        """
        query_emb = self.model.embed_query(query)
        candidate_embs = self.model.embed(candidates)
        
        # Compute all similarities
        similarities = np.dot(candidate_embs, query_emb)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [(candidates[i], float(similarities[i])) for i in top_indices]
    
    def analyze_clustering(
        self,
        texts: List[str]
    ) -> dict:
        """
        Analyze how texts cluster in embedding space.
        
        Useful for understanding document similarity patterns.
        """
        embeddings = self.model.embed(texts)
        
        # Compute pairwise similarities
        n = len(texts)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                similarity_matrix[i, j] = np.dot(
                    embeddings[i], embeddings[j]
                )
        
        # Average similarity
        avg_similarity = float(similarity_matrix.sum() - n) / (n * n - n)
        
        # Min and max non-diagonal
        mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask, False)
        
        min_similarity = float(similarity_matrix[mask].min())
        max_similarity = float(similarity_matrix[mask].max())
        
        return {
            "num_texts": n,
            "embedding_dimension": embeddings.shape[1],
            "avg_similarity": avg_similarity,
            "min_similarity": min_similarity,
            "max_similarity": max_similarity
        }


if __name__ == "__main__":
    # Demo and testing
    print("=" * 60)
    print("Testing Embedding Models")
    print("=" * 60)
    
    # Test Sentence Transformer
    model = SentenceTransformerEmbedding()
    
    texts = [
        "Fall protection is required when working at heights above 6 feet.",
        "Workers must wear safety harnesses at elevated locations.",
        "Concrete should cure for at least 28 days.",
        "PPE includes hard hats, safety glasses, and steel-toed boots."
    ]
    
    print(f"\nModel: {model.model_name}")
    print(f"Dimension: {model.dimension}")
    
    embeddings = model.embed(texts)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Test similarity
    analyzer = EmbeddingAnalyzer(model)
    
    print("\n" + "-" * 40)
    print("Similarity Analysis")
    print("-" * 40)
    
    # Similar texts
    sim1 = analyzer.compute_similarity(texts[0], texts[1])
    print(f"\nFall protection vs Safety harness: {sim1:.4f}")
    
    # Different topics
    sim2 = analyzer.compute_similarity(texts[0], texts[2])
    print(f"Fall protection vs Concrete curing: {sim2:.4f}")
    
    # Find nearest
    print("\n" + "-" * 40)
    print("Finding nearest to: 'What safety equipment do I need?'")
    print("-" * 40)
    
    query = "What safety equipment do I need?"
    nearest = analyzer.find_nearest(query, texts, top_k=3)
    
    for text, similarity in nearest:
        print(f"\n  [{similarity:.4f}] {text}")
