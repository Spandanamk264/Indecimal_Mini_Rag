"""
Retrieval Module for Construction RAG System
=============================================

This module implements the retrieval pipeline that connects
queries to relevant document chunks.

Retrieval is the "R" in RAG - it's the critical bridge between
user questions and the knowledge in your documents.

Retrieval Strategies:
---------------------
1. Dense Retrieval (what we implement)
   - Convert query to embedding
   - Find nearest neighbors in vector space
   - Works well for semantic similarity

2. Sparse Retrieval (traditional)
   - Use TF-IDF or BM25 scoring
   - Works well for exact keyword matching

3. Hybrid Retrieval (best of both)
   - Combine dense and sparse results
   - Re-rank using cross-encoder

Key Concepts:
-------------
1. Query Embedding: Transform user question to vector

2. Semantic Search: Find chunks with similar meaning
   - "What PPE do I need?" matches "Personal Protective Equipment requirements"

3. Top-K Retrieval: Return K most similar chunks
   - Too few: Miss relevant information
   - Too many: Add noise to context

4. Context Filtering: Remove low-quality results
   - Similarity threshold
   - Metadata filtering
   - Deduplication

ML Fundamentals Connection:
---------------------------
Retrieval is a ranking problem (similar to recommendation systems):
- Query = user preference vector
- Documents = item vectors
- Similarity = relevance score

We can apply ML ranking techniques:
- Learning to rank (LTR)
- Cross-encoder reranking
- Query expansion
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
from loguru import logger

from ..config import settings
from ..embeddings import get_embedding_model, BaseEmbeddingModel
from ..vectordb import get_vector_store, BaseVectorStore, SearchResult


@dataclass
class RetrievalResult:
    """
    Enhanced retrieval result with additional context.
    
    Extends SearchResult with retrieval-specific metadata
    for debugging and quality assessment.
    """
    content: str
    score: float
    chunk_id: str
    source: str
    metadata: Dict[str, Any]
    
    # Retrieval quality indicators
    above_threshold: bool = True
    rank: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "content": self.content,
            "score": self.score,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "rank": self.rank,
            "metadata": self.metadata
        }


class Retriever:
    """
    Main retrieval class for the RAG pipeline.
    
    This class orchestrates:
    1. Query embedding
    2. Vector search
    3. Result filtering
    4. Context assembly
    """
    
    def __init__(
        self,
        embedding_model: Optional[BaseEmbeddingModel] = None,
        vector_store: Optional[BaseVectorStore] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        include_metadata: bool = True
    ):
        """
        Initialize the retriever.
        
        Args:
            embedding_model: Model for query embedding
            vector_store: Vector database for search
            top_k: Number of results to retrieve
            similarity_threshold: Minimum similarity score
            include_metadata: Include metadata in results
        """
        self.embedding_model = embedding_model or get_embedding_model()
        self.vector_store = vector_store or get_vector_store()
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.include_metadata = include_metadata
        
        logger.info(
            f"Retriever initialized: top_k={top_k}, threshold={similarity_threshold}"
        )
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        include_scores: bool = True
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User query string
            top_k: Override default top_k
            filter_dict: Metadata filter constraints
            include_scores: Include similarity scores
        
        Returns:
            List of RetrievalResult objects
        """
        top_k = top_k or self.top_k
        
        logger.debug(f"Retrieving for query: '{query[:50]}...'")
        
        # Step 1: Embed the query
        query_embedding = self.embedding_model.embed_query(query)
        
        # Step 2: Search vector store
        search_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Get extra results for filtering
            filter_dict=filter_dict
        )
        
        # Step 3: Filter and rank results
        retrieval_results = []
        
        for rank, result in enumerate(search_results):
            above_threshold = result.score >= self.similarity_threshold
            
            retrieval_result = RetrievalResult(
                content=result.content,
                score=result.score,
                chunk_id=result.chunk_id,
                source=result.metadata.get("source", "unknown"),
                metadata=result.metadata if self.include_metadata else {},
                above_threshold=above_threshold,
                rank=rank + 1
            )
            
            retrieval_results.append(retrieval_result)
        
        # Step 4: Filter by threshold and limit
        filtered_results = [
            r for r in retrieval_results if r.above_threshold
        ][:top_k]
        
        logger.info(
            f"Retrieved {len(filtered_results)} results "
            f"(filtered from {len(retrieval_results)})"
        )
        
        return filtered_results
    
    def retrieve_with_context(
        self,
        query: str,
        max_context_length: int = 4000,
        **kwargs
    ) -> Tuple[List[RetrievalResult], str]:
        """
        Retrieve chunks and assemble context string.
        
        Returns both the results and a formatted context string
        ready for LLM prompting.
        
        Args:
            query: User query
            max_context_length: Maximum characters in context
            **kwargs: Additional arguments for retrieve()
        
        Returns:
            Tuple of (results, context_string)
        """
        results = self.retrieve(query, **kwargs)
        
        # Assemble context with source attribution
        context_parts = []
        current_length = 0
        
        for result in results:
            # Format each chunk with source
            chunk_context = (
                f"[Source: {result.source}]\n"
                f"{result.content}\n"
            )
            
            if current_length + len(chunk_context) > max_context_length:
                break
            
            context_parts.append(chunk_context)
            current_length += len(chunk_context)
        
        context = "\n---\n".join(context_parts)
        
        return results, context
    
    def multi_query_retrieve(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        deduplicate: bool = True
    ) -> List[RetrievalResult]:
        """
        Retrieve for multiple query variations.
        
        This is useful for query expansion - generate multiple
        phrasings of the question and combine results.
        
        Args:
            queries: List of query variations
            top_k: Results per query
            deduplicate: Remove duplicate chunks
        
        Returns:
            Combined results from all queries
        """
        all_results = []
        seen_chunks = set()
        
        for query in queries:
            results = self.retrieve(query, top_k=top_k)
            
            for result in results:
                if deduplicate and result.chunk_id in seen_chunks:
                    continue
                
                seen_chunks.add(result.chunk_id)
                all_results.append(result)
        
        # Re-rank combined results by score
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        return all_results


class HybridRetriever(Retriever):
    """
    Hybrid retriever combining dense and sparse search.
    
    Uses both:
    - Dense retrieval (semantic similarity)
    - Sparse retrieval (keyword matching with BM25)
    
    Combines results using reciprocal rank fusion.
    """
    
    def __init__(
        self,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        **kwargs
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            dense_weight: Weight for dense retrieval scores
            sparse_weight: Weight for sparse retrieval scores
            **kwargs: Arguments for base Retriever
        """
        super().__init__(**kwargs)
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        
        # Initialize BM25 index (if available)
        self._init_sparse_index()
    
    def _init_sparse_index(self):
        """Initialize sparse retrieval index."""
        try:
            from rank_bm25 import BM25Okapi
            self.bm25_available = True
            self.bm25_index = None
            self.bm25_corpus = []
            self.bm25_chunk_ids = []
        except ImportError:
            logger.warning("rank_bm25 not available, using dense-only retrieval")
            self.bm25_available = False
    
    def update_sparse_index(self, chunks: List):
        """Update BM25 index with chunks."""
        if not self.bm25_available:
            return
        
        from rank_bm25 import BM25Okapi
        
        # Tokenize documents
        self.bm25_corpus = [
            chunk.content.lower().split() for chunk in chunks
        ]
        self.bm25_chunk_ids = [chunk.chunk_id for chunk in chunks]
        
        self.bm25_index = BM25Okapi(self.bm25_corpus)
        
        logger.info(f"Updated BM25 index with {len(chunks)} documents")
    
    def _sparse_search(
        self,
        query: str,
        top_k: int
    ) -> Dict[str, float]:
        """Perform sparse (BM25) search."""
        if not self.bm25_available or self.bm25_index is None:
            return {}
        
        query_tokens = query.lower().split()
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        return {
            self.bm25_chunk_ids[i]: scores[i]
            for i in top_indices
            if scores[i] > 0
        }
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Hybrid retrieval combining dense and sparse search.
        """
        top_k = top_k or self.top_k
        
        # Dense retrieval
        dense_results = super().retrieve(query, top_k=top_k * 2, **kwargs)
        
        if not self.bm25_available or self.bm25_index is None:
            return dense_results[:top_k]
        
        # Sparse retrieval
        sparse_scores = self._sparse_search(query, top_k * 2)
        
        # Combine using reciprocal rank fusion
        chunk_scores = {}
        
        # Add dense scores
        for rank, result in enumerate(dense_results):
            rrf_score = 1 / (60 + rank + 1)  # RRF formula
            chunk_scores[result.chunk_id] = {
                "result": result,
                "score": self.dense_weight * rrf_score
            }
        
        # Add sparse scores
        for rank, (chunk_id, _) in enumerate(
            sorted(sparse_scores.items(), key=lambda x: -x[1])
        ):
            rrf_score = 1 / (60 + rank + 1)
            if chunk_id in chunk_scores:
                chunk_scores[chunk_id]["score"] += self.sparse_weight * rrf_score
            # Note: if chunk not in dense results, we skip it 
            # (would need full chunk data)
        
        # Sort by combined score
        sorted_results = sorted(
            chunk_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        
        return [item["result"] for item in sorted_results[:top_k]]


class ContextualRetriever(Retriever):
    """
    Retriever with conversation context awareness.
    
    Incorporates previous conversation turns to improve
    retrieval for follow-up questions.
    """
    
    def __init__(
        self,
        context_window: int = 3,
        **kwargs
    ):
        """
        Initialize contextual retriever.
        
        Args:
            context_window: Number of previous turns to consider
            **kwargs: Arguments for base Retriever
        """
        super().__init__(**kwargs)
        self.context_window = context_window
        self.conversation_history: List[Dict[str, str]] = []
    
    def add_turn(self, query: str, response: str):
        """Add a conversation turn to history."""
        self.conversation_history.append({
            "query": query,
            "response": response
        })
        
        # Keep only recent turns
        if len(self.conversation_history) > self.context_window:
            self.conversation_history = self.conversation_history[-self.context_window:]
    
    def retrieve(
        self,
        query: str,
        use_context: bool = True,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        Retrieve with conversation context.
        
        Expands the query with relevant context from previous turns.
        """
        if not use_context or not self.conversation_history:
            return super().retrieve(query, **kwargs)
        
        # Build contextual query
        context_parts = [query]
        
        for turn in self.conversation_history[-self.context_window:]:
            context_parts.append(turn["query"])
        
        # Use multi-query retrieval
        return self.multi_query_retrieve(context_parts, **kwargs)
    
    def clear_context(self):
        """Clear conversation history."""
        self.conversation_history = []


# Factory function
def create_retriever(
    retriever_type: str = "standard",
    **kwargs
) -> Retriever:
    """
    Create a retriever instance.
    
    Args:
        retriever_type: "standard", "hybrid", or "contextual"
        **kwargs: Arguments for retriever
    
    Returns:
        Configured retriever
    """
    retrievers = {
        "standard": Retriever,
        "hybrid": HybridRetriever,
        "contextual": ContextualRetriever
    }
    
    if retriever_type not in retrievers:
        raise ValueError(f"Unknown retriever type: {retriever_type}")
    
    return retrievers[retriever_type](**kwargs)


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("Retriever Demo")
    print("=" * 60)
    
    # This would work with actual data
    retriever = Retriever(top_k=3, similarity_threshold=0.5)
    
    print(f"Retriever configured:")
    print(f"  top_k: {retriever.top_k}")
    print(f"  threshold: {retriever.similarity_threshold}")
    print(f"  embedding model: {retriever.embedding_model.model_name}")
