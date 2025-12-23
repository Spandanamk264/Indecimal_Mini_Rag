"""
Vector Database Module for Construction RAG System
===================================================

This module provides a unified interface to multiple vector databases.
Vector databases are specialized storage systems optimized for similarity search.

Why Vector Databases?
---------------------
Traditional databases use exact matching (SQL WHERE clauses).
Vector databases use similarity matching (nearest neighbor search).

When you search for "fall protection equipment", a vector DB finds:
- "safety harnesses for elevated work" (semantically similar)
- "fall protection requirements" (conceptually related)

Even though neither contains the exact query words!

Available Backends:
------------------
1. ChromaDB (Default)
   - Pros: Easy setup, no server needed, persistent storage
   - Cons: Not as scalable as distributed solutions
   - Best for: Development, small-medium datasets (<1M vectors)

2. FAISS (Facebook AI Similarity Search)
   - Pros: Very fast, efficient memory usage, mature library
   - Cons: No built-in persistence, manual metadata handling
   - Best for: Large-scale production, GPU acceleration

3. Pinecone (Cloud-based)
   - Pros: Fully managed, scales infinitely, real-time updates
   - Cons: Requires API key, costs money, network dependency
   - Best for: Production at scale, multi-tenant applications

ML Fundamentals Connection:
---------------------------
Vector similarity search is essentially a k-NN (k-Nearest Neighbors) problem.

The core challenge is making k-NN fast:
- Brute force: O(N) - Check every vector
- Approximate (ANN): O(log N) - Use index structures

ANN techniques:
- LSH (Locality Sensitive Hashing): Hash similar items together
- HNSW (Hierarchical Navigable Small World): Graph-based navigation
- IVF (Inverted File Index): Cluster vectors, search relevant clusters

Trade-off: Speed vs Accuracy (recall@k)
"""

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from ..config import settings
from ..ingestion.chunker import Chunk


@dataclass
class SearchResult:
    """
    Represents a single search result.
    
    Contains both the chunk content and retrieval metadata
    for debugging and filtering.
    """
    chunk_id: str
    content: str
    score: float  # Similarity score (higher = more similar)
    metadata: Dict[str, Any]
    
    def __repr__(self) -> str:
        preview = self.content[:60].replace('\n', ' ')
        return f"SearchResult(score={self.score:.4f}, '{preview}...')"


class BaseVectorStore(ABC):
    """
    Abstract base class for vector stores.
    
    Defines the interface that all vector database implementations
    must follow, enabling easy switching between backends.
    """
    
    @abstractmethod
    def add_chunks(
        self,
        chunks: List[Chunk],
        embeddings: np.ndarray
    ) -> None:
        """Add chunks with their embeddings to the store."""
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar chunks."""
        pass
    
    @abstractmethod
    def delete(
        self,
        chunk_ids: Optional[List[str]] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> int:
        """Delete chunks by ID or filter. Returns count deleted."""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Return total number of chunks in store."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all data from the store."""
        pass


class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB implementation of vector store.
    
    ChromaDB is an open-source embedding database that's easy to use
    and requires no external server setup.
    
    Features:
    - Persistent storage to disk
    - Metadata filtering
    - Multiple distance metrics
    - Automatic deduplication
    """
    
    def __init__(
        self,
        collection_name: str = "construction_docs",
        persist_directory: Optional[str] = None,
        embedding_function = None
    ):
        """
        Initialize ChromaDB vector store.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist data
            embedding_function: Optional custom embedding function
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or settings.chroma_persist_directory
        
        logger.info(f"Initializing ChromaDB: {collection_name}")
        
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            
            # Create persistent client
            Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
            
            self.client = chromadb.Client(
                ChromaSettings(
                    persist_directory=self.persist_directory,
                    anonymized_telemetry=False
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            
            logger.info(
                f"ChromaDB initialized: {self.count()} existing vectors"
            )
            
        except ImportError:
            logger.error("chromadb not installed. Run: pip install chromadb")
            raise
    
    def add_chunks(
        self,
        chunks: List[Chunk],
        embeddings: np.ndarray
    ) -> None:
        """Add chunks to ChromaDB collection."""
        if len(chunks) == 0:
            logger.warning("No chunks to add")
            return
        
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
            )
        
        # Prepare data for ChromaDB
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = []
        
        for chunk in chunks:
            # ChromaDB requires string/int/float values in metadata
            meta = {}
            for key, value in chunk.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    meta[key] = value
                else:
                    meta[key] = str(value)
            
            meta["chunk_index"] = chunk.chunk_index
            meta["total_chunks"] = chunk.total_chunks
            meta["token_count"] = chunk.token_count
            metadatas.append(meta)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(chunks)} chunks to ChromaDB")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar chunks in ChromaDB.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_dict: Metadata filter (e.g., {"source": "safety_manual.txt"})
        
        Returns:
            List of SearchResult objects sorted by similarity
        """
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Build query arguments
        query_args = {
            "query_embeddings": query_embedding.tolist(),
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"]
        }
        
        if filter_dict:
            query_args["where"] = filter_dict
        
        # Execute search
        results = self.collection.query(**query_args)
        
        # Convert to SearchResult objects
        search_results = []
        
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                # ChromaDB returns L2 distance, convert to similarity
                # For cosine distance: similarity = 1 - distance
                distance = results["distances"][0][i]
                similarity = 1 - distance  # Convert distance to similarity
                
                search_results.append(SearchResult(
                    chunk_id=chunk_id,
                    content=results["documents"][0][i],
                    score=similarity,
                    metadata=results["metadatas"][0][i]
                ))
        
        return search_results
    
    def delete(
        self,
        chunk_ids: Optional[List[str]] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> int:
        """Delete chunks by ID or filter."""
        if chunk_ids:
            self.collection.delete(ids=chunk_ids)
            return len(chunk_ids)
        elif filter_dict:
            self.collection.delete(where=filter_dict)
            return -1  # ChromaDB doesn't return count for filter deletes
        else:
            raise ValueError("Must specify either chunk_ids or filter_dict")
    
    def count(self) -> int:
        """Return total count of vectors."""
        return self.collection.count()
    
    def clear(self) -> None:
        """Delete all data from collection."""
        # Get all IDs and delete them
        all_data = self.collection.get()
        if all_data["ids"]:
            self.collection.delete(ids=all_data["ids"])
        logger.info("ChromaDB collection cleared")


class FAISSVectorStore(BaseVectorStore):
    """
    FAISS implementation of vector store.
    
    FAISS (Facebook AI Similarity Search) is highly optimized for
    similarity search, especially on large datasets.
    
    Features:
    - Very fast search (uses SIMD, optional GPU)
    - Multiple index types (Flat, IVF, HNSW)
    - Memory efficient
    
    Index Types:
    - IndexFlatL2/IP: Exact search (brute force)
    - IndexIVFFlat: Cluster-based approximate search
    - IndexHNSWFlat: Graph-based approximate search
    """
    
    def __init__(
        self,
        dimension: int = 384,
        index_type: str = "flat",
        index_path: Optional[str] = None
    ):
        """
        Initialize FAISS vector store.
        
        Args:
            dimension: Embedding dimension
            index_type: "flat" (exact) or "ivf" (approximate)
            index_path: Path to save/load index
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index_path = index_path or settings.faiss_index_path
        
        # Storage for chunk metadata (FAISS only stores vectors)
        self.chunk_data: Dict[int, Dict] = {}
        self.id_to_idx: Dict[str, int] = {}
        self.idx_to_id: Dict[int, str] = {}
        self.next_idx = 0
        
        try:
            import faiss
            self.faiss = faiss
            
            # Create index
            if index_type == "flat":
                # Exact search with inner product (cosine on normalized vectors)
                self.index = faiss.IndexFlatIP(dimension)
            elif index_type == "ivf":
                # Approximate search with IVF
                quantizer = faiss.IndexFlatIP(dimension)
                self.index = faiss.IndexIVFFlat(
                    quantizer, dimension, 100  # 100 clusters
                )
            else:
                raise ValueError(f"Unknown index type: {index_type}")
            
            # Try to load existing index
            self._load_if_exists()
            
            logger.info(
                f"FAISS initialized: type={index_type}, dim={dimension}, "
                f"vectors={self.index.ntotal}"
            )
            
        except ImportError:
            logger.error("faiss not installed. Run: pip install faiss-cpu")
            raise
    
    def _load_if_exists(self):
        """Load index and metadata if they exist."""
        index_file = Path(f"{self.index_path}/index.faiss")
        meta_file = Path(f"{self.index_path}/metadata.npy")
        
        if index_file.exists() and meta_file.exists():
            self.index = self.faiss.read_index(str(index_file))
            
            import pickle
            with open(meta_file, "rb") as f:
                data = pickle.load(f)
                self.chunk_data = data["chunk_data"]
                self.id_to_idx = data["id_to_idx"]
                self.idx_to_id = data["idx_to_id"]
                self.next_idx = data["next_idx"]
            
            logger.info(f"Loaded FAISS index from {self.index_path}")
    
    def _save(self):
        """Save index and metadata to disk."""
        Path(self.index_path).mkdir(parents=True, exist_ok=True)
        
        index_file = Path(f"{self.index_path}/index.faiss")
        meta_file = Path(f"{self.index_path}/metadata.npy")
        
        self.faiss.write_index(self.index, str(index_file))
        
        import pickle
        with open(meta_file, "wb") as f:
            pickle.dump({
                "chunk_data": self.chunk_data,
                "id_to_idx": self.id_to_idx,
                "idx_to_id": self.idx_to_id,
                "next_idx": self.next_idx
            }, f)
        
        logger.debug(f"Saved FAISS index to {self.index_path}")
    
    def add_chunks(
        self,
        chunks: List[Chunk],
        embeddings: np.ndarray
    ) -> None:
        """Add chunks to FAISS index."""
        if len(chunks) == 0:
            return
        
        # Ensure embeddings are float32 and normalized
        embeddings = embeddings.astype(np.float32)
        self.faiss.normalize_L2(embeddings)
        
        # Store metadata and mappings
        for i, chunk in enumerate(chunks):
            idx = self.next_idx
            self.chunk_data[idx] = {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "token_count": chunk.token_count
            }
            self.id_to_idx[chunk.chunk_id] = idx
            self.idx_to_id[idx] = chunk.chunk_id
            self.next_idx += 1
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Save to disk
        self._save()
        
        logger.info(f"Added {len(chunks)} chunks to FAISS")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search FAISS index."""
        if self.index.ntotal == 0:
            return []
        
        # Prepare query
        query = query_embedding.astype(np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        self.faiss.normalize_L2(query)
        
        # Search with extra results if filtering (post-filter strategy)
        search_k = top_k * 3 if filter_dict else top_k
        scores, indices = self.index.search(query, search_k)
        
        # Convert to SearchResults with filtering
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for no result
                continue
            
            chunk_data = self.chunk_data.get(idx)
            if not chunk_data:
                continue
            
            # Apply filter if specified
            if filter_dict:
                metadata = chunk_data["metadata"]
                if not all(
                    metadata.get(k) == v for k, v in filter_dict.items()
                ):
                    continue
            
            results.append(SearchResult(
                chunk_id=chunk_data["chunk_id"],
                content=chunk_data["content"],
                score=float(score),
                metadata=chunk_data["metadata"]
            ))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def delete(
        self,
        chunk_ids: Optional[List[str]] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Delete chunks from FAISS.
        
        Note: FAISS doesn't support direct deletion.
        We mark as deleted and rebuild periodically.
        """
        deleted = 0
        
        if chunk_ids:
            for chunk_id in chunk_ids:
                if chunk_id in self.id_to_idx:
                    idx = self.id_to_idx[chunk_id]
                    del self.chunk_data[idx]
                    del self.idx_to_id[idx]
                    del self.id_to_idx[chunk_id]
                    deleted += 1
        
        # For filter deletion, would need to rebuild index
        # This is a limitation of FAISS
        
        return deleted
    
    def count(self) -> int:
        """Return total count of vectors."""
        return self.index.ntotal
    
    def clear(self) -> None:
        """Clear all data."""
        self.index = self.faiss.IndexFlatIP(self.dimension)
        self.chunk_data = {}
        self.id_to_idx = {}
        self.idx_to_id = {}
        self.next_idx = 0
        
        # Delete saved files
        import shutil
        if Path(self.index_path).exists():
            shutil.rmtree(self.index_path)
        
        logger.info("FAISS index cleared")


class PineconeVectorStore(BaseVectorStore):
    """
    Pinecone implementation of vector store.
    
    Pinecone is a fully managed vector database service.
    Ideal for production deployments at scale.
    
    Features:
    - Serverless operation
    - Real-time updates
    - Hybrid search (vector + metadata)
    - Multi-tenant support
    """
    
    def __init__(
        self,
        index_name: Optional[str] = None,
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
        dimension: int = 384
    ):
        """
        Initialize Pinecone vector store.
        
        Args:
            index_name: Name of Pinecone index
            api_key: Pinecone API key
            environment: Pinecone environment
            dimension: Embedding dimension
        """
        self.index_name = index_name or settings.pinecone_index_name
        self.dimension = dimension
        
        try:
            from pinecone import Pinecone, ServerlessSpec
            
            # Initialize Pinecone client
            self.pc = Pinecone(
                api_key=api_key or settings.pinecone_api_key
            )
            
            # Create index if it doesn't exist
            existing_indexes = self.pc.list_indexes().names()
            
            if self.index_name not in existing_indexes:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                logger.info(f"Created Pinecone index: {self.index_name}")
            
            self.index = self.pc.Index(self.index_name)
            
            stats = self.index.describe_index_stats()
            logger.info(
                f"Pinecone initialized: {stats.total_vector_count} vectors"
            )
            
        except ImportError:
            logger.error("pinecone-client not installed. Run: pip install pinecone-client")
            raise
    
    def add_chunks(
        self,
        chunks: List[Chunk],
        embeddings: np.ndarray
    ) -> None:
        """Add chunks to Pinecone index."""
        if len(chunks) == 0:
            return
        
        # Prepare vectors for upsert
        vectors = []
        for i, chunk in enumerate(chunks):
            # Clean metadata for Pinecone (only strings, numbers, booleans)
            metadata = {"content": chunk.content[:1000]}  # Limit content size
            for key, value in chunk.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = value
            
            metadata["chunk_index"] = chunk.chunk_index
            metadata["token_count"] = chunk.token_count
            
            vectors.append({
                "id": chunk.chunk_id,
                "values": embeddings[i].tolist(),
                "metadata": metadata
            })
        
        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            self.index.upsert(vectors=batch)
        
        logger.info(f"Added {len(chunks)} chunks to Pinecone")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search Pinecone index."""
        query_args = {
            "vector": query_embedding.tolist(),
            "top_k": top_k,
            "include_metadata": True
        }
        
        if filter_dict:
            query_args["filter"] = filter_dict
        
        results = self.index.query(**query_args)
        
        search_results = []
        for match in results.matches:
            metadata = match.metadata or {}
            content = metadata.pop("content", "")
            
            search_results.append(SearchResult(
                chunk_id=match.id,
                content=content,
                score=match.score,
                metadata=metadata
            ))
        
        return search_results
    
    def delete(
        self,
        chunk_ids: Optional[List[str]] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> int:
        """Delete from Pinecone."""
        if chunk_ids:
            self.index.delete(ids=chunk_ids)
            return len(chunk_ids)
        elif filter_dict:
            self.index.delete(filter=filter_dict)
            return -1
        return 0
    
    def count(self) -> int:
        """Return total count."""
        stats = self.index.describe_index_stats()
        return stats.total_vector_count
    
    def clear(self) -> None:
        """Clear all data from index."""
        self.index.delete(delete_all=True)
        logger.info("Pinecone index cleared")


class VectorStoreFactory:
    """Factory for creating vector stores."""
    
    @staticmethod
    def create(
        store_type: Optional[str] = None,
        **kwargs
    ) -> BaseVectorStore:
        """
        Create a vector store instance.
        
        Args:
            store_type: "chroma", "faiss", or "pinecone"
            **kwargs: Additional arguments for store
        
        Returns:
            Configured vector store
        """
        store_type = store_type or settings.vector_db_type
        
        if store_type == "chroma":
            return ChromaVectorStore(**kwargs)
        elif store_type == "faiss":
            return FAISSVectorStore(**kwargs)
        elif store_type == "pinecone":
            return PineconeVectorStore(**kwargs)
        else:
            raise ValueError(f"Unknown store type: {store_type}")


# Convenience function
def get_vector_store(**kwargs) -> BaseVectorStore:
    """Get vector store based on settings."""
    return VectorStoreFactory.create(**kwargs)


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("Vector Store Demo")
    print("=" * 60)
    
    # Create a ChromaDB store
    store = ChromaVectorStore(
        collection_name="test_collection",
        persist_directory="./test_chroma"
    )
    
    # Create some test chunks
    from ingestion.chunker import Chunk
    
    chunks = [
        Chunk(
            content="Fall protection is required when working at heights above 6 feet.",
            chunk_id="chunk_1",
            doc_id="doc_1",
            metadata={"source": "safety_manual.txt"},
            chunk_index=0,
            total_chunks=3,
            token_count=15
        ),
        Chunk(
            content="Workers must wear PPE including hard hats, safety glasses, and high-viz vests.",
            chunk_id="chunk_2",
            doc_id="doc_1",
            metadata={"source": "safety_manual.txt"},
            chunk_index=1,
            total_chunks=3,
            token_count=20
        ),
        Chunk(
            content="The subcontractor shall maintain liability insurance of $2,000,000 per occurrence.",
            chunk_id="chunk_3",
            doc_id="doc_2",
            metadata={"source": "contract.txt"},
            chunk_index=0,
            total_chunks=1,
            token_count=18
        ),
    ]
    
    # Generate dummy embeddings
    embeddings = np.random.randn(3, 384).astype(np.float32)
    
    # Add chunks
    store.add_chunks(chunks, embeddings)
    
    print(f"\nTotal vectors: {store.count()}")
    
    # Search
    query_embedding = embeddings[0]  # Use first chunk as query
    results = store.search(query_embedding, top_k=2)
    
    print("\nSearch Results:")
    for result in results:
        print(f"  {result}")
    
    # Cleanup
    store.clear()
    print(f"\nAfter clear: {store.count()} vectors")
