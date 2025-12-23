"""
Semantic Chunking Module for Construction RAG System
=====================================================

This module implements advanced chunking strategies for splitting documents
into semantically meaningful segments suitable for embedding and retrieval.

Why Chunking Matters:
--------------------
Chunking is one of the most critical design decisions in RAG systems.
Poor chunking leads to:
- Lost context (chunks too small)
- Diluted relevance (chunks too large)
- Broken semantic units (splitting mid-sentence)

This directly affects retrieval quality, which cascades to final answer quality.

ML Fundamentals Connection:
---------------------------
Chunking is analogous to feature windowing in time series:
- Window too small: Loses patterns (high variance)
- Window too large: Mixes signals (high bias)
- Overlap preserves continuity (like stride in CNNs)

The chunk_size and overlap are hyperparameters that require tuning
based on document characteristics and use case.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Generator
from dataclasses import dataclass, field

import tiktoken
from loguru import logger

from .document_loader import Document


@dataclass 
class Chunk:
    """
    Represents a chunk of text with metadata.
    
    A chunk is the atomic unit we embed and retrieve.
    Rich metadata enables filtering and attribution.
    """
    content: str
    chunk_id: str
    doc_id: str
    metadata: Dict = field(default_factory=dict)
    
    # Position information for context
    chunk_index: int = 0
    total_chunks: int = 0
    
    # Token information
    token_count: int = 0
    
    def __repr__(self) -> str:
        preview = self.content[:80].replace('\n', ' ')
        return f"Chunk({self.chunk_id}, tokens={self.token_count}, '{preview}...')"


class BaseChunker(ABC):
    """Abstract base class for chunking strategies."""
    
    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """Split a document into chunks."""
        pass


class TokenBasedChunker(BaseChunker):
    """
    Token-based chunking with overlap.
    
    This is the most reliable chunking method because:
    1. LLMs and embeddings work with tokens, not characters
    2. Guarantees chunks fit within model limits
    3. Consistent chunk sizes for fair comparison during retrieval
    
    Design Parameters:
    ------------------
    chunk_size: Target tokens per chunk
        - 256 tokens: Fine-grained, good for specific facts
        - 512 tokens: Balanced (our default)
        - 1024 tokens: More context, fewer chunks
    
    chunk_overlap: Tokens shared between consecutive chunks
        - 0: No overlap, may break context
        - 50: 10% overlap, good balance
        - 100+: May cause redundancy in retrieval
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        model_name: str = "cl100k_base"
    ):
        """
        Initialize the token-based chunker.
        
        Args:
            chunk_size: Target number of tokens per chunk
            chunk_overlap: Number of overlapping tokens between chunks
            model_name: Tiktoken encoding model (cl100k_base for GPT-4)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding(model_name)
        except Exception:
            logger.warning(f"Failed to load {model_name}, using cl100k_base")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        logger.info(
            f"TokenBasedChunker initialized: size={chunk_size}, overlap={chunk_overlap}"
        )
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.tokenizer.encode(text))
    
    def _tokens_to_text(self, tokens: List[int]) -> str:
        """Convert token IDs back to text."""
        return self.tokenizer.decode(tokens)
    
    def chunk(self, document: Document) -> List[Chunk]:
        """
        Split document into token-bounded chunks with overlap.
        
        Algorithm:
        1. Tokenize entire document
        2. Create windows of chunk_size tokens
        3. Move window by (chunk_size - overlap) for each chunk
        4. Decode tokens back to text
        """
        content = document.content
        if not content.strip():
            logger.warning(f"Empty document: {document.doc_id}")
            return []
        
        # Tokenize the full document
        tokens = self.tokenizer.encode(content)
        total_tokens = len(tokens)
        
        if total_tokens <= self.chunk_size:
            # Document fits in one chunk
            chunk = Chunk(
                content=content,
                chunk_id=f"{document.doc_id}_0",
                doc_id=document.doc_id,
                metadata=document.metadata.copy(),
                chunk_index=0,
                total_chunks=1,
                token_count=total_tokens
            )
            return [chunk]
        
        # Calculate stride (how many tokens to move for each chunk)
        stride = self.chunk_size - self.chunk_overlap
        
        chunks = []
        chunk_idx = 0
        
        for start in range(0, total_tokens, stride):
            end = min(start + self.chunk_size, total_tokens)
            chunk_tokens = tokens[start:end]
            chunk_text = self._tokens_to_text(chunk_tokens)
            
            chunk = Chunk(
                content=chunk_text,
                chunk_id=f"{document.doc_id}_{chunk_idx}",
                doc_id=document.doc_id,
                metadata=document.metadata.copy(),
                chunk_index=chunk_idx,
                token_count=len(chunk_tokens)
            )
            
            # Add position metadata
            chunk.metadata["chunk_start_token"] = start
            chunk.metadata["chunk_end_token"] = end
            
            chunks.append(chunk)
            chunk_idx += 1
            
            # Stop if we've covered all tokens
            if end == total_tokens:
                break
        
        # Update total_chunks for all chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        logger.debug(
            f"Document {document.doc_id}: {total_tokens} tokens -> {len(chunks)} chunks"
        )
        
        return chunks


class SemanticChunker(BaseChunker):
    """
    Semantic chunking based on document structure.
    
    This chunker respects natural document boundaries:
    - Section headers
    - Paragraph breaks
    - Sentence boundaries
    
    Trade-offs:
    + More coherent chunks (respect meaning)
    + Better for structured documents (specs, manuals)
    - Variable chunk sizes
    - May create very small or very large chunks
    
    We use this for construction documents which are highly structured.
    """
    
    def __init__(
        self,
        max_chunk_size: int = 1024,
        min_chunk_size: int = 100,
        chunk_overlap: int = 50,
        section_patterns: Optional[List[str]] = None
    ):
        """
        Initialize semantic chunker.
        
        Args:
            max_chunk_size: Maximum tokens per chunk
            min_chunk_size: Minimum tokens per chunk (merge smaller)
            chunk_overlap: Overlap tokens at boundaries
            section_patterns: Regex patterns for section boundaries
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.chunk_overlap = chunk_overlap
        
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = tiktoken.get_encoding("gpt2")
        
        # Default section patterns for construction documents
        self.section_patterns = section_patterns or [
            r'^(?:SECTION|ARTICLE|PART)\s+\d+',  # SECTION 1, ARTICLE 2
            r'^\d+\.\d+\s+\w',  # 1.1 Title
            r'^={3,}',  # ====== (separator)
            r'^-{3,}',  # ------ (separator)
            r'^(?:CHAPTER|APPENDIX)\s+\w+',  # CHAPTER A
        ]
        
        # Compile patterns
        self.section_regex = re.compile(
            '|'.join(f'({p})' for p in self.section_patterns),
            re.MULTILINE | re.IGNORECASE
        )
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))
    
    def _find_section_boundaries(self, text: str) -> List[int]:
        """Find positions where new sections begin."""
        boundaries = [0]  # Start of document is always a boundary
        
        for match in self.section_regex.finditer(text):
            boundaries.append(match.start())
        
        boundaries.append(len(text))  # End of document
        
        return sorted(set(boundaries))
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split on double newlines (paragraph breaks)
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """Merge chunks that are too small."""
        merged = []
        current = ""
        
        for chunk in chunks:
            if not chunk.strip():
                continue
            
            combined = current + "\n\n" + chunk if current else chunk
            combined_tokens = self._count_tokens(combined)
            
            if combined_tokens < self.min_chunk_size:
                current = combined
            elif combined_tokens <= self.max_chunk_size:
                current = combined
            else:
                if current:
                    merged.append(current)
                current = chunk
        
        if current:
            merged.append(current)
        
        return merged
    
    def _split_large_chunk(self, text: str) -> List[str]:
        """Split a chunk that exceeds max_chunk_size."""
        # First try splitting by paragraphs
        paragraphs = self._split_by_paragraphs(text)
        
        result = []
        current = ""
        
        for para in paragraphs:
            para_tokens = self._count_tokens(para)
            
            if para_tokens > self.max_chunk_size:
                # Paragraph itself is too large, split by sentences
                if current:
                    result.append(current)
                    current = ""
                
                # Split by sentences and recombine
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sent in sentences:
                    sent_tokens = self._count_tokens(sent)
                    combined = current + " " + sent if current else sent
                    combined_tokens = self._count_tokens(combined)
                    
                    if combined_tokens <= self.max_chunk_size:
                        current = combined
                    else:
                        if current:
                            result.append(current)
                        current = sent
            else:
                combined = current + "\n\n" + para if current else para
                combined_tokens = self._count_tokens(combined)
                
                if combined_tokens <= self.max_chunk_size:
                    current = combined
                else:
                    result.append(current)
                    current = para
        
        if current:
            result.append(current)
        
        return result
    
    def chunk(self, document: Document) -> List[Chunk]:
        """
        Split document based on semantic boundaries.
        
        Algorithm:
        1. Find section boundaries
        2. Extract text between boundaries
        3. Merge small sections
        4. Split large sections
        5. Create Chunk objects with metadata
        """
        content = document.content
        if not content.strip():
            return []
        
        # Find section boundaries
        boundaries = self._find_section_boundaries(content)
        
        # Extract sections
        raw_sections = []
        for i in range(len(boundaries) - 1):
            section_text = content[boundaries[i]:boundaries[i+1]].strip()
            if section_text:
                raw_sections.append(section_text)
        
        # Merge small sections
        merged_sections = self._merge_small_chunks(raw_sections)
        
        # Split large sections
        final_chunks = []
        for section in merged_sections:
            tokens = self._count_tokens(section)
            if tokens > self.max_chunk_size:
                final_chunks.extend(self._split_large_chunk(section))
            else:
                final_chunks.append(section)
        
        # Create Chunk objects
        chunks = []
        for idx, chunk_text in enumerate(final_chunks):
            chunk = Chunk(
                content=chunk_text,
                chunk_id=f"{document.doc_id}_{idx}",
                doc_id=document.doc_id,
                metadata=document.metadata.copy(),
                chunk_index=idx,
                total_chunks=len(final_chunks),
                token_count=self._count_tokens(chunk_text)
            )
            
            # Try to extract section title
            first_line = chunk_text.split('\n')[0][:100]
            chunk.metadata["section_preview"] = first_line
            
            chunks.append(chunk)
        
        logger.debug(
            f"Document {document.doc_id}: semantic chunking -> {len(chunks)} chunks"
        )
        
        return chunks


class HybridChunker(BaseChunker):
    """
    Hybrid chunking combining semantic and token-based approaches.
    
    Best of both worlds:
    - Respects document structure when possible
    - Guarantees token limits
    - Adds overlap for context preservation
    
    This is the recommended chunker for production RAG systems.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100,
    ):
        """Initialize hybrid chunker."""
        self.semantic_chunker = SemanticChunker(
            max_chunk_size=chunk_size,
            min_chunk_size=min_chunk_size,
            chunk_overlap=0  # We'll add overlap ourselves
        )
        self.token_chunker = TokenBasedChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk(self, document: Document) -> List[Chunk]:
        """
        Chunk using hybrid strategy.
        
        Strategy:
        1. First try semantic chunking
        2. If chunks are still too large, apply token chunking
        3. Add overlap context between chunks
        """
        # Start with semantic chunks
        semantic_chunks = self.semantic_chunker.chunk(document)
        
        # Verify all chunks are within size limit
        final_chunks = []
        for c in semantic_chunks:
            if c.token_count > self.chunk_size:
                # This chunk is too large, apply token chunking
                sub_doc = Document(
                    content=c.content,
                    metadata=c.metadata,
                    doc_id=c.doc_id
                )
                sub_chunks = self.token_chunker.chunk(sub_doc)
                
                # Update chunk IDs
                for idx, sc in enumerate(sub_chunks):
                    sc.chunk_id = f"{c.chunk_id}_{idx}"
                
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(c)
        
        # Re-index chunks
        for idx, chunk in enumerate(final_chunks):
            chunk.chunk_index = idx
            chunk.total_chunks = len(final_chunks)
        
        logger.info(
            f"Hybrid chunking: {document.doc_id} -> {len(final_chunks)} chunks"
        )
        
        return final_chunks


class ChunkingPipeline:
    """
    Complete chunking pipeline for batch processing.
    
    Combines:
    - Document loading
    - Text cleaning
    - Chunking
    - Metadata enrichment
    """
    
    def __init__(
        self,
        chunker: Optional[BaseChunker] = None,
        add_context_metadata: bool = True
    ):
        """
        Initialize chunking pipeline.
        
        Args:
            chunker: Chunking strategy to use (default: HybridChunker)
            add_context_metadata: Whether to add neighboring chunk info
        """
        self.chunker = chunker or HybridChunker()
        self.add_context_metadata = add_context_metadata
    
    def process_document(self, document: Document) -> List[Chunk]:
        """Process a single document into chunks."""
        chunks = self.chunker.chunk(document)
        
        if self.add_context_metadata and len(chunks) > 1:
            self._add_context_metadata(chunks)
        
        return chunks
    
    def process_documents(
        self,
        documents: List[Document]
    ) -> Generator[Chunk, None, None]:
        """
        Process multiple documents, yielding chunks.
        
        Uses generator for memory efficiency with large document sets.
        """
        for doc in documents:
            chunks = self.process_document(doc)
            for chunk in chunks:
                yield chunk
    
    def _add_context_metadata(self, chunks: List[Chunk]):
        """Add metadata about neighboring chunks for context."""
        for i, chunk in enumerate(chunks):
            # Add preview of previous chunk
            if i > 0:
                prev_content = chunks[i-1].content
                chunk.metadata["prev_chunk_preview"] = prev_content[-100:]
            
            # Add preview of next chunk
            if i < len(chunks) - 1:
                next_content = chunks[i+1].content
                chunk.metadata["next_chunk_preview"] = next_content[:100]


# Factory function for easy usage
def create_chunker(
    strategy: str = "hybrid",
    chunk_size: int = 512,
    chunk_overlap: int = 50
) -> BaseChunker:
    """
    Create a chunker with specified strategy.
    
    Args:
        strategy: "token", "semantic", or "hybrid"
        chunk_size: Target tokens per chunk
        chunk_overlap: Overlap tokens
    
    Returns:
        Configured chunker instance
    """
    strategies = {
        "token": TokenBasedChunker,
        "semantic": SemanticChunker,
        "hybrid": HybridChunker
    }
    
    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from {list(strategies.keys())}")
    
    return strategies[strategy](
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )


if __name__ == "__main__":
    # Test the chunkers
    from document_loader import Document
    
    sample_text = """
    SECTION 1: INTRODUCTION
    
    This is the introduction section. It provides an overview of the document
    and explains the purpose and scope.
    
    SECTION 2: SAFETY REQUIREMENTS
    
    2.1 Personal Protective Equipment
    
    All workers must wear appropriate PPE at all times. This includes:
    - Hard hats (ANSI Z89.1 certified)
    - Safety glasses
    - High-visibility vests
    - Steel-toed boots
    
    2.2 Fall Protection
    
    Fall protection is required when working at heights of 6 feet or more.
    OSHA 29 CFR 1926.501 specifies fall protection requirements.
    
    SECTION 3: EMERGENCY PROCEDURES
    
    In case of emergency, evacuate immediately using the designated routes.
    Assembly point is the main parking lot.
    """
    
    doc = Document(content=sample_text, metadata={"source": "test.txt"})
    
    print("=" * 60)
    print("Testing Token-Based Chunker")
    print("=" * 60)
    token_chunker = TokenBasedChunker(chunk_size=100, chunk_overlap=20)
    token_chunks = token_chunker.chunk(doc)
    for chunk in token_chunks:
        print(f"\n{chunk}")
        print(f"Content preview: {chunk.content[:100]}...")
    
    print("\n" + "=" * 60)
    print("Testing Semantic Chunker")
    print("=" * 60)
    semantic_chunker = SemanticChunker(max_chunk_size=200, min_chunk_size=50)
    semantic_chunks = semantic_chunker.chunk(doc)
    for chunk in semantic_chunks:
        print(f"\n{chunk}")
        print(f"Content preview: {chunk.content[:100]}...")
    
    print("\n" + "=" * 60)
    print("Testing Hybrid Chunker")
    print("=" * 60)
    hybrid_chunker = HybridChunker(chunk_size=150, chunk_overlap=20)
    hybrid_chunks = hybrid_chunker.chunk(doc)
    for chunk in hybrid_chunks:
        print(f"\n{chunk}")
        print(f"Content preview: {chunk.content[:100]}...")
