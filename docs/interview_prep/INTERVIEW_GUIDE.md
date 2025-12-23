# Interview Preparation Guide

## Project Summary (30-second pitch)

"I built a production-grade RAG system for construction industry documents. It enables instant, accurate question-answering from safety manuals, contracts, and inspection reports using semantic search and GPT-4. The key differentiator is an agentic extension that handles complex multi-step queries—like comparing policies across multiple documents—using the REACT reasoning framework."

---

## Core Technical Questions

### 1. "Walk me through the RAG pipeline"

**Answer Structure:**

1. **Indexing Phase** (offline)
   - Documents are loaded and cleaned
   - Split into chunks (512 tokens with 50 overlap)
   - Each chunk is embedded using Sentence Transformers
   - Embeddings stored in ChromaDB vector database

2. **Query Phase** (online)
   - User query is embedded using same model
   - Cosine similarity search finds top-K chunks
   - Chunks are concatenated as context
   - GPT-4 generates answer grounded in context

3. **Key insight**: "We're essentially creating a two-stage retrieval-then-generation pipeline that allows the LLM to access domain-specific knowledge it wasn't trained on."

---

### 2. "How do you choose the chunk size?"

**Answer:**

"This is a classic bias-variance tradeoff:

**Small chunks (128-256 tokens):**
- Pros: Precise retrieval, specific information
- Cons: Lost context, fragmented concepts, more chunks to search

**Large chunks (1024+ tokens):**
- Pros: Complete context, fewer chunks
- Cons: May include irrelevant info, dilutes relevance

**Our choice: 512 tokens with 50-token overlap**
- Sweet spot for construction documents
- Overlap preserves cross-boundary context
- Empirically tuned on sample queries"

---

### 3. "How do you prevent hallucinations?"

**Answer (give multiple strategies):**

1. **Temperature control**: Use 0.1 for deterministic responses
2. **Explicit grounding**: Prompt says "ONLY use information from context"
3. **Source citation**: Require LLM to cite sources
4. **Post-generation check**: Compare answer words against context
5. **No-context handling**: If retrieval fails, admit "I don't have this information"
6. **Confidence scoring**: Flag low-confidence answers

---

### 4. "Explain the embedding model selection"

**Answer:**

"I evaluated three options:

| Model | Dimensions | Speed | Quality |
|-------|------------|-------|---------|
| all-MiniLM-L6-v2 | 384 | 14K words/sec | Good |
| all-mpnet-base-v2 | 768 | 2K words/sec | Better |
| OpenAI text-embedding-3-small | 1536 | API latency | Best |

**Choice: MiniLM** because:
1. 5x faster than mpnet
2. Saves ~2GB memory for 10K documents
3. Quality difference minimal for our use case
4. Free to run locally

For production with higher quality needs, I'd consider OpenAI embeddings."

---

### 5. "What's the REACT agent and why did you implement it?"

**Answer:**

"REACT = Reasoning + Acting. It's a framework for building AI agents.

**Standard RAG limitation**: Can only answer single-hop questions

**REACT enables**:
- Multi-step reasoning
- Tool usage (search, compare, calculate)
- Query decomposition

**Example**:
```
Query: "Compare PPE requirements in safety manual vs contract"

Agent:
  Thought: I need info from both documents
  Action: search_documents(query="PPE", source="safety_manual")
  Observation: "Required: hard hats, safety glasses..."
  
  Thought: Now search the contract
  Action: search_documents(query="PPE", source="contract")
  Observation: "Subcontractor shall maintain PPE..."
  
  Thought: I can now compare
  Final Answer: "The safety manual specifies... while the contract requires..."
```

This makes the system handle complex, real-world queries that simple RAG cannot."

---

### 6. "How do you evaluate RAG systems?"

**Answer:**

"RAG has two components to evaluate:

**Retrieval metrics**:
- Precision@K: Are retrieved docs relevant?
- Recall@K: Did we get all relevant docs?
- MRR: How early is first relevant result?
- NDCG: Accounts for ranking position

**Generation metrics**:
- Faithfulness: Is answer grounded in context?
- Relevance: Does it answer the question?
- Completeness: Does it cover all aspects?

**End-to-end**:
- Answer correctness vs ground truth
- Latency and cost
- User satisfaction (if available)

I implemented automated evaluation using heuristics, with option to use LLM-as-judge for production."

---

## ML Fundamentals Questions

### 7. "How do embeddings work?"

**Answer:**

"Embeddings are learned vector representations of text.

**How they're created**:
1. Text is tokenized into subwords
2. Each token gets an initial embedding (learned lookup table)
3. Transformer layers update embeddings with context
4. Final token embeddings are pooled (mean/CLS) into single vector

**Why they work for similarity**:
- Training objective: similar texts → close vectors
- Uses contrastive loss: minimize distance between positive pairs
- Similar to how CNN features capture image patterns

**Key insight**: Embeddings capture semantic meaning, not just keywords.
'fall protection' and 'safety harness' have similar embeddings even without shared words."

---

### 8. "Explain the loss function for embedding models"

**Answer:**

"Sentence Transformers use contrastive loss:

```
Loss = max(0, margin - sim(anchor, positive) + sim(anchor, negative))
```

**Intuition**:
- Want positive pairs closer than negative pairs by at least 'margin'
- If already satisfied, loss is 0 (no gradient)
- This pushes similar texts together, different texts apart

**Connection to traditional ML**:
- Similar to hinge loss in SVM
- Creates decision boundary in embedding space
- Margin concept from maximum margin classifiers"

---

### 9. "What's the bias-variance tradeoff in RAG?"

**Answer:**

"Several places it appears:

**1. Chunk size**:
- Small = high variance (fragmented), low bias
- Large = low variance, high bias (includes noise)

**2. Top-K retrieval**:
- Small K = precise but may miss info (high variance)
- Large K = complete but noisy (high bias)

**3. Temperature in LLM**:
- Low temp = deterministic (low variance, may repeat)
- High temp = creative (high variance, may hallucinate)

**4. Similarity threshold**:
- High threshold = precise, may miss relevant (high bias)
- Low threshold = complete, includes noise (high variance)

The art is finding the sweet spot for each use case."

---

### 10. "How would you improve retrieval quality?"

**Answer (give multiple strategies):**

1. **Hybrid retrieval**: Combine dense (semantic) + sparse (BM25) search
2. **Re-ranking**: Use cross-encoder to re-score top results
3. **Query expansion**: Generate multiple phrasings of query
4. **Fine-tuning embeddings**: Train on domain-specific pairs
5. **Metadata filtering**: Use document type, date, source
6. **Chunk optimization**: Experiment with sizes and overlap
7. **Multi-vector retrieval**: Embed at multiple granularities

---

## System Design Questions

### 11. "How would you scale this to 1 million documents?"

**Answer:**

```
Current architecture (< 100K docs):
- Single server, ChromaDB
- ~10ms retrieval latency

Scaled architecture:

1. Vector DB: Switch to Pinecone (managed, distributed)
   - Handles billions of vectors
   - Auto-scaling
   - ~50ms latency at scale

2. Caching layer: Add Redis
   - Cache frequent queries
   - Store embeddings for common terms
   - 5x latency reduction for cache hits

3. Async processing:
   - Background indexing with Celery
   - Queue for document uploads
   - Non-blocking API responses

4. Load balancing:
   - Multiple API servers behind nginx
   - Horizontal scaling for concurrent users

5. Batching:
   - Batch embedding generation
   - Batch vector insertions
```

---

### 12. "How do you handle document updates?"

**Answer:**

"Three strategies:

1. **Full re-index** (simplest):
   - Delete all vectors, re-index everything
   - Simple but slow for large datasets

2. **Delta updates** (balanced):
   - Track document hashes
   - Only re-index changed/new documents
   - Remove deleted document vectors

3. **Versioning** (most robust):
   - Keep old versions with timestamps
   - Query can specify 'as of' date
   - Garbage collect old versions periodically

Our implementation uses delta updates with document hashing."

---

## Behavioral Questions

### 13. "Why did you choose this project?"

**Answer:**

"Three reasons:

1. **Practical impact**: Construction industry has real documentation challenges—this solves a genuine problem

2. **Technical depth**: RAG combines multiple ML concepts—embeddings, retrieval, generation—giving me broad exposure

3. **Production focus**: Not just a notebook demo—includes API, frontend, evaluation, monitoring"

---

### 14. "What was the hardest part?"

**Answer:**

"Balancing retrieval precision with recall.

**The problem**: Safety documents have similar language across sections. Query 'fall protection requirements' would return many somewhat-relevant chunks.

**My solution**:
1. Experimented with chunk sizes (256, 512, 1024)
2. Added semantic chunking to respect section boundaries
3. Implemented similarity threshold filtering
4. Added source diversity (don't return 5 chunks from same page)

**Learning**: RAG quality is mostly about retrieval quality—garbage in, garbage out."

---

### 15. "What would you do differently?"

**Answer:**

"Three things:

1. **Start with evaluation first**: Built components before metrics. Should have defined success criteria and created test set upfront.

2. **More domain customization**: Would fine-tune embeddings on construction vocabulary if I had more time.

3. **Better error handling**: Current system fails silently on some edge cases. Would add more robust error recovery."

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG SYSTEM CHEAT SHEET                    │
├─────────────────────────────────────────────────────────────┤
│ Chunk size: 512 tokens                                       │
│ Overlap: 50 tokens (10%)                                     │
│ Embedding: all-MiniLM-L6-v2 (384 dim)                       │
│ Vector DB: ChromaDB                                          │
│ Similarity metric: Cosine                                    │
│ Top-K: 5                                                     │
│ Threshold: 0.7                                               │
│ LLM: GPT-4 Turbo                                             │
│ Temperature: 0.1                                             │
├─────────────────────────────────────────────────────────────┤
│ Retrieval metrics: Precision, Recall, MRR, NDCG             │
│ Generation metrics: Faithfulness, Relevance, Completeness   │
├─────────────────────────────────────────────────────────────┤
│ Agent: REACT framework                                       │
│ Tools: search_documents, compare_sources, calculate          │
│ Max iterations: 5                                            │
└─────────────────────────────────────────────────────────────┘
```
