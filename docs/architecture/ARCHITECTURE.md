# Architecture Deep Dive

## System Components

### 1. Document Ingestion Pipeline

```
Raw Documents (PDF, DOCX, TXT)
         │
         ▼
┌─────────────────┐
│ Document Loader │
│  - Factory pattern for file types
│  - Metadata extraction
│  - Error handling
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  Text Cleaner   │
│  - Unicode normalization
│  - Noise removal
│  - Code/regulation preservation
└─────────────────┘
         │
         ▼
┌─────────────────┐
│    Chunker      │
│  - Semantic boundaries
│  - Token-based fallback
│  - Overlap handling
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Embeddings    │
│  - Sentence Transformers
│  - OpenAI option
│  - Caching
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  Vector Store   │
│  - ChromaDB (default)
│  - FAISS option
│  - Pinecone option
└─────────────────┘
```

### 2. Query Pipeline

```
User Query
    │
    ▼
┌──────────────────┐
│   Query Router   │
│  - Complexity analysis
│  - Mode selection
│  - Keyword detection
└──────────────────┘
    │
    ├── Simple Query ──────────────────────┐
    │                                       │
    │   ┌──────────────────┐              │
    │   │  Query Embedding │              │
    │   └──────────────────┘              │
    │         │                            │
    │         ▼                            │
    │   ┌──────────────────┐              │
    │   │ Vector Similarity│              │
    │   │     Search       │              │
    │   └──────────────────┘              │
    │         │                            │
    │         ▼                            │
    │   ┌──────────────────┐              │
    │   │  Context Filter  │              │
    │   │  (threshold +    │              │
    │   │   deduplication) │              │
    │   └──────────────────┘              │
    │         │                            │
    │         ▼                            │
    │   ┌──────────────────┐              │
    │   │  LLM Generation  │◄─────────────┘
    │   │  with Grounding  │
    │   └──────────────────┘
    │         │
    │         ▼
    │   Response with Sources
    │
    └── Complex Query (Agent Mode) ────────┐
                                            │
        ┌───────────────────────────────────┘
        │
        ▼
    ┌──────────────────┐
    │   REACT Agent    │
    │                  │
    │ ┌──────────────┐ │
    │ │   Thought    │ │
    │ └──────────────┘ │
    │       │          │
    │       ▼          │
    │ ┌──────────────┐ │
    │ │   Action     │ │  → Tools: search, compare, calculate
    │ └──────────────┘ │
    │       │          │
    │       ▼          │
    │ ┌──────────────┐ │
    │ │ Observation  │ │
    │ └──────────────┘ │
    │       │          │
    │       ▼          │
    │  [Loop until    │
    │   Final Answer] │
    └──────────────────┘
            │
            ▼
    Response with Reasoning Trace
```

### 3. Data Flow Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐  │
│  │ Query   │  │Response │  │ Sources │  │ Agent Reasoning │  │
│  │ Input   │  │ Display │  │  List   │  │     Trace       │  │
│  └────┬────┘  └────▲────┘  └────▲────┘  └───────▲─────────┘  │
└───────│────────────│────────────│───────────────│─────────────┘
        │            │            │               │
        │ HTTP POST  │ JSON       │               │
        ▼            │            │               │
┌───────────────────────────────────────────────────────────────┐
│                         API LAYER                              │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                    /api/v1/query                          │ │
│  │  1. Validate request                                      │ │
│  │  2. Route to pipeline                                     │ │
│  │  3. Format response                                       │ │
│  │  4. Log metrics                                           │ │
│  └──────────────────────────────────────────────────────────┘ │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                      RAG PIPELINE                              │
│                                                                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐   │
│  │  Embedding  │───▶│  Retriever  │───▶│   Generator     │   │
│  │   Model     │    │             │    │                 │   │
│  │             │    │ • Embed     │    │ • Build prompt  │   │
│  │ • encode()  │    │   query     │    │ • Call LLM      │   │
│  │             │    │ • Search    │    │ • Check ground  │   │
│  │             │    │   vectors   │    │ • Format resp   │   │
│  └─────────────┘    │ • Filter    │    │                 │   │
│                     │   results   │    │                 │   │
│                     └─────────────┘    └─────────────────┘   │
│                            │                    │              │
│                            ▼                    ▼              │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                      EVALUATION                          │ │
│  │  • Retrieval metrics (Precision, Recall, MRR)           │ │
│  │  • Generation metrics (Faithfulness, Relevance)         │ │
│  │  • End-to-end metrics (Latency, Cost)                   │ │
│  └─────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                     STORAGE LAYER                              │
│                                                                │
│  ┌─────────────────┐  ┌─────────────────┐                    │
│  │   Vector DB     │  │   Document      │                    │
│  │   (ChromaDB)    │  │   Storage       │                    │
│  │                 │  │                 │                    │
│  │ • embeddings    │  │ • raw files     │                    │
│  │ • metadata      │  │ • chunk cache   │                    │
│  │ • indices       │  │                 │                    │
│  └─────────────────┘  └─────────────────┘                    │
│                                                                │
└───────────────────────────────────────────────────────────────┘
```

## Design Decisions

### Why These Choices?

| Decision | Options Considered | Choice | Rationale |
|----------|-------------------|--------|-----------|
| Vector DB | Chroma, FAISS, Pinecone | ChromaDB | Easy setup, good for dev/small-mid scale |
| Embedding | MiniLM, MPNET, OpenAI | MiniLM | Fast, free, good quality |
| Chunking | Fixed, Semantic, Hybrid | Hybrid | Best of both worlds |
| LLM | GPT-3.5, GPT-4, Claude | GPT-4 | Best quality for grounded answers |
| Framework | LangChain, Custom | Custom | Full control, learning opportunity |

### Scalability Considerations

```
Current: Single-server setup
         ↓
Future:  Distributed architecture

┌─────────────────────────────────────────────────────────────┐
│                    LOAD BALANCER                             │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   API Server  │   │   API Server  │   │   API Server  │
│      #1       │   │      #2       │   │      #3       │
└───────────────┘   └───────────────┘   └───────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     PINECONE (Managed)                       │
│           Distributed Vector Storage                         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                       REDIS CACHE                            │
│           Query caching, Session storage                     │
└─────────────────────────────────────────────────────────────┘
```
