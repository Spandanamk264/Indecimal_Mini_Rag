"""
Standalone FastAPI for Construction RAG - Quick Demo
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

app = FastAPI(title="Construction RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    use_agent: Optional[bool] = None

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    mode_used: str
    processing_time_ms: float
    token_usage: Dict[str, int]

# Mock data for demo
MOCK_RESPONSES = {
    "ppe": {
        "answer": "Required PPE on construction sites includes:\n\n1. **Hard Hats** - ANSI Z89.1 certified, required at all times\n2. **Safety Glasses** - ANSI Z87.1 rated eye protection\n3. **High-Visibility Vests** - Class 2 or higher\n4. **Steel-Toed Boots** - ASTM F2413 rated footwear\n5. **Hearing Protection** - When noise exceeds 85 dB\n6. **Gloves** - Task-appropriate hand protection\n\n[Source: Construction Safety Manual, Section 3.2]",
        "sources": [{"source": "construction_safety_manual.txt", "score": 0.95, "preview": "Personal Protective Equipment requirements..."}],
        "confidence": 0.92
    },
    "fall": {
        "answer": "Fall protection is required when working at heights of **6 feet or more** above lower levels (per OSHA 29 CFR 1926.501).\n\nRequired measures include:\n- Guardrail systems (42\" high)\n- Personal fall arrest systems (harness + lanyard)\n- Safety nets\n- Floor hole covers\n\n**Critical**: Workers must be trained before working at heights.\n\n[Source: Construction Safety Manual, Section 4.1]",
        "sources": [{"source": "construction_safety_manual.txt", "score": 0.93, "preview": "Fall Protection Requirements..."}],
        "confidence": 0.90
    },
    "contract": {
        "answer": "Based on the Subcontractor Agreement for the Riverside Tower Construction Project:\n\n- **Contract Value**: $4,852,000.00\n- **Scope**: Electrical Systems Installation\n- **Duration**: January 15, 2024 - December 31, 2024\n- **Insurance Required**: $2,000,000 per occurrence\n\n[Source: Subcontractor Agreement, Article 2]",
        "sources": [{"source": "subcontractor_agreement.txt", "score": 0.91, "preview": "Contract Terms and Conditions..."}],
        "confidence": 0.88
    },
    "default": {
        "answer": "I found relevant information in the construction documents. The system is ready to answer questions about:\n\n- Safety requirements and PPE\n- Fall protection regulations\n- Contract terms and values\n- Inspection findings\n\nPlease ask a specific question about construction safety, contracts, or inspections.",
        "sources": [{"source": "construction_safety_manual.txt", "score": 0.75}],
        "confidence": 0.70
    }
}

@app.get("/")
async def root():
    return {"name": "Construction RAG API", "status": "running", "docs": "/docs"}

@app.get("/api/v1/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat(), "version": "1.0.0", "vector_count": 156}

@app.get("/api/v1/stats")
async def stats():
    return {
        "total_vectors": 156,
        "queries_processed": 0,
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dimension": 384,
        "top_k": 5,
        "similarity_threshold": 0.7
    }

@app.post("/api/v1/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    q = request.question.lower()
    
    if "ppe" in q or "equipment" in q or "wear" in q:
        resp = MOCK_RESPONSES["ppe"]
    elif "fall" in q or "height" in q or "protection" in q:
        resp = MOCK_RESPONSES["fall"]
    elif "contract" in q or "value" in q or "electrical" in q:
        resp = MOCK_RESPONSES["contract"]
    else:
        resp = MOCK_RESPONSES["default"]
    
    return QueryResponse(
        answer=resp["answer"],
        sources=resp["sources"],
        confidence=resp["confidence"],
        mode_used="agent" if request.use_agent else "simple",
        processing_time_ms=245.5,
        token_usage={"prompt": 450, "completion": 120, "total": 570}
    )

@app.post("/api/v1/index")
async def index_documents():
    return {"status": "completed", "documents_processed": 3, "chunks_created": 156, "total_vectors": 156, "processing_time_seconds": 2.5}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
