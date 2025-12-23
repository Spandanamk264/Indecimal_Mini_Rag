"""
Indecimal mini RAG - Robust Backend API
"""
import os
import re
import json
import http.client
import logging
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Indecimal mini RAG", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq Configuration - Use environment variable for security
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

import requests

def call_groq_llm(system_prompt: str, user_prompt: str) -> str:
    """Calls Groq API using requests library for better reliability"""
    url = "https://api.groq.com/openai/v1/chat/completions"
    
    headers = {
        'Authorization': f'Bearer {GROQ_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1024
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code != 200:
            print(f"!!! GROQ API ERROR: {response.status_code} - {response.text}")
            logger.error(f"Groq API Error {response.status_code}: {response.text}")
            return f"Error: Unable to generate answer. (API Status: {response.status_code})"
            
        data = response.json()
        return data["choices"][0]["message"]["content"]
        
    except Exception as e:
        print(f"!!! GROQ CONNECTION ERROR: {e}")
        logger.error(f"Groq Connection Failed: {e}")
        return "I encountered a connection issue with the AI service. Please try again."

# Document Store
class DocumentStore:
    def __init__(self):
        self.documents: Dict[str, str] = {}
        self.chunks: List[Dict[str, Any]] = []
        self.vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1, 2))
        self.vectors = None
        self.indexed = False
    
    def add_document(self, name: str, content: str):
        self.documents[name] = content
        self.indexed = False
        self._rebuild_index()
    
    def _chunk_text(self, text: str, name: str, size: int = 400) -> List[Dict]:
        # Simple chunking by words
        text = re.sub(r'\s+', ' ', text).strip()
        words = text.split()
        chunks = []
        for i in range(0, len(words), size - 50):
            chunk = ' '.join(words[i:i + size])
            if len(chunk) > 50:
                chunks.append({'content': chunk, 'source': name, 'idx': len(chunks)})
        return chunks
    
    def _rebuild_index(self):
        self.chunks = []
        for name, content in self.documents.items():
            self.chunks.extend(self._chunk_text(content, name))
        
        if self.chunks:
            texts = [c['content'] for c in self.chunks]
            try:
                self.vectors = self.vectorizer.fit_transform(texts)
                self.indexed = True
                logger.info(f"Indexed {len(self.chunks)} chunks from {len(self.documents)} docs")
            except Exception as e:
                logger.error(f"Indexing error: {e}")
                self.indexed = False
        else:
            self.indexed = False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if not self.indexed or not self.chunks:
            return []
        
        try:
            query_vec = self.vectorizer.transform([query])
            scores = cosine_similarity(query_vec, self.vectors)[0]
            
            results = []
            for idx in np.argsort(scores)[::-1][:top_k]:
                if scores[idx] > 0.01:
                    results.append({
                        'content': self.chunks[idx]['content'],
                        'source': self.chunks[idx]['source'],
                        'score': float(scores[idx])
                    })
            return results
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

store = DocumentStore()

# Models
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    processing_time_ms: float

# Helper
def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_answer_groq(question: str, results: List[Dict]) -> str:
    if not results:
        return "I couldn't find relevant information in the uploaded documents to answer your question."
    
    # Prepare Context
    context_text = ""
    for r in results[:4]:
        name = os.path.basename(r['source'])
        context_text += f"-- Source: {name}\n{r['content']}\n\n"
    
    system_prompt = """You are Indecimal's Intelligent Document Assistant.
Generate a structured, professional, and neat answer based ONLY on the provided context.

Rules:
- Use Markdown formatting (bold, lists).
- Be concise and direct.
- Do NOT make up info not in the context.
- If the answer isn't in the docs, state that kindly.
"""
    
    user_prompt = f"""Context:
{context_text}

Question: {question}

Answer:"""
    
    return call_groq_llm(system_prompt, user_prompt)

# Endpoints
@app.get("/")
async def root():
    return {"name": "Indecimal mini RAG", "status": "running"}

@app.get("/api/v1/stats")
async def stats():
    return {
        "total_vectors": len(store.chunks),
        "documents_loaded": len(store.documents),
        "document_names": [os.path.basename(n) for n in store.documents.keys()]
    }

@app.post("/api/v1/upload")
async def upload(file: UploadFile = File(...)):
    filename = file.filename
    logger.info(f"Receiving file: {filename}")
    
    content = await file.read()
    text = ""
    
    try:
        # Simple text decoding for .txt and .md
        try:
            text = content.decode('utf-8')
        except:
            text = content.decode('latin-1', errors='ignore')
            
        # Basic check to avoid indexing binary garbage if someone uploads a real PDF without extraction
        if "\0" in text[:1000]:
            return {
                "status": "error",
                "message": "Binary file detected. Please upload .txt or .md files for best results."
            }
            
    except Exception as e:
        logger.error(f"File decode error: {e}")
        raise HTTPException(status_code=400, detail="Could not read file content")
    
    store.add_document(filename, text)
    
    return {
        "status": "success",
        "filename": filename,
        "chunks_created": len(store.chunks),
        "total_documents": len(store.documents)
    }

@app.post("/api/v1/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    import time
    start = time.time()
    
    results = store.search(req.question, req.top_k)
    answer = generate_answer_groq(req.question, results)
    
    # Calc confidence
    raw_score = results[0]['score'] if results else 0.0
    display_confidence = min(raw_score * 3.5, 0.99) if results else 0.0
    
    sources = []
    if results:
        for r in results:
            sources.append({
                "source": os.path.basename(r['source']),
                "score": r['score'],
                "preview": clean_text(r['content'][:150])
            })
    
    return QueryResponse(
        answer=answer,
        sources=sources,
        confidence=display_confidence,
        processing_time_ms=(time.time() - start) * 1000
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
