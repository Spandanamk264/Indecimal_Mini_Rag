# Indecimal Mini RAG

A document question-answering system built for querying construction documentation. Upload your docs, ask questions, get answers grounded in the actual content.

## Overview

This project implements a retrieval-augmented generation (RAG) pipeline. The system ingests documents, chunks them into smaller segments, and uses vector similarity to find relevant context when answering user queries. An LLM then synthesizes the retrieved context into a coherent response.

## Tech Stack

Backend: Python with FastAPI. Vector search uses TF-IDF and cosine similarity via scikit-learn. LLM inference handled through Groq API (Llama 3).

Frontend: React with Vite for the build tooling.

## Setup

### Backend

Navigate to the backend directory and install dependencies:

```
cd backend
pip install -r requirements.txt
```

Set your Groq API key as an environment variable:

```
set GROQ_API_KEY=your_api_key
```

Start the server:

```
python real_rag_api.py
```

The API runs on port 8000.

### Frontend

Navigate to the frontend directory:

```
cd frontend
npm install
npm run dev
```

The client runs on port 5173.

## Usage

1. Open the web interface
2. Upload documents using the sidebar (supports txt, md, pdf, docx)
3. Type your question in the chat input
4. View the answer along with source citations

## Architecture

The pipeline follows these steps:

1. Document ingestion: Files are read and split into overlapping chunks
2. Indexing: Chunks are vectorized using TF-IDF
3. Retrieval: User query is vectorized and matched against indexed chunks
4. Generation: Top matching chunks are sent to the LLM as context
5. Response: LLM generates an answer based on the provided context

## Limitations

The current implementation uses in-memory storage, so uploaded documents do not persist across server restarts. For production use, a persistent vector store would be needed.

## File Structure

```
backend/
  real_rag_api.py
  requirements.txt
frontend/
  src/
    App.jsx
    App.css
  package.json
README.md
```

## Author

Spandana M K
