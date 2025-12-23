@echo off
echo ===================================================
echo Construction RAG System - Setup Script
echo ===================================================

echo [1/4] checking Python environment...
if not exist "..\construction-rag-env" (
    echo Creating virtual environment...
    python -m venv ..\construction-rag-env
)

echo [2/4] Installing Backend Dependencies...
call ..\construction-rag-env\Scripts\activate
cd backend
pip install --upgrade pip
pip install fastapi uvicorn pydantic pydantic-settings python-dotenv loguru
pip install tiktoken sentence-transformers chromadb openai
pip install -r requirements.txt

echo [3/4] Setting up Environment Variables...
if not exist .env (
    copy .env.example .env
    echo Created .env file. Please edit it with your API keys.
)

echo [4/4] Installing Frontend Dependencies...
cd ..\frontend
call npm install

echo ===================================================
echo Setup Complete!
echo.
echo To run the Backend:
echo   cd backend
echo   ..\..\construction-rag-env\Scripts\python -m uvicorn src.api.main:app --reload
echo.
echo To run the Frontend:
echo   cd frontend
echo   npm run dev
echo ===================================================
pause
