"""
FastAPI Server for RAG-Based AI Teaching Assistant.

Provides REST endpoints for questioning the course index and monitoring health.
"""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import ALLOWED_ORIGINS, HOST, PORT
from rag_engine import (
    EmbeddingsNotFoundError,
    OllamaUnavailableError,
    RagEngineError,
    ask_question,
    check_health,
)

app = FastAPI(
    title="RAG-Based AI Teaching Assistant API",
    description="Query course lectures by topic to get AI-generated explanations with exact video numbers and timestamp citations.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The question to ask about the course material.",
        examples=["Where is the CSS box model explained?"],
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=15,
        description="Number of relevant chunks to retrieve.",
    )


class Source(BaseModel):
    video_number: str
    title: str
    start: str
    end: str
    score: float
    excerpt: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]


class HealthResponse(BaseModel):
    ready: bool = Field(..., description="True if both embeddings and Ollama are ready.")
    embeddings_loaded: bool
    ollama_reachable: bool
    chunk_count: int
    models_available: list[str] = []


@app.get("/", tags=["Info"])
def root():
    return {
        "title": "RAG-Based AI Teaching Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "POST /ask": "Submit a question to receive AI answer + video timestamp citations",
            "GET /health": "Check system readiness, index state, and Ollama connectivity",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return check_health()


@app.post("/ask", response_model=AskResponse, tags=["Q&A"])
def ask(body: AskRequest):
    try:
        if body.top_k is not None:
            result = ask_question(body.question, top_k=body.top_k)
        else:
            result = ask_question(body.question)
        return result
    except EmbeddingsNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Embeddings unavailable: {exc}",
        ) from exc
    except OllamaUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Ollama service error: {exc}",
        ) from exc
    except RagEngineError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your request.",
        ) from exc


if __name__ == "__main__":
    uvicorn.run("api:app", host=HOST, port=PORT, reload=True)
