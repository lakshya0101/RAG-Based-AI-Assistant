"""
Core RAG Engine: Embed query, retrieve relevant course chunks, generate answer via LLM.

Shared by the CLI client (process_incoming.py) and the FastAPI web server (api.py).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    EMBED_MODEL,
    EMBEDDINGS_FILE,
    LLM_MODEL,
    OLLAMA_BASE_URL,
    TOP_K_RESULTS,
)

_df_cache: pd.DataFrame | None = None


class RagEngineError(Exception):
    """Base exception for RAG engine failures."""


class OllamaUnavailableError(RagEngineError):
    """Raised when the Ollama server cannot be reached."""


class EmbeddingsNotFoundError(RagEngineError):
    """Raised when the precomputed embeddings file is missing."""


def seconds_to_mmss(seconds: float | int) -> str:
    """Format numeric seconds as 'mm:ss' string."""
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    return f"{minutes}:{secs:02d}"


def load_embeddings(force_reload: bool = False) -> pd.DataFrame:
    """Load and cache the DataFrame containing chunk embeddings."""
    global _df_cache
    if _df_cache is not None and not force_reload:
        return _df_cache

    path = Path(EMBEDDINGS_FILE)
    if not path.exists():
        raise EmbeddingsNotFoundError(
            f"Embeddings file '{EMBEDDINGS_FILE}' not found. "
            "Run `preprocess_new_json.py` or `generate_sample_data.py` to create the vector index."
        )

    try:
        _df_cache = joblib.load(path)
        return _df_cache
    except Exception as exc:
        raise RagEngineError(f"Failed to load embeddings from '{path.resolve()}': {exc}") from exc


def create_embedding(texts: list[str]) -> list[list[float]]:
    """Generate vector embeddings for input texts using Ollama."""
    url = f"{OLLAMA_BASE_URL}/api/embed"
    try:
        response = requests.post(
            url,
            json={"model": EMBED_MODEL, "input": texts},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as exc:
        raise OllamaUnavailableError(
            f"Could not connect to Ollama at '{OLLAMA_BASE_URL}'. "
            f"Please ensure Ollama is running (`ollama serve`) and model '{EMBED_MODEL}' is pulled."
        ) from exc

    if "embeddings" in data and isinstance(data["embeddings"], list):
        return data["embeddings"]

    raise RagEngineError(f"Unexpected Ollama embed response format: {data}")


def run_inference(prompt: str, model: str = LLM_MODEL) -> str:
    """Send prompt to Ollama LLM and return generated text answer."""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    try:
        response = requests.post(
            url,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.exceptions.RequestException as exc:
        raise OllamaUnavailableError(
            f"Could not connect to Ollama at '{OLLAMA_BASE_URL}'. "
            f"Please ensure Ollama is running (`ollama serve`) and model '{model}' is pulled."
        ) from exc


def retrieve_relevant_chunks(
    df: pd.DataFrame,
    question_embedding: list[float],
    top_k: int = TOP_K_RESULTS,
) -> pd.DataFrame:
    """Find top K chunks with highest cosine similarity to question embedding."""
    if df.empty:
        raise RagEngineError("Embeddings index is empty.")

    embeddings_matrix = np.vstack(df["embedding"].values)
    similarities = cosine_similarity(
        embeddings_matrix,
        [question_embedding],
    ).flatten()

    effective_k = min(top_k, len(df))
    best_indices = similarities.argsort()[::-1][:effective_k]

    results = df.iloc[best_indices].copy()
    results["score"] = [float(similarities[i]) for i in best_indices]
    results = results.sort_values("start")

    results["start_mmss"] = results["start"].apply(seconds_to_mmss)
    results["end_mmss"] = results["end"].apply(seconds_to_mmss)

    return results


def build_prompt(query: str, context_df: pd.DataFrame) -> str:
    """Format retrieval chunks and query into a structured system prompt."""
    records = []
    for _, row in context_df.iterrows():
        records.append({
            "video_number": str(row.get("number", "00")),
            "video_title": str(row.get("title", "")),
            "timestamp": f"{row.get('start_mmss', '0:00')} - {row.get('end_mmss', '0:00')}",
            "content": str(row.get("text", "")).strip(),
        })

    context_json = json.dumps(records, indent=2)

    return f"""\
You are an expert AI Teaching Assistant for a Web Development course.

Below are subtitle excerpts retrieved from course lecture videos:
{context_json}

--------------------------------------------------------------
Student Question:
"{query}"

Instructions for your answer:
1. Answer clearly, concisely, and accurately as a supportive instructor.
2. Explicitly cite the most relevant Video Number, Video Title, and the exact Timestamp range (mm:ss format).
3. If multiple relevant sections exist, mention the primary one first.
4. If the question is outside the scope of the course materials, politely state that you can only answer topics covered in the course videos.
"""


def chunks_to_sources(context_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert retrieved context DataFrame into a clean list of serializable source dicts."""
    sources = []
    for _, row in context_df.iterrows():
        sources.append({
            "video_number": str(row.get("number", "00")).strip(),
            "title": str(row.get("title", "")).strip(),
            "start": str(row.get("start_mmss", "0:00")),
            "end": str(row.get("end_mmss", "0:00")),
            "score": round(float(row.get("score", 0.0)), 3),
            "excerpt": str(row.get("text", "")).strip()[:200],
        })
    return sources


def ask_question(query: str, top_k: int = TOP_K_RESULTS) -> dict[str, Any]:
    """
    End-to-end RAG query:
    1. Validates query
    2. Embeds question
    3. Retrieves top relevant chunks
    4. Generates response via Ollama LLM
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        raise RagEngineError("Question cannot be empty.")

    df = load_embeddings()
    question_embedding = create_embedding([cleaned_query])[0]
    relevant_chunks = retrieve_relevant_chunks(df, question_embedding, top_k)
    prompt = build_prompt(cleaned_query, relevant_chunks)
    answer = run_inference(prompt)

    return {
        "question": cleaned_query,
        "answer": answer,
        "sources": chunks_to_sources(relevant_chunks),
    }


def check_health() -> dict[str, Any]:
    """Inspect status of embeddings index and Ollama service connectivity."""
    status: dict[str, Any] = {
        "ready": False,
        "embeddings_loaded": False,
        "ollama_reachable": False,
        "chunk_count": 0,
        "models_available": [],
    }

    # Check embeddings
    try:
        df = load_embeddings()
        status["embeddings_loaded"] = True
        status["chunk_count"] = len(df)
    except Exception:
        status["embeddings_loaded"] = False

    # Check Ollama connectivity
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if response.ok:
            status["ollama_reachable"] = True
            tags_data = response.json()
            status["models_available"] = [m.get("name") for m in tags_data.get("models", [])]
    except Exception:
        status["ollama_reachable"] = False

    status["ready"] = bool(status["embeddings_loaded"] and status["ollama_reachable"])
    return status
