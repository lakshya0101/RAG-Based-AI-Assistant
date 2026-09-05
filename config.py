"""
Configuration loader for RAG Teaching Assistant.
Loads environment variables from .env file if available, providing fallbacks.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Automatically load .env if present
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = BASE_DIR / "videos"
AUDIOS_DIR = BASE_DIR / "audios"
JSONS_DIR = BASE_DIR / "jsons"
NEW_JSONS_DIR = BASE_DIR / "new_jsons"

# Ollama & LLM settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")

# Retrieval settings
EMBEDDINGS_FILE = os.getenv("EMBEDDINGS_FILE", "new_embeddings.joblib")
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))

# Audio / Whisper settings
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
SOURCE_LANGUAGE = os.getenv("SOURCE_LANGUAGE", "hi")

# API Server settings
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
