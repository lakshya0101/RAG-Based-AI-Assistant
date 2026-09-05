"""
Step 4a of the RAG pipeline: Embed raw Whisper JSON chunks via Ollama.

Reads raw segment JSON from `jsons/`, calls the embedding model via Ollama,
and saves a pandas DataFrame to `embeddings.joblib` for vector retrieval.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import joblib
import pandas as pd
import requests

from config import EMBED_MODEL, JSONS_DIR, OLLAMA_BASE_URL

BATCH_SIZE = 8
OUTPUT_FILE = Path("embeddings.joblib")


def create_embedding(texts: list[str]) -> list[list[float]]:
    """Request vector embeddings for a batch of texts from Ollama."""
    url = f"{OLLAMA_BASE_URL}/api/embed"
    try:
        response = requests.post(
            url,
            json={"model": EMBED_MODEL, "input": texts},
            timeout=300,
        )
        response.raise_for_status()
        data = response.json()
        if "embeddings" in data and isinstance(data["embeddings"], list):
            return data["embeddings"]
        print("Unexpected response format from Ollama:", data)
        return []
    except requests.exceptions.RequestException as exc:
        print(f"Failed to connect to Ollama at {url}: {exc}")
        print("Make sure Ollama is running (`ollama serve`) and model is pulled (`ollama pull bge-m3`).")
        return []


def batch_items(items: list[Any], size: int) -> Iterator[tuple[int, list[Any]]]:
    """Yield (start_index, batch_slice) pairs."""
    for i in range(0, len(items), size):
        yield i, items[i : i + size]


def load_valid_chunks(content: dict[str, Any]) -> list[dict[str, Any]]:
    """Filter for chunks that contain non-empty text."""
    chunks = content.get("chunks", [])
    return [
        chunk for chunk in chunks
        if isinstance(chunk, dict) and chunk.get("text", "").strip()
    ]


def main() -> None:
    JSONS_DIR.mkdir(parents=True, exist_ok=True)
    json_files = [f for f in JSONS_DIR.iterdir() if f.is_file() and f.suffix.lower() == ".json"]

    if not json_files:
        print(f"No JSON files found in {JSONS_DIR.resolve()}.")
        print("Run `mp3_to_json.py` first.")
        return

    records = []
    chunk_id = 0

    print(f"Embedding raw chunks with model '{EMBED_MODEL}' from {JSONS_DIR.name}...")

    for file_path in sorted(json_files):
        print(f"\nProcessing: {file_path.name}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = json.load(f)
        except Exception as exc:
            print(f"Skipping unreadable file {file_path.name}: {exc}")
            continue

        valid_chunks = load_valid_chunks(content)
        if not valid_chunks:
            print("No valid text chunks found, skipping.")
            continue

        texts = [chunk["text"] for chunk in valid_chunks]

        for start_idx, text_batch in batch_items(texts, BATCH_SIZE):
            embeddings = create_embedding(text_batch)
            if not embeddings:
                print(f"Skipping batch at index {start_idx} due to embedding failure.")
                continue

            for chunk, embedding in zip(
                valid_chunks[start_idx : start_idx + len(embeddings)],
                embeddings,
            ):
                record = dict(chunk)
                record["chunk_id"] = chunk_id
                record["embedding"] = embedding
                records.append(record)
                chunk_id += 1

    if not records:
        print("No embeddings were generated. Ensure Ollama is running.")
        return

    print(f"\nFinished. Total embedded chunks: {len(records)}")
    df = pd.DataFrame.from_records(records)
    joblib.dump(df, OUTPUT_FILE)
    print(f"Saved embedding DataFrame to {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
