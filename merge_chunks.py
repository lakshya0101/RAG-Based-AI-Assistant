"""
Step 3 of the RAG pipeline: Merge small Whisper segments into larger retrieval chunks.

Groups every N consecutive segments from `jsons/` into one chunk and writes
the result to `new_jsons/`. Larger chunks provide the LLM with richer context per hit.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from config import JSONS_DIR, NEW_JSONS_DIR

CHUNKS_PER_GROUP = int(os.getenv("CHUNKS_PER_GROUP", "6"))


def merge_chunk_group(chunk_group: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine adjacent subtitle segments into a single consolidated chunk."""
    combined_text = " ".join(chunk["text"].strip() for chunk in chunk_group if chunk.get("text"))
    return {
        "number": str(chunk_group[0].get("number", "00")),
        "title": str(chunk_group[0].get("title", "Untitled")),
        "start": float(chunk_group[0].get("start", 0.0)),
        "end": float(chunk_group[-1].get("end", 0.0)),
        "text": combined_text,
    }


def merge_file_chunks(chunks: list[dict[str, Any]], group_size: int = CHUNKS_PER_GROUP) -> list[dict[str, Any]]:
    """Split a list of segments into fixed-size groups and merge each group."""
    if not chunks:
        return []

    num_groups = math.ceil(len(chunks) / group_size)
    merged = []

    for i in range(num_groups):
        start = i * group_size
        end = min((i + 1) * group_size, len(chunks))
        group = chunks[start:end]
        if group:
            merged.append(merge_chunk_group(group))

    return merged


def main() -> None:
    JSONS_DIR.mkdir(parents=True, exist_ok=True)
    NEW_JSONS_DIR.mkdir(parents=True, exist_ok=True)

    json_files = [f for f in JSONS_DIR.iterdir() if f.is_file() and f.suffix.lower() == ".json"]

    if not json_files:
        print(f"No JSON transcript files found in {JSONS_DIR.resolve()}.")
        print("Run `mp3_to_json.py` first or run `generate_sample_data.py`.")
        return

    total_orig = 0
    total_merged = 0

    for json_file in sorted(json_files):
        with open(json_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as exc:
                print(f"Skipping corrupt JSON file {json_file.name}: {exc}")
                continue

        chunks = data.get("chunks", [])
        if not chunks:
            continue

        merged_chunks = merge_file_chunks(chunks, CHUNKS_PER_GROUP)
        total_orig += len(chunks)
        total_merged += len(merged_chunks)

        output_path = NEW_JSONS_DIR / json_file.name
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "number": data.get("number", "00"),
                    "title": data.get("title", ""),
                    "full_text": data.get("full_text", ""),
                    "chunks": merged_chunks,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"Merged {json_file.name}: {len(chunks)} segments -> {len(merged_chunks)} chunks (group size: {CHUNKS_PER_GROUP})")

    print(f"\nDone. Consolidated {total_orig} segments into {total_merged} chunks in {NEW_JSONS_DIR.resolve()}.")


if __name__ == "__main__":
    main()
