"""
Step 2 of the RAG pipeline: Transcribe audio to timestamped English JSON.

Uses OpenAI Whisper to transcribe/translate each MP3 in `audios/` and saves
structured JSON transcripts with timestamped subtitle chunks into `jsons/`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from config import AUDIOS_DIR, JSONS_DIR, SOURCE_LANGUAGE, WHISPER_MODEL


def parse_audio_filename(filename: str) -> tuple[str, str]:
    """
    Safely extract tutorial number and title from audio filename.
    Handles '03_Basic Structure.mp3' or '03 - Basic Structure.mp3'.
    """
    stem = Path(filename).stem
    match = re.match(r"^(\d+)[_-\s]+(.*)$", stem)
    if match:
        return match.group(1).zfill(2), match.group(2).strip()
    return "00", stem


def segments_to_chunks(segments: list[dict[str, Any]], number: str, title: str) -> list[dict[str, Any]]:
    """Convert Whisper raw segments into standardized chunk schema."""
    return [
        {
            "number": number,
            "title": title,
            "start": round(float(segment["start"]), 2),
            "end": round(float(segment["end"]), 2),
            "text": segment["text"].strip(),
        }
        for segment in segments
        if segment.get("text", "").strip()
    ]


def main() -> None:
    try:
        import whisper
    except ModuleNotFoundError:
        print("Error: openai-whisper is not installed. Please run `pip install -r requirements.txt`.")
        return

    JSONS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIOS_DIR.mkdir(parents=True, exist_ok=True)

    audio_files = [f for f in AUDIOS_DIR.iterdir() if f.is_file() and f.suffix.lower() == ".mp3"]

    if not audio_files:
        print(f"No .mp3 audio files found in {AUDIOS_DIR.resolve()}.")
        print("Run `video_to_mp3.py` first.")
        return

    print(f"Loading Whisper model '{WHISPER_MODEL}'...")
    try:
        model = whisper.load_model(WHISPER_MODEL)
    except Exception as exc:
        print(f"Failed to load Whisper model '{WHISPER_MODEL}': {exc}")
        return

    print(f"Processing {len(audio_files)} audio file(s)...")

    for audio_file in sorted(audio_files):
        number, title = parse_audio_filename(audio_file.name)
        json_filename = f"{audio_file.stem}.json"
        json_path = JSONS_DIR / json_filename

        print(f"\nTranscribing: {audio_file.name} (Source language: {SOURCE_LANGUAGE})")

        try:
            result = model.transcribe(
                audio=str(audio_file),
                language=SOURCE_LANGUAGE,
                task="translate",  # Translate non-English audio to English subtitles
                word_timestamps=False,
            )

            chunks = segments_to_chunks(result.get("segments", []), number, title)
            output = {
                "number": number,
                "title": title,
                "chunks": chunks,
                "full_text": result.get("text", "").strip(),
            }

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            print(f"Saved: {json_path.name} ({len(chunks)} subtitle segments)")

        except Exception as exc:
            print(f"Error transcribing {audio_file.name}: {exc}")

    print(f"\nTranscription complete. Transcripts saved to {JSONS_DIR.resolve()}.")


if __name__ == "__main__":
    main()
