"""
Step 1 of the RAG pipeline: Extract audio from course videos as MP3 files.

Expects video files in `videos/` and writes one MP3 per video to `audios/`.
Filenames are safely parsed to extract the tutorial index and title.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from config import AUDIOS_DIR, VIDEOS_DIR

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".webm", ".mkv", ".avi", ".mov"}


def parse_video_filename(filename: str) -> tuple[str, str]:
    """
    Safely extract (tutorial_number, title) from video filename.
    Handles formats like:
      - '03 - Basic Structure of HTML.webm'
      - '03 _ Basic Structure.mp4'
      - '03. Introduction.mp4'
      - 'Lesson 3 - Introduction.mp4'
    """
    stem = Path(filename).stem
    # Remove separators like pipes
    cleaned = stem.split("｜")[0].strip()

    # Pattern: Match leading number optionally prefixed by words
    match = re.match(r"^(?:(?:Lesson|Video|Tutorial)\s*)?(\d+)\s*[-_.:\s]+(.*)$", cleaned, re.IGNORECASE)
    if match:
        tutorial_number = match.group(1).zfill(2)
        title = match.group(2).strip() or "Untitled"
        # Sanitize title for filesystem safety
        title = re.sub(r'[\\/*?:"<>|]', "", title).strip()
        return tutorial_number, title

    # Fallback if no numeric prefix found
    sanitized = re.sub(r'[\\/*?:"<>|]', "", cleaned).strip()
    return "00", sanitized or "video"


def convert_video_to_mp3(video_path: Path, output_path: Path) -> bool:
    """Extract the audio track from a video file using ffmpeg."""
    try:
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite without asking
            "-hide_banner",
            "-loglevel", "error",
            "-i", str(video_path),
            "-vn",  # Disable video recording
            "-acodec", "libmp3lame",
            "-q:a", "2",
            str(output_path),
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except FileNotFoundError:
        print("Error: ffmpeg is not installed or not in PATH.")
        return False
    except subprocess.CalledProcessError as exc:
        print(f"Error converting {video_path.name}: {exc.stderr.strip() if exc.stderr else exc}")
        return False


def main() -> None:
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIOS_DIR.mkdir(parents=True, exist_ok=True)

    video_files = [
        f for f in VIDEOS_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    ]

    if not video_files:
        print(f"No supported video files found in {VIDEOS_DIR.resolve()}.")
        print(f"Supported formats: {', '.join(SUPPORTED_VIDEO_EXTENSIONS)}")
        return

    print(f"Found {len(video_files)} video file(s) to process.")

    converted = 0
    for video_file in sorted(video_files):
        tutorial_number, title = parse_video_filename(video_file.name)
        output_name = f"{tutorial_number}_{title}.mp3"
        output_path = AUDIOS_DIR / output_name

        print(f"Converting: {video_file.name} -> {output_name}")
        if convert_video_to_mp3(video_file, output_path):
            converted += 1

    print(f"\nFinished converting {converted}/{len(video_files)} video(s) to MP3 in {AUDIOS_DIR.resolve()}.")


if __name__ == "__main__":
    main()
