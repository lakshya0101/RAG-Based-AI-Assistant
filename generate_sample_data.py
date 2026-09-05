"""
Sample Data Generator for RAG-Based AI Teaching Assistant.

Creates sample course transcripts in `jsons/` and `new_jsons/`, and generates
an initial `new_embeddings.joblib` vector index so the application can be
tested and verified immediately after cloning, even before processing raw videos.
"""

from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import requests

from config import EMBED_MODEL, JSONS_DIR, NEW_JSONS_DIR, OLLAMA_BASE_URL

SAMPLE_LESSONS = [
    {
        "number": "01",
        "title": "Introduction to Web Development & Setup",
        "full_text": "Welcome to the web development course. In this video, we set up VS Code, install extensions, and understand how the internet works.",
        "chunks": [
            {
                "number": "01",
                "title": "Introduction to Web Development & Setup",
                "start": 0.0,
                "end": 65.0,
                "text": "Welcome everyone to the web development bootcamp. In this lecture, we will understand how browsers request HTML pages from web servers.",
            },
            {
                "number": "01",
                "title": "Introduction to Web Development & Setup",
                "start": 66.0,
                "end": 140.0,
                "text": "Let us install VS Code, the live server extension, and configure our workspace for efficient frontend development.",
            },
            {
                "number": "01",
                "title": "Introduction to Web Development & Setup",
                "start": 141.0,
                "end": 220.0,
                "text": "An overview of frontend technologies: HTML structures content, CSS handles styling and layout, and JavaScript provides interactivity.",
            },
        ],
    },
    {
        "number": "02",
        "title": "HTML Fundamentals & Semantic Tags",
        "full_text": "HTML tags, document structure, headings, paragraphs, links, images, and semantic HTML elements.",
        "chunks": [
            {
                "number": "02",
                "title": "HTML Fundamentals & Semantic Tags",
                "start": 0.0,
                "end": 75.0,
                "text": "Understanding the HTML5 boilerplate, doctype declaration, html, head, title, and body tags.",
            },
            {
                "number": "02",
                "title": "HTML Fundamentals & Semantic Tags",
                "start": 76.0,
                "end": 155.0,
                "text": "Working with text elements: headings h1 through h6, paragraphs, strong, em, and the quotation tag blockquote and q tags.",
            },
            {
                "number": "02",
                "title": "HTML Fundamentals & Semantic Tags",
                "start": 156.0,
                "end": 240.0,
                "text": "Semantic HTML tags including header, nav, main, section, article, aside, and footer improve accessibility and SEO.",
            },
        ],
    },
    {
        "number": "03",
        "title": "CSS Box Model & Styling Fundamentals",
        "full_text": "CSS selectors, colors, typography, and the complete CSS box model including margin, border, padding, and content.",
        "chunks": [
            {
                "number": "03",
                "title": "CSS Box Model & Styling Fundamentals",
                "start": 0.0,
                "end": 80.0,
                "text": "CSS syntax: selector, property, and value. Linking external stylesheets via the link tag in the HTML head.",
            },
            {
                "number": "03",
                "title": "CSS Box Model & Styling Fundamentals",
                "start": 81.0,
                "end": 170.0,
                "text": "The CSS Box Model is fundamental: every element consists of content, padding, border, and margin. Box-sizing border-box simplifies sizing calculations.",
            },
            {
                "number": "03",
                "title": "CSS Box Model & Styling Fundamentals",
                "start": 171.0,
                "end": 260.0,
                "text": "CSS specificity and selectors: element selectors, class selectors with dot notation, ID selectors with hash, and pseudo-classes like hover.",
            },
        ],
    },
    {
        "number": "04",
        "title": "Modern Layouts with CSS Flexbox",
        "full_text": "Building responsive layouts using CSS Flexbox, flex containers, flex items, justify-content, and align-items.",
        "chunks": [
            {
                "number": "04",
                "title": "Modern Layouts with CSS Flexbox",
                "start": 0.0,
                "end": 90.0,
                "text": "Display flex activates flexbox on a parent container. Understand the main axis and cross axis directions.",
            },
            {
                "number": "04",
                "title": "Modern Layouts with CSS Flexbox",
                "start": 91.0,
                "end": 185.0,
                "text": "Aligning items: justify-content controls main axis alignment (center, space-between, space-around) and align-items controls cross axis.",
            },
            {
                "number": "04",
                "title": "Modern Layouts with CSS Flexbox",
                "start": 186.0,
                "end": 275.0,
                "text": "Flex properties for items: flex-grow, flex-shrink, and flex-basis. Creating a responsive navigation bar and card grid.",
            },
        ],
    },
    {
        "number": "05",
        "title": "JavaScript DOM Manipulation & Event Handling",
        "full_text": "Introduction to JavaScript, selecting DOM elements, modifying text and styles, and attaching event listeners.",
        "chunks": [
            {
                "number": "05",
                "title": "JavaScript DOM Manipulation & Event Handling",
                "start": 0.0,
                "end": 85.0,
                "text": "What is the Document Object Model (DOM)? Selecting elements with document.querySelector and document.querySelectorAll.",
            },
            {
                "number": "05",
                "title": "JavaScript DOM Manipulation & Event Handling",
                "start": 86.0,
                "end": 175.0,
                "text": "Modifying elements dynamically: textContent, innerHTML, classList.add, classList.remove, and modifying CSS styles in JavaScript.",
            },
            {
                "number": "05",
                "title": "JavaScript DOM Manipulation & Event Handling",
                "start": 176.0,
                "end": 265.0,
                "text": "Event listeners: element.addEventListener for click, input, and submit events. Handling user interactions and form validation.",
            },
        ],
    },
]


def generate_synthetic_embedding(text: str, dim: int = 1024) -> list[float]:
    """Deterministic normalized pseudo-embedding based on hash for offline testing."""
    import hashlib
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim)
    norm = np.linalg.norm(vec)
    return (vec / norm).tolist()


def get_embeddings_for_texts(texts: list[str]) -> list[list[float]]:
    """Try getting real embeddings from Ollama, fallback to deterministic synthetic vectors."""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": texts},
            timeout=5,
        )
        if response.ok:
            data = response.json()
            if "embeddings" in data and isinstance(data["embeddings"], list):
                print("  Using real Ollama bge-m3 embeddings.")
                return data["embeddings"]
    except Exception:
        pass

    print("  Ollama not detected; generating deterministic sample vector embeddings for immediate testing.")
    return [generate_synthetic_embedding(t) for t in texts]


def main() -> None:
    JSONS_DIR.mkdir(parents=True, exist_ok=True)
    NEW_JSONS_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    chunk_id = 0

    print("Generating sample course data...")

    for lesson in SAMPLE_LESSONS:
        num = lesson["number"]
        filename = f"{num}_{lesson['title'].replace(' ', '_')}.json"

        # Write to jsons/
        with open(JSONS_DIR / filename, "w", encoding="utf-8") as f:
            json.dump(lesson, f, indent=2, ensure_ascii=False)

        # Write to new_jsons/
        with open(NEW_JSONS_DIR / filename, "w", encoding="utf-8") as f:
            json.dump(lesson, f, indent=2, ensure_ascii=False)

        print(f"Created sample transcript: {filename}")

        texts = [c["text"] for c in lesson["chunks"]]
        embeddings = get_embeddings_for_texts(texts)

        for chunk, emb in zip(lesson["chunks"], embeddings):
            rec = dict(chunk)
            rec["chunk_id"] = chunk_id
            rec["embedding"] = emb
            records.append(rec)
            chunk_id += 1

    df = pd.DataFrame.from_records(records)
    joblib.dump(df, "new_embeddings.joblib")
    print(f"\nCreated new_embeddings.joblib with {len(records)} sample chunks.")
    print("You can now test the API (`python api.py`) and UI immediately!")


if __name__ == "__main__":
    main()
