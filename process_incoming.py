"""
CLI Query Client for RAG-Based AI Teaching Assistant.

Allows testing questions directly from the terminal.
"""

from __future__ import annotations

import sys
from rag_engine import RagEngineError, ask_question


def main() -> None:
    print("=" * 60)
    print(" RAG-Based AI Teaching Assistant — CLI Query Mode")
    print("=" * 60)
    print("Type your question below (or 'exit' / 'quit' to exit):\n")

    while True:
        try:
            query = input("Ask a question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not query:
            continue

        if query.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        print("\nSearching relevant course lecture chunks and generating answer...\n")
        try:
            result = ask_question(query)

            print("=" * 60)
            print(" ANSWER")
            print("=" * 60)
            print(result["answer"])

            if result["sources"]:
                print("\n" + "=" * 60)
                print(" RETRIEVED SOURCES & TIMESTAMPS")
                print("=" * 60)
                for i, src in enumerate(result["sources"], 1):
                    print(
                        f" [{i}] Video {src['video_number']}: {src['title']}\n"
                        f"     Timestamp: {src['start']} - {src['end']} | Similarity: {src['score']}\n"
                        f"     Excerpt: {src['excerpt']}...\n"
                    )
            print("-" * 60 + "\n")

        except RagEngineError as exc:
            print(f"[Error] {exc}\n")
        except Exception as exc:
            print(f"[Unexpected Error] {exc}\n")


if __name__ == "__main__":
    main()
