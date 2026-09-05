# RAG-Based AI Teaching Assistant

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.3-61DAFB.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.6-3178C6.svg)](https://www.typescriptlang.org/)
[![Vite](https://img.shields.io/badge/Vite-5.4-646CFF.svg)](https://vitejs.dev/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black.svg)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end local **Retrieval-Augmented Generation (RAG)** system that converts course video lectures into an interactive teaching assistant, providing conversational answers cited with **exact video numbers, titles, and timestamp ranges**.

---

## Overview

Online courses often contain dozens of hours of video content, making it difficult for students to find specific concepts or re-watch exact explanations. 

**RAG-Based AI Teaching Assistant** automates the lecture ingestion and retrieval pipeline:
1. Extracts audio tracks from lecture video containers via FFmpeg.
2. Transcribes and translates speech into timestamped subtitles using OpenAI Whisper.
3. Groups small subtitle segments into multi-sentence context chunks.
4. Generates dense vector embeddings using the `BGE-M3` model via Ollama.
5. Performs cosine similarity retrieval to extract top-matching lecture segments for user queries.
6. Synthesizes clear, instructional responses using the `Llama 3.2` LLM with exact video timestamp citations.
7. Serves responses through a high-performance FastAPI backend and a modern React + TypeScript web interface.

---

## Key Features

- **Automated Video Ingestion**: Batch audio extraction from standard video formats (`.mp4`, `.webm`, `.mkv`, `.avi`, `.mov`).
- **Timestamped Transcription**: Multilingual speech-to-text with start and end timestamps via OpenAI Whisper.
- **Sliding Context Windowing**: Aggregates atomic subtitle segments into cohesive chunks to preserve semantic context.
- **Dense Vector Retrieval**: 1024-dimensional semantic search powered by `BGE-M3` embeddings and scikit-learn cosine similarity.
- **LLM-Powered Answer Synthesis**: Context-grounded instruction with strict timestamp citations using local `Llama 3.2`.
- **FastAPI Backend**: Asynchronous REST API with Pydantic validation, CORS middleware, and structured diagnostics.
- **Real-Time Health Monitoring**: Dedicated `/health` endpoint and live UI indicator tracking vector index and Ollama availability.
- **React + TypeScript UI**: Modern glassmorphism chat interface with example queries, loading skeletons, and interactive source cards.
- **Zero-Video Sample Mode**: Built-in synthetic course generator for immediate offline testing without raw video files.

---

## Architecture & Data Flow

```
                     ┌───────────────────────────────┐
                     │      Course Video Files       │
                     │         (videos/*.mp4)        │
                     └──────────────┬────────────────┘
                                    │ (video_to_mp3.py / FFmpeg)
                                    ▼
                     ┌───────────────────────────────┐
                     │     Extracted Audio Track     │
                     │         (audios/*.mp3)        │
                     └──────────────┬────────────────┘
                                    │ (mp3_to_json.py / OpenAI Whisper)
                                    ▼
                     ┌───────────────────────────────┐
                     │     Raw Subtitle Segments     │
                     │         (jsons/*.json)        │
                     └──────────────┬────────────────┘
                                    │ (merge_chunks.py - windowing)
                                    ▼
                     ┌───────────────────────────────┐
                     │     Merged Context Chunks     │
                     │       (new_jsons/*.json)      │
                     └──────────────┬────────────────┘
                                    │ (preprocess_new_json.py / BGE-M3)
                                    ▼
                     ┌───────────────────────────────┐
                     │     Vector Index DataFrame    │
                     │     (new_embeddings.joblib)   │
                     └──────────────┬────────────────┘
                                    │
       ┌────────────────────────────┴────────────────────────────┐
       ▼                                                         ▼
┌──────────────┐                                          ┌──────────────┐
│ CLI Client   │                                          │ FastAPI REST │
│ (terminal)   │                                          │   (:8000)    │
└──────────────┘                                          └──────┬───────┘
                                                                 │
                                                                 ▼
                                                          ┌──────────────┐
                                                          │ React + Vite │
                                                          │  Frontend UI │
                                                          └──────────────┘
```

---

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Audio Processing** | `FFmpeg` | High-efficiency audio extraction from video files |
| **Speech-to-Text** | `OpenAI Whisper` | Speech transcription and translation to timestamped subtitles |
| **Embedding Model** | `BGE-M3` (Ollama) | 1024-dimensional dense semantic text representations |
| **Generative Model** | `Llama 3.2` (Ollama) | Contextual answer synthesis with timestamp citation formatting |
| **Vector Retrieval** | `scikit-learn` / `NumPy` | Cosine similarity ranking over serialized DataFrame index |
| **Backend API** | `FastAPI` + `Uvicorn` | Asynchronous REST endpoints, CORS, and Pydantic validation |
| **Data Serialization** | `Pandas` / `Joblib` | Efficient in-memory index caching and DataFrame serialization |
| **Frontend Framework** | `React 18` + `TypeScript` | Type-safe interactive user interface |
| **Build & Styling** | `Vite` + `Tailwind CSS` | Fast module bundling and responsive dark glassmorphism styling |

---

## Project Structure

```
├── .env.example               # Environment variable configuration template
├── .gitignore                 # Ignore rules for virtualenvs, artifacts, and node_modules
├── LICENSE                    # MIT License
├── README.md                  # Main documentation
├── DEPLOY.md                  # Cloudflare Tunnel / Vercel deployment guide
├── requirements.txt           # Python backend dependencies
├── config.py                  # Centralized .env configuration loader
├── rag_engine.py              # Retrieval & LLM inference engine
├── api.py                     # FastAPI server with health checks & CORS
├── process_incoming.py        # CLI interactive query interface
├── generate_sample_data.py    # Zero-video synthetic dataset generator
│
├── video_to_mp3.py            # Step 1: Audio extraction with FFmpeg
├── mp3_to_json.py             # Step 2: Whisper transcription & translation
├── merge_chunks.py            # Step 3: Context chunk windowing
├── preprocess_json.py         # Step 4a: Embed raw segments
├── preprocess_new_json.py     # Step 4b: Embed merged chunks
│
├── videos/ (.gitkeep)         # Input course videos
├── audios/ (.gitkeep)         # Extracted MP3s
├── jsons/ (.gitkeep)          # Subtitle transcripts
├── new_jsons/ (.gitkeep)      # Merged chunk JSONs
│
└── frontend/                  # React + TypeScript + Vite web interface
    ├── src/
    │   ├── components/        # Header, ChatMessage, SourceCard, ExampleQuestions
    │   ├── lib/api.ts         # Type-safe API client
    │   ├── App.tsx            # Main chat application layout
    │   └── index.css          # Tailwind CSS and glassmorphism styling
    └── package.json
```

---

## Getting Started

### Prerequisites

- **Python 3.10+** and **Node.js 18+**
- **FFmpeg** installed and accessible on system `PATH`
  - *Windows:* `winget install Gyan.FFmpeg`
  - *macOS:* `brew install ffmpeg`
  - *Linux:* `sudo apt install ffmpeg`
- **Ollama** installed locally with models pulled:
  ```bash
  ollama pull bge-m3
  ollama pull llama3.2
  ```

---

## Quick Start (Sample Data Mode)

To run and evaluate the system immediately without processing custom video files:

### 1. Clone & Set Up Backend

```bash
git clone https://github.com/lakshya0101/RAG-Based-AI-Assistant.git
cd RAG-Based-AI-Assistant

# Create and activate virtual environment
python -m venv venv
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Generate synthetic course transcripts & vector index
python generate_sample_data.py

# Start FastAPI server
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation will be available at `http://localhost:8000/docs`.

### 2. Set Up Frontend

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

Open `http://localhost:5173` in your browser to interact with the teaching assistant.

---

## Configuration

Settings can be customized via `.env` in the project root:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Endpoint for the local Ollama instance |
| `EMBED_MODEL` | `bge-m3` | Embedding model for semantic vector representations |
| `LLM_MODEL` | `llama3.2` | Generative LLM for conversational responses |
| `EMBEDDINGS_FILE` | `new_embeddings.joblib` | Target file for the serialized embeddings index |
| `TOP_K_RESULTS` | `5` | Number of context chunks retrieved per query |
| `WHISPER_MODEL` | `base` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v2`) |
| `SOURCE_LANGUAGE` | `hi` | Source language code for lecture audio translation |
| `ALLOWED_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Allowed CORS origins for API requests |
| `PORT` | `8000` | Port for the FastAPI server |

---

## Custom Video Processing Pipeline

To index your own lecture videos:

1. Place your video files (`.mp4`, `.webm`, etc.) into `videos/` (e.g., `01 - Introduction.mp4`, `02 - HTML Fundamentals.mp4`).
2. Run the pipeline stages sequentially:

```bash
# Step 1: Extract audio tracks
python video_to_mp3.py

# Step 2: Transcribe speech to timestamped subtitle JSONs
python mp3_to_json.py

# Step 3: Aggregate small segments into richer context chunks
python merge_chunks.py

# Step 4: Generate dense vector embeddings via BGE-M3
python preprocess_new_json.py

# Step 5 (Option A): Query via CLI terminal
python process_incoming.py

# Step 5 (Option B): Start REST API backend
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Reference

### `GET /health`
Returns system readiness, vector index status, chunk count, and Ollama connectivity:
```json
{
  "ready": true,
  "embeddings_loaded": true,
  "ollama_reachable": true,
  "chunk_count": 15,
  "models_available": ["bge-m3:latest", "llama3.2:latest"]
}
```

### `POST /ask`
Submit a question to retrieve relevant lecture segments and generate an answer:

**Request:**
```json
{
  "question": "Where is the CSS box model explained?",
  "top_k": 5
}
```

**Response:**
```json
{
  "question": "Where is the CSS box model explained?",
  "answer": "The CSS Box Model is explained in Video 03: CSS Box Model & Styling Fundamentals at timestamp 1:21 - 2:50...",
  "sources": [
    {
      "video_number": "03",
      "title": "CSS Box Model & Styling Fundamentals",
      "start": "1:21",
      "end": "2:50",
      "score": 0.884,
      "excerpt": "The CSS Box Model is fundamental: every element consists of content, padding, border, and margin..."
    }
  ]
}
```

---

## Deployment

For deploying the React frontend to Vercel and exposing the local backend via Cloudflare Tunnel or ngrok, see **[DEPLOY.md](DEPLOY.md)**.

---

## Limitations

- **Hardware Compute**: Full local pipeline execution requires sufficient CPU/GPU RAM to run OpenAI Whisper and Ollama models (`bge-m3` and `llama3.2`).
- **Initial Inference Cold Starts**: The first LLM response in a session may exhibit a brief delay while Ollama initializes model weights in memory.
- **Language Configurations**: Whisper translation default is configured for Hindi to English (`SOURCE_LANGUAGE=hi`), but can be adjusted in `.env` for other source languages.

---

## Author

**Lakshya Dogra**  
GitHub: [@lakshya0101](https://github.com/lakshya0101)  
Repository: [https://github.com/lakshya0101/RAG-Based-AI-Assistant](https://github.com/lakshya0101/RAG-Based-AI-Assistant)

---

## License

This project is licensed under the [MIT License](LICENSE).
