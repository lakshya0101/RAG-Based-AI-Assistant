# RAG-Based AI Teaching Assistant

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.3-61DAFB.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-5.4-646CFF.svg)](https://vitejs.dev/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black.svg)](https://ollama.com/)
[![Whisper](https://img.shields.io/badge/OpenAI-Whisper-412991.svg)](https://github.com/openai/whisper)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end local **Retrieval-Augmented Generation (RAG)** teaching assistant that ingests course lecture videos and provides intelligent Q&A with **exact video numbers, titles, and timestamp citations**.

---

## Architecture & Data Flow

```
                     ┌───────────────────────────────┐
                     │     Course Video Files        │
                     │       (videos/*.mp4)          │
                     └──────────────┬────────────────┘
                                    │ (video_to_mp3.py / ffmpeg)
                                    ▼
                     ┌───────────────────────────────┐
                     │      Extracted Audio Track    │
                     │       (audios/*.mp3)          │
                     └──────────────┬────────────────┘
                                    │ (mp3_to_json.py / OpenAI Whisper)
                                    ▼
                     ┌───────────────────────────────┐
                     │    Raw Subtitle Segments      │
                     │        (jsons/*.json)         │
                     └──────────────┬────────────────┘
                                    │ (merge_chunks.py - windowing)
                                    ▼
                     ┌───────────────────────────────┐
                     │     Merged Context Chunks     │
                     │      (new_jsons/*.json)       │
                     └──────────────┬────────────────┘
                                    │ (preprocess_new_json.py / Ollama bge-m3)
                                    ▼
                     ┌───────────────────────────────┐
                     │     Vector Index DataFrame    │
                     │    (new_embeddings.joblib)    │
                     └──────────────┬────────────────┘
                                    │
       ┌────────────────────────────┴────────────────────────────┐
       ▼                                                         ▼
┌──────────────┐                                          ┌──────────────┐
│ CLI Client   │                                          │ FastAPI REST │
│ (terminal)   │                                          │  (:8000)     │
└──────────────┘                                          └──────┬───────┘
                                                                 │
                                                                 ▼
                                                          ┌──────────────┐
                                                          │ React + Vite │
                                                          │ Frontend UI  │
                                                          └──────────────┘
```

---

## Tech Stack

| Component | Technology | Description |
|-----------|------------|-------------|
| **Audio Extraction** | `ffmpeg` | High-efficiency audio stream extraction from video containers |
| **Speech-to-Text** | `OpenAI Whisper` | Multilingual transcription and translation to English subtitles |
| **Embedding Model** | `BGE-M3` (via Ollama) | 1024-dimensional dense semantic text representations |
| **Generative LLM** | `Llama 3.2` (via Ollama) | Context-aware reasoning and citation generation |
| **Vector Retrieval** | `scikit-learn` / `NumPy` | Cosine similarity ranking over serialized DataFrame index |
| **Backend API** | `FastAPI` + `Uvicorn` | Asynchronous REST endpoints with Pydantic validation and CORS |
| **Frontend UI** | `React` + `TypeScript` + `Tailwind CSS` | Modern glassmorphism chat interface with live health status |

---

## Prerequisites

1. **Python 3.10+** and **Node.js 18+**
2. **ffmpeg** installed and available in system `PATH`
   - *Windows:* `winget install Gyan.FFmpeg` or download from [ffmpeg.org](https://ffmpeg.org/)
   - *macOS:* `brew install ffmpeg`
   - *Linux:* `sudo apt install ffmpeg`
3. **Ollama** installed and models downloaded:
   ```bash
   ollama pull bge-m3
   ollama pull llama3.2
   ```

---

## Quick Start (Zero-Video Sample Mode)

You can run and test the complete system immediately without processing raw videos by generating sample course data:

### 1. Clone & Setup Backend

```bash
# Clone the repository
git clone https://github.com/lakshya0101/RAG-Based-AI-Assistant.git
cd RAG-Based-AI-Assistant

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# macOS / Linux:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Create sample lesson transcripts and embedding index
python generate_sample_data.py

# Start the FastAPI server
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Backend will be accessible at `http://localhost:8000` (Swagger docs at `http://localhost:8000/docs`).

### 2. Setup Frontend

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

Open `http://localhost:5173` in your browser to interact with the teaching assistant!

---

## Full Video Pipeline Execution

To index custom course videos:

1. Place your video files (`.mp4`, `.webm`, `.mkv`, etc.) into `videos/` with standard naming (e.g., `01 - Introduction.mp4`, `02 - HTML Basics.mp4`).
2. Run the pipeline stages sequentially:

```bash
# Step 1: Extract MP3 audio tracks
python video_to_mp3.py

# Step 2: Transcribe and translate to timestamped English subtitles
python mp3_to_json.py

# Step 3: Merge small subtitle segments into richer context chunks
python merge_chunks.py

# Step 4: Generate vector embeddings via Ollama bge-m3
python preprocess_new_json.py

# Step 5 (Option A): Test via CLI interactive prompt
python process_incoming.py

# Step 5 (Option B): Launch REST API server
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

---

## Configuration

Settings can be customized in a `.env` file in the project root:

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama HTTP endpoint |
| `EMBED_MODEL` | `bge-m3` | Vector embedding model name |
| `LLM_MODEL` | `llama3.2` | Generative LLM name |
| `EMBEDDINGS_FILE` | `new_embeddings.joblib` | Path to vector index file |
| `TOP_K_RESULTS` | `5` | Number of context chunks retrieved per question |
| `WHISPER_MODEL` | `base` | Whisper model size (`tiny`, `base`, `small`, `medium`, `large-v2`) |
| `SOURCE_LANGUAGE` | `hi` | Source audio language code |
| `ALLOWED_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Allowed CORS origins for API |
| `PORT` | `8000` | API port |

---

## API Reference

### `GET /health`
Returns system status, vector index statistics, and Ollama service connectivity:
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
Submit a question to receive an explanation citing video timestamps:

**Request Body:**
```json
{
  "question": "Where is the CSS box model explained?",
  "top_k": 5
}
```

**Response Body:**
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

## Project Structure

```
├── .env.example               # Environment variable configuration template
├── .gitignore                 # Ignore rules for virtualenvs, artifacts, and node_modules
├── LICENSE                    # MIT License with original author attribution
├── README.md                  # Main project documentation
├── DEPLOY.md                  # Production & public demo deployment guide
├── requirements.txt           # Python backend dependencies
├── config.py                  # Centralized configuration helper
├── rag_engine.py              # Core retrieval & Ollama generation logic
├── api.py                     # FastAPI REST server with health checks & CORS
├── process_incoming.py        # Interactive CLI query tool
├── generate_sample_data.py    # Zero-video synthetic course index generator
│
├── video_to_mp3.py            # Step 1: ffmpeg audio extraction
├── mp3_to_json.py             # Step 2: Whisper transcription / translation
├── merge_chunks.py            # Step 3: Context chunk windowing & merging
├── preprocess_json.py         # Step 4a: Embedding raw segments
├── preprocess_new_json.py     # Step 4b: Embedding merged chunks
│
├── videos/                    # Raw course videos (gitkept)
├── audios/                    # Extracted MP3 audio files (gitkept)
├── jsons/                     # Raw transcript JSONs (gitkept)
├── new_jsons/                 # Merged chunk JSONs (gitkept)
│
└── frontend/                  # React + TypeScript + Vite web interface
    ├── src/
    │   ├── components/        # Header, ChatMessage, SourceCard, ExampleQuestions
    │   ├── lib/api.ts         # Type-safe API client
    │   ├── App.tsx            # Main application layout
    │   └── index.css          # Tailwind CSS styles & glassmorphism
    └── package.json
```

---

## Limitations

- **Local Inference Requirements**: Full local pipeline execution requires sufficient hardware (RAM/VRAM) to run Whisper and Ollama models (`bge-m3` and `llama3.2`).
- **Cold-Start Latency**: The first LLM response in a session may experience a brief cold-start while Ollama loads the model weights into memory.
- **Language Transcriptions**: Whisper translation default is configured for Hindi to English (`SOURCE_LANGUAGE=hi`), but can be overridden in `.env` for other languages.

---

## Attribution & Acknowledgements

- **Original Implementation**: [Lakshya Arora (`Lakshya8725/Rag-based-ai-assistant`)](https://github.com/Lakshya8725/Rag-based-ai-assistant)
- **Current Modified & Maintained Version**: [Lakshya Dogra (`lakshya0101/RAG-Based-AI-Assistant`)](https://github.com/lakshya0101/RAG-Based-AI-Assistant)
- Re-architected with centralized configuration, robust cross-platform error handling, automated sample data synthesis, enhanced API schemas, and upgraded frontend polish.

---

## License

This project is open-source software licensed under the [MIT License](LICENSE).
