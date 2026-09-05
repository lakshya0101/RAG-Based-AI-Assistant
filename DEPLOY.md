# Deployment & Public Demo Guide

This guide covers running the **FastAPI + Ollama backend** locally (on Windows, macOS, or Linux) and exposing it via a secure tunnel (Cloudflare Tunnel or ngrok) to connect to a **React frontend** deployed on Vercel.

```
┌────────────────────────┐         ┌─────────────────────────┐         ┌───────────────────────────┐
│     Vercel Deploy      │  HTTP   │    Cloudflare Tunnel    │  HTTP   │   Local Machine (Backend) │
│ React + TypeScript UI  │ ──────> │ *.trycloudflare.com     │ ──────> │ FastAPI (:8000) + Ollama  │
└────────────────────────┘         └─────────────────────────┘         └───────────────────────────┘
```

---

## Part 1 — Local Backend Setup

### 1. Prerequisites & Ollama Service

Make sure [Ollama](https://ollama.com/) is installed and running:

```bash
# Verify Ollama is running and pull models
ollama pull bge-m3
ollama pull llama3.2
```

### 2. Python Environment

#### On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

#### On macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Generate Course Embeddings Index

If you have raw course video files, run the indexing pipeline:
```bash
python video_to_mp3.py
python mp3_to_json.py
python merge_chunks.py
python preprocess_new_json.py
```

*Quick Test / Demo Mode:* If you do not have raw video files on hand, you can immediately generate sample lesson data and index vectors:
```bash
python generate_sample_data.py
```

### 4. Start the FastAPI Server

Set your CORS allowed origins and launch the server:

#### Windows (PowerShell):
```powershell
$env:ALLOWED_ORIGINS="http://localhost:5173,https://your-frontend.vercel.app"
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

#### macOS / Linux:
```bash
export ALLOWED_ORIGINS="http://localhost:5173,https://your-frontend.vercel.app"
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Verify backend health in your browser at `http://localhost:8000/health`. It should return:
```json
{
  "ready": true,
  "embeddings_loaded": true,
  "ollama_reachable": true,
  "chunk_count": 15,
  "models_available": ["bge-m3:latest", "llama3.2:latest"]
}
```

---

## Part 2 — Exposing the Backend (Cloudflare Tunnel / ngrok)

### Option A: Cloudflare Tunnel (Free, No Account Required)

1. Download and install `cloudflared` from [Cloudflare](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/):
   - **Windows:** `winget install --id Cloudflare.cloudflared`
   - **macOS:** `brew install cloudflared`
   - **Linux:** `sudo apt-get install cloudflared`

2. In a new terminal, launch the quick tunnel:
   ```bash
   cloudflared tunnel --url http://localhost:8000
   ```

3. Note down the public URL output (e.g., `https://random-subdomain.trycloudflare.com`).

### Option B: ngrok

```bash
ngrok http 8000
```
Copy the generated `https://...ngrok-free.app` URL.

---

## Part 3 — Frontend Deployment (Vercel)

### 1. Local Testing

1. Navigate to the `frontend/` directory:
   ```bash
   cd frontend
   cp .env.example .env
   ```
2. Set `VITE_API_URL` inside `.env` to your tunnel URL or `http://localhost:8000`.
3. Install dependencies and start the dev server:
   ```bash
   npm install
   npm run dev
   ```
4. Open `http://localhost:5173` to test the interface.

### 2. Deploying to Vercel

1. Push your repository to your GitHub account.
2. Log in to [Vercel](https://vercel.com) and click **Add New Project**.
3. Import your GitHub repository `RAG-Based-AI-Assistant`.
4. In the configuration screen:
   - **Framework Preset**: Vite
   - **Root Directory**: `frontend`
5. Under **Environment Variables**, add:
   - `VITE_API_URL` = `https://your-tunnel-subdomain.trycloudflare.com`
6. Click **Deploy**.
7. Once deployed, add your Vercel URL to `ALLOWED_ORIGINS` in your local backend environment and restart the backend.

---

## Demo Checklist

- [ ] Ollama service active (`ollama serve` or background service)
- [ ] Required models present (`ollama list` shows `bge-m3` and `llama3.2`)
- [ ] Embedding vector index present (`new_embeddings.joblib`)
- [ ] FastAPI backend running (`uvicorn api:app --host 0.0.0.0 --port 8000`)
- [ ] Tunnel active (`cloudflared tunnel --url http://localhost:8000`)
- [ ] Vercel frontend has `VITE_API_URL` matching the tunnel endpoint
- [ ] Backend status indicator on UI displays **Backend Ready**
