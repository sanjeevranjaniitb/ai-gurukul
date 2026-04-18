# Live Talking Head Avatar

An AI-powered conversational system that transforms a static photograph into an animated talking head avatar. Upload a photo and PDF documents, then ask questions — the avatar narrates answers with synchronized lip movements.

Built entirely with open-source models. Runs locally on edge devices (8 GB RAM, no GPU required).

## How It Works

```
Photo + PDF → RAG retrieval → LLM answer → Text-to-Speech → Lip-sync video → Streaming playback
```

**Stack:** FastAPI · React · Ollama (Llama 3.2 3B) · Piper TTS · Wav2Lip · ChromaDB · all-MiniLM-L6-v2

## Quick Start (macOS)

### Prerequisites

| Tool | Install |
|------|---------|
| Python 3.10+ | `brew install python@3.10` |
| Node.js 20+ | `brew install node` |
| Ollama | [ollama.com](https://ollama.com) |
| ffmpeg | `brew install ffmpeg` |
| Git | `brew install git` |

### 1. Clone and set up

```bash
git clone <your-repo-url>
cd AiGurukul

# Create Python virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r backend/requirements.txt

# Install frontend dependencies
cd frontend && npm install && cd ..
```

### 2. Download AI models

```bash
bash scripts/setup_models.sh
```

This downloads:
- Piper TTS voice model (~60 MB)
- Wav2Lip checkpoint (~416 MB)
- GFPGAN face detection weights (~300 MB)
- Sentence-transformers embedding model (~80 MB, cached by HuggingFace)

### 3. Start Ollama and pull the LLM

```bash
# Start Ollama (runs in background)
ollama serve &

# Pull the language model (~2 GB download, one time)
ollama pull llama3.2:3b
```

### 4. Run the backend

```bash
source .venv/bin/activate
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Backend runs at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

### 5. Run the frontend

In a separate terminal:

```bash
cd frontend
npm run dev
```

Frontend runs at `http://localhost:5173`. The Vite dev server proxies `/api` requests to the backend automatically.

### 6. Use the app

1. Open `http://localhost:5173`
2. Upload a photo (PNG/JPG, must contain a face)
3. Upload a PDF document
4. Ask a question — the avatar will answer with lip-synced video

## Quick Start (Docker)

```bash
# Download models first
bash scripts/setup_models.sh

# Start everything
docker compose up --build

# Pull the LLM (first time only, in another terminal)
docker compose exec ollama ollama pull llama3.2:3b
```

- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:8000`
- Ollama: `http://localhost:11434`

## Running Tests

```bash
source .venv/bin/activate

# All backend tests
python -m pytest backend/app/ -v

# Specific test file
python -m pytest backend/app/test_embedding_store.py -v

# With coverage
python -m pytest backend/app/ -v --cov=backend/app
```

## Project Structure

```
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI endpoints
│   │   ├── orchestrator.py      # Streaming pipeline coordinator
│   │   ├── pdf_parser.py        # PDF text extraction (PyMuPDF)
│   │   ├── chunking.py          # Text chunking (LangChain)
│   │   ├── embedding_store.py   # Vector store (ChromaDB + MiniLM)
│   │   ├── llm_service.py       # LLM client (Ollama)
│   │   ├── tts_engine.py        # Speech synthesis (Piper TTS)
│   │   ├── avatar_engine.py     # Lip-sync video (Wav2Lip)
│   │   ├── rag_pipeline.py      # RAG retrieval + generation
│   │   ├── evaluation.py        # RAG quality metrics (RAGAS)
│   │   ├── config.py            # Configuration loader
│   │   ├── models.py            # Shared data models
│   │   └── test_*.py            # Unit tests
│   ├── eval.py                  # CLI evaluation runner
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/          # React components
│   │   ├── hooks/useSSE.ts      # Server-Sent Events hook
│   │   ├── App.tsx              # Main app layout
│   │   └── api.ts               # API client
│   ├── Dockerfile
│   └── package.json
├── config/config.yaml           # Tunable parameters
├── data/                        # Runtime data (avatars, documents, chroma)
├── models/                      # AI model weights (downloaded by setup script)
├── scripts/setup_models.sh      # Model download script
├── docker-compose.yml           # Full stack deployment
└── HARDWARE_REQUIREMENTS.md
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/upload/avatar` | Upload avatar image |
| `POST` | `/api/upload/pdf` | Upload PDF document |
| `POST` | `/api/ask` | Ask a question (SSE stream) |
| `GET` | `/api/health` | Health check |

Full OpenAPI docs available at `/docs` when the backend is running.

## Configuration

Edit `config/config.yaml` to tune parameters:

```yaml
chunk_size: 512          # Tokens per chunk
chunk_overlap: 50        # Overlap between chunks
retrieval_top_k: 5       # Number of chunks to retrieve
llm_model: "llama3.2:3b" # Ollama model name
llm_temperature: 0.1     # LLM temperature
```

## Hardware Requirements

| Resource | Minimum |
|----------|---------|
| RAM | 8 GB |
| CPU | 4-core (x86_64 or ARM64) |
| Disk | 10 GB free |
| GPU | Not required |

## Licenses

All models and libraries are open-source. See [LICENSES.md](LICENSES.md) for details.

| Component | License |
|-----------|---------|
| Llama 3.2 3B | Llama Community License |
| Wav2Lip | MIT |
| Piper TTS | MIT |
| all-MiniLM-L6-v2 | Apache 2.0 |
| ChromaDB | Apache 2.0 |
| FastAPI | MIT |
| React | MIT |
