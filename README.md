<p align="center">
  <img src="frontend/src/assets/hero.png" alt="AI Gurukul" width="120" />
</p>

<h1 align="center">AI Gurukul</h1>

<p align="center">
  <strong>Talk to your documents through a living avatar.</strong>
</p>

<p align="center">
  Upload a photograph. Upload your PDFs. Ask a question.<br/>
  The avatar reads your documents, speaks the answer, and moves its lips — all running locally on your machine.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/react-19-61DAFB?logo=react&logoColor=white" alt="React" />
  <img src="https://img.shields.io/badge/fastapi-0.104+-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/ollama-llama3.2-black?logo=ollama" alt="Ollama" />
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License" />
  <img src="https://img.shields.io/badge/GPU-not%20required-orange" alt="No GPU" />
</p>

---

## What is AI Gurukul?

AI Gurukul is an end-to-end conversational AI system that chains together four capabilities into a single seamless experience:

| Step | What happens | Technology |
|------|-------------|------------|
| **1. Understand** | Your PDF documents are parsed, chunked, and embedded into a vector store | PyMuPDF · LangChain · ChromaDB · all-MiniLM-L6-v2 |
| **2. Think** | A local LLM retrieves relevant context and generates an answer | Ollama · Llama 3.2 3B (Q4) |
| **3. Speak** | The text answer is converted to natural speech | Piper TTS (ONNX) |
| **4. Animate** | Your uploaded photograph comes alive with synchronized lip movements | Wav2Lip |

Everything streams in real time — you see text tokens appear, hear audio play, and watch the avatar speak, all progressively as the answer is generated.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser (React)                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │  Upload   │  │   Chat   │  │  Avatar  │  │  Processing   │  │
│  │  Panel    │  │  History  │  │  Player  │  │  Indicators   │  │
│  └──────────┘  └──────────┘  └──────────┘  └───────────────┘  │
└────────────────────────┬────────────────────────────────────────┘
                         │ SSE (Server-Sent Events)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                              │
│                                                                 │
│  ┌─────────────────── Orchestrator ──────────────────────────┐  │
│  │                                                           │  │
│  │  Question ──► Retrieve ──► Stream LLM ──► TTS ──► Animate │  │
│  │              (ChromaDB)    (Ollama)      (Piper)  (Wav2Lip)│  │
│  │                                                           │  │
│  │  Each sentence is pipelined: while sentence N animates,   │  │
│  │  sentence N+1 is being synthesized, and N+2 is streaming  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │PDF Parser│  │Embedding │  │   RAG    │  │  Evaluation  │   │
│  │(PyMuPDF) │  │  Store   │  │ Pipeline │  │   (RAGAS)    │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Features

- **Talking avatar from any photo** — upload a face, get a lip-synced video narrator
- **RAG over your PDFs** — ask questions grounded in your uploaded documents
- **Real-time streaming** — text, audio, and video stream progressively via SSE
- **Fully local** — no cloud APIs, no paid services, no data leaves your machine
- **Edge-optimized** — runs on 8 GB RAM, 4-core CPU, no GPU required
- **Open source stack** — every model and library is permissively licensed
- **Docker deployment** — single `docker compose up` to run everything
- **RAG evaluation** — built-in RAGAS metrics (faithfulness, relevance) with CLI benchmarking
- **Production patterns** — structured logging, error handling, correlation IDs, unit tests

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | React 19 · TypeScript · Vite | Chat UI, file uploads, video playback |
| **Backend** | FastAPI · Python 3.10 | API gateway, streaming orchestration |
| **LLM** | Ollama · Llama 3.2 3B (Q4_K_M) | Question answering (~2 GB) |
| **Embeddings** | all-MiniLM-L6-v2 | Semantic search (384-dim, ~80 MB) |
| **Vector Store** | ChromaDB | Persistent document embeddings |
| **TTS** | Piper TTS (ONNX) | Speech synthesis (~60 MB) |
| **Avatar** | Wav2Lip | Audio-driven lip sync (~416 MB) |
| **PDF Parsing** | PyMuPDF | Text + table extraction |
| **Chunking** | LangChain + tiktoken | 512-token chunks, 50-token overlap |
| **Evaluation** | RAGAS | Faithfulness, context & answer relevance |
| **Deployment** | Docker Compose | Multi-service orchestration |

---

## Getting Started

### Prerequisites

| Tool | Version | Install (macOS) |
|------|---------|-----------------|
| Python | 3.10+ | `brew install python@3.10` |
| Node.js | 20+ | `brew install node` |
| Ollama | latest | [ollama.com](https://ollama.com) |
| ffmpeg | any | `brew install ffmpeg` |
| Git | any | `brew install git` |

### Step 1 — Clone the repository

```bash
git clone https://github.com/sanjeevranjaniitb/ai-gurukul.git
cd ai-gurukul
```

### Step 2 — Set up the Python environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

### Step 3 — Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### Step 4 — Download AI models

```bash
bash scripts/setup_models.sh
```

This downloads (~860 MB total, one-time):

| Model | Size | What it does |
|-------|------|-------------|
| Piper TTS voice | ~60 MB | Speech synthesis |
| Wav2Lip checkpoint | ~416 MB | Lip-sync animation |
| GFPGAN weights | ~300 MB | Face detection for Wav2Lip |
| all-MiniLM-L6-v2 | ~80 MB | Document embeddings (auto-cached) |

### Step 5 — Start Ollama and pull the language model

```bash
# Start Ollama (if not already running)
ollama serve &

# Download the LLM (~2 GB, one-time)
ollama pull llama3.2:3b
```

### Step 6 — Start the backend

```bash
source .venv/bin/activate
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API is now live at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Step 7 — Start the frontend

Open a new terminal:

```bash
cd frontend
npm run dev
```

The app is now live at **`http://localhost:5173`**.

### Step 8 — Use the app

1. Open `http://localhost:5173` in your browser
2. **Upload a photo** — any image with a clear, front-facing human face (PNG/JPG, ≥256×256)
3. **Upload a PDF** — the document you want to ask questions about
4. **Ask a question** — type your question and watch the avatar answer

---

## Docker Deployment (Alternative)

For a single-command deployment without manual setup:

```bash
# Download models first
bash scripts/setup_models.sh

# Build and start all services
docker compose up --build

# Pull the LLM (first time only, in a separate terminal)
docker compose exec ollama ollama pull llama3.2:3b
```

| Service | URL | Memory Limit |
|---------|-----|-------------|
| Frontend | `http://localhost:3000` | 256 MB |
| Backend API | `http://localhost:8000` | 2 GB |
| Ollama LLM | `http://localhost:11434` | 3 GB |

Total peak memory: **< 6 GB**

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload/avatar` | Upload an avatar image (multipart) |
| `POST` | `/api/upload/pdf` | Upload a PDF document (multipart) |
| `POST` | `/api/ask` | Ask a question → SSE stream response |
| `GET` | `/api/health` | Health check with memory usage |

### SSE Event Types (from `/api/ask`)

| Event | Payload | Description |
|-------|---------|-------------|
| `text_token` | `{ token }` | Incremental LLM text |
| `audio_chunk` | `{ chunk_url, chunk_index }` | Synthesized audio segment URL |
| `video_chunk` | `{ chunk_url, chunk_index }` | Animated video segment URL |
| `stage_update` | `{ stage, duration_ms }` | Pipeline progress |
| `sources` | `{ sources[] }` | Retrieved document chunks |
| `done` | `{ total_duration_ms }` | Stream complete |
| `error` | `{ message }` | Error details |

Full OpenAPI documentation is auto-generated at `/docs` when the backend is running.

---

## Running Tests

```bash
source .venv/bin/activate

# Run all backend unit tests
python -m pytest backend/app/ -v

# Run a specific test file
python -m pytest backend/app/test_embedding_store.py -v

# Run with coverage report
python -m pytest backend/app/ -v --cov=backend/app --cov-report=term-missing
```

### RAG Evaluation

Run the built-in RAGAS evaluation benchmark:

```bash
source .venv/bin/activate
python -m backend.eval
```

This evaluates the RAG pipeline against `data/eval/test_dataset.json` and outputs faithfulness, context relevance, and answer relevance scores.

---

## Configuration

All tunable parameters live in `config/config.yaml`:

```yaml
# RAG settings
chunk_size: 512              # Tokens per chunk
chunk_overlap: 50            # Overlap between chunks
retrieval_top_k: 5           # Chunks retrieved per query
embedding_model: "all-MiniLM-L6-v2"

# LLM settings
llm_model: "llama3.2:3b"    # Ollama model name
llm_base_url: "http://localhost:11434"
llm_temperature: 0.1
llm_max_tokens: 512

# TTS settings
tts_model_path: "./models/piper/en_US-lessac-medium.onnx"
tts_sample_rate: 22050

# Avatar settings
avatar_fps: 25
avatar_resolution: [256, 256]

# Validation limits
max_image_size_mb: 10
max_pdf_size_mb: 50
```

---

## Project Structure

```
ai-gurukul/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application & endpoints
│   │   ├── orchestrator.py      # Streaming pipeline coordinator
│   │   ├── pdf_parser.py        # PDF text extraction (PyMuPDF)
│   │   ├── chunking.py          # Text chunking (LangChain + tiktoken)
│   │   ├── embedding_store.py   # Vector store (ChromaDB + MiniLM)
│   │   ├── llm_service.py       # LLM client (Ollama HTTP API)
│   │   ├── rag_pipeline.py      # RAG retrieval + generation
│   │   ├── tts_engine.py        # Speech synthesis (Piper TTS)
│   │   ├── avatar_engine.py     # Lip-sync video (Wav2Lip)
│   │   ├── evaluation.py        # RAG quality metrics (RAGAS)
│   │   ├── config.py            # Configuration loader
│   │   ├── models.py            # Shared data models
│   │   ├── logging_utils.py     # Structured logging
│   │   └── test_*.py            # Unit tests (11 test files)
│   ├── eval.py                  # CLI evaluation runner
│   ├── Dockerfile               # Multi-stage backend image
│   └── requirements.txt         # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.tsx              # Main application layout
│   │   ├── api.ts               # API client utilities
│   │   ├── components/
│   │   │   ├── AvatarUpload.tsx     # Image upload with validation
│   │   │   ├── PdfUpload.tsx        # PDF upload component
│   │   │   ├── ChatInterface.tsx    # Chat history & question input
│   │   │   ├── AvatarPlayer.tsx     # Video segment queue & playback
│   │   │   └── ProcessingIndicator.tsx  # Pipeline stage display
│   │   └── hooks/
│   │       └── useSSE.ts           # Server-Sent Events consumer
│   ├── Dockerfile               # Build + nginx serve image
│   ├── nginx.conf               # Production reverse proxy config
│   └── package.json
├── config/
│   └── config.yaml              # Application configuration
├── data/
│   ├── avatars/                 # Uploaded avatar images
│   ├── documents/               # Uploaded PDF files
│   ├── chroma/                  # ChromaDB persistent storage
│   └── eval/
│       └── test_dataset.json    # RAG evaluation dataset
├── models/                      # AI model weights (via setup script)
├── scripts/
│   └── setup_models.sh          # One-command model download
├── docker-compose.yml           # Full stack deployment
├── HARDWARE_REQUIREMENTS.md     # Minimum specs & memory budget
└── LICENSES.md                  # Open-source license manifest
```

---

## Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| CPU | 4-core (x86_64 or ARM64) | 8-core |
| Disk | 10 GB free | 15 GB free |
| GPU | Not required | — |
| OS | macOS / Linux | — |

### Memory Budget

| Component | Allocation |
|-----------|-----------|
| Ollama (Llama 3.2 3B) | 3 GB |
| Backend (FastAPI + models) | 2 GB |
| Frontend (nginx / Vite) | 256 MB |
| **Peak total** | **< 6 GB** |

---

## Model Disk Usage

| Model | Size | License |
|-------|------|---------|
| Llama 3.2 3B (Q4_K_M) | ~2.0 GB | Llama Community License |
| Wav2Lip checkpoint | ~416 MB | MIT |
| GFPGAN weights | ~300 MB | Apache 2.0 |
| Piper TTS (en_US-lessac-medium) | ~60 MB | MIT |
| all-MiniLM-L6-v2 | ~80 MB | Apache 2.0 |
| **Total** | **~2.9 GB** | — |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` when running tests | Use `.venv/bin/python -m pytest` — the system Python doesn't have the dependencies |
| Backend hangs on first request | Models load lazily on first use. The first request takes 5–10 seconds. |
| `Cannot connect to Ollama` | Make sure `ollama serve` is running. Check `curl http://localhost:11434/api/tags` |
| Wav2Lip not producing video | The system falls back to static image + audio. Check `models/wav2lip/checkpoints/wav2lip_gan.pth` exists |
| Frontend can't reach backend | In dev mode, Vite proxies `/api` to `localhost:8000`. Make sure the backend is running on port 8000 |
| `piper-tts` import error | Some platforms need: `pip install piper-tts --no-build-isolation` |
| Docker build fails on ARM Mac | The Dockerfiles support ARM64. Make sure Docker Desktop has Rosetta emulation enabled |

---

## Licenses

All components use open-source licenses. No paid APIs, subscriptions, or cloud services required.

See [LICENSES.md](LICENSES.md) for the complete dependency manifest.

---

<p align="center">
  Built with open-source AI · Runs on your machine · No cloud required
</p>
