<p align="center">
  <img src="frontend/src/assets/hero.png" alt="AI Gurukul" width="120" />
</p>

<h1 align="center">🎓 AI Gurukul</h1>

<p align="center">
  <strong>Talk to your documents through a living avatar.</strong>
</p>

<p align="center">
  Upload a photograph. Upload your PDFs. Ask a question.<br/>
  The avatar reads your documents, speaks the answer, and moves its lips — all running locally on your machine.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10-blue?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/react-19-61DAFB?logo=react&logoColor=white" alt="React" />
  <img src="https://img.shields.io/badge/fastapi-0.104+-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/ollama-llama3.2-black?logo=ollama" alt="Ollama" />
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License" />
  <img src="https://img.shields.io/badge/GPU-not%20required-orange" alt="No GPU" />
</p>

---

## What is AI Gurukul?

AI Gurukul is an end-to-end multimodal conversational AI system. It chains together RAG-based question answering, neural text-to-speech, and real-time facial animation into a single seamless experience:

| Step | What happens | Technology |
|------|-------------|------------|
| **1. Understand** | PDF documents are parsed, chunked, and embedded into a vector store | PyMuPDF · LangChain · ChromaDB · all-MiniLM-L6-v2 |
| **2. Think** | A local LLM retrieves relevant context and generates a concise answer | Ollama · Llama 3.2 3B (Q4) |
| **3. Speak** | The text answer is converted to natural, high-quality speech | Edge-TTS (primary) · Piper TTS (fallback) |
| **4. Animate** | Your uploaded photograph comes alive with synchronized lip movements | Wav2Lip neural network · Viseme-based canvas animation |

---

## Two Lip-Sync Modes

AI Gurukul offers two avatar animation modes, selectable via a dropdown in the UI:

### ⚡ Animated Lip Sync (Fast)
- Pre-generates 5 neural viseme images (idle, a, e, o, m) using Wav2Lip + Poisson seamless cloning on avatar upload
- During playback, a canvas renderer swaps viseme frames at 60fps in sync with audio
- Latency: **~2 seconds** from question to first audio + animation
- Best for: real-time conversational feel

### 🎬 Real Lip Sync (High Quality)
- Streams text first, then generates a single Wav2Lip video for the full answer
- Produces actual neural lip-synced video with the avatar speaking every word
- Latency: **~20-30 seconds** for video generation (plays smoothly once ready)
- Best for: presentation-quality output

---

## Features

- **Any face, any document** — upload any photo with a face and any PDF
- **Dual lip-sync modes** — fast animated visemes or high-quality Wav2Lip video
- **Real-time streaming** — text tokens stream autoregressively via SSE
- **Edge-TTS** — high-quality Microsoft neural voice synthesis (free, no API key)
- **Collapsible sidebar** — auto-collapses when chatting starts for more screen space
- **Response time tracking** — latency shown next to each answer
- **Replay button** — replay any previous answer with lip-sync animation
- **Session cleanup** — auto-cleans temp data on browser close and server shutdown
- **Fully local** — no cloud APIs, no paid services, all data stays on your machine
- **Edge-optimized** — runs on 8 GB RAM, 4-core CPU, no GPU required
- **RAG evaluation** — built-in RAGAS metrics with CLI benchmarking
- **Production patterns** — structured logging, correlation IDs, error handling, 170+ unit tests

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Browser (React + Canvas)                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ Sidebar  │  │  Canvas  │  │   Chat   │  │  Processing   │  │
│  │ (upload) │  │ (avatar) │  │ (stream) │  │  Indicators   │  │
│  └──────────┘  └──────────┘  └──────────┘  └───────────────┘  │
└────────────────────────┬────────────────────────────────────────┘
                         │ SSE (Server-Sent Events)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                              │
│                                                                 │
│  ┌─────────────────── Orchestrator ──────────────────────────┐  │
│  │                                                           │  │
│  │  Animated: Retrieve → Stream LLM → Edge-TTS per sentence  │  │
│  │            → viseme images (base64) → canvas animation     │  │
│  │                                                           │  │
│  │  Real:     Retrieve → Stream LLM → Edge-TTS full answer   │  │
│  │            → Wav2Lip video → play in browser               │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │PDF Parser│  │Embedding │  │  Viseme  │  │  Evaluation  │   │
│  │(PyMuPDF) │  │  Store   │  │  Engine  │  │   (RAGAS)    │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | React 19 · TypeScript · Vite · Canvas API | Chat UI, viseme animation, video playback |
| **Backend** | FastAPI · Python 3.10 | API gateway, streaming orchestration |
| **LLM** | Ollama · Llama 3.2 3B (Q4_K_M) | Question answering (~2 GB) |
| **Embeddings** | all-MiniLM-L6-v2 | Semantic search (384-dim, ~80 MB) |
| **Vector Store** | ChromaDB | Persistent document embeddings |
| **TTS (primary)** | Edge-TTS | High-quality neural voice (free, ~1s per sentence) |
| **TTS (fallback)** | Piper TTS (ONNX) | Offline speech synthesis (~60 MB) |
| **Avatar** | Wav2Lip | Neural lip-sync + viseme generation (~416 MB) |
| **PDF Parsing** | PyMuPDF | Text + table extraction |
| **Chunking** | LangChain + tiktoken | 512-token chunks, 50-token overlap |
| **Evaluation** | RAGAS | Faithfulness, context & answer relevance |
| **Deployment** | Docker Compose · Conda | Multi-service orchestration |

---

## Quick Start (One Command)

```bash
git clone https://github.com/sanjeevranjaniitb/ai-gurukul.git
cd ai-gurukul
bash run.sh
```

The `run.sh` script handles everything:
1. Creates conda environment `ai-gurukul` (Python 3.10)
2. Installs all Python and Node.js dependencies
3. Downloads AI models (~860 MB, one-time)
4. Starts Ollama and pulls `llama3.2:3b`
5. Starts backend (port 8000) and frontend (port 5173)
6. Cleans up on Ctrl+C

Open **`http://localhost:5173`** when ready.

---

## Manual Setup

### Prerequisites

| Tool | Version | Install (macOS) |
|------|---------|-----------------|
| Conda | any | [anaconda.com](https://www.anaconda.com/) or `brew install miniconda` |
| Node.js | 20+ | `brew install node` |
| Ollama | latest | [ollama.com](https://ollama.com) |
| ffmpeg | any | `brew install ffmpeg` |

### Step-by-step

```bash
# 1. Clone
git clone https://github.com/sanjeevranjaniitb/ai-gurukul.git
cd ai-gurukul

# 2. Create conda environment
conda env create -f environment.yml
conda activate ai-gurukul

# 3. Download AI models
bash scripts/setup_models.sh

# 4. Download Wav2Lip checkpoint (~416 MB)
#    If setup_models.sh didn't get it, download manually from:
#    https://huggingface.co/tensorbanana/wav2lip/resolve/main/wav2lip_gan.pth
#    Place at: models/wav2lip/checkpoints/wav2lip_gan.pth

# 5. Install frontend
cd frontend && npm install && cd ..

# 6. Start Ollama + pull model
ollama serve &
ollama pull llama3.2:3b

# 7. Start backend
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

# 8. Start frontend (new terminal)
cd frontend && npm run dev
```

Open **`http://localhost:5173`**.

---

## How to Use

1. **Upload a photo** (left sidebar) — any image with a clear, front-facing human face
2. **Upload a PDF** (left sidebar) — the document you want to ask questions about
3. **Select lip-sync mode** (dropdown above avatar) — Animated (fast) or Real (high quality)
4. **Ask a question** — type in the chat box and press Send
5. **Watch the avatar speak** — text streams on the right, avatar animates on the left
6. **Replay** — click 🔄 Replay on any previous answer to hear it again
7. **Sidebar auto-collapses** when you start chatting — click ◀/▶ to toggle

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload/avatar` | Upload avatar image → returns viseme data |
| `POST` | `/api/upload/pdf` | Upload PDF → parse, chunk, embed |
| `POST` | `/api/ask` | Ask question → SSE stream (supports `mode`: `animated` or `real`) |
| `POST` | `/api/reset` | Clear all session data |
| `GET` | `/api/health` | Health check with memory usage |

### SSE Event Types (from `/api/ask`)

| Event | Payload | Description |
|-------|---------|-------------|
| `text_token` | `{ token }` | Incremental LLM text |
| `audio_chunk` | `{ chunk_url, sentence, duration_seconds }` | Synthesized audio + text for viseme sync |
| `video_chunk` | `{ chunk_url, duration_seconds }` | Wav2Lip video (real mode only) |
| `stage_update` | `{ stage, status }` | Pipeline progress |
| `sources` | `{ sources[] }` | Retrieved document chunks |
| `done` | `{ total_duration_ms }` | Stream complete |
| `error` | `{ message }` | Error details |

Full OpenAPI docs at `/docs` when backend is running.

---

## Running Tests

```bash
conda activate ai-gurukul

# Backend tests (170+ tests)
python -m pytest backend/app/ -v

# Frontend tests (27 tests)
cd frontend && npx vitest run

# With coverage
python -m pytest backend/app/ -v --cov=backend/app
```

---

## Configuration

All tunable parameters in `config/config.yaml`:

```yaml
chunk_size: 512
chunk_overlap: 50
retrieval_top_k: 5
llm_model: "llama3.2:3b"
llm_temperature: 0.1
llm_max_tokens: 150        # Short, concise answers for avatar speech
tts_sample_rate: 22050
avatar_fps: 25
max_image_size_mb: 10
max_pdf_size_mb: 50
```

---

## Project Structure

```
ai-gurukul/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI endpoints + session management
│   │   ├── orchestrator.py      # Dual-mode streaming pipeline
│   │   ├── viseme_engine.py     # Wav2Lip neural viseme generation
│   │   ├── edge_tts_engine.py   # Edge-TTS speech synthesis
│   │   ├── avatar_engine.py     # Wav2Lip video generation
│   │   ├── tts_engine.py        # Piper TTS fallback
│   │   ├── pdf_parser.py        # PDF text extraction
│   │   ├── chunking.py          # Text chunking
│   │   ├── embedding_store.py   # ChromaDB vector store
│   │   ├── llm_service.py       # Ollama LLM client
│   │   ├── rag_pipeline.py      # RAG retrieval + generation
│   │   ├── evaluation.py        # RAGAS evaluation
│   │   ├── config.py            # Configuration loader
│   │   ├── models.py            # Shared data models
│   │   └── test_*.py            # Unit tests (12 test files)
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx              # Main layout + state management
│   │   ├── components/
│   │   │   ├── AvatarPlayer.tsx     # Canvas-based viseme animation
│   │   │   ├── RealVideoPlayer.tsx  # Wav2Lip video playback
│   │   │   ├── ChatInterface.tsx    # Chat + replay + latency display
│   │   │   ├── AvatarUpload.tsx     # Image upload with validation
│   │   │   ├── PdfUpload.tsx        # PDF upload
│   │   │   └── ProcessingIndicator.tsx
│   │   └── hooks/useSSE.ts         # SSE consumer (dual mode)
│   ├── Dockerfile
│   └── package.json
├── models/wav2lip/              # Wav2Lip model + checkpoints
├── config/config.yaml           # Application configuration
├── environment.yml              # Conda environment spec
├── run.sh                       # Single-command launcher
├── scripts/setup_models.sh      # Model download script
├── docker-compose.yml           # Docker deployment
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

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Lip sync not working | Verify `models/wav2lip/checkpoints/wav2lip_gan.pth` is ~416 MB (not a text file). Re-download from HuggingFace if needed. |
| `ModuleNotFoundError` | Use `conda activate ai-gurukul` or `.venv/bin/python` — system Python doesn't have dependencies |
| Backend slow on first request | Models load lazily. First avatar upload takes ~15s to load Wav2Lip. |
| `Cannot connect to Ollama` | Run `ollama serve` first. Check with `curl http://localhost:11434/api/tags` |
| Frontend can't reach backend | Vite proxies `/api` to `localhost:8000`. Make sure backend is running on port 8000. |
| Real lip sync takes too long | Expected: ~20-30s for video generation on CPU. Use Animated mode for faster interaction. |

---

## Licenses

All components use open-source licenses. No paid APIs or cloud services required.

See [LICENSES.md](LICENSES.md) for the complete dependency manifest.

---

<p align="center">
  Built with open-source AI · Runs on your machine · No cloud required
</p>
