#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
# AI Gurukul — Backend API Server (headless, no frontend)
#
# Starts ONLY the backend API server so edge devices on the network
# can call the Avatar, Quiz, and Core endpoints.
#
# Usage:
#   bash start_api.sh              # default port 8000
#   bash start_api.sh 9000         # custom port
#
# Endpoints exposed:
#   POST /api/avatar/register      — Register avatar image
#   POST /api/avatar/tts           — Text to speech (MP3)
#   POST /api/avatar/video         — Lip-synced video (MP4)
#   POST /api/avatar/visemes       — 20 viseme images (base64)
#   GET  /api/avatar/list          — List registered avatars
#   POST /api/quiz/generate        — Quiz from Q&A session
#   POST /api/quiz/generate-from-doc — Quiz from document
#   POST /api/upload/pdf           — Upload PDF document
#   POST /api/ask                  — Ask question (SSE stream)
#   GET  /api/health               — Health check
#   GET  /docs                     — Interactive API docs
#
# See AVATAR_API.md for full edge device integration guide.
# ══════════════════════════════════════════════════════════════════

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

PORT="${1:-8000}"
VENV_DIR="$PROJECT_ROOT/.venv"

# Colors
G='\033[0;32m'; Y='\033[1;33m'; C='\033[0;36m'; R='\033[0;31m'; N='\033[0m'
log()  { echo -e "${G}[✓]${N} $1"; }
warn() { echo -e "${Y}[!]${N} $1"; }
info() { echo -e "${C}[→]${N} $1"; }
err()  { echo -e "${R}[✗]${N} $1"; }

cleanup() {
    echo ""
    info "Shutting down API server..."
    curl -sf -X POST "http://localhost:${PORT}/api/reset" >/dev/null 2>&1 && log "Session data cleaned"
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null && wait "$SERVER_PID" 2>/dev/null
    rm -rf "$PROJECT_ROOT/data/media/"* 2>/dev/null
    echo -e "${G}API server stopped.${N}"
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

echo ""
echo "🎓 AI Gurukul — API Server"
echo "═══════════════════════════════════════════"

# ── 1. Check venv — create and install deps if missing ──
PYTHON_BIN=""
for candidate in python3.10 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        PY_VER=$("$candidate" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
        if [ "$PY_VER" = "3.10" ]; then
            PYTHON_BIN="$candidate"
            break
        fi
    fi
done

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    if [ -z "$PYTHON_BIN" ]; then
        err "Python 3.10 not found and no existing venv."
        err "Install Python 3.10: brew install python@3.10 (macOS) or sudo apt install python3.10 (Ubuntu)"
        exit 1
    fi
    info "Creating virtual environment..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    log "Virtual environment created"
    FRESH_VENV=true
else
    FRESH_VENV=false
fi

source "$VENV_DIR/bin/activate"
log "Python venv activated — $(python --version)"

# ── 2. Install dependencies if needed ──
if [ "$FRESH_VENV" = true ] || [ ! -f "$VENV_DIR/.deps_installed" ]; then
    info "Installing Python dependencies..."
    pip install --upgrade pip --quiet
    pip install -r backend/requirements.txt --quiet
    log "Backend dependencies installed"
    pip install librosa scipy torch torchvision torchaudio --quiet 2>/dev/null
    log "Wav2Lip dependencies installed"
    touch "$VENV_DIR/.deps_installed"
else
    log "Python dependencies already installed"
fi

# ── 3. Download AI models if needed ──
if [ -f scripts/setup_models.sh ]; then
    info "Checking AI models..."
    bash scripts/setup_models.sh 2>&1 | grep -E "^\[" || true
fi

# ── 4. Check Ollama ──
if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    log "Ollama is running"
else
    info "Starting Ollama..."
    if command -v ollama &>/dev/null; then
        nohup ollama serve >/dev/null 2>&1 &
        for i in $(seq 1 15); do
            sleep 1
            curl -sf http://localhost:11434/api/tags >/dev/null 2>&1 && break
        done
        if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
            log "Ollama started"
        else
            warn "Could not start Ollama — quiz and Q&A features will be unavailable"
        fi
    else
        warn "Ollama not installed — quiz and Q&A features will be unavailable"
        warn "Install from https://ollama.com"
    fi
fi

# Check LLM model
if curl -sf http://localhost:11434/api/tags 2>/dev/null | grep -q "llama3.2:3b"; then
    log "LLM llama3.2:3b ready"
else
    if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
        info "Pulling llama3.2:3b (~2 GB)..."
        ollama pull llama3.2:3b
        log "LLM pulled"
    fi
fi

# ── 5. Get network IP ──
LOCAL_IP=$(ifconfig 2>/dev/null | grep "inet " | grep -v 127.0.0.1 | head -1 | awk '{print $2}')
if [ -z "$LOCAL_IP" ]; then
    LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
fi
[ -z "$LOCAL_IP" ] && LOCAL_IP="<your-ip>"

# ── 6. Clean previous session ──
rm -rf data/media/* 2>/dev/null
log "Previous session data cleaned"

# ── 7. Start API server ──
info "Starting API server on 0.0.0.0:${PORT}..."
python -m uvicorn backend.app.main:app \
    --host 0.0.0.0 --port "$PORT" \
    --log-level info &
SERVER_PID=$!

# Wait for ready
for i in $(seq 1 30); do
    sleep 1
    if curl -sf "http://localhost:${PORT}/api/health" >/dev/null 2>&1; then
        break
    fi
    [ "$i" -eq 30 ] && warn "Server still loading — will be ready on first request"
done

echo ""
echo "═══════════════════════════════════════════"
echo -e "🎓 ${G}API Server is ready!${N}"
echo ""
echo -e "   ${C}Local:${N}    http://localhost:${PORT}"
echo -e "   ${C}Network:${N}  http://${LOCAL_IP}:${PORT}"
echo -e "   ${C}API Docs:${N} http://${LOCAL_IP}:${PORT}/docs"
echo ""
echo -e "   ${C}Health:${N}   curl http://${LOCAL_IP}:${PORT}/api/health"
echo -e "   ${C}Register:${N} curl -X POST http://${LOCAL_IP}:${PORT}/api/avatar/register -F 'file=@face.jpg'"
echo ""
echo "   See AVATAR_API.md for full edge device guide"
echo "   Press Ctrl+C to stop"
echo "═══════════════════════════════════════════"
echo ""

wait
