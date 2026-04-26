#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
# AI Gurukul — Single-command launcher
# Usage: bash run.sh
#
# Creates conda env "ai-gurukul" if needed, starts Ollama, backend,
# and frontend. Cleans up session data on exit.
# ══════════════════════════════════════════════════════════════════

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

CONDA_ENV="ai-gurukul"
BACKEND_PORT=8000
FRONTEND_PORT=5173
BACKEND_PID=""
FRONTEND_PID=""

# Colors
G='\033[0;32m'; Y='\033[1;33m'; C='\033[0;36m'; R='\033[0;31m'; N='\033[0m'
log()  { echo -e "${G}[✓]${N} $1"; }
warn() { echo -e "${Y}[!]${N} $1"; }
info() { echo -e "${C}[→]${N} $1"; }
err()  { echo -e "${R}[✗]${N} $1"; }

cleanup() {
    echo ""
    info "Shutting down..."

    # Call backend reset to clean session data
    curl -sf -X POST "http://localhost:${BACKEND_PORT}/api/reset" >/dev/null 2>&1 && log "Session data cleaned"

    # Kill processes
    if [ -n "$BACKEND_PID" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        kill "$BACKEND_PID" 2>/dev/null
        wait "$BACKEND_PID" 2>/dev/null
        log "Backend stopped"
    fi
    if [ -n "$FRONTEND_PID" ] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        kill "$FRONTEND_PID" 2>/dev/null
        wait "$FRONTEND_PID" 2>/dev/null
        log "Frontend stopped"
    fi

    # Clean temp files
    rm -rf "$PROJECT_ROOT/data/media/"* 2>/dev/null
    rm -rf "$PROJECT_ROOT/temp/"* 2>/dev/null

    echo -e "${G}Goodbye!${N}"
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# ══════════════════════════════════════════════════════════════════
echo ""
echo "🎓 AI Gurukul — Starting up"
echo "═══════════════════════════════════════════"

# ── 1. Check prerequisites ──
for cmd in conda node; do
    if ! command -v "$cmd" &>/dev/null; then
        err "$cmd not found. Install it first."
        exit 1
    fi
done

if ! command -v ollama &>/dev/null; then
    err "ollama not found. Install from https://ollama.com"
    exit 1
fi

command -v ffmpeg &>/dev/null && log "ffmpeg found" || warn "ffmpeg not found (install with: brew install ffmpeg)"

# ── 2. Conda environment ──
# Source conda so 'conda activate' works in this script
CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    err "Cannot find conda.sh — is conda installed correctly?"
    exit 1
fi

if conda env list 2>/dev/null | grep -qw "$CONDA_ENV"; then
    log "Conda env '$CONDA_ENV' exists"
else
    info "Creating conda env '$CONDA_ENV' (Python 3.10)..."
    conda env create -f environment.yml -y
    log "Conda env created"
fi

conda activate "$CONDA_ENV"
log "Activated '$CONDA_ENV' — $(python --version)"

# ── 3. Download AI models ──
if [ -f scripts/setup_models.sh ]; then
    info "Checking AI models..."
    bash scripts/setup_models.sh 2>&1 | grep -E "^\[" || true
fi

# ── 4. Ollama + LLM model ──
if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    log "Ollama is running"
else
    info "Starting Ollama..."
    nohup ollama serve >/dev/null 2>&1 &
    for i in $(seq 1 10); do
        sleep 1
        curl -sf http://localhost:11434/api/tags >/dev/null 2>&1 && break
    done
    if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
        log "Ollama started"
    else
        err "Could not start Ollama. Run 'ollama serve' manually."
        exit 1
    fi
fi

if curl -sf http://localhost:11434/api/tags 2>/dev/null | grep -q "llama3.2:3b"; then
    log "LLM llama3.2:3b ready"
else
    info "Pulling llama3.2:3b (~2 GB)..."
    ollama pull llama3.2:3b
    log "LLM pulled"
fi

# ── 5. Frontend dependencies ──
if [ ! -d frontend/node_modules ]; then
    info "Installing frontend dependencies..."
    (cd frontend && npm install --silent)
    log "Frontend deps installed"
else
    log "Frontend deps ready"
fi

# ── 6. Clean previous session data ──
rm -rf data/media/* 2>/dev/null
rm -rf temp/* 2>/dev/null
log "Previous session data cleaned"

# ── 7. Start backend ──
info "Starting backend on :${BACKEND_PORT}..."
python -m uvicorn backend.app.main:app \
    --host 0.0.0.0 --port "$BACKEND_PORT" \
    --log-level warning &
BACKEND_PID=$!

# Wait for backend to be ready (up to 30s for model loading)
for i in $(seq 1 30); do
    sleep 1
    if curl -sf "http://localhost:${BACKEND_PORT}/api/health" >/dev/null 2>&1; then
        log "Backend ready at http://localhost:${BACKEND_PORT}"
        break
    fi
    [ "$i" -eq 30 ] && warn "Backend still loading — it will be ready on first request"
done

# ── 8. Start frontend ──
info "Starting frontend on :${FRONTEND_PORT}..."
(cd frontend && npx vite --port "$FRONTEND_PORT" --host 2>/dev/null) &
FRONTEND_PID=$!
sleep 2
log "Frontend ready at http://localhost:${FRONTEND_PORT}"

# ── 9. Done ──
echo ""
echo "═══════════════════════════════════════════"
echo -e "🎓 ${G}AI Gurukul is ready!${N}"
echo ""
echo -e "   ${C}App:${N}      http://localhost:${FRONTEND_PORT}"
echo -e "   ${C}API:${N}      http://localhost:${BACKEND_PORT}"
echo -e "   ${C}API Docs:${N} http://localhost:${BACKEND_PORT}/docs"
echo ""
echo "   Press Ctrl+C to stop"
echo "═══════════════════════════════════════════"
echo ""

# Keep running until Ctrl+C
wait
