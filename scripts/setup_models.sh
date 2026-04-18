#!/usr/bin/env bash
# Download all required AI models for the Live Talking Head Avatar project.
# Usage: bash scripts/setup_models.sh
#
# This script is idempotent — it skips files that already exist.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Live Talking Head Avatar — Model Setup ==="
echo ""

# ---------------------------------------------------------------
# 1. Piper TTS model (~60 MB)
# ---------------------------------------------------------------
PIPER_DIR="models/piper"
PIPER_ONNX="$PIPER_DIR/en_US-lessac-medium.onnx"
PIPER_JSON="$PIPER_DIR/en_US-lessac-medium.onnx.json"
PIPER_BASE_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"

mkdir -p "$PIPER_DIR"

if [ -f "$PIPER_ONNX" ]; then
    echo "[✓] Piper TTS model already exists"
else
    echo "[↓] Downloading Piper TTS model (~60 MB)..."
    curl -L -o "$PIPER_ONNX" "$PIPER_BASE_URL/en_US-lessac-medium.onnx"
    echo "[✓] Piper TTS model downloaded"
fi

if [ -f "$PIPER_JSON" ]; then
    echo "[✓] Piper TTS config already exists"
else
    echo "[↓] Downloading Piper TTS config..."
    curl -L -o "$PIPER_JSON" "$PIPER_BASE_URL/en_US-lessac-medium.onnx.json"
    echo "[✓] Piper TTS config downloaded"
fi

# ---------------------------------------------------------------
# 2. Wav2Lip (~416 MB checkpoint + repo)
# ---------------------------------------------------------------
WAV2LIP_DIR="models/wav2lip"
WAV2LIP_CKPT="$WAV2LIP_DIR/checkpoints/wav2lip_gan.pth"

if [ -d "$WAV2LIP_DIR/.git" ]; then
    echo "[✓] Wav2Lip repo already cloned"
else
    echo "[↓] Cloning Wav2Lip repository..."
    rm -rf "$WAV2LIP_DIR"
    git clone --depth 1 https://github.com/Rudrabha/Wav2Lip.git "$WAV2LIP_DIR"
    echo "[✓] Wav2Lip repo cloned"
fi

mkdir -p "$WAV2LIP_DIR/checkpoints"
if [ -f "$WAV2LIP_CKPT" ]; then
    echo "[✓] Wav2Lip checkpoint already exists"
else
    echo "[↓] Downloading Wav2Lip GAN checkpoint (~416 MB)..."
    echo "    NOTE: If this URL fails, download manually from:"
    echo "    https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/Eb3LEzbfuKlJiR600lQWRxgBIY27JZg80YsvvDUNsBtfPA?e=n9ljGW"
    echo "    and place at: $WAV2LIP_CKPT"
    # Try the direct link; Wav2Lip hosts on SharePoint which may require manual download
    curl -L -o "$WAV2LIP_CKPT" \
        "https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/Eb3LEzbfuKlJiR600lQWRxgBIY27JZg80YsvvDUNsBtfPA?download=1" \
        2>/dev/null || echo "    [!] Auto-download failed. Please download manually (see above)."
    if [ -f "$WAV2LIP_CKPT" ] && [ "$(stat -f%z "$WAV2LIP_CKPT" 2>/dev/null || stat -c%s "$WAV2LIP_CKPT" 2>/dev/null)" -gt 1000000 ]; then
        echo "[✓] Wav2Lip checkpoint downloaded"
    else
        echo "[!] Wav2Lip checkpoint may not have downloaded correctly."
        echo "    The system will fall back to static image video without it."
    fi
fi

# ---------------------------------------------------------------
# 3. GFPGAN face detection weights (~300 MB)
#    Required by Wav2Lip's face detection module
# ---------------------------------------------------------------
GFPGAN_DIR="gfpgan/weights"
ALIGNMENT_PTH="$GFPGAN_DIR/alignment_WFLW_4HG.pth"
DETECTION_PTH="$GFPGAN_DIR/detection_Resnet50_Final.pth"

mkdir -p "$GFPGAN_DIR"

if [ -f "$ALIGNMENT_PTH" ]; then
    echo "[✓] GFPGAN alignment weights already exist"
else
    echo "[↓] Downloading GFPGAN alignment weights (~185 MB)..."
    curl -L -o "$ALIGNMENT_PTH" \
        "https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth"
    echo "[✓] GFPGAN alignment weights downloaded"
fi

if [ -f "$DETECTION_PTH" ]; then
    echo "[✓] GFPGAN detection weights already exist"
else
    echo "[↓] Downloading GFPGAN detection weights (~104 MB)..."
    curl -L -o "$DETECTION_PTH" \
        "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth"
    echo "[✓] GFPGAN detection weights downloaded"
fi

# ---------------------------------------------------------------
# 4. SadTalker (placeholder — not currently used)
# ---------------------------------------------------------------
mkdir -p models/sadtalker
echo "[·] SadTalker directory created (models not required — Wav2Lip is used instead)"

# ---------------------------------------------------------------
# 5. Sentence-transformers embedding model
#    Downloaded automatically on first use by sentence-transformers.
#    Pre-download here to avoid delays on first request.
# ---------------------------------------------------------------
echo ""
echo "[↓] Pre-downloading sentence-transformers embedding model..."
python3 -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print('[✓] all-MiniLM-L6-v2 cached')
" 2>/dev/null || echo "[·] Skipped (install Python deps first: pip install -r backend/requirements.txt)"

# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------
echo ""
echo "=== Model Setup Complete ==="
echo ""
echo "Model sizes:"
du -sh models/piper/*.onnx models/wav2lip/checkpoints/*.pth gfpgan/weights/*.pth 2>/dev/null || true
echo ""
echo "Next steps:"
echo "  1. Install Ollama: https://ollama.com"
echo "  2. Pull the LLM:   ollama pull llama3.2:3b"
echo "  3. Start the app:  see README.md"
