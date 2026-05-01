# Avatar Generation REST API — Edge Device Guide

This document describes the standalone REST API for generating lip-synced avatar audio and video from any edge device. The API runs on your laptop/server and edge devices call it over HTTP.

## Interactive Demo

A visual demo page is included that calls all REST endpoints — no Swagger needed:

**`http://localhost:5173/avatar-demo.html`**

The demo lets you:
1. Upload a face image → calls `POST /api/avatar/register`
2. Generate audio from text → calls `POST /api/avatar/tts`
3. Play viseme lip-sync animation → calls `POST /api/avatar/visemes`
4. Generate full Wav2Lip video → calls `POST /api/avatar/video`

All interactions use pure HTTP `fetch()` to the REST API — identical to what any edge device client would do.

## Prerequisites

- AI Gurukul backend running on your laptop/server
- Your laptop and edge device on the same network
- Find your laptop's IP: `ifconfig | grep "inet "` (use the 192.168.x.x or 10.x.x.x address)

Replace `LAPTOP_IP` in all examples below with your actual IP (e.g., `192.168.1.42`).

### Quick Start (API only — no frontend needed)

```bash
# Start the API server (backend only, accessible from any device on the network)
bash start_api.sh

# Or with a custom port
bash start_api.sh 9000
```

The script activates the venv, starts Ollama + LLM, launches the API server on `0.0.0.0`, and prints your network IP with ready-to-copy curl commands.

### Full Application (with web UI)

```bash
# Start everything — backend + frontend
bash run.sh
```

### Manual Start

```bash
source .venv/bin/activate
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

---

## API Overview

### Avatar Generation (Edge Device)

| Endpoint | Method | Purpose | Latency |
|----------|--------|---------|---------|
| `/api/avatar/register` | POST | Upload and register an avatar face image | ~1-15s (first load) |
| `/api/avatar/tts` | POST | Convert text to speech audio (MP3) | ~1-2s |
| `/api/avatar/video` | POST | Generate lip-synced video from text (MP4) | ~20-30s |
| `/api/avatar/visemes` | POST | Get 20 pre-generated viseme images (base64) | <1s |
| `/api/avatar/list` | GET | List all registered avatars | <1s |

### Quiz Generation

| Endpoint | Method | Purpose | Latency |
|----------|--------|---------|---------|
| `/api/quiz/generate` | POST | Generate MCQs from session Q&A history | ~5-15s |
| `/api/quiz/generate-from-doc` | POST | Generate MCQs from uploaded document content | ~30-60s |

### Core Application

| Endpoint | Method | Purpose | Latency |
|----------|--------|---------|---------|
| `/api/upload/avatar` | POST | Upload avatar for the web UI | ~1-15s |
| `/api/upload/pdf` | POST | Upload PDF → parse, chunk, embed | ~2-10s |
| `/api/ask` | POST | Ask question → SSE stream | ~3-8s |
| `/api/health` | GET | Health check with memory usage | <1s |
| `/api/reset` | POST | Clear all session data | <1s |

---

## Step 1: Register an Avatar

Upload a face image once. The server preprocesses it (face detection, cropping, viseme generation) and returns an `avatar_id` for all subsequent calls.

**Request:**
```bash
curl -X POST http://LAPTOP_IP:8000/api/avatar/register \
    -F "file=@/path/to/face.jpg"
```

**Response:**
```json
{
    "avatar_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "frame_url": "/api/data/avatars/a1b2c3d4-.../frame.jpg",
    "viseme_count": 20,
    "duration_ms": 1250.3
}
```

**Notes:**
- Accepts PNG, JPG, JPEG images up to 10 MB
- Image must contain a clear, front-facing human face
- First call takes ~15s (loads Wav2Lip model). Subsequent calls take ~1s
- Save the `avatar_id` — you need it for all other calls

---

## Step 2: Generate Audio (Text-to-Speech)

Convert any text to natural-sounding speech. Returns an MP3 file.

**Request:**
```bash
curl -X POST http://LAPTOP_IP:8000/api/avatar/tts \
    -H "Content-Type: application/json" \
    -d '{"text": "The attention mechanism allows the model to focus on relevant parts of the input."}' \
    -o speech.mp3
```

**Request body:**
```json
{
    "text": "Your text here",
    "voice": "en-IN-PrabhatNeural"
}
```

**Available voices:**
| Voice | Language | Style |
|-------|----------|-------|
| `en-IN-PrabhatNeural` | English (India) | Male, professional (default) |
| `en-US-GuyNeural` | English (US) | Male, casual |
| `en-US-JennyNeural` | English (US) | Female, friendly |
| `en-GB-RyanNeural` | English (UK) | Male, formal |

Full voice list: [Edge-TTS voices](https://github.com/rany2/edge-tts)

**Response:** MP3 audio file (binary)

**Response headers:**
- `X-Duration-Seconds`: Audio duration in seconds

---

## Step 3a: Generate Lip-Synced Video (Real Mode)

Generate a full Wav2Lip lip-synced MP4 video of the avatar speaking the text. This is the highest quality output but takes ~20-30 seconds.

**Request:**
```bash
curl -X POST http://LAPTOP_IP:8000/api/avatar/video \
    -H "Content-Type: application/json" \
    -d '{
        "text": "The attention mechanism allows the model to focus on relevant parts of the input.",
        "avatar_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
    }' \
    -o avatar_video.mp4
```

**Response:** MP4 video file (binary) with embedded audio

**Response headers:**
- `X-Duration-Seconds`: Video duration in seconds
- `X-Generation-Ms`: Time taken to generate in milliseconds

**Pipeline:** Text → Edge-TTS (MP3) → Convert to WAV → Wav2Lip → MP4

---

## Step 3b: Get Viseme Images (Animated Mode)

For faster, client-side lip-sync animation, get pre-generated viseme images and animate them on the edge device in sync with audio playback.

**Request:**
```bash
curl -X POST http://LAPTOP_IP:8000/api/avatar/visemes \
    -H "Content-Type: application/json" \
    -d '{"avatar_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"}'
```

**Response:**
```json
{
    "avatar_id": "a1b2c3d4-...",
    "visemes": {
        "idle": "data:image/jpeg;base64,/9j/4AAQ...",
        "ah": "data:image/jpeg;base64,...",
        "ae": "data:image/jpeg;base64,...",
        "eh": "data:image/jpeg;base64,...",
        "ee": "data:image/jpeg;base64,...",
        "ih": "data:image/jpeg;base64,...",
        "oh": "data:image/jpeg;base64,...",
        "oo": "data:image/jpeg;base64,...",
        "uh": "data:image/jpeg;base64,...",
        "mm": "data:image/jpeg;base64,...",
        "bv": "data:image/jpeg;base64,...",
        "ff": "data:image/jpeg;base64,...",
        "th": "data:image/jpeg;base64,...",
        "td": "data:image/jpeg;base64,...",
        "nn": "data:image/jpeg;base64,...",
        "ll": "data:image/jpeg;base64,...",
        "ss": "data:image/jpeg;base64,...",
        "sh": "data:image/jpeg;base64,...",
        "kk": "data:image/jpeg;base64,...",
        "ww": "data:image/jpeg;base64,...",
        "rr": "data:image/jpeg;base64,..."
    },
    "char_to_viseme": {
        "a": "ah", "h": "ah",
        "e": "eh", "i": "ih",
        "o": "oh", "u": "oo", "q": "oo",
        "m": "mm", "b": "bv", "p": "mm",
        "f": "ff", "v": "ff",
        "t": "td", "d": "td", "n": "nn", "l": "ll",
        "s": "ss", "z": "ss", "j": "sh", "c": "sh", "x": "sh",
        "k": "kk", "g": "kk",
        "w": "ww", "r": "rr", "y": "ee",
        " ": "idle", ".": "idle", ",": "idle"
    }
}
```

**20 viseme phoneme groups:**

| Viseme | Mouth Shape | Letters |
|--------|------------|---------|
| `idle` | Neutral/closed | spaces, punctuation |
| `ah` | Wide open | a, h |
| `ae` | Half open | (default fallback) |
| `eh` | Mid open | e |
| `ee` | Spread smile | y |
| `ih` | Slight smile | i |
| `oh` | Rounded | o |
| `oo` | Tight round | u, q |
| `uh` | Relaxed round | (variant) |
| `mm` | Closed lips | m, p |
| `bv` | Near-closed | b |
| `ff` | Lower lip tuck | f, v |
| `th` | Tongue tip | (th sound) |
| `td` | Alveolar stop | t, d |
| `nn` | Nasal | n |
| `ll` | Lateral | l |
| `ss` | Hiss | s, z |
| `sh` | Postalveolar | j, c, x |
| `kk` | Velar | k, g |
| `ww` | Pursed | w |
| `rr` | Retroflex | r |

**Client-side animation algorithm:**
```python
# Pseudocode for edge device playback
import time

# 1. Get audio + visemes
audio = call_tts_api(text)
visemes = call_visemes_api(avatar_id)
images = decode_base64_images(visemes["visemes"])  # 20 viseme images
char_map = visemes["char_to_viseme"]  # char → viseme name

# 2. Play audio and animate simultaneously
audio_duration = get_duration(audio)
char_speed = len(text) / audio_duration  # chars per second

play_audio(audio)
start_time = time.time()

while audio_is_playing():
    elapsed = time.time() - start_time
    char_idx = int(elapsed * char_speed)
    
    if char_idx < len(text):
        char = text[char_idx].lower()
        viseme_name = char_map.get(char, "ae")  # default: half-open
        display_image(images[viseme_name])
    
    time.sleep(1/60)  # 60 FPS for smooth animation

display_image(images["idle"])  # Reset to idle when audio ends
```

---

## List Registered Avatars

**Request:**
```bash
curl http://LAPTOP_IP:8000/api/avatar/list
```

**Response:**
```json
{
    "avatars": [
        {
            "avatar_id": "a1b2c3d4-...",
            "frame_url": "/api/data/avatars/a1b2c3d4-.../frame.jpg"
        }
    ]
}
```

---

## Complete Edge Device Example (Python)

```python
"""Complete example: edge device calling AI Gurukul avatar API."""

import requests
import base64
import time
import io

SERVER = "http://192.168.1.42:8000"

# ── Step 1: Register avatar ──
print("Registering avatar...")
with open("face.jpg", "rb") as f:
    resp = requests.post(f"{SERVER}/api/avatar/register", files={"file": f})
avatar_id = resp.json()["avatar_id"]
print(f"Avatar ID: {avatar_id}")

# ── Step 2: Generate audio ──
text = "The transformer architecture uses self-attention to process sequences in parallel."
print(f"Generating audio for: {text[:50]}...")
resp = requests.post(f"{SERVER}/api/avatar/tts", json={"text": text})
with open("speech.mp3", "wb") as f:
    f.write(resp.content)
duration = float(resp.headers.get("X-Duration-Seconds", "3"))
print(f"Audio: {duration:.1f}s")

# ── Step 3a: Option A — Real lip-synced video ──
print("Generating lip-synced video (this takes ~20-30s)...")
resp = requests.post(f"{SERVER}/api/avatar/video", json={
    "text": text,
    "avatar_id": avatar_id,
})
with open("avatar.mp4", "wb") as f:
    f.write(resp.content)
gen_ms = resp.headers.get("X-Generation-Ms", "?")
print(f"Video saved: avatar.mp4 (generated in {gen_ms}ms)")

# ── Step 3b: Option B — Animated visemes (fast) ──
print("Getting viseme images...")
resp = requests.post(f"{SERVER}/api/avatar/visemes", json={"avatar_id": avatar_id})
data = resp.json()
visemes = data["visemes"]       # {name: "data:image/jpeg;base64,..."}
char_map = data["char_to_viseme"]  # {char: viseme_name}
print(f"Got {len(visemes)} viseme images")  # Should be 20+idle

# Play audio + animate visemes (pseudocode — adapt to your display)
# See the animation algorithm above
```

---

## Complete Edge Device Example (cURL + Shell)

```bash
#!/bin/bash
SERVER="http://192.168.1.42:8000"

# Register avatar
AVATAR_ID=$(curl -s -X POST "$SERVER/api/avatar/register" \
    -F "file=@face.jpg" | python3 -c "import sys,json; print(json.load(sys.stdin)['avatar_id'])")
echo "Avatar: $AVATAR_ID"

# Generate audio
curl -s -X POST "$SERVER/api/avatar/tts" \
    -H "Content-Type: application/json" \
    -d '{"text": "Hello from the edge device"}' \
    -o speech.mp3
echo "Audio saved: speech.mp3"

# Generate video
curl -s -X POST "$SERVER/api/avatar/video" \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"Hello from the edge device\", \"avatar_id\": \"$AVATAR_ID\"}" \
    -o avatar.mp4
echo "Video saved: avatar.mp4"
```

---

## Error Handling

All errors return JSON with an HTTP status code:

```json
{"detail": "Error description"}
```

| Status | Meaning |
|--------|---------|
| 400 | Bad request (empty text, invalid format, file too large) |
| 404 | Avatar not registered (call `/register` first) |
| 422 | Image processing failed (no face detected) |
| 500 | Server error (TTS or video generation failed) |

---

## Performance Notes

| Operation | Typical Latency | Notes |
|-----------|----------------|-------|
| Register avatar (first time) | ~15s | Loads Wav2Lip model |
| Register avatar (subsequent) | ~1s | Model already loaded |
| TTS (Edge-TTS) | ~1-2s | Depends on text length |
| Video generation | ~20-30s | CPU-bound Wav2Lip inference |
| Viseme images | <1s | Pre-generated during registration |

For real-time interaction, use the **animated viseme approach** (Step 3b). For presentation-quality output, use the **video approach** (Step 3a).

---

## Network Configuration

If your edge device can't reach the server:

1. Ensure both devices are on the same WiFi/LAN
2. Check firewall: `sudo ufw allow 8000` (Linux) or System Preferences → Firewall (macOS)
3. Start the backend with `--host 0.0.0.0` to listen on all interfaces
4. Test connectivity: `curl http://LAPTOP_IP:8000/api/health`
