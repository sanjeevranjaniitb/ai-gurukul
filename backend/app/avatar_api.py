"""Standalone REST API for avatar video/audio generation.

Edge devices can call these endpoints to generate:
1. TTS audio from text
2. Lip-synced video from text + avatar image
3. Viseme images from an avatar image

All endpoints are mounted under /api/avatar/ in the main app.

Example usage from an edge device:

    # Upload avatar once
    curl -X POST http://<laptop-ip>:8000/api/avatar/register \
        -F "file=@face.jpg"

    # Generate audio from text
    curl -X POST http://<laptop-ip>:8000/api/avatar/tts \
        -H "Content-Type: application/json" \
        -d '{"text": "Hello world", "avatar_id": "abc123"}' \
        --output response.mp3

    # Generate lip-synced video from text
    curl -X POST http://<laptop-ip>:8000/api/avatar/video \
        -H "Content-Type: application/json" \
        -d '{"text": "Hello world", "avatar_id": "abc123"}' \
        --output response.mp4

    # Generate viseme images for client-side animation
    curl -X POST http://<laptop-ip>:8000/api/avatar/visemes \
        -H "Content-Type: application/json" \
        -d '{"avatar_id": "abc123"}'
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from backend.app.config import load_config
from backend.app.logging_utils import get_logger

logger = get_logger("avatar_api")
router = APIRouter(prefix="/api/avatar", tags=["Avatar Generation"])

_config = load_config()

# Lazy singletons
_avatar_engine = None
_viseme_engine = None
_tts_voice = None

# In-memory avatar registry
_registered_avatars: dict[str, dict] = {}


def _get_avatar_engine():
    global _avatar_engine
    if _avatar_engine is None:
        from backend.app.avatar_engine import AvatarEngine
        _avatar_engine = AvatarEngine(config=_config)
    return _avatar_engine


def _get_viseme_engine():
    global _viseme_engine
    if _viseme_engine is None:
        from backend.app.viseme_engine import VisemeEngine
        _viseme_engine = VisemeEngine()
    return _viseme_engine


# ── Request schemas ──

class TTSRequest(BaseModel):
    """Generate audio from text."""
    text: str
    avatar_id: str | None = None  # Optional — not needed for TTS-only
    voice: str = "en-IN-PrabhatNeural"


class VideoRequest(BaseModel):
    """Generate lip-synced video from text."""
    text: str
    avatar_id: str


class VisemeRequest(BaseModel):
    """Get viseme images for an avatar."""
    avatar_id: str


# ── Endpoints ──

@router.post(
    "/register",
    summary="Register an avatar image",
    response_description="Avatar ID and metadata",
)
async def register_avatar(file: UploadFile = File(...)):
    """Upload and register an avatar image for video generation.

    The image is preprocessed (face-centered crop) and viseme images
    are pre-generated. Returns an avatar_id to use in subsequent calls.
    """
    start = time.monotonic()

    if file.filename is None:
        raise HTTPException(400, "No filename")

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ("png", "jpg", "jpeg"):
        raise HTTPException(400, f"Unsupported format: {ext}")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(400, "Image exceeds 10 MB")

    avatar_id = str(uuid.uuid4())
    avatar_dir = Path("data/avatars") / avatar_id
    avatar_dir.mkdir(parents=True, exist_ok=True)

    img_path = avatar_dir / f"original.{ext}"
    img_path.write_bytes(contents)

    # Preprocess face
    engine = _get_avatar_engine()
    try:
        profile = engine.preprocess(str(img_path))
    except Exception as e:
        raise HTTPException(422, str(e))

    # Generate frame
    from backend.app.viseme_engine import preprocess_avatar_frame
    frame_path = str(avatar_dir / "frame.jpg")
    preprocess_avatar_frame(str(img_path), frame_path)

    # Generate visemes
    viseme_engine = _get_viseme_engine()
    viseme_dir = str(avatar_dir / "visemes")
    viseme_paths = viseme_engine.generate_visemes(
        frame_path if os.path.exists(frame_path) else str(img_path),
        viseme_dir,
    )

    _registered_avatars[avatar_id] = {
        "image_path": str(img_path.resolve()),
        "frame_path": frame_path if os.path.exists(frame_path) else str(img_path),
        "profile": profile,
        "viseme_paths": viseme_paths,
    }

    duration_ms = (time.monotonic() - start) * 1000
    logger.info("Avatar registered: %s in %.1fms", avatar_id, duration_ms)

    return {
        "avatar_id": avatar_id,
        "frame_url": f"/api/data/avatars/{avatar_id}/frame.jpg",
        "viseme_count": len(viseme_paths),
        "duration_ms": round(duration_ms, 1),
    }


@router.post(
    "/tts",
    summary="Text to speech",
    response_description="MP3 audio file",
)
async def text_to_speech(req: TTSRequest):
    """Convert text to speech audio. Returns an MP3 file.

    Uses Edge-TTS (high quality, free) with Piper TTS as fallback.
    """
    if not req.text.strip():
        raise HTTPException(400, "Text is empty")

    output_dir = Path("data/media") / f"tts_{uuid.uuid4().hex[:12]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = str(output_dir / "speech.mp3")

    # Try Edge-TTS
    try:
        from backend.app.edge_tts_engine import generate_edge_tts
        result = generate_edge_tts(req.text, audio_path, voice=req.voice)
        if result and os.path.exists(audio_path):
            return FileResponse(
                audio_path,
                media_type="audio/mpeg",
                filename="speech.mp3",
                headers={"X-Duration-Seconds": str(result.duration_seconds)},
            )
    except Exception as e:
        logger.warning("Edge-TTS failed: %s", e)

    # Fallback to Piper
    try:
        from backend.app.tts_engine import TTSEngine
        tts = TTSEngine(config=_config)
        wav_path = str(output_dir / "speech.wav")
        result = tts.synthesize(req.text, wav_path)
        return FileResponse(
            wav_path,
            media_type="audio/wav",
            filename="speech.wav",
            headers={"X-Duration-Seconds": str(result.duration_seconds)},
        )
    except Exception as e:
        raise HTTPException(500, f"TTS failed: {e}")


@router.post(
    "/video",
    summary="Generate lip-synced video",
    response_description="MP4 video file with lip-synced avatar",
)
async def generate_video(req: VideoRequest):
    """Generate a lip-synced MP4 video of the avatar speaking the given text.

    Pipeline: text → Edge-TTS → MP3 → WAV conversion → Wav2Lip → MP4

    This is the Real Lip Sync mode. Expect ~20-30 seconds for generation.
    """
    if not req.text.strip():
        raise HTTPException(400, "Text is empty")

    avatar = _registered_avatars.get(req.avatar_id)
    if not avatar:
        raise HTTPException(404, f"Avatar '{req.avatar_id}' not registered. Call /api/avatar/register first.")

    output_dir = Path("data/media") / f"video_{uuid.uuid4().hex[:12]}"
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()

    # Step 1: TTS
    audio_mp3 = str(output_dir / "audio.mp3")
    try:
        from backend.app.edge_tts_engine import generate_edge_tts
        audio_result = generate_edge_tts(req.text, audio_mp3)
        if not audio_result:
            raise RuntimeError("Edge-TTS returned None")
    except Exception as e:
        raise HTTPException(500, f"TTS failed: {e}")

    # Step 2: Convert MP3 → WAV for Wav2Lip
    audio_wav = str(output_dir / "audio.wav")
    try:
        from pydub import AudioSegment
        sound = AudioSegment.from_mp3(audio_mp3)
        sound.export(audio_wav, format="wav")
    except Exception:
        import subprocess
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_mp3, "-ar", "16000", "-ac", "1", audio_wav],
            capture_output=True, timeout=30,
        )

    # Step 3: Wav2Lip video generation
    video_path = str(output_dir / "video.mp4")
    engine = _get_avatar_engine()
    try:
        video_result = engine.animate_chunk(
            profile=avatar["profile"],
            audio_path=audio_wav,
            output_path=video_path,
            chunk_index=0,
        )
    except Exception as e:
        raise HTTPException(500, f"Video generation failed: {e}")

    duration_ms = (time.monotonic() - start) * 1000
    logger.info("Video generated: %.1fms, %.2fs duration", duration_ms, video_result.duration_seconds)

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename="avatar_video.mp4",
        headers={
            "X-Duration-Seconds": str(video_result.duration_seconds),
            "X-Generation-Ms": str(round(duration_ms, 1)),
        },
    )


@router.post(
    "/visemes",
    summary="Get viseme images",
    response_description="Base64-encoded viseme images for client-side animation",
)
async def get_visemes(req: VisemeRequest):
    """Return base64-encoded viseme images for client-side lip-sync animation.

    Returns 5 images: idle, a, e, o, m — each as a JPEG data URL.
    The edge device can swap between these in sync with audio playback.
    """
    avatar = _registered_avatars.get(req.avatar_id)
    if not avatar:
        raise HTTPException(404, f"Avatar '{req.avatar_id}' not registered.")

    import base64
    viseme_data = {}
    for name, path in avatar.get("viseme_paths", {}).items():
        try:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            viseme_data[name] = f"data:image/jpeg;base64,{b64}"
        except Exception:
            viseme_data[name] = ""

    return {
        "avatar_id": req.avatar_id,
        "visemes": viseme_data,
        "char_to_viseme": {
            'a': 'a', 'h': 'a', 'r': 'a', 'l': 'a',
            'e': 'e', 'i': 'e', 'y': 'e', 's': 'e', 'z': 'e',
            'o': 'o', 'u': 'o', 'w': 'o', 'q': 'o',
            'm': 'm', 'b': 'm', 'p': 'm', 'f': 'm', 'v': 'm',
            ' ': 'idle', '.': 'idle', ',': 'idle',
        },
    }


@router.get(
    "/list",
    summary="List registered avatars",
)
async def list_avatars():
    """List all registered avatar IDs."""
    return {
        "avatars": [
            {"avatar_id": aid, "frame_url": f"/api/data/avatars/{aid}/frame.jpg"}
            for aid in _registered_avatars
        ]
    }
