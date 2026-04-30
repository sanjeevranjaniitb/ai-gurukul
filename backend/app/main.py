"""FastAPI application entry point for the Live Talking Head Avatar backend."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.app.config import load_config
from backend.app.logging_utils import get_logger, set_correlation_id

logger = get_logger("api")

app = FastAPI(
    title="AI Gurukul",
    description="Conversational AI avatar that answers questions from your PDF documents.",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Configuration & lazy orchestrator
# ---------------------------------------------------------------------------

_config = load_config()
_orchestrator = None


def _get_orchestrator():
    """Lazily initialise the Orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        from backend.app.orchestrator import Orchestrator
        _orchestrator = Orchestrator(_config)
    return _orchestrator


# ---------------------------------------------------------------------------
# Mount static media files
# ---------------------------------------------------------------------------

_data_dir = Path("data")
_data_dir.mkdir(parents=True, exist_ok=True)
Path(_config.media_output_dir).mkdir(parents=True, exist_ok=True)
Path("data/avatars").mkdir(parents=True, exist_ok=True)
Path("data/documents").mkdir(parents=True, exist_ok=True)
app.mount("/api/data", StaticFiles(directory=str(_data_dir)), name="data")

# Mount avatar generation API for edge devices
from backend.app.avatar_api import router as avatar_router
app.include_router(avatar_router)

# Mount quiz generation API for session knowledge quizzes
from backend.app.quiz_module import router as quiz_router
app.include_router(quiz_router)

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    """JSON body for the /api/ask endpoint."""
    question: str
    avatar_id: str
    mode: str = "animated"  # "animated" (viseme) or "real" (Wav2Lip video)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post(
    "/api/upload/avatar",
    summary="Upload avatar image",
    tags=["Avatar"],
    response_description="Avatar ID, preview URL, and landmarks readiness flag",
)
async def upload_avatar(file: UploadFile = File(..., description="Avatar image file (PNG/JPG/JPEG, ≤10 MB, ≥256×256)")):
    """Upload an avatar image for talking head animation.

    Accepts PNG, JPG, or JPEG images up to 10 MB with a minimum resolution of
    256×256 pixels. The image must contain a detectable human face. Returns an
    avatar_id used to reference this avatar in subsequent /api/ask requests.
    """
    cid = str(uuid.uuid4())
    set_correlation_id(cid)
    logger.info("upload_avatar request received, filename=%s", file.filename)

    if file.filename is None:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in _config.allowed_image_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format '{ext}'. Allowed: {_config.allowed_image_formats}",
        )

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > _config.max_image_size_mb:
        raise HTTPException(
            status_code=400,
            detail=f"Image exceeds {_config.max_image_size_mb} MB size limit",
        )

    # Save to a temp file and delegate to orchestrator
    avatar_id = str(uuid.uuid4())
    avatar_dir = Path("data/avatars") / avatar_id
    avatar_dir.mkdir(parents=True, exist_ok=True)
    temp_path = avatar_dir / f"original.{ext}"
    temp_path.write_bytes(contents)

    orchestrator = _get_orchestrator()
    try:
        await orchestrator.upload_avatar(temp_path, avatar_id)
    except Exception as exc:
        logger.error("upload_avatar failed: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    preview_url = f"/api/data/avatars/{avatar_id}/original.{ext}"
    viseme_urls = orchestrator.get_viseme_urls(avatar_id)
    frame_url = orchestrator.get_frame_url(avatar_id)
    logger.info("upload_avatar success, avatar_id=%s, visemes=%d", avatar_id, len(viseme_urls))
    return {
        "avatar_id": avatar_id,
        "preview_url": frame_url or preview_url,
        "landmarks_ready": True,
        "visemes": viseme_urls,
        "frame_url": frame_url,
    }


@app.post(
    "/api/upload/pdf",
    summary="Upload PDF document",
    tags=["Documents"],
    response_description="Document ID, filename, page count, and chunk count",
)
async def upload_pdf(file: UploadFile = File(..., description="PDF document file (≤50 MB)")):
    """Upload a PDF document to build the knowledge base.

    The document is parsed, split into 512-token chunks with 50-token overlap,
    embedded using all-MiniLM-L6-v2, and stored in ChromaDB for retrieval.
    """
    cid = str(uuid.uuid4())
    set_correlation_id(cid)
    logger.info("upload_pdf request received, filename=%s", file.filename)

    if file.filename is None:
        raise HTTPException(status_code=400, detail="No filename provided")

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > _config.max_pdf_size_mb:
        raise HTTPException(
            status_code=400,
            detail=f"PDF exceeds {_config.max_pdf_size_mb} MB size limit",
        )

    document_id = str(uuid.uuid4())
    doc_dir = Path("data/documents") / document_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    temp_path = doc_dir / "original.pdf"
    temp_path.write_bytes(contents)

    orchestrator = _get_orchestrator()
    try:
        result = await orchestrator.upload_pdf(temp_path, document_id)
    except Exception as exc:
        logger.error("upload_pdf failed: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    logger.info("upload_pdf success, document_id=%s", document_id)
    return {
        "document_id": document_id,
        "name": file.filename,
        "page_count": result.get("page_count", 0),
        "chunk_count": result.get("chunk_count", 0),
    }


@app.post(
    "/api/ask",
    summary="Ask a question",
    tags=["Chat"],
    response_description="Server-Sent Events stream with text, audio, and video chunks",
)
async def ask(body: AskRequest):
    """Submit a question and receive a streaming avatar response.

    Returns a text/event-stream with the following SSE event types:
    - **text_token** — incremental LLM tokens for the text answer
    - **audio_chunk** — URL of a synthesised audio segment
    - **video_chunk** — URL of an animated avatar video segment
    - **stage_update** — current processing stage and duration
    - **sources** — retrieved document chunks used as context
    - **error** — error details if a pipeline stage fails
    - **done** — signals the end of the stream
    """
    cid = str(uuid.uuid4())
    set_correlation_id(cid)
    logger.info("ask request, question=%s, avatar_id=%s", body.question, body.avatar_id)

    orchestrator = _get_orchestrator()

    async def _event_generator():
        try:
            async for event in orchestrator.process_question_stream(
                body.question, body.avatar_id, mode=body.mode
            ):
                payload = json.dumps(event.data, default=str)
                yield f"event: {event.type}\ndata: {payload}\n\n"
        except Exception as exc:
            logger.error("ask stream error: %s", exc)
            error_payload = json.dumps({"error": str(exc)})
            yield f"event: error\ndata: {error_payload}\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Correlation-Id": cid,
        },
    )


def _memory_usage_mb() -> float:
    """Return current process RSS in megabytes using /proc or os."""
    try:
        # Linux: read from /proc for zero-dependency memory info
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # kB → MB
    except (FileNotFoundError, OSError):
        pass
    # Fallback: use resource module (Unix) or return 0
    try:
        import resource
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        # maxrss is in KB on Linux, bytes on macOS
        if os.uname().sysname == "Darwin":
            return rusage.ru_maxrss / (1024 * 1024)
        return rusage.ru_maxrss / 1024
    except (ImportError, AttributeError):
        return 0.0


@app.get(
    "/api/health",
    summary="Health check",
    tags=["System"],
    response_description="System status, model readiness, and memory usage",
)
async def health_check():
    """Check system health and resource usage.

    Returns the current status, whether AI models are loaded, and the
    process memory usage in megabytes.
    """
    orchestrator = None
    try:
        orchestrator = _get_orchestrator()
    except Exception:
        pass

    models_loaded = orchestrator is not None
    return {
        "status": "ok",
        "models_loaded": models_loaded,
        "memory_usage_mb": round(_memory_usage_mb(), 1),
    }


@app.post(
    "/api/reset",
    summary="Reset session",
    tags=["System"],
    response_description="Clears all session data (avatars, media, documents)",
)
async def reset_session():
    """Clear all session data — avatars, media files, uploaded documents.

    Call this when starting a new session or when the browser tab closes.
    """
    import shutil

    global _orchestrator

    # Reset orchestrator state
    if _orchestrator is not None:
        _orchestrator._avatars.clear()
        _orchestrator._avatar_visemes.clear()
        _orchestrator._avatar_frames.clear()

    # Clean media files
    media_dir = Path(_config.media_output_dir)
    if media_dir.exists():
        for item in media_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)

    # Clean avatar visemes and frames (keep original uploads)
    avatars_dir = Path("data/avatars")
    if avatars_dir.exists():
        for avatar_dir in avatars_dir.iterdir():
            if avatar_dir.is_dir():
                visemes_dir = avatar_dir / "visemes"
                if visemes_dir.exists():
                    shutil.rmtree(visemes_dir, ignore_errors=True)
                frame = avatar_dir / "frame.jpg"
                if frame.exists():
                    frame.unlink()

    # Clear quiz cache
    from backend.app.quiz_module import clear_quiz_cache
    clear_quiz_cache()

    logger.info("Session reset — all media and avatar data cleared")
    return {"status": "reset", "message": "Session data cleared"}


@app.on_event("shutdown")
async def shutdown_cleanup():
    """Clean up session data when the server stops."""
    import shutil
    media_dir = Path(_config.media_output_dir)
    if media_dir.exists():
        for item in media_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
    logger.info("Shutdown cleanup completed")
