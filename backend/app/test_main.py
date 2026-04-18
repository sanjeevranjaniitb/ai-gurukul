"""Unit tests for FastAPI API endpoints (task 8.2)."""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.app.models import StreamEvent


@pytest.fixture(autouse=True)
def _reset_orchestrator():
    """Reset the global orchestrator between tests."""
    import backend.app.main as main_mod
    main_mod._orchestrator = None
    yield
    main_mod._orchestrator = None


@pytest.fixture()
def mock_orchestrator():
    """Return a mock Orchestrator wired into the app."""
    orch = MagicMock()
    orch.upload_avatar = AsyncMock()
    orch.upload_pdf = AsyncMock(return_value={"page_count": 3, "chunk_count": 12})

    import backend.app.main as main_mod
    main_mod._orchestrator = orch
    return orch


@pytest.fixture()
def client():
    from backend.app.main import app
    return TestClient(app)


# ---- Health endpoint -------------------------------------------------------

def test_health_returns_status(client, mock_orchestrator):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "models_loaded" in data
    assert "memory_usage_mb" in data


def test_health_models_loaded_true_when_orchestrator_exists(client, mock_orchestrator):
    resp = client.get("/api/health")
    assert resp.json()["models_loaded"] is True


# ---- Upload avatar endpoint ------------------------------------------------

def test_upload_avatar_success(client, mock_orchestrator, tmp_path):
    fake_image = b"\x89PNG\r\n" + b"\x00" * 100
    resp = client.post(
        "/api/upload/avatar",
        files={"file": ("face.png", io.BytesIO(fake_image), "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "avatar_id" in data
    assert data["landmarks_ready"] is True
    assert data["preview_url"].startswith("/api/media/avatars/")
    assert data["preview_url"].endswith(".png")
    mock_orchestrator.upload_avatar.assert_awaited_once()


def test_upload_avatar_rejects_unsupported_format(client, mock_orchestrator):
    resp = client.post(
        "/api/upload/avatar",
        files={"file": ("face.bmp", io.BytesIO(b"\x00" * 10), "image/bmp")},
    )
    assert resp.status_code == 400
    assert "Unsupported image format" in resp.json()["detail"]


def test_upload_avatar_rejects_oversized_file(client, mock_orchestrator):
    big = b"\x00" * (11 * 1024 * 1024)  # 11 MB
    resp = client.post(
        "/api/upload/avatar",
        files={"file": ("face.png", io.BytesIO(big), "image/png")},
    )
    assert resp.status_code == 400
    assert "size limit" in resp.json()["detail"]


def test_upload_avatar_orchestrator_error(client, mock_orchestrator):
    mock_orchestrator.upload_avatar = AsyncMock(side_effect=ValueError("No face detected"))
    resp = client.post(
        "/api/upload/avatar",
        files={"file": ("face.jpg", io.BytesIO(b"\xff\xd8" + b"\x00" * 50), "image/jpeg")},
    )
    assert resp.status_code == 422
    assert "No face detected" in resp.json()["detail"]


# ---- Upload PDF endpoint ---------------------------------------------------

def test_upload_pdf_success(client, mock_orchestrator):
    fake_pdf = b"%PDF-1.4" + b"\x00" * 100
    resp = client.post(
        "/api/upload/pdf",
        files={"file": ("doc.pdf", io.BytesIO(fake_pdf), "application/pdf")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "document_id" in data
    assert data["name"] == "doc.pdf"
    assert data["page_count"] == 3
    assert data["chunk_count"] == 12
    mock_orchestrator.upload_pdf.assert_awaited_once()


def test_upload_pdf_rejects_oversized(client, mock_orchestrator):
    big = b"\x00" * (51 * 1024 * 1024)
    resp = client.post(
        "/api/upload/pdf",
        files={"file": ("big.pdf", io.BytesIO(big), "application/pdf")},
    )
    assert resp.status_code == 400
    assert "size limit" in resp.json()["detail"]


def test_upload_pdf_orchestrator_error(client, mock_orchestrator):
    mock_orchestrator.upload_pdf = AsyncMock(side_effect=RuntimeError("Corrupt PDF"))
    resp = client.post(
        "/api/upload/pdf",
        files={"file": ("bad.pdf", io.BytesIO(b"%PDF" + b"\x00" * 10), "application/pdf")},
    )
    assert resp.status_code == 422
    assert "Corrupt PDF" in resp.json()["detail"]


# ---- Ask endpoint (SSE) ----------------------------------------------------

def test_ask_streams_sse_events(client, mock_orchestrator):
    events = [
        StreamEvent(type="text_token", data={"token": "Hello"}),
        StreamEvent(type="stage_update", data={"stage": "generating", "duration_ms": 100}),
        StreamEvent(type="done", data={"total_duration_ms": 500}),
    ]

    async def fake_stream(question, avatar_id):
        for e in events:
            yield e

    mock_orchestrator.process_question_stream = fake_stream

    resp = client.post(
        "/api/ask",
        json={"question": "What is AI?", "avatar_id": "abc-123"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    body = resp.text
    assert "event: text_token" in body
    assert '"Hello"' in body
    assert "event: stage_update" in body
    assert "event: done" in body


def test_ask_handles_stream_error(client, mock_orchestrator):
    async def failing_stream(question, avatar_id):
        raise RuntimeError("LLM unavailable")
        yield  # noqa: unreachable — makes this an async generator

    mock_orchestrator.process_question_stream = failing_stream

    resp = client.post(
        "/api/ask",
        json={"question": "test", "avatar_id": "x"},
    )
    assert resp.status_code == 200  # SSE stream still returns 200
    assert "event: error" in resp.text
    assert "LLM unavailable" in resp.text
