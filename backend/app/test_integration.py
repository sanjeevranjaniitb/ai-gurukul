"""Integration tests for the end-to-end flow (task 15.1).

Validates: Requirements 9.4

These tests exercise the full pipeline from HTTP request through FastAPI
routing, orchestrator wiring, and SSE streaming — mocking only the heavy
AI components (SentenceTransformer, Ollama, Piper TTS, Wav2Lip/AvatarEngine).
"""

from __future__ import annotations

import io
import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.app.config import AppConfig
from backend.app.models import (
    AudioResult,
    AvatarProfile,
    DocumentChunk,
    PageContent,
    ParsedDocument,
    RetrievalResult,
    VideoResult,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _parse_sse_events(body: str) -> list[dict]:
    """Parse a text/event-stream body into a list of {type, data} dicts."""
    events = []
    current_type = None
    current_data_lines = []

    for line in body.splitlines():
        if line.startswith("event: "):
            # If we had a previous event being built, flush it
            if current_type is not None and current_data_lines:
                raw = "\n".join(current_data_lines)
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    data = raw
                events.append({"type": current_type, "data": data})
                current_data_lines = []
            current_type = line[len("event: "):].strip()
        elif line.startswith("data: "):
            current_data_lines.append(line[len("data: "):])
        elif line == "" and current_type is not None and current_data_lines:
            raw = "\n".join(current_data_lines)
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = raw
            events.append({"type": current_type, "data": data})
            current_type = None
            current_data_lines = []

    # Flush any trailing event
    if current_type is not None and current_data_lines:
        raw = "\n".join(current_data_lines)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = raw
        events.append({"type": current_type, "data": data})

    return events


def _make_profile(avatar_id: str = "test-avatar") -> AvatarProfile:
    return AvatarProfile(
        avatar_id=avatar_id,
        image_path="/tmp/avatar.png",
        landmarks={"face_rect": {"x": 0, "y": 0, "w": 100, "h": 100}},
        preprocessed_at=datetime.now(timezone.utc).isoformat(),
    )


def _make_parsed_doc() -> ParsedDocument:
    return ParsedDocument(
        text="Machine learning is a subset of artificial intelligence.",
        pages=[
            PageContent(
                page_number=1,
                text="Machine learning is a subset of artificial intelligence.",
                tables=[],
            )
        ],
        page_count=1,
        metadata={"title": "ML Intro"},
    )


def _make_chunks(document_id: str = "doc-123") -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="chunk-1",
            text="Machine learning is a subset of artificial intelligence.",
            document_id=document_id,
            page_number=1,
            token_count=10,
            start_char=0,
            end_char=55,
        )
    ]


def _make_retrieval_results() -> list[RetrievalResult]:
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        text="Machine learning is a subset of artificial intelligence.",
        document_id="doc-123",
        page_number=1,
        token_count=10,
        start_char=0,
        end_char=55,
    )
    return [RetrievalResult(chunk=chunk, score=0.92, distance=0.16)]


def _make_audio_result(path: str = "data/media/test/audio_0.wav") -> AudioResult:
    return AudioResult(
        file_path=path, duration_seconds=1.5, sample_rate=22050, format="wav"
    )


def _make_video_result(path: str = "data/media/test/video_0.mp4") -> VideoResult:
    return VideoResult(
        file_path=path,
        duration_seconds=1.5,
        fps=25,
        resolution=(256, 256),
        format="mp4",
    )


def _build_orchestrator_with_mocks(*, with_context: bool = True):
    """Create a real Orchestrator with mocked sub-components.

    Returns (orchestrator, mocks_dict).
    """
    from backend.app.orchestrator import Orchestrator

    config = AppConfig(
        media_output_dir="data/media",
        chroma_persist_dir="/tmp/test_chroma_integration",
    )
    orch = Orchestrator(config)

    mock_store = MagicMock()
    mock_store.search.return_value = (
        _make_retrieval_results() if with_context else []
    )
    mock_store.add_chunks = MagicMock()

    mock_llm = MagicMock()
    mock_llm.generate_stream.return_value = iter(
        ["Machine", " learning", " is", " great", "."]
    )

    mock_tts = MagicMock()
    mock_tts.synthesize_chunk.return_value = _make_audio_result()

    mock_avatar = MagicMock()
    mock_avatar.preprocess.return_value = _make_profile()
    mock_avatar.animate_chunk.return_value = _make_video_result()

    mock_parser = MagicMock()
    mock_parser.parse.return_value = _make_parsed_doc()

    mock_chunking = MagicMock()
    mock_chunking.chunk.return_value = _make_chunks()

    orch._embedding_store = mock_store
    orch._llm_service = mock_llm
    orch._tts_engine = mock_tts
    orch._avatar_engine = mock_avatar
    orch._pdf_parser = mock_parser
    orch._chunking = mock_chunking

    mocks = {
        "embedding_store": mock_store,
        "llm_service": mock_llm,
        "tts_engine": mock_tts,
        "avatar_engine": mock_avatar,
        "pdf_parser": mock_parser,
        "chunking": mock_chunking,
    }
    return orch, mocks


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_orchestrator():
    """Reset the global orchestrator between tests."""
    import backend.app.main as main_mod
    main_mod._orchestrator = None
    yield
    main_mod._orchestrator = None


@pytest.fixture()
def integration_setup():
    """Wire a real Orchestrator (with mocked AI components) into the app."""
    orch, mocks = _build_orchestrator_with_mocks(with_context=True)

    import backend.app.main as main_mod
    main_mod._orchestrator = orch

    from backend.app.main import app
    client = TestClient(app)

    return client, orch, mocks


@pytest.fixture()
def client():
    """Bare TestClient with no orchestrator pre-wired."""
    from backend.app.main import app
    return TestClient(app)


# ------------------------------------------------------------------
# 1. Health endpoint returns correct status
# ------------------------------------------------------------------

class TestHealthIntegration:
    def test_health_returns_ok_status(self, integration_setup):
        client, _, _ = integration_setup
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_health_reports_models_loaded(self, integration_setup):
        client, _, _ = integration_setup
        resp = client.get("/api/health")
        data = resp.json()
        assert data["models_loaded"] is True

    def test_health_includes_memory_usage(self, integration_setup):
        client, _, _ = integration_setup
        resp = client.get("/api/health")
        data = resp.json()
        assert "memory_usage_mb" in data
        assert isinstance(data["memory_usage_mb"], (int, float))


# ------------------------------------------------------------------
# 2. Avatar upload → returns avatar_id and preview_url
# ------------------------------------------------------------------

class TestAvatarUploadIntegration:
    def test_avatar_upload_returns_avatar_id_and_preview(self, integration_setup):
        client, _, mocks = integration_setup
        fake_image = b"\x89PNG\r\n" + b"\x00" * 100

        resp = client.post(
            "/api/upload/avatar",
            files={"file": ("face.png", io.BytesIO(fake_image), "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "avatar_id" in data
        assert len(data["avatar_id"]) > 0
        assert "preview_url" in data
        assert data["preview_url"].startswith("/api/data/avatars/")
        assert data["landmarks_ready"] is True

    def test_avatar_upload_calls_orchestrator_pipeline(self, integration_setup):
        client, _, mocks = integration_setup
        fake_image = b"\x89PNG\r\n" + b"\x00" * 100

        client.post(
            "/api/upload/avatar",
            files={"file": ("face.jpg", io.BytesIO(fake_image), "image/jpeg")},
        )
        mocks["avatar_engine"].preprocess.assert_called_once()

    def test_avatar_upload_rejects_invalid_format(self, integration_setup):
        client, _, _ = integration_setup
        resp = client.post(
            "/api/upload/avatar",
            files={"file": ("face.bmp", io.BytesIO(b"\x00" * 10), "image/bmp")},
        )
        assert resp.status_code == 400
        assert "Unsupported image format" in resp.json()["detail"]


# ------------------------------------------------------------------
# 3. PDF upload → returns document_id, page_count, chunk_count
# ------------------------------------------------------------------

class TestPdfUploadIntegration:
    def test_pdf_upload_returns_document_metadata(self, integration_setup):
        client, _, mocks = integration_setup
        fake_pdf = b"%PDF-1.4" + b"\x00" * 100

        resp = client.post(
            "/api/upload/pdf",
            files={"file": ("report.pdf", io.BytesIO(fake_pdf), "application/pdf")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "document_id" in data
        assert len(data["document_id"]) > 0
        assert data["name"] == "report.pdf"
        assert data["page_count"] == 1
        assert data["chunk_count"] == 1

    def test_pdf_upload_triggers_parse_chunk_embed_pipeline(self, integration_setup):
        client, _, mocks = integration_setup
        fake_pdf = b"%PDF-1.4" + b"\x00" * 100

        client.post(
            "/api/upload/pdf",
            files={"file": ("doc.pdf", io.BytesIO(fake_pdf), "application/pdf")},
        )
        mocks["pdf_parser"].parse.assert_called_once()
        mocks["chunking"].chunk.assert_called_once()
        mocks["embedding_store"].add_chunks.assert_called_once()

    def test_pdf_upload_rejects_oversized_file(self, integration_setup):
        client, _, _ = integration_setup
        big = b"\x00" * (51 * 1024 * 1024)
        resp = client.post(
            "/api/upload/pdf",
            files={"file": ("big.pdf", io.BytesIO(big), "application/pdf")},
        )
        assert resp.status_code == 400
        assert "size limit" in resp.json()["detail"]


# ------------------------------------------------------------------
# 4. Ask endpoint → returns SSE stream with expected events
# ------------------------------------------------------------------

class TestAskSSEIntegration:
    def test_ask_returns_sse_stream_with_text_tokens(self, integration_setup):
        client, orch, mocks = integration_setup

        # Register an avatar so the pipeline doesn't error
        profile = _make_profile("sse-avatar")
        orch._avatars["sse-avatar"] = profile

        resp = client.post(
            "/api/ask",
            json={"question": "What is ML?", "avatar_id": "sse-avatar"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        events = _parse_sse_events(resp.text)
        event_types = [e["type"] for e in events]

        assert "text_token" in event_types
        assert "stage_update" in event_types
        assert "done" in event_types

    def test_ask_sse_contains_stage_updates(self, integration_setup):
        client, orch, mocks = integration_setup

        profile = _make_profile("stage-avatar")
        orch._avatars["stage-avatar"] = profile

        resp = client.post(
            "/api/ask",
            json={"question": "Explain AI", "avatar_id": "stage-avatar"},
        )
        events = _parse_sse_events(resp.text)
        stage_events = [e for e in events if e["type"] == "stage_update"]

        # Should have at least retrieving started/completed and generating started/completed
        assert len(stage_events) >= 2
        stages = [e["data"].get("stage") for e in stage_events]
        assert "retrieving" in stages
        assert "generating" in stages

    def test_ask_sse_done_event_has_duration(self, integration_setup):
        client, orch, mocks = integration_setup

        profile = _make_profile("done-avatar")
        orch._avatars["done-avatar"] = profile

        resp = client.post(
            "/api/ask",
            json={"question": "Hello?", "avatar_id": "done-avatar"},
        )
        events = _parse_sse_events(resp.text)
        done_events = [e for e in events if e["type"] == "done"]

        assert len(done_events) == 1
        assert "total_duration_ms" in done_events[0]["data"]

    def test_ask_sse_text_tokens_reconstruct_answer(self, integration_setup):
        client, orch, mocks = integration_setup

        profile = _make_profile("text-avatar")
        orch._avatars["text-avatar"] = profile
        mocks["llm_service"].generate_stream.return_value = iter(
            ["Hello", " world", "."]
        )

        resp = client.post(
            "/api/ask",
            json={"question": "Greet me", "avatar_id": "text-avatar"},
        )
        events = _parse_sse_events(resp.text)
        tokens = [
            e["data"]["token"]
            for e in events
            if e["type"] == "text_token"
        ]
        assert "".join(tokens) == "Hello world."

    def test_ask_sse_includes_sources_when_context_found(self, integration_setup):
        client, orch, mocks = integration_setup

        profile = _make_profile("src-avatar")
        orch._avatars["src-avatar"] = profile

        resp = client.post(
            "/api/ask",
            json={"question": "What is ML?", "avatar_id": "src-avatar"},
        )
        events = _parse_sse_events(resp.text)
        source_events = [e for e in events if e["type"] == "sources"]

        assert len(source_events) == 1
        assert "sources" in source_events[0]["data"]
        assert len(source_events[0]["data"]["sources"]) > 0


# ------------------------------------------------------------------
# 5. Ask with unknown avatar_id → returns error event
# ------------------------------------------------------------------

class TestAskUnknownAvatarIntegration:
    def test_ask_unknown_avatar_returns_error_event(self, integration_setup):
        client, _, _ = integration_setup

        resp = client.post(
            "/api/ask",
            json={"question": "Hello?", "avatar_id": "nonexistent-avatar"},
        )
        # SSE streams always return 200
        assert resp.status_code == 200

        events = _parse_sse_events(resp.text)
        error_events = [e for e in events if e["type"] == "error"]

        assert len(error_events) == 1
        assert "not found" in error_events[0]["data"]["message"].lower()

    def test_ask_unknown_avatar_no_done_event(self, integration_setup):
        client, _, _ = integration_setup

        resp = client.post(
            "/api/ask",
            json={"question": "Hello?", "avatar_id": "ghost-avatar"},
        )
        events = _parse_sse_events(resp.text)
        done_events = [e for e in events if e["type"] == "done"]

        # When avatar is not found, the stream ends with error, no done event
        assert len(done_events) == 0


# ------------------------------------------------------------------
# 6. Full flow: upload avatar → upload PDF → ask → streaming response
# ------------------------------------------------------------------

class TestFullFlowIntegration:
    def test_full_pipeline_avatar_pdf_ask(self, integration_setup):
        """End-to-end: upload avatar, upload PDF, ask question, get SSE response."""
        client, orch, mocks = integration_setup

        # Step 1: Upload avatar
        fake_image = b"\x89PNG\r\n" + b"\x00" * 100
        avatar_resp = client.post(
            "/api/upload/avatar",
            files={"file": ("face.png", io.BytesIO(fake_image), "image/png")},
        )
        assert avatar_resp.status_code == 200
        avatar_id = avatar_resp.json()["avatar_id"]
        assert avatar_id

        # The orchestrator stores the avatar profile via upload_avatar.
        # Since the mock avatar_engine.preprocess returns a profile with
        # avatar_id="test-avatar", but the API layer overrides it with
        # its own UUID, we need to verify the avatar is stored.
        assert avatar_id in orch._avatars

        # Step 2: Upload PDF
        fake_pdf = b"%PDF-1.4" + b"\x00" * 100
        pdf_resp = client.post(
            "/api/upload/pdf",
            files={"file": ("knowledge.pdf", io.BytesIO(fake_pdf), "application/pdf")},
        )
        assert pdf_resp.status_code == 200
        pdf_data = pdf_resp.json()
        assert pdf_data["document_id"]
        assert pdf_data["page_count"] == 1
        assert pdf_data["chunk_count"] == 1

        # Step 3: Ask a question referencing the uploaded avatar
        mocks["llm_service"].generate_stream.return_value = iter(
            ["ML", " is", " a", " branch", " of", " AI", "."]
        )

        ask_resp = client.post(
            "/api/ask",
            json={"question": "What is machine learning?", "avatar_id": avatar_id},
        )
        assert ask_resp.status_code == 200
        assert ask_resp.headers["content-type"].startswith("text/event-stream")

        events = _parse_sse_events(ask_resp.text)
        event_types = [e["type"] for e in events]

        # Verify the full event sequence
        assert "stage_update" in event_types
        assert "text_token" in event_types
        assert "done" in event_types

        # Verify text tokens reconstruct the answer
        tokens = [
            e["data"]["token"]
            for e in events
            if e["type"] == "text_token"
        ]
        full_answer = "".join(tokens)
        assert "ML" in full_answer
        assert "AI" in full_answer

        # Verify the done event has timing info
        done_events = [e for e in events if e["type"] == "done"]
        assert len(done_events) == 1
        assert done_events[0]["data"]["total_duration_ms"] > 0

    def test_full_flow_verifies_pipeline_wiring(self, integration_setup):
        """Verify that all pipeline components are called in the correct order."""
        client, orch, mocks = integration_setup

        # Upload avatar
        fake_image = b"\x89PNG\r\n" + b"\x00" * 100
        avatar_resp = client.post(
            "/api/upload/avatar",
            files={"file": ("face.png", io.BytesIO(fake_image), "image/png")},
        )
        avatar_id = avatar_resp.json()["avatar_id"]

        # Upload PDF
        fake_pdf = b"%PDF-1.4" + b"\x00" * 100
        client.post(
            "/api/upload/pdf",
            files={"file": ("doc.pdf", io.BytesIO(fake_pdf), "application/pdf")},
        )

        # Verify PDF pipeline: parse → chunk → embed
        mocks["pdf_parser"].parse.assert_called_once()
        mocks["chunking"].chunk.assert_called_once()
        mocks["embedding_store"].add_chunks.assert_called_once()

        # Ask question
        mocks["llm_service"].generate_stream.return_value = iter(["Answer."])
        client.post(
            "/api/ask",
            json={"question": "What is this about?", "avatar_id": avatar_id},
        )

        # Verify ask pipeline: retrieve → generate → TTS → animate
        mocks["embedding_store"].search.assert_called_once()
        mocks["llm_service"].generate_stream.assert_called_once()
        mocks["tts_engine"].synthesize_chunk.assert_called()
        mocks["avatar_engine"].animate_chunk.assert_called()

    def test_full_flow_audio_and_video_chunks_in_stream(self, integration_setup):
        """Verify that audio_chunk and video_chunk events appear in the SSE stream."""
        client, orch, mocks = integration_setup

        # Upload avatar
        fake_image = b"\x89PNG\r\n" + b"\x00" * 100
        avatar_resp = client.post(
            "/api/upload/avatar",
            files={"file": ("face.png", io.BytesIO(fake_image), "image/png")},
        )
        avatar_id = avatar_resp.json()["avatar_id"]

        # Set up LLM to produce a complete sentence (triggers TTS + animation)
        mocks["llm_service"].generate_stream.return_value = iter(
            ["This", " is", " a", " test", "."]
        )

        ask_resp = client.post(
            "/api/ask",
            json={"question": "Test?", "avatar_id": avatar_id},
        )
        events = _parse_sse_events(ask_resp.text)
        event_types = [e["type"] for e in events]

        assert "audio_chunk" in event_types
        assert "video_chunk" in event_types

        # Verify audio chunk has expected fields
        audio_events = [e for e in events if e["type"] == "audio_chunk"]
        assert len(audio_events) >= 1
        assert "chunk_url" in audio_events[0]["data"]
        assert "chunk_index" in audio_events[0]["data"]

        # Verify video chunk has expected fields
        video_events = [e for e in events if e["type"] == "video_chunk"]
        assert len(video_events) >= 1
        assert "chunk_url" in video_events[0]["data"]
        assert "chunk_index" in video_events[0]["data"]
