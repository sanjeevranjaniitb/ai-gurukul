"""Tests for the Orchestrator class."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from backend.app.config import AppConfig
from backend.app.models import (
    AudioResult,
    AvatarProfile,
    DocumentChunk,
    PageContent,
    ParsedDocument,
    RetrievalResult,
    StreamEvent,
    VideoResult,
)
from backend.app.orchestrator import Orchestrator, _split_sentences


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_config() -> AppConfig:
    return AppConfig(
        media_output_dir="/tmp/test_media",
        chroma_persist_dir="/tmp/test_chroma",
    )


def _make_profile() -> AvatarProfile:
    return AvatarProfile(
        avatar_id="avatar123",
        image_path="/tmp/avatar.png",
        landmarks={"face_rect": {"x": 0, "y": 0, "w": 100, "h": 100}},
        preprocessed_at="2024-01-01T00:00:00Z",
    )


def _make_retrieval_results() -> list[RetrievalResult]:
    chunk = DocumentChunk(
        chunk_id="c1", text="Some relevant context.", document_id="doc1",
        page_number=1, token_count=5, start_char=0, end_char=22,
    )
    return [RetrievalResult(chunk=chunk, score=0.9, distance=0.2)]


def _make_audio_result(path: str = "/tmp/audio.wav") -> AudioResult:
    return AudioResult(file_path=path, duration_seconds=1.5, sample_rate=22050, format="wav")


def _make_video_result(path: str = "/tmp/video.mp4") -> VideoResult:
    return VideoResult(file_path=path, duration_seconds=1.5, fps=25, resolution=(256, 256), format="mp4")


async def _collect_events(orch: Orchestrator, question: str, avatar_id: str) -> list[StreamEvent]:
    events: list[StreamEvent] = []
    async for event in orch.process_question_stream(question, avatar_id):
        events.append(event)
    return events


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ------------------------------------------------------------------
# Tests: _split_sentences helper
# ------------------------------------------------------------------

class TestSplitSentences:
    def test_single_incomplete(self):
        assert _split_sentences("hello world") == ["hello world"]

    def test_single_complete(self):
        result = _split_sentences("Hello world. ")
        assert len(result) >= 2
        assert result[0].strip() == "Hello world."

    def test_multiple_sentences(self):
        result = _split_sentences("First. Second! Third? Incomplete")
        complete = [s.strip() for s in result[:-1] if s.strip()]
        assert len(complete) >= 3

    def test_empty_string(self):
        assert _split_sentences("") == [""]


# ------------------------------------------------------------------
# Tests: Orchestrator construction and lazy init
# ------------------------------------------------------------------

class TestOrchestratorInit:
    def test_construction_is_cheap(self):
        orch = Orchestrator(_make_config())
        assert orch._embedding_store is None
        assert orch._llm_service is None
        assert orch._tts_engine is None
        assert orch._avatar_engine is None
        assert orch._pdf_parser is None
        assert orch._chunking is None

    def test_avatars_dict_starts_empty(self):
        orch = Orchestrator(_make_config())
        assert orch._avatars == {}


# ------------------------------------------------------------------
# Tests: upload_avatar
# ------------------------------------------------------------------

class TestUploadAvatar:
    def test_upload_avatar_stores_profile(self):
        orch = Orchestrator(_make_config())
        profile = _make_profile()

        mock_engine = MagicMock()
        mock_engine.preprocess.return_value = profile
        orch._avatar_engine = mock_engine

        result = _run(orch.upload_avatar("/tmp/photo.png"))

        assert result.avatar_id == "avatar123"
        assert orch._avatars["avatar123"] is profile
        mock_engine.preprocess.assert_called_once_with("/tmp/photo.png")

    def test_upload_avatar_with_caller_id(self):
        orch = Orchestrator(_make_config())
        profile = _make_profile()

        mock_engine = MagicMock()
        mock_engine.preprocess.return_value = profile
        orch._avatar_engine = mock_engine

        result = _run(orch.upload_avatar("/tmp/photo.png", avatar_id="custom-id"))

        assert result.avatar_id == "custom-id"
        assert "custom-id" in orch._avatars


# ------------------------------------------------------------------
# Tests: upload_pdf
# ------------------------------------------------------------------

class TestUploadPdf:
    def test_upload_pdf_returns_summary(self):
        orch = Orchestrator(_make_config())

        parsed = ParsedDocument(
            text="Hello world",
            pages=[PageContent(page_number=1, text="Hello world", tables=[])],
            page_count=1,
            metadata={},
        )
        chunks = [
            DocumentChunk(
                chunk_id="c1", text="Hello world", document_id="d1",
                page_number=1, token_count=2, start_char=0, end_char=11,
            )
        ]

        mock_parser = MagicMock()
        mock_parser.parse.return_value = parsed
        mock_chunking = MagicMock()
        mock_chunking.chunk.return_value = chunks
        mock_store = MagicMock()

        orch._pdf_parser = mock_parser
        orch._chunking = mock_chunking
        orch._embedding_store = mock_store

        result = _run(orch.upload_pdf("/tmp/doc.pdf", document_id="my-doc-id"))

        assert result["name"] == "doc.pdf"
        assert result["page_count"] == 1
        assert result["chunk_count"] == 1
        assert result["document_id"] == "my-doc-id"
        mock_store.add_chunks.assert_called_once_with(chunks)


# ------------------------------------------------------------------
# Tests: process_question_stream
# ------------------------------------------------------------------

class TestProcessQuestionStream:
    def test_error_event_on_missing_avatar(self):
        orch = Orchestrator(_make_config())
        events = _run(_collect_events(orch, "What is AI?", "nonexistent"))

        assert len(events) == 1
        assert events[0].type == "error"
        assert "not found" in events[0].data["message"]

    def test_full_pipeline_yields_expected_events(self):
        orch = Orchestrator(_make_config())
        profile = _make_profile()
        orch._avatars[profile.avatar_id] = profile

        mock_store = MagicMock()
        mock_store.search.return_value = _make_retrieval_results()
        orch._embedding_store = mock_store

        mock_llm = MagicMock()
        mock_llm.generate_stream.return_value = iter(
            ["Hello", " world", ".", " Good", " bye", "!"]
        )
        orch._llm_service = mock_llm

        mock_tts = MagicMock()
        mock_tts.synthesize_chunk.return_value = _make_audio_result()
        orch._tts_engine = mock_tts

        mock_avatar = MagicMock()
        mock_avatar.animate_chunk.return_value = _make_video_result()
        orch._avatar_engine = mock_avatar

        events = _run(_collect_events(orch, "What is AI?", profile.avatar_id))
        event_types = [e.type for e in events]

        assert "text_token" in event_types
        assert "audio_chunk" in event_types
        assert "video_chunk" in event_types
        assert "stage_update" in event_types
        assert "sources" in event_types
        assert events[-1].type == "done"
        assert "total_duration_ms" in events[-1].data

    def test_text_tokens_contain_all_content(self):
        orch = Orchestrator(_make_config())
        profile = _make_profile()
        orch._avatars[profile.avatar_id] = profile

        mock_store = MagicMock()
        mock_store.search.return_value = []
        orch._embedding_store = mock_store

        mock_llm = MagicMock()
        mock_llm.generate_stream.return_value = iter(["Hi", " there", "."])
        orch._llm_service = mock_llm

        mock_tts = MagicMock()
        mock_tts.synthesize_chunk.return_value = _make_audio_result()
        orch._tts_engine = mock_tts

        mock_avatar = MagicMock()
        mock_avatar.animate_chunk.return_value = _make_video_result()
        orch._avatar_engine = mock_avatar

        events = _run(_collect_events(orch, "Hello?", profile.avatar_id))
        tokens = [e.data["token"] for e in events if e.type == "text_token"]
        assert "".join(tokens) == "Hi there."

    def test_error_event_on_exception(self):
        orch = Orchestrator(_make_config())
        profile = _make_profile()
        orch._avatars[profile.avatar_id] = profile

        mock_store = MagicMock()
        mock_store.search.side_effect = RuntimeError("DB crashed")
        orch._embedding_store = mock_store

        events = _run(_collect_events(orch, "Boom?", profile.avatar_id))
        error_events = [e for e in events if e.type == "error"]
        assert len(error_events) == 1
        assert "internal error" in error_events[0].data["message"].lower()

    def test_no_sources_when_context_empty(self):
        orch = Orchestrator(_make_config())
        profile = _make_profile()
        orch._avatars[profile.avatar_id] = profile

        mock_store = MagicMock()
        mock_store.search.return_value = []
        orch._embedding_store = mock_store

        mock_llm = MagicMock()
        mock_llm.generate_stream.return_value = iter(["Done."])
        orch._llm_service = mock_llm

        mock_tts = MagicMock()
        mock_tts.synthesize_chunk.return_value = _make_audio_result()
        orch._tts_engine = mock_tts

        mock_avatar = MagicMock()
        mock_avatar.animate_chunk.return_value = _make_video_result()
        orch._avatar_engine = mock_avatar

        events = _run(_collect_events(orch, "Hello?", profile.avatar_id))
        source_events = [e for e in events if e.type == "sources"]
        assert len(source_events) == 0
