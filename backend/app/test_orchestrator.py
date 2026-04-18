"""Tests for the Orchestrator class.

Covers:
  - _split_sentences helper
  - Orchestrator construction and lazy init
  - upload_avatar() — calls avatar_engine.preprocess(), stores profile
  - upload_pdf() — calls pdf_parser, chunking, embedding_store
  - process_question_stream() — yields correct StreamEvent sequence
  - Error handling — unknown avatar_id, unhandled exceptions
  - Structured logging — verify log calls happen

Requirements: 9.1, 9.2, 9.3
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio

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
        media_output_dir="data/media",
        chroma_persist_dir="/tmp/test_chroma",
    )


def _make_profile(avatar_id: str = "avatar123") -> AvatarProfile:
    return AvatarProfile(
        avatar_id=avatar_id,
        image_path="/tmp/avatar.png",
        landmarks={"face_rect": {"x": 0, "y": 0, "w": 100, "h": 100}},
        preprocessed_at="2024-01-01T00:00:00Z",
    )


def _make_retrieval_results() -> list[RetrievalResult]:
    chunk = DocumentChunk(
        chunk_id="c1",
        text="Some relevant context.",
        document_id="doc1",
        page_number=1,
        token_count=5,
        start_char=0,
        end_char=22,
    )
    return [RetrievalResult(chunk=chunk, score=0.9, distance=0.2)]


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


async def _collect_events(
    orch: Orchestrator, question: str, avatar_id: str
) -> list[StreamEvent]:
    events: list[StreamEvent] = []
    async for event in orch.process_question_stream(question, avatar_id):
        events.append(event)
    return events


def _inject_mocks(orch: Orchestrator, *, with_context: bool = True) -> dict[str, MagicMock]:
    """Inject mock sub-components into an Orchestrator and return them."""
    mock_store = MagicMock()
    mock_store.search.return_value = (
        _make_retrieval_results() if with_context else []
    )

    mock_llm = MagicMock()
    mock_llm.generate_stream.return_value = iter(["Hello", " world", "."])

    mock_tts = MagicMock()
    mock_tts.synthesize_chunk.return_value = _make_audio_result()

    mock_avatar = MagicMock()
    mock_avatar.preprocess.return_value = _make_profile()
    mock_avatar.animate_chunk.return_value = _make_video_result()

    mock_parser = MagicMock()
    mock_chunking = MagicMock()

    orch._embedding_store = mock_store
    orch._llm_service = mock_llm
    orch._tts_engine = mock_tts
    orch._avatar_engine = mock_avatar
    orch._pdf_parser = mock_parser
    orch._chunking = mock_chunking

    return {
        "embedding_store": mock_store,
        "llm_service": mock_llm,
        "tts_engine": mock_tts,
        "avatar_engine": mock_avatar,
        "pdf_parser": mock_parser,
        "chunking": mock_chunking,
    }


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
    @pytest.mark.asyncio
    async def test_upload_avatar_stores_profile(self):
        orch = Orchestrator(_make_config())
        profile = _make_profile()

        mock_engine = MagicMock()
        mock_engine.preprocess.return_value = profile
        orch._avatar_engine = mock_engine

        result = await orch.upload_avatar("/tmp/photo.png")

        assert result.avatar_id == "avatar123"
        assert orch._avatars["avatar123"] is profile
        mock_engine.preprocess.assert_called_once_with("/tmp/photo.png")

    @pytest.mark.asyncio
    async def test_upload_avatar_with_caller_id(self):
        orch = Orchestrator(_make_config())
        profile = _make_profile()

        mock_engine = MagicMock()
        mock_engine.preprocess.return_value = profile
        orch._avatar_engine = mock_engine

        result = await orch.upload_avatar("/tmp/photo.png", avatar_id="custom-id")

        assert result.avatar_id == "custom-id"
        assert "custom-id" in orch._avatars

    @pytest.mark.asyncio
    async def test_upload_avatar_calls_preprocess(self):
        """Verify avatar_engine.preprocess() is called with the file path."""
        orch = Orchestrator(_make_config())
        mocks = _inject_mocks(orch)

        await orch.upload_avatar("/some/image.png")

        mocks["avatar_engine"].preprocess.assert_called_once_with("/some/image.png")


# ------------------------------------------------------------------
# Tests: upload_pdf
# ------------------------------------------------------------------

class TestUploadPdf:
    @pytest.mark.asyncio
    async def test_upload_pdf_returns_summary(self):
        orch = Orchestrator(_make_config())

        parsed = ParsedDocument(
            text="Hello world",
            pages=[PageContent(page_number=1, text="Hello world", tables=[])],
            page_count=1,
            metadata={},
        )
        chunks = [
            DocumentChunk(
                chunk_id="c1",
                text="Hello world",
                document_id="d1",
                page_number=1,
                token_count=2,
                start_char=0,
                end_char=11,
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

        result = await orch.upload_pdf("/tmp/doc.pdf", document_id="my-doc-id")

        assert result["name"] == "doc.pdf"
        assert result["page_count"] == 1
        assert result["chunk_count"] == 1
        assert result["document_id"] == "my-doc-id"

    @pytest.mark.asyncio
    async def test_upload_pdf_calls_parser_chunking_store(self):
        """Verify the full parse → chunk → embed pipeline is called."""
        orch = Orchestrator(_make_config())

        parsed = ParsedDocument(
            text="Content",
            pages=[PageContent(page_number=1, text="Content", tables=[])],
            page_count=1,
            metadata={},
        )
        chunks = [
            DocumentChunk(
                chunk_id="c1",
                text="Content",
                document_id="d1",
                page_number=1,
                token_count=1,
                start_char=0,
                end_char=7,
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

        await orch.upload_pdf("/tmp/report.pdf", document_id="doc-abc")

        mock_parser.parse.assert_called_once_with("/tmp/report.pdf")
        mock_chunking.chunk.assert_called_once()
        mock_store.add_chunks.assert_called_once_with(chunks)

    @pytest.mark.asyncio
    async def test_upload_pdf_generates_document_id_when_none(self):
        """When no document_id is supplied, one is auto-generated."""
        orch = Orchestrator(_make_config())

        parsed = ParsedDocument(
            text="X", pages=[], page_count=0, metadata={}
        )
        mock_parser = MagicMock()
        mock_parser.parse.return_value = parsed
        mock_chunking = MagicMock()
        mock_chunking.chunk.return_value = []
        mock_store = MagicMock()

        orch._pdf_parser = mock_parser
        orch._chunking = mock_chunking
        orch._embedding_store = mock_store

        result = await orch.upload_pdf("/tmp/file.pdf")

        assert "document_id" in result
        assert len(result["document_id"]) > 0


# ------------------------------------------------------------------
# Tests: process_question_stream — full pipeline
# ------------------------------------------------------------------

class TestProcessQuestionStream:
    @pytest.mark.asyncio
    async def test_error_event_on_missing_avatar(self):
        orch = Orchestrator(_make_config())
        events = await _collect_events(orch, "What is AI?", "nonexistent")

        assert len(events) == 1
        assert events[0].type == "error"
        assert "not found" in events[0].data["message"]

    @pytest.mark.asyncio
    async def test_full_pipeline_yields_expected_event_types(self):
        """The stream should contain stage_update, text_token, audio_chunk,
        video_chunk, sources, and done events."""
        orch = Orchestrator(_make_config())
        profile = _make_profile()
        orch._avatars[profile.avatar_id] = profile
        _inject_mocks(orch, with_context=True)

        # Override LLM to produce a complete sentence
        orch._llm_service.generate_stream.return_value = iter(
            ["Hello", " world", ".", " Good", " bye", "!"]
        )

        events = await _collect_events(orch, "What is AI?", profile.avatar_id)
        event_types = [e.type for e in events]

        assert "text_token" in event_types
        assert "audio_chunk" in event_types
        assert "video_chunk" in event_types
        assert "stage_update" in event_types
        assert "sources" in event_types
        assert events[-1].type == "done"
        assert "total_duration_ms" in events[-1].data

    @pytest.mark.asyncio
    async def test_text_tokens_contain_all_content(self):
        orch = Orchestrator(_make_config())
        profile = _make_profile()
        orch._avatars[profile.avatar_id] = profile
        _inject_mocks(orch, with_context=False)

        orch._llm_service.generate_stream.return_value = iter(["Hi", " there", "."])

        events = await _collect_events(orch, "Hello?", profile.avatar_id)
        tokens = [e.data["token"] for e in events if e.type == "text_token"]
        assert "".join(tokens) == "Hi there."

    @pytest.mark.asyncio
    async def test_error_event_on_unhandled_exception(self):
        """Requirement 9.2: unhandled exceptions yield a user-friendly error event."""
        orch = Orchestrator(_make_config())
        profile = _make_profile()
        orch._avatars[profile.avatar_id] = profile

        mock_store = MagicMock()
        mock_store.search.side_effect = RuntimeError("DB crashed")
        orch._embedding_store = mock_store

        events = await _collect_events(orch, "Boom?", profile.avatar_id)
        error_events = [e for e in events if e.type == "error"]
        assert len(error_events) == 1
        assert "internal error" in error_events[0].data["message"].lower()

    @pytest.mark.asyncio
    async def test_no_sources_when_context_empty(self):
        orch = Orchestrator(_make_config())
        profile = _make_profile()
        orch._avatars[profile.avatar_id] = profile
        _inject_mocks(orch, with_context=False)

        orch._llm_service.generate_stream.return_value = iter(["Done."])

        events = await _collect_events(orch, "Hello?", profile.avatar_id)
        source_events = [e for e in events if e.type == "sources"]
        assert len(source_events) == 0

    @pytest.mark.asyncio
    async def test_sources_present_when_context_found(self):
        """When retrieval returns results, a sources event should be emitted."""
        orch = Orchestrator(_make_config())
        profile = _make_profile()
        orch._avatars[profile.avatar_id] = profile
        _inject_mocks(orch, with_context=True)

        orch._llm_service.generate_stream.return_value = iter(["Answer."])

        events = await _collect_events(orch, "Tell me?", profile.avatar_id)
        source_events = [e for e in events if e.type == "sources"]
        assert len(source_events) == 1
        assert "sources" in source_events[0].data
        assert len(source_events[0].data["sources"]) > 0

    @pytest.mark.asyncio
    async def test_done_event_is_last(self):
        """The final event must always be 'done'."""
        orch = Orchestrator(_make_config())
        profile = _make_profile()
        orch._avatars[profile.avatar_id] = profile
        _inject_mocks(orch, with_context=False)

        orch._llm_service.generate_stream.return_value = iter(["OK."])

        events = await _collect_events(orch, "Hi?", profile.avatar_id)
        assert events[-1].type == "done"

    @pytest.mark.asyncio
    async def test_stage_update_events_have_stage_and_status(self):
        """Stage update events should carry stage name and status."""
        orch = Orchestrator(_make_config())
        profile = _make_profile()
        orch._avatars[profile.avatar_id] = profile
        _inject_mocks(orch, with_context=False)

        orch._llm_service.generate_stream.return_value = iter(["OK."])

        events = await _collect_events(orch, "Hi?", profile.avatar_id)
        stage_events = [e for e in events if e.type == "stage_update"]
        assert len(stage_events) >= 2  # at least retrieving started/completed

        for se in stage_events:
            assert "stage" in se.data
            assert "status" in se.data

    @pytest.mark.asyncio
    async def test_tts_called_for_each_sentence(self):
        """TTS synthesize_chunk should be called once per complete sentence."""
        orch = Orchestrator(_make_config())
        profile = _make_profile()
        orch._avatars[profile.avatar_id] = profile
        mocks = _inject_mocks(orch, with_context=False)

        # Two complete sentences
        orch._llm_service.generate_stream.return_value = iter(
            ["First", " sentence", ".", " Second", " sentence", "!"]
        )

        await _collect_events(orch, "Go?", profile.avatar_id)

        # Should be called at least twice (one per sentence)
        assert mocks["tts_engine"].synthesize_chunk.call_count >= 2

    @pytest.mark.asyncio
    async def test_avatar_animate_called_for_each_sentence(self):
        """Avatar animate_chunk should be called once per audio chunk."""
        orch = Orchestrator(_make_config())
        profile = _make_profile()
        orch._avatars[profile.avatar_id] = profile
        mocks = _inject_mocks(orch, with_context=False)

        orch._llm_service.generate_stream.return_value = iter(
            ["One", ".", " Two", "."]
        )

        await _collect_events(orch, "Go?", profile.avatar_id)

        assert mocks["avatar_engine"].animate_chunk.call_count >= 2

    @pytest.mark.asyncio
    async def test_error_during_tts_yields_error_event(self):
        """If TTS raises, the pipeline should catch it and yield an error event."""
        orch = Orchestrator(_make_config())
        profile = _make_profile()
        orch._avatars[profile.avatar_id] = profile
        mocks = _inject_mocks(orch, with_context=False)

        orch._llm_service.generate_stream.return_value = iter(["Crash."])
        mocks["tts_engine"].synthesize_chunk.side_effect = RuntimeError("TTS failed")

        events = await _collect_events(orch, "Go?", profile.avatar_id)
        error_events = [e for e in events if e.type == "error"]
        assert len(error_events) == 1

    @pytest.mark.asyncio
    async def test_error_during_avatar_yields_error_event(self):
        """If avatar engine raises, the pipeline should catch it and yield an error event."""
        orch = Orchestrator(_make_config())
        profile = _make_profile()
        orch._avatars[profile.avatar_id] = profile
        mocks = _inject_mocks(orch, with_context=False)

        orch._llm_service.generate_stream.return_value = iter(["Crash."])
        mocks["avatar_engine"].animate_chunk.side_effect = RuntimeError("Avatar failed")

        events = await _collect_events(orch, "Go?", profile.avatar_id)
        error_events = [e for e in events if e.type == "error"]
        assert len(error_events) == 1


# ------------------------------------------------------------------
# Tests: Structured logging (Requirement 9.1)
# ------------------------------------------------------------------

class TestStructuredLogging:
    @pytest.mark.asyncio
    async def test_upload_avatar_logs_start_and_completion(self):
        """Requirement 9.1: upload_avatar logs start and completion with durations."""
        orch = Orchestrator(_make_config())
        mocks = _inject_mocks(orch)

        with patch("backend.app.orchestrator.logger") as mock_logger:
            await orch.upload_avatar("/tmp/photo.png")

            # Should have at least 2 info calls: started + completed
            info_calls = mock_logger.info.call_args_list
            messages = [call.args[0] for call in info_calls]
            assert any("upload_avatar started" in m for m in messages)
            assert any("upload_avatar completed" in m for m in messages)

    @pytest.mark.asyncio
    async def test_upload_pdf_logs_start_and_completion(self):
        """Requirement 9.1: upload_pdf logs start and completion."""
        orch = Orchestrator(_make_config())

        parsed = ParsedDocument(
            text="X",
            pages=[PageContent(page_number=1, text="X", tables=[])],
            page_count=1,
            metadata={},
        )
        chunks = [
            DocumentChunk(
                chunk_id="c1", text="X", document_id="d1",
                page_number=1, token_count=1, start_char=0, end_char=1,
            )
        ]

        orch._pdf_parser = MagicMock()
        orch._pdf_parser.parse.return_value = parsed
        orch._chunking = MagicMock()
        orch._chunking.chunk.return_value = chunks
        orch._embedding_store = MagicMock()

        with patch("backend.app.orchestrator.logger") as mock_logger:
            await orch.upload_pdf("/tmp/doc.pdf", document_id="d1")

            info_calls = mock_logger.info.call_args_list
            messages = [call.args[0] for call in info_calls]
            assert any("upload_pdf started" in m for m in messages)
            assert any("upload_pdf completed" in m for m in messages)

    @pytest.mark.asyncio
    async def test_process_question_stream_logs_pipeline_start(self):
        """Requirement 9.1: process_question_stream logs pipeline start."""
        orch = Orchestrator(_make_config())
        profile = _make_profile()
        orch._avatars[profile.avatar_id] = profile
        _inject_mocks(orch, with_context=False)
        orch._llm_service.generate_stream.return_value = iter(["OK."])

        with patch("backend.app.orchestrator.logger") as mock_logger:
            await _collect_events(orch, "Hi?", profile.avatar_id)

            info_calls = mock_logger.info.call_args_list
            messages = [call.args[0] for call in info_calls]
            assert any("process_question_stream started" in m for m in messages)

    @pytest.mark.asyncio
    async def test_unhandled_exception_logs_error(self):
        """Requirement 9.2: unhandled exceptions are logged with stack traces."""
        orch = Orchestrator(_make_config())
        profile = _make_profile()
        orch._avatars[profile.avatar_id] = profile

        mock_store = MagicMock()
        mock_store.search.side_effect = RuntimeError("Kaboom")
        orch._embedding_store = mock_store

        with patch("backend.app.orchestrator.logger") as mock_logger:
            await _collect_events(orch, "Fail?", profile.avatar_id)

            error_calls = mock_logger.error.call_args_list
            assert len(error_calls) >= 1
            # The error log should contain the traceback
            error_msg = error_calls[0].args[0]
            assert "exception" in error_msg.lower() or "Unhandled" in error_msg

    @pytest.mark.asyncio
    async def test_pipeline_completion_logs_total_duration(self):
        """Requirement 9.1: pipeline completion is logged with total duration."""
        orch = Orchestrator(_make_config())
        profile = _make_profile()
        orch._avatars[profile.avatar_id] = profile
        _inject_mocks(orch, with_context=False)
        orch._llm_service.generate_stream.return_value = iter(["Done."])

        with patch("backend.app.orchestrator.logger") as mock_logger:
            await _collect_events(orch, "Hi?", profile.avatar_id)

            info_calls = mock_logger.info.call_args_list
            messages = [call.args[0] for call in info_calls]
            assert any("process_question_stream completed" in m for m in messages)
