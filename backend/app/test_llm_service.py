"""Tests for LLMService — Ollama HTTP interactions are mocked.

Validates: Requirements 3.2, 3.3, 3.5, 9.3
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from backend.app.config import AppConfig
from backend.app.llm_service import LLMService, _build_prompt, _SYSTEM_PROMPT
from backend.app.models import DocumentChunk, RetrievalResult


# ── Helpers ───────────────────────────────────────────────────────────


def _make_retrieval(text: str, score: float = 0.9) -> RetrievalResult:
    chunk = DocumentChunk(
        chunk_id="c1",
        text=text,
        document_id="d1",
        page_number=1,
        token_count=10,
        start_char=0,
        end_char=len(text),
    )
    return RetrievalResult(chunk=chunk, score=score, distance=1 - score)


@pytest.fixture
def service() -> LLMService:
    cfg = AppConfig(llm_base_url="http://test:11434")
    return LLMService(config=cfg)


# ── Prompt building ──────────────────────────────────────────────────


class TestBuildPrompt:
    """Tests for _build_prompt() — prompt construction logic."""

    def test_includes_system_prompt(self):
        prompt = _build_prompt("Q?", [])
        assert _SYSTEM_PROMPT in prompt

    def test_with_context_includes_numbered_chunks(self):
        ctx = [_make_retrieval("chunk A"), _make_retrieval("chunk B")]
        prompt = _build_prompt("What is X?", ctx)
        assert "Context:" in prompt
        assert "[1] chunk A" in prompt
        assert "[2] chunk B" in prompt
        assert "Question: What is X?" in prompt
        assert "Answer:" in prompt

    def test_without_context_omits_context_section(self):
        prompt = _build_prompt("Hello?", [])
        assert "Context:" not in prompt
        assert "Question: Hello?" in prompt
        assert "Answer:" in prompt

    def test_preserves_question_text_exactly(self):
        prompt = _build_prompt("What is the capital of France?", [])
        assert "Question: What is the capital of France?" in prompt

    def test_multiple_chunks_numbered_sequentially(self):
        ctx = [
            _make_retrieval("first"),
            _make_retrieval("second"),
            _make_retrieval("third"),
        ]
        prompt = _build_prompt("Q?", ctx)
        assert "[1] first" in prompt
        assert "[2] second" in prompt
        assert "[3] third" in prompt


# ── generate() ───────────────────────────────────────────────────────


class TestGenerate:
    """Tests for LLMService.generate() — non-streaming generation."""

    def test_success_returns_generation_result(self, service: LLMService):
        body = {
            "response": "The answer is 42.",
            "model": "llama3.2:3b",
            "prompt_eval_count": 50,
            "eval_count": 12,
        }
        mock_resp = httpx.Response(
            200,
            json=body,
            request=httpx.Request("POST", "http://test:11434/api/generate"),
        )

        with patch("backend.app.llm_service.httpx.Client") as MockClient:
            MockClient.return_value.__enter__ = lambda s: s
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            MockClient.return_value.post = MagicMock(return_value=mock_resp)

            result = service.generate("question", [])

        assert result.answer == "The answer is 42."
        assert result.model == "llama3.2:3b"
        assert result.prompt_tokens == 50
        assert result.completion_tokens == 12
        assert result.duration_ms > 0

    def test_connect_error_returns_friendly_message(self, service: LLMService):
        """Req 3.5: graceful error handling on connection failure."""
        with patch("backend.app.llm_service.httpx.Client") as MockClient:
            MockClient.return_value.__enter__ = lambda s: s
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            MockClient.return_value.post = MagicMock(
                side_effect=httpx.ConnectError("refused")
            )

            result = service.generate("q", [])

        assert "unable to generate" in result.answer.lower()
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0

    def test_timeout_returns_friendly_message(self, service: LLMService):
        """Req 3.5: graceful error handling on timeout."""
        with patch("backend.app.llm_service.httpx.Client") as MockClient:
            MockClient.return_value.__enter__ = lambda s: s
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            MockClient.return_value.post = MagicMock(
                side_effect=httpx.TimeoutException("timeout")
            )

            result = service.generate("q", [])

        assert "unable to generate" in result.answer.lower()
        assert result.duration_ms > 0

    def test_http_status_error_returns_friendly_message(self, service: LLMService):
        """Req 3.5: graceful error handling on HTTP error status."""
        error_resp = httpx.Response(
            500,
            request=httpx.Request("POST", "http://test:11434/api/generate"),
        )

        with patch("backend.app.llm_service.httpx.Client") as MockClient:
            MockClient.return_value.__enter__ = lambda s: s
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            MockClient.return_value.post = MagicMock(
                side_effect=httpx.HTTPStatusError(
                    "Server Error", request=error_resp.request, response=error_resp
                )
            )

            result = service.generate("q", [])

        assert "unable to generate" in result.answer.lower()
        assert result.prompt_tokens == 0

    def test_passes_context_to_prompt(self, service: LLMService):
        """Req 3.2: context chunks are included in the LLM prompt."""
        ctx = [_make_retrieval("important context")]
        body = {
            "response": "answer",
            "model": "llama3.2:3b",
            "prompt_eval_count": 10,
            "eval_count": 5,
        }
        mock_resp = httpx.Response(
            200,
            json=body,
            request=httpx.Request("POST", "http://test:11434/api/generate"),
        )

        with patch("backend.app.llm_service.httpx.Client") as MockClient:
            MockClient.return_value.__enter__ = lambda s: s
            MockClient.return_value.__exit__ = MagicMock(return_value=False)
            mock_post = MagicMock(return_value=mock_resp)
            MockClient.return_value.post = mock_post

            service.generate("What is it?", ctx)

        # Verify the prompt sent to Ollama contains the context
        call_args = mock_post.call_args
        payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
        assert "important context" in payload["prompt"]

    def test_uses_configured_model_and_temperature(self):
        """Req 3.2: LLM uses configured model settings."""
        cfg = AppConfig(
            llm_base_url="http://test:11434",
            llm_model="custom-model:7b",
            llm_temperature=0.5,
            llm_max_tokens=256,
        )
        svc = LLMService(config=cfg)
        assert svc.model_name == "custom-model:7b"
        assert svc.temperature == 0.5
        assert svc.max_tokens == 256


# ── generate_stream() ────────────────────────────────────────────────


class TestGenerateStream:
    """Tests for LLMService.generate_stream() — streaming token generation."""

    def _make_streaming_client(self, lines: list[str]):
        """Build a mock httpx.Client that streams the given JSON lines."""
        mock_stream_resp = MagicMock()
        mock_stream_resp.raise_for_status = MagicMock()
        mock_stream_resp.iter_lines = MagicMock(return_value=iter(lines))
        mock_stream_resp.__enter__ = lambda s: s
        mock_stream_resp.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_stream_resp)
        mock_client.__enter__ = lambda s: s
        mock_client.__exit__ = MagicMock(return_value=False)
        return mock_client

    def test_yields_tokens_until_done(self, service: LLMService):
        lines = [
            json.dumps({"response": "Hello", "done": False}),
            json.dumps({"response": " world", "done": False}),
            json.dumps({"response": "", "done": True}),
        ]
        mock_client = self._make_streaming_client(lines)

        with patch("backend.app.llm_service.httpx.Client", return_value=mock_client):
            tokens = list(service.generate_stream("hi", []))

        assert tokens == ["Hello", " world"]

    def test_connect_error_yields_nothing(self, service: LLMService):
        """Req 3.5: connection errors are handled gracefully in streaming."""
        mock_client = MagicMock()
        mock_client.stream = MagicMock(side_effect=httpx.ConnectError("refused"))
        mock_client.__enter__ = lambda s: s
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("backend.app.llm_service.httpx.Client", return_value=mock_client):
            tokens = list(service.generate_stream("hi", []))

        assert tokens == []

    def test_timeout_error_yields_nothing(self, service: LLMService):
        """Req 3.5: timeout errors are handled gracefully in streaming."""
        mock_client = MagicMock()
        mock_client.stream = MagicMock(side_effect=httpx.TimeoutException("timeout"))
        mock_client.__enter__ = lambda s: s
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("backend.app.llm_service.httpx.Client", return_value=mock_client):
            tokens = list(service.generate_stream("hi", []))

        assert tokens == []

    def test_http_status_error_yields_nothing(self, service: LLMService):
        """Req 3.5: HTTP errors are handled gracefully in streaming."""
        error_resp = httpx.Response(
            503,
            request=httpx.Request("POST", "http://test:11434/api/generate"),
        )
        mock_client = MagicMock()
        mock_client.stream = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Service Unavailable",
                request=error_resp.request,
                response=error_resp,
            )
        )
        mock_client.__enter__ = lambda s: s
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("backend.app.llm_service.httpx.Client", return_value=mock_client):
            tokens = list(service.generate_stream("hi", []))

        assert tokens == []

    def test_skips_empty_lines(self, service: LLMService):
        """Empty lines in the stream should be silently skipped."""
        lines = [
            "",
            json.dumps({"response": "token", "done": False}),
            "",
            json.dumps({"response": "", "done": True}),
        ]
        mock_client = self._make_streaming_client(lines)

        with patch("backend.app.llm_service.httpx.Client", return_value=mock_client):
            tokens = list(service.generate_stream("hi", []))

        assert tokens == ["token"]

    def test_skips_malformed_json_lines(self, service: LLMService):
        """Malformed JSON lines should be skipped without crashing."""
        lines = [
            "not valid json",
            json.dumps({"response": "ok", "done": False}),
            json.dumps({"response": "", "done": True}),
        ]
        mock_client = self._make_streaming_client(lines)

        with patch("backend.app.llm_service.httpx.Client", return_value=mock_client):
            tokens = list(service.generate_stream("hi", []))

        assert tokens == ["ok"]

    def test_streams_with_context(self, service: LLMService):
        """Req 3.3: streaming generation uses retrieved context."""
        ctx = [_make_retrieval("context chunk")]
        lines = [
            json.dumps({"response": "answer", "done": False}),
            json.dumps({"response": "", "done": True}),
        ]
        mock_client = self._make_streaming_client(lines)

        with patch("backend.app.llm_service.httpx.Client", return_value=mock_client):
            tokens = list(service.generate_stream("Q?", ctx))

        assert tokens == ["answer"]
        # Verify the prompt included context
        call_args = mock_client.stream.call_args
        payload = call_args[1].get("json") or call_args[0][2]
        assert "context chunk" in payload["prompt"]
