"""Tests for LLMService — Ollama HTTP interactions are mocked."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from backend.app.config import AppConfig
from backend.app.llm_service import LLMService, _build_prompt
from backend.app.models import DocumentChunk, RetrievalResult


# ── Fixtures ──────────────────────────────────────────────────────────


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


def test_build_prompt_with_context():
    ctx = [_make_retrieval("chunk A"), _make_retrieval("chunk B")]
    prompt = _build_prompt("What is X?", ctx)
    assert "Context:" in prompt
    assert "[1] chunk A" in prompt
    assert "[2] chunk B" in prompt
    assert "Question: What is X?" in prompt


def test_build_prompt_without_context():
    prompt = _build_prompt("Hello?", [])
    assert "Context:" not in prompt
    assert "Question: Hello?" in prompt


# ── generate() ───────────────────────────────────────────────────────


def test_generate_success(service: LLMService):
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


def test_generate_connect_error(service: LLMService):
    with patch("backend.app.llm_service.httpx.Client") as MockClient:
        MockClient.return_value.__enter__ = lambda s: s
        MockClient.return_value.__exit__ = MagicMock(return_value=False)
        MockClient.return_value.post = MagicMock(
            side_effect=httpx.ConnectError("refused")
        )

        result = service.generate("q", [])

    assert "unable to generate" in result.answer.lower()
    assert result.prompt_tokens == 0


def test_generate_timeout(service: LLMService):
    with patch("backend.app.llm_service.httpx.Client") as MockClient:
        MockClient.return_value.__enter__ = lambda s: s
        MockClient.return_value.__exit__ = MagicMock(return_value=False)
        MockClient.return_value.post = MagicMock(
            side_effect=httpx.TimeoutException("timeout")
        )

        result = service.generate("q", [])

    assert "unable to generate" in result.answer.lower()


# ── generate_stream() ────────────────────────────────────────────────


def test_generate_stream_yields_tokens(service: LLMService):
    lines = [
        json.dumps({"response": "Hello", "done": False}),
        json.dumps({"response": " world", "done": False}),
        json.dumps({"response": "", "done": True}),
    ]

    mock_stream_resp = MagicMock()
    mock_stream_resp.raise_for_status = MagicMock()
    mock_stream_resp.iter_lines = MagicMock(return_value=iter(lines))
    mock_stream_resp.__enter__ = lambda s: s
    mock_stream_resp.__exit__ = MagicMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=mock_stream_resp)
    mock_client.__enter__ = lambda s: s
    mock_client.__exit__ = MagicMock(return_value=False)

    with patch("backend.app.llm_service.httpx.Client", return_value=mock_client):
        tokens = list(service.generate_stream("hi", []))

    assert tokens == ["Hello", " world"]


def test_generate_stream_connect_error(service: LLMService):
    mock_client = MagicMock()
    mock_client.stream = MagicMock(side_effect=httpx.ConnectError("refused"))
    mock_client.__enter__ = lambda s: s
    mock_client.__exit__ = MagicMock(return_value=False)

    with patch("backend.app.llm_service.httpx.Client", return_value=mock_client):
        tokens = list(service.generate_stream("hi", []))

    assert tokens == []
