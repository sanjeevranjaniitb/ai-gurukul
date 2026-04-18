"""Unit tests for RAGPipeline.

Validates: Requirements 3.2, 3.3, 3.4, 3.5, 9.3
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.app.config import AppConfig
from backend.app.models import DocumentChunk, GenerationResult, RetrievalResult
from backend.app.rag_pipeline import RAGPipeline, _NO_CONTEXT_MESSAGE


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_result(score: float, text: str = "chunk", chunk_id: str = "c1") -> RetrievalResult:
    chunk = DocumentChunk(
        chunk_id=chunk_id,
        text=text,
        document_id="d1",
        page_number=1,
        token_count=10,
        start_char=0,
        end_char=len(text),
    )
    return RetrievalResult(chunk=chunk, score=score, distance=1.0 - score)


def _mock_services(
    search_results: list[RetrievalResult] | None = None,
    generate_result: GenerationResult | None = None,
    stream_tokens: list[str] | None = None,
):
    store = MagicMock()
    store.search.return_value = search_results or []

    llm = MagicMock()
    llm.model_name = "test-model"
    llm.generate.return_value = generate_result or GenerationResult(
        answer="test answer",
        model="test-model",
        prompt_tokens=10,
        completion_tokens=5,
        duration_ms=100.0,
    )
    llm.generate_stream.return_value = iter(stream_tokens or ["hello", " world"])
    return store, llm


# ------------------------------------------------------------------
# Tests – query()
# ------------------------------------------------------------------


class TestQuery:
    """Tests for RAGPipeline.query() — full answer generation."""

    def test_returns_llm_answer_when_relevant_context(self):
        """Req 3.2, 3.3: retrieves context and passes to LLM for generation."""
        results = [_make_result(0.8, "relevant chunk")]
        store, llm = _mock_services(search_results=results)
        pipeline = RAGPipeline(store, llm)

        result = pipeline.query("What is X?")

        store.search.assert_called_once_with("What is X?", top_k=5)
        llm.generate.assert_called_once_with("What is X?", results)
        assert result.answer == "test answer"

    def test_returns_no_context_message_when_all_below_threshold(self):
        """Req 3.4: no relevant context triggers fallback message."""
        results = [_make_result(0.1), _make_result(0.2)]
        store, llm = _mock_services(search_results=results)
        pipeline = RAGPipeline(store, llm, relevance_threshold=0.3)

        result = pipeline.query("Unknown topic?")

        llm.generate.assert_not_called()
        assert result.answer == _NO_CONTEXT_MESSAGE

    def test_returns_no_context_message_when_store_empty(self):
        """Req 3.4: empty store triggers fallback message."""
        store, llm = _mock_services(search_results=[])
        pipeline = RAGPipeline(store, llm)

        result = pipeline.query("Anything?")

        llm.generate.assert_not_called()
        assert result.answer == _NO_CONTEXT_MESSAGE

    def test_filters_out_low_score_chunks(self):
        """Req 3.2: only relevant chunks above threshold are passed to LLM."""
        high = _make_result(0.9, "good", chunk_id="c-high")
        low = _make_result(0.1, "bad", chunk_id="c-low")
        store, llm = _mock_services(search_results=[high, low])
        pipeline = RAGPipeline(store, llm, relevance_threshold=0.3)

        pipeline.query("Q?")

        # Only the high-score chunk should be passed to the LLM
        llm.generate.assert_called_once_with("Q?", [high])

    def test_uses_config_top_k(self):
        """Req 3.2: retrieval uses configured top_k value."""
        cfg = AppConfig()
        cfg.retrieval_top_k = 3
        store, llm = _mock_services(search_results=[_make_result(0.5)])
        pipeline = RAGPipeline(store, llm, config=cfg)

        pipeline.query("Q?")

        store.search.assert_called_once_with("Q?", top_k=3)

    def test_no_context_fallback_includes_model_name(self):
        """Req 3.4: fallback GenerationResult carries the model name."""
        store, llm = _mock_services(search_results=[])
        pipeline = RAGPipeline(store, llm)

        result = pipeline.query("Q?")

        assert result.model == "test-model"
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.duration_ms == 0.0

    def test_default_relevance_threshold(self):
        """Default threshold is 0.3 — chunks at exactly 0.3 should pass."""
        at_threshold = _make_result(0.3, "borderline")
        store, llm = _mock_services(search_results=[at_threshold])
        pipeline = RAGPipeline(store, llm)  # default threshold=0.3

        pipeline.query("Q?")

        llm.generate.assert_called_once_with("Q?", [at_threshold])

    def test_chunks_just_below_threshold_are_excluded(self):
        """Chunks scoring below the threshold should not reach the LLM."""
        below = _make_result(0.29, "almost")
        store, llm = _mock_services(search_results=[below])
        pipeline = RAGPipeline(store, llm)  # default threshold=0.3

        result = pipeline.query("Q?")

        llm.generate.assert_not_called()
        assert result.answer == _NO_CONTEXT_MESSAGE

    def test_llm_error_propagates_through_pipeline(self):
        """Req 3.5: LLM errors should propagate as error GenerationResult."""
        results = [_make_result(0.8)]
        error_result = GenerationResult(
            answer="I'm sorry, I was unable to generate a response.",
            model="test-model",
            prompt_tokens=0,
            completion_tokens=0,
            duration_ms=50.0,
        )
        store, llm = _mock_services(
            search_results=results, generate_result=error_result
        )
        pipeline = RAGPipeline(store, llm)

        result = pipeline.query("Q?")

        assert "unable to generate" in result.answer.lower()


# ------------------------------------------------------------------
# Tests – query_stream()
# ------------------------------------------------------------------


class TestQueryStream:
    """Tests for RAGPipeline.query_stream() — streaming token generation."""

    def test_streams_tokens_when_relevant_context(self):
        """Req 3.3: streams LLM tokens when relevant context exists."""
        results = [_make_result(0.8)]
        store, llm = _mock_services(search_results=results, stream_tokens=["a", "b"])
        pipeline = RAGPipeline(store, llm)

        tokens = list(pipeline.query_stream("Q?"))

        assert tokens == ["a", "b"]
        llm.generate_stream.assert_called_once_with("Q?", results)

    def test_yields_no_context_message_when_irrelevant(self):
        """Req 3.4: no-context fallback in streaming mode."""
        store, llm = _mock_services(search_results=[_make_result(0.1)])
        pipeline = RAGPipeline(store, llm, relevance_threshold=0.3)

        tokens = list(pipeline.query_stream("Q?"))

        assert tokens == [_NO_CONTEXT_MESSAGE]
        llm.generate_stream.assert_not_called()

    def test_yields_no_context_message_when_empty(self):
        """Req 3.4: empty store fallback in streaming mode."""
        store, llm = _mock_services(search_results=[])
        pipeline = RAGPipeline(store, llm)

        tokens = list(pipeline.query_stream("Q?"))

        assert tokens == [_NO_CONTEXT_MESSAGE]
        llm.generate_stream.assert_not_called()

    def test_filters_low_score_chunks_before_streaming(self):
        """Req 3.2: only relevant chunks are passed to generate_stream."""
        high = _make_result(0.9, "good", chunk_id="c-high")
        low = _make_result(0.1, "bad", chunk_id="c-low")
        store, llm = _mock_services(
            search_results=[high, low], stream_tokens=["tok"]
        )
        pipeline = RAGPipeline(store, llm, relevance_threshold=0.3)

        tokens = list(pipeline.query_stream("Q?"))

        assert tokens == ["tok"]
        llm.generate_stream.assert_called_once_with("Q?", [high])

    def test_streams_multiple_tokens(self):
        """Verify streaming yields all tokens in order."""
        results = [_make_result(0.8)]
        expected = ["The ", "answer ", "is ", "42."]
        store, llm = _mock_services(
            search_results=results, stream_tokens=expected
        )
        pipeline = RAGPipeline(store, llm)

        tokens = list(pipeline.query_stream("Q?"))

        assert tokens == expected
