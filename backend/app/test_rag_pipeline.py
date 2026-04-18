"""Unit tests for RAGPipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from backend.app.config import AppConfig
from backend.app.models import DocumentChunk, GenerationResult, RetrievalResult
from backend.app.rag_pipeline import RAGPipeline, _NO_CONTEXT_MESSAGE


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_result(score: float, text: str = "chunk") -> RetrievalResult:
    chunk = DocumentChunk(
        chunk_id="c1",
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
    def test_returns_llm_answer_when_relevant_context(self):
        results = [_make_result(0.8, "relevant chunk")]
        store, llm = _mock_services(search_results=results)
        pipeline = RAGPipeline(store, llm)

        result = pipeline.query("What is X?")

        store.search.assert_called_once_with("What is X?", top_k=5)
        llm.generate.assert_called_once_with("What is X?", results)
        assert result.answer == "test answer"

    def test_returns_no_context_message_when_all_below_threshold(self):
        results = [_make_result(0.1), _make_result(0.2)]
        store, llm = _mock_services(search_results=results)
        pipeline = RAGPipeline(store, llm, relevance_threshold=0.3)

        result = pipeline.query("Unknown topic?")

        llm.generate.assert_not_called()
        assert result.answer == _NO_CONTEXT_MESSAGE

    def test_returns_no_context_message_when_store_empty(self):
        store, llm = _mock_services(search_results=[])
        pipeline = RAGPipeline(store, llm)

        result = pipeline.query("Anything?")

        llm.generate.assert_not_called()
        assert result.answer == _NO_CONTEXT_MESSAGE

    def test_filters_out_low_score_chunks(self):
        high = _make_result(0.9, "good")
        low = _make_result(0.1, "bad")
        store, llm = _mock_services(search_results=[high, low])
        pipeline = RAGPipeline(store, llm, relevance_threshold=0.3)

        pipeline.query("Q?")

        # Only the high-score chunk should be passed to the LLM
        llm.generate.assert_called_once_with("Q?", [high])

    def test_uses_config_top_k(self):
        cfg = AppConfig()
        cfg.retrieval_top_k = 3
        store, llm = _mock_services(search_results=[_make_result(0.5)])
        pipeline = RAGPipeline(store, llm, config=cfg)

        pipeline.query("Q?")

        store.search.assert_called_once_with("Q?", top_k=3)


# ------------------------------------------------------------------
# Tests – query_stream()
# ------------------------------------------------------------------

class TestQueryStream:
    def test_streams_tokens_when_relevant_context(self):
        results = [_make_result(0.8)]
        store, llm = _mock_services(search_results=results, stream_tokens=["a", "b"])
        pipeline = RAGPipeline(store, llm)

        tokens = list(pipeline.query_stream("Q?"))

        assert tokens == ["a", "b"]
        llm.generate_stream.assert_called_once_with("Q?", results)

    def test_yields_no_context_message_when_irrelevant(self):
        store, llm = _mock_services(search_results=[_make_result(0.1)])
        pipeline = RAGPipeline(store, llm, relevance_threshold=0.3)

        tokens = list(pipeline.query_stream("Q?"))

        assert tokens == [_NO_CONTEXT_MESSAGE]
        llm.generate_stream.assert_not_called()

    def test_yields_no_context_message_when_empty(self):
        store, llm = _mock_services(search_results=[])
        pipeline = RAGPipeline(store, llm)

        tokens = list(pipeline.query_stream("Q?"))

        assert tokens == [_NO_CONTEXT_MESSAGE]
        llm.generate_stream.assert_not_called()
