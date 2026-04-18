"""Unit tests for the EvaluationModule."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from backend.app.evaluation import (
    EvaluationModule,
    _word_overlap,
    results_to_json,
)
from backend.app.llm_service import LLMService
from backend.app.models import EvalResult


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def llm() -> LLMService:
    """Return an LLMService (not called during heuristic tests)."""
    return LLMService()


@pytest.fixture()
def evaluator(llm: LLMService) -> EvaluationModule:
    return EvaluationModule(llm)


@pytest.fixture()
def sample_dataset(tmp_path):
    """Write a small dataset file and return its path."""
    data = [
        {
            "question": "What is RAG?",
            "answer": "RAG is retrieval augmented generation.",
            "context": [
                "Retrieval augmented generation combines retrieval with generation."
            ],
        },
        {
            "question": "What is chunking?",
            "answer": "Chunking splits text into smaller pieces.",
            "context": ["Chunking divides documents into smaller segments."],
        },
    ]
    path = tmp_path / "test_dataset.json"
    path.write_text(json.dumps(data))
    return str(path)


# ------------------------------------------------------------------
# _word_overlap helper
# ------------------------------------------------------------------


class TestWordOverlap:
    def test_full_overlap(self):
        assert _word_overlap("hello world", "hello world") == 1.0

    def test_partial_overlap(self):
        score = _word_overlap("hello world foo", "hello world bar")
        assert 0.0 < score < 1.0

    def test_no_overlap(self):
        assert _word_overlap("alpha beta", "gamma delta") == 0.0

    def test_empty_source(self):
        assert _word_overlap("", "some text") == 0.0

    def test_case_insensitive(self):
        assert _word_overlap("Hello", "hello world") == 1.0


# ------------------------------------------------------------------
# evaluate_single (heuristic fallback)
# ------------------------------------------------------------------


class TestEvaluateSingle:
    def test_returns_eval_result(self, evaluator: EvaluationModule):
        result = evaluator.evaluate_single(
            question="What is RAG?",
            answer="RAG is retrieval augmented generation.",
            context=["Retrieval augmented generation uses retrieval."],
        )
        assert isinstance(result, EvalResult)
        assert result.question == "What is RAG?"

    def test_scores_in_range(self, evaluator: EvaluationModule):
        result = evaluator.evaluate_single(
            question="What is RAG?",
            answer="RAG is retrieval augmented generation.",
            context=["Retrieval augmented generation uses retrieval."],
        )
        assert 0.0 <= result.faithfulness <= 1.0
        assert 0.0 <= result.context_relevance <= 1.0
        assert 0.0 <= result.answer_relevance <= 1.0

    def test_metadata_contains_evaluator(self, evaluator: EvaluationModule):
        result = evaluator.evaluate_single(
            question="test",
            answer="test answer",
            context=["test context"],
        )
        assert "evaluator" in result.metadata

    def test_empty_context(self, evaluator: EvaluationModule):
        result = evaluator.evaluate_single(
            question="What is RAG?",
            answer="RAG is great.",
            context=[],
        )
        assert result.faithfulness == 0.0
        assert result.context_relevance == 0.0


# ------------------------------------------------------------------
# evaluate_dataset
# ------------------------------------------------------------------


class TestEvaluateDataset:
    def test_loads_and_evaluates(
        self, evaluator: EvaluationModule, sample_dataset: str
    ):
        results = evaluator.evaluate_dataset(sample_dataset)
        assert len(results) == 2
        assert all(isinstance(r, EvalResult) for r in results)

    def test_missing_file_raises(self, evaluator: EvaluationModule):
        with pytest.raises(FileNotFoundError):
            evaluator.evaluate_dataset("/nonexistent/path.json")

    def test_invalid_json_raises(self, evaluator: EvaluationModule, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text('{"not": "an array"}')
        with pytest.raises(ValueError, match="JSON array"):
            evaluator.evaluate_dataset(str(bad))

    def test_skips_entries_without_question(
        self, evaluator: EvaluationModule, tmp_path
    ):
        data = [{"answer": "no question here", "context": []}]
        path = tmp_path / "partial.json"
        path.write_text(json.dumps(data))
        results = evaluator.evaluate_dataset(str(path))
        assert len(results) == 0


# ------------------------------------------------------------------
# results_to_json
# ------------------------------------------------------------------


class TestResultsToJson:
    def test_valid_json_output(self):
        results = [
            EvalResult(
                question="Q1",
                faithfulness=0.8,
                context_relevance=0.7,
                answer_relevance=0.9,
                metadata={"evaluator": "heuristic"},
            )
        ]
        output = results_to_json(results)
        parsed = json.loads(output)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert parsed[0]["question"] == "Q1"
        assert parsed[0]["faithfulness"] == 0.8

    def test_empty_results(self):
        output = results_to_json([])
        assert json.loads(output) == []
