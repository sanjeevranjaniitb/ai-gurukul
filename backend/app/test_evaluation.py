"""Unit tests for the EvaluationModule.

Validates: Requirements 8.1, 8.2, 8.3, 8.4, 9.3
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from backend.app.evaluation import (
    EvaluationModule,
    _word_overlap,
    results_to_json,
)
from backend.app.models import EvalResult


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def mock_llm() -> MagicMock:
    """Return a mocked LLMService (no real Ollama calls)."""
    return MagicMock()


@pytest.fixture()
def evaluator(mock_llm: MagicMock) -> EvaluationModule:
    return EvaluationModule(mock_llm)


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
# evaluate_single (heuristic fallback — RAGAS not available)
# ------------------------------------------------------------------


class TestEvaluateSingleHeuristic:
    """Test evaluate_single when RAGAS is NOT available (heuristic path)."""

    def test_returns_eval_result(self, evaluator: EvaluationModule):
        result = evaluator.evaluate_single(
            question="What is RAG?",
            answer="RAG is retrieval augmented generation.",
            context=["Retrieval augmented generation uses retrieval."],
        )
        assert isinstance(result, EvalResult)
        assert result.question == "What is RAG?"

    def test_scores_in_range(self, evaluator: EvaluationModule):
        """Validates: Requirement 8.4 — scores are floats in [0, 1]."""
        result = evaluator.evaluate_single(
            question="What is RAG?",
            answer="RAG is retrieval augmented generation.",
            context=["Retrieval augmented generation uses retrieval."],
        )
        assert 0.0 <= result.faithfulness <= 1.0
        assert 0.0 <= result.context_relevance <= 1.0
        assert 0.0 <= result.answer_relevance <= 1.0

    def test_all_three_metrics_present(self, evaluator: EvaluationModule):
        """Validates: Requirements 8.1, 8.2, 8.3 — all three metrics computed."""
        result = evaluator.evaluate_single(
            question="What is chunking?",
            answer="Chunking splits text into smaller pieces.",
            context=["Chunking divides documents into smaller segments."],
        )
        assert hasattr(result, "faithfulness")
        assert hasattr(result, "context_relevance")
        assert hasattr(result, "answer_relevance")
        assert isinstance(result.faithfulness, float)
        assert isinstance(result.context_relevance, float)
        assert isinstance(result.answer_relevance, float)

    def test_metadata_contains_evaluator(self, evaluator: EvaluationModule):
        result = evaluator.evaluate_single(
            question="test",
            answer="test answer",
            context=["test context"],
        )
        assert "evaluator" in result.metadata
        assert result.metadata["evaluator"] == "heuristic"

    def test_empty_context(self, evaluator: EvaluationModule):
        result = evaluator.evaluate_single(
            question="What is RAG?",
            answer="RAG is great.",
            context=[],
        )
        assert result.faithfulness == 0.0
        assert result.context_relevance == 0.0


# ------------------------------------------------------------------
# evaluate_single (mocked RAGAS path)
# ------------------------------------------------------------------


class TestEvaluateSingleRagas:
    """Test evaluate_single when RAGAS IS available (mocked)."""

    @patch("backend.app.evaluation._RAGAS_AVAILABLE", True)
    @patch("backend.app.evaluation.ragas_evaluate")
    @patch("backend.app.evaluation.Dataset")
    def test_ragas_path_returns_eval_result(
        self, mock_dataset_cls, mock_ragas_eval, evaluator: EvaluationModule
    ):
        """Validates: Requirements 8.1, 8.2, 8.3 — RAGAS computes all three metrics."""
        # Set up mock RAGAS result as a pandas DataFrame
        mock_df = MagicMock()
        mock_df.iloc.__getitem__ = MagicMock(
            return_value={
                "faithfulness": 0.85,
                "context_precision": 0.72,
                "answer_relevancy": 0.91,
            }
        )
        mock_result = MagicMock()
        mock_result.to_pandas.return_value = mock_df
        mock_ragas_eval.return_value = mock_result

        mock_dataset_cls.from_dict.return_value = MagicMock()

        result = evaluator.evaluate_single(
            question="What is RAG?",
            answer="RAG is retrieval augmented generation.",
            context=["Retrieval augmented generation combines retrieval with generation."],
        )

        assert isinstance(result, EvalResult)
        assert result.question == "What is RAG?"
        assert result.faithfulness == 0.85
        assert result.context_relevance == 0.72
        assert result.answer_relevance == 0.91
        assert result.metadata["evaluator"] == "ragas"

    @patch("backend.app.evaluation._RAGAS_AVAILABLE", True)
    @patch("backend.app.evaluation.ragas_evaluate")
    @patch("backend.app.evaluation.Dataset")
    def test_ragas_scores_in_range(
        self, mock_dataset_cls, mock_ragas_eval, evaluator: EvaluationModule
    ):
        """Validates: Requirement 8.4 — RAGAS scores are floats in [0, 1]."""
        mock_df = MagicMock()
        mock_df.iloc.__getitem__ = MagicMock(
            return_value={
                "faithfulness": 0.5,
                "context_precision": 0.6,
                "answer_relevancy": 0.7,
            }
        )
        mock_result = MagicMock()
        mock_result.to_pandas.return_value = mock_df
        mock_ragas_eval.return_value = mock_result
        mock_dataset_cls.from_dict.return_value = MagicMock()

        result = evaluator.evaluate_single(
            question="Q", answer="A", context=["C"]
        )

        assert 0.0 <= result.faithfulness <= 1.0
        assert 0.0 <= result.context_relevance <= 1.0
        assert 0.0 <= result.answer_relevance <= 1.0

    @patch("backend.app.evaluation._RAGAS_AVAILABLE", True)
    @patch("backend.app.evaluation.ragas_evaluate")
    @patch("backend.app.evaluation.Dataset")
    def test_ragas_failure_falls_back_to_heuristic(
        self, mock_dataset_cls, mock_ragas_eval, evaluator: EvaluationModule
    ):
        """When RAGAS raises an exception, the module falls back to heuristic scoring."""
        mock_ragas_eval.side_effect = RuntimeError("RAGAS internal error")
        mock_dataset_cls.from_dict.return_value = MagicMock()

        result = evaluator.evaluate_single(
            question="What is RAG?",
            answer="RAG is retrieval augmented generation.",
            context=["Retrieval augmented generation uses retrieval."],
        )

        assert isinstance(result, EvalResult)
        assert result.metadata["evaluator"] == "heuristic"
        assert 0.0 <= result.faithfulness <= 1.0


# ------------------------------------------------------------------
# evaluate_dataset
# ------------------------------------------------------------------


class TestEvaluateDataset:
    def test_loads_and_evaluates(
        self, evaluator: EvaluationModule, sample_dataset: str
    ):
        """Validates: Requirement 8.4 — evaluates dataset and returns results."""
        results = evaluator.evaluate_dataset(sample_dataset)
        assert len(results) == 2
        assert all(isinstance(r, EvalResult) for r in results)

    def test_each_result_has_all_metrics(
        self, evaluator: EvaluationModule, sample_dataset: str
    ):
        """Validates: Requirements 8.1, 8.2, 8.3 — all metrics present in dataset results."""
        results = evaluator.evaluate_dataset(sample_dataset)
        for r in results:
            assert isinstance(r.faithfulness, float)
            assert isinstance(r.context_relevance, float)
            assert isinstance(r.answer_relevance, float)
            assert 0.0 <= r.faithfulness <= 1.0
            assert 0.0 <= r.context_relevance <= 1.0
            assert 0.0 <= r.answer_relevance <= 1.0

    def test_missing_file_raises(self, evaluator: EvaluationModule):
        with pytest.raises(FileNotFoundError):
            evaluator.evaluate_dataset("/nonexistent/path.json")

    def test_invalid_json_raises(self, evaluator: EvaluationModule, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text('{"not": "an array"}')
        with pytest.raises(ValueError, match="JSON array"):
            evaluator.evaluate_dataset(str(bad))

    def test_malformed_json_raises(self, evaluator: EvaluationModule, tmp_path):
        """Malformed JSON (not valid JSON at all) raises an error."""
        bad = tmp_path / "malformed.json"
        bad.write_text("{broken json content!!!")
        with pytest.raises(json.JSONDecodeError):
            evaluator.evaluate_dataset(str(bad))

    def test_empty_dataset(self, evaluator: EvaluationModule, tmp_path):
        """An empty JSON array returns an empty list of results."""
        empty = tmp_path / "empty.json"
        empty.write_text("[]")
        results = evaluator.evaluate_dataset(str(empty))
        assert results == []

    def test_skips_entries_without_question(
        self, evaluator: EvaluationModule, tmp_path
    ):
        data = [{"answer": "no question here", "context": []}]
        path = tmp_path / "partial.json"
        path.write_text(json.dumps(data))
        results = evaluator.evaluate_dataset(str(path))
        assert len(results) == 0


# ------------------------------------------------------------------
# results_to_json — JSON output format
# ------------------------------------------------------------------


class TestResultsToJson:
    def test_valid_json_output(self):
        """Validates: Requirement 8.4 — structured JSON output format."""
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
        assert parsed[0]["context_relevance"] == 0.7
        assert parsed[0]["answer_relevance"] == 0.9

    def test_json_scores_are_floats(self):
        """Validates: Requirement 8.4 — scores in JSON are floats in [0, 1]."""
        results = [
            EvalResult(
                question="Q",
                faithfulness=0.0,
                context_relevance=1.0,
                answer_relevance=0.55,
                metadata={},
            )
        ]
        output = results_to_json(results)
        parsed = json.loads(output)
        entry = parsed[0]
        for key in ("faithfulness", "context_relevance", "answer_relevance"):
            assert isinstance(entry[key], float)
            assert 0.0 <= entry[key] <= 1.0

    def test_json_contains_metadata(self):
        results = [
            EvalResult(
                question="Q",
                faithfulness=0.5,
                context_relevance=0.5,
                answer_relevance=0.5,
                metadata={"evaluator": "ragas", "extra": "info"},
            )
        ]
        output = results_to_json(results)
        parsed = json.loads(output)
        assert parsed[0]["metadata"]["evaluator"] == "ragas"
        assert parsed[0]["metadata"]["extra"] == "info"

    def test_empty_results(self):
        output = results_to_json([])
        assert json.loads(output) == []

    def test_multiple_results(self):
        results = [
            EvalResult(
                question=f"Q{i}",
                faithfulness=0.1 * i,
                context_relevance=0.1 * i,
                answer_relevance=0.1 * i,
                metadata={},
            )
            for i in range(1, 4)
        ]
        output = results_to_json(results)
        parsed = json.loads(output)
        assert len(parsed) == 3
        assert [e["question"] for e in parsed] == ["Q1", "Q2", "Q3"]
