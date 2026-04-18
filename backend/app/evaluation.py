"""RAG evaluation module with RAGAS integration and fallback heuristics.

Computes faithfulness, context_relevance, and answer_relevance scores
for question-answer pairs. Uses RAGAS library when available, otherwise
falls back to simple heuristic scoring using the local Ollama LLM.
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.app.llm_service import LLMService
from backend.app.logging_utils import get_logger
from backend.app.models import EvalResult

logger = get_logger("evaluation")

# ---------------------------------------------------------------------------
# Try importing RAGAS; set a flag so we can fall back gracefully.
# ---------------------------------------------------------------------------
try:
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    _RAGAS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RAGAS_AVAILABLE = False


class EvaluationModule:
    """Evaluate RAG pipeline quality using RAGAS or heuristic fallback.

    Parameters
    ----------
    llm_service:
        An ``LLMService`` instance used as the local evaluator (Ollama).
    """

    def __init__(self, llm_service: LLMService) -> None:
        self._llm = llm_service

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_single(
        self,
        question: str,
        answer: str,
        context: list[str],
    ) -> EvalResult:
        """Evaluate a single question-answer-context triple.

        Returns an ``EvalResult`` with faithfulness, context_relevance,
        and answer_relevance scores in [0, 1].
        """
        if _RAGAS_AVAILABLE:
            return self._evaluate_single_ragas(question, answer, context)
        return self._evaluate_single_heuristic(question, answer, context)

    def evaluate_dataset(self, dataset_path: str) -> list[EvalResult]:
        """Evaluate all entries in a JSON test dataset.

        The JSON file must be an array of objects, each with keys:
        ``question``, ``answer``, and ``context`` (list of strings).

        Returns a list of ``EvalResult`` objects.
        """
        path = Path(dataset_path)
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        if not isinstance(data, list):
            raise ValueError("Dataset must be a JSON array of objects")

        results: list[EvalResult] = []
        for idx, entry in enumerate(data):
            question = entry.get("question", "")
            answer = entry.get("answer", "")
            context = entry.get("context", [])
            if not question:
                logger.warning("Skipping entry %d: missing question", idx)
                continue
            result = self.evaluate_single(question, answer, context)
            results.append(result)

        logger.info(
            "Evaluated %d entries from %s", len(results), dataset_path
        )
        return results

    # ------------------------------------------------------------------
    # RAGAS-based evaluation
    # ------------------------------------------------------------------

    def _evaluate_single_ragas(
        self,
        question: str,
        answer: str,
        context: list[str],
    ) -> EvalResult:
        """Use RAGAS library for evaluation."""
        dataset = Dataset.from_dict(
            {
                "question": [question],
                "answer": [answer],
                "contexts": [context],
            }
        )
        try:
            result = ragas_evaluate(
                dataset,
                metrics=[faithfulness, context_precision, answer_relevancy],
            )
            scores = result.to_pandas().iloc[0]
            return EvalResult(
                question=question,
                faithfulness=float(scores.get("faithfulness", 0.0)),
                context_relevance=float(
                    scores.get("context_precision", 0.0)
                ),
                answer_relevance=float(
                    scores.get("answer_relevancy", 0.0)
                ),
                metadata={"evaluator": "ragas"},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("RAGAS evaluation failed, using fallback: %s", exc)
            return self._evaluate_single_heuristic(question, answer, context)

    # ------------------------------------------------------------------
    # Heuristic fallback evaluation
    # ------------------------------------------------------------------

    def _evaluate_single_heuristic(
        self,
        question: str,
        answer: str,
        context: list[str],
    ) -> EvalResult:
        """Simple heuristic scoring without RAGAS.

        Scores are computed via lightweight text-overlap heuristics:
        - **faithfulness**: fraction of answer words found in context
        - **context_relevance**: fraction of question words found in context
        - **answer_relevance**: fraction of question words found in answer
        """
        faithfulness = _word_overlap(answer, " ".join(context))
        context_relevance = _word_overlap(question, " ".join(context))
        answer_relevance = _word_overlap(question, answer)

        return EvalResult(
            question=question,
            faithfulness=faithfulness,
            context_relevance=context_relevance,
            answer_relevance=answer_relevance,
            metadata={"evaluator": "heuristic"},
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _word_overlap(source: str, target: str) -> float:
    """Return the fraction of *source* words present in *target*.

    Both strings are lowercased and split on whitespace.  Returns 0.0
    when *source* is empty.
    """
    source_words = set(source.lower().split())
    if not source_words:
        return 0.0
    target_words = set(target.lower().split())
    overlap = source_words & target_words
    return round(len(overlap) / len(source_words), 4)


def results_to_json(results: list[EvalResult]) -> str:
    """Serialize a list of ``EvalResult`` objects to a JSON string."""
    return json.dumps(
        [
            {
                "question": r.question,
                "faithfulness": r.faithfulness,
                "context_relevance": r.context_relevance,
                "answer_relevance": r.answer_relevance,
                "metadata": r.metadata,
            }
            for r in results
        ],
        indent=2,
    )
