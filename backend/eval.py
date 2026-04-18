"""CLI entry point for RAG evaluation.

Usage::

    python -m backend.eval
    python -m backend.eval --dataset path/to/dataset.json
"""

from __future__ import annotations

import argparse
import sys

from backend.app.config import load_config
from backend.app.evaluation import EvaluationModule, results_to_json
from backend.app.llm_service import LLMService

_DEFAULT_DATASET = "data/eval/test_dataset.json"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run RAG evaluation against a test dataset"
    )
    parser.add_argument(
        "--dataset",
        default=_DEFAULT_DATASET,
        help=f"Path to JSON test dataset (default: {_DEFAULT_DATASET})",
    )
    args = parser.parse_args(argv)

    config = load_config()
    llm = LLMService(config)
    evaluator = EvaluationModule(llm)

    results = evaluator.evaluate_dataset(args.dataset)
    print(results_to_json(results))


if __name__ == "__main__":
    main()
