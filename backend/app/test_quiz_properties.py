"""Property-based tests for quiz generation backend using Hypothesis.

Tests the core correctness properties of parse_and_validate_questions,
build_quiz_prompt, and the /api/quiz/generate endpoint validation.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from backend.app.quiz_module import (
    QAPair,
    parse_and_validate_questions,
    build_quiz_prompt,
    router as quiz_router,
)
from backend.app.models import GenerationResult


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy for a valid option string (non-empty text)
option_text_st = st.text(min_size=1, max_size=80).filter(lambda s: s.strip())


def raw_question_st(max_options: int = 4):
    """Generate a raw question dict that mirrors LLM JSON output."""
    return st.integers(min_value=2, max_value=max_options).flatmap(
        lambda n_opts: st.fixed_dictionaries(
            {
                "question": st.text(min_size=1, max_size=120).filter(
                    lambda s: s.strip()
                ),
                "options": st.lists(option_text_st, min_size=n_opts, max_size=n_opts),
                "correct_answer": st.integers(min_value=0, max_value=n_opts - 1),
                "explanation": st.text(max_size=200),
            }
        )
    )


# Strategy for a list of raw question dicts (1–10 questions)
raw_questions_list_st = st.lists(raw_question_st(), min_size=1, max_size=10)

# Strategy for num_questions parameter (1–5 as per QuizRequest validation)
num_questions_st = st.integers(min_value=1, max_value=5)

# Strategy for a QAPair
qa_pair_st = st.builds(
    QAPair,
    question=st.text(min_size=1, max_size=200).filter(lambda s: s.strip()),
    answer=st.text(min_size=1, max_size=500).filter(lambda s: s.strip()),
)

# Strategy for a non-empty qa_history list (1–10 pairs)
qa_history_st = st.lists(qa_pair_st, min_size=1, max_size=10)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def quiz_client():
    """Create a test client with the quiz router mounted."""
    test_app = FastAPI()
    test_app.include_router(quiz_router)
    return TestClient(test_app)


# ---------------------------------------------------------------------------
# Property 1: Question count bound
# Validates: Requirements 1.2
# ---------------------------------------------------------------------------


class TestQuestionCountBound:
    """Property 1: ∀ quiz response R: len(R.questions) ≤ request.num_questions"""

    @given(raw_questions=raw_questions_list_st, num_questions=num_questions_st)
    @settings(max_examples=200, deadline=None)
    def test_parse_never_exceeds_requested_count(
        self, raw_questions: list[dict], num_questions: int
    ):
        """**Validates: Requirements 1.2**

        parse_and_validate_questions must never return more questions
        than the requested num_questions, regardless of how many raw
        questions the LLM produces.
        """
        json_str = json.dumps(raw_questions)
        result = parse_and_validate_questions(json_str, num_questions)
        assert len(result) <= num_questions


# ---------------------------------------------------------------------------
# Property 2: Valid correct_answer index
# Validates: Requirements 1.2
# ---------------------------------------------------------------------------


class TestValidCorrectAnswerIndex:
    """Property 2: ∀ question Q in R.questions: 0 ≤ Q.correct_answer < len(Q.options)"""

    @given(raw_questions=raw_questions_list_st, num_questions=num_questions_st)
    @settings(max_examples=200, deadline=None)
    def test_correct_answer_always_valid_index(
        self, raw_questions: list[dict], num_questions: int
    ):
        """**Validates: Requirements 1.2**

        Every validated question must have a correct_answer that is a
        valid 0-based index into its options list.
        """
        json_str = json.dumps(raw_questions)
        result = parse_and_validate_questions(json_str, num_questions)
        for q in result:
            assert 0 <= q.correct_answer < len(q.options), (
                f"correct_answer={q.correct_answer} out of range for "
                f"{len(q.options)} options"
            )


# ---------------------------------------------------------------------------
# Property 3: Option count range
# Validates: Requirements 1.2
# ---------------------------------------------------------------------------


class TestOptionCountRange:
    """Property 3: ∀ question Q in R.questions: 2 ≤ len(Q.options) ≤ 4"""

    @given(raw_questions=raw_questions_list_st, num_questions=num_questions_st)
    @settings(max_examples=200, deadline=None)
    def test_options_count_within_bounds(
        self, raw_questions: list[dict], num_questions: int
    ):
        """**Validates: Requirements 1.2**

        Every validated question must have between 2 and 4 options
        (inclusive).
        """
        json_str = json.dumps(raw_questions)
        result = parse_and_validate_questions(json_str, num_questions)
        for q in result:
            assert 2 <= len(q.options) <= 4, (
                f"Expected 2-4 options, got {len(q.options)}"
            )


# ---------------------------------------------------------------------------
# Property 5: Empty history rejection
# Validates: Requirements 1.3
# ---------------------------------------------------------------------------


class TestEmptyHistoryRejection:
    """Property 5: ∀ quiz request with empty qa_history: returns HTTP 422"""

    @given(num_questions=st.integers(min_value=1, max_value=5))
    @settings(max_examples=50, deadline=None)
    def test_empty_qa_history_returns_422(self, num_questions: int):
        """**Validates: Requirements 1.3**

        Any quiz request with an empty qa_history list must be rejected
        with HTTP 422 by Pydantic validation, regardless of num_questions.
        """
        test_app = FastAPI()
        test_app.include_router(quiz_router)
        client = TestClient(test_app)

        resp = client.post(
            "/api/quiz/generate",
            json={"qa_history": [], "num_questions": num_questions},
        )
        assert resp.status_code == 422, (
            f"Expected 422 for empty qa_history, got {resp.status_code}"
        )
