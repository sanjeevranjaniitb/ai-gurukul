"""Unit tests for quiz_module — prompt building, JSON extraction, validation, and endpoint.

Validates: Requirements 7.1, 7.2
"""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.models import GenerationResult
from backend.app.quiz_module import (
    QAPair,
    QuizOption,
    QuizQuestion,
    QuizRequest,
    QuizResponse,
    build_quiz_prompt,
    extract_json_from_response,
    parse_and_validate_questions,
    router as quiz_router,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_QA = [
    QAPair(question="What is photosynthesis?", answer="The process by which plants convert sunlight into energy."),
    QAPair(question="What are the products?", answer="Glucose and oxygen."),
]

_VALID_LLM_JSON = json.dumps([
    {
        "question": "What organelle is responsible for photosynthesis?",
        "options": ["A) Mitochondria", "B) Nucleus", "C) Chloroplast", "D) Ribosome"],
        "correct_answer": 2,
        "explanation": "Chloroplasts contain chlorophyll.",
    },
    {
        "question": "Which is a product of photosynthesis?",
        "options": ["A) Carbon dioxide", "B) Glucose", "C) Nitrogen", "D) Water"],
        "correct_answer": 1,
        "explanation": "Glucose is produced during photosynthesis.",
    },
])


def _make_generation_result(answer: str) -> GenerationResult:
    """Build a GenerationResult with the given answer text."""
    return GenerationResult(
        answer=answer,
        model="llama3.2:3b",
        prompt_tokens=100,
        completion_tokens=50,
        duration_ms=500.0,
    )


@pytest.fixture()
def quiz_client():
    """Create a TestClient with only the quiz router mounted."""
    test_app = FastAPI()
    test_app.include_router(quiz_router)
    return TestClient(test_app)


# ---------------------------------------------------------------------------
# build_quiz_prompt()
# ---------------------------------------------------------------------------


class TestBuildQuizPrompt:
    """Tests for build_quiz_prompt — prompt construction logic."""

    def test_includes_all_qa_pairs(self):
        prompt = build_quiz_prompt(_SAMPLE_QA, 2)
        assert "Q1: What is photosynthesis?" in prompt
        assert "A1: The process by which plants convert sunlight into energy." in prompt
        assert "Q2: What are the products?" in prompt
        assert "A2: Glucose and oxygen." in prompt

    def test_specifies_correct_question_count(self):
        prompt = build_quiz_prompt(_SAMPLE_QA, 3)
        assert "exactly 3" in prompt

    def test_requests_json_format(self):
        prompt = build_quiz_prompt(_SAMPLE_QA, 2)
        assert "JSON" in prompt

    def test_single_qa_pair(self):
        single = [QAPair(question="What is AI?", answer="Artificial Intelligence.")]
        prompt = build_quiz_prompt(single, 1)
        assert "Q1: What is AI?" in prompt
        assert "A1: Artificial Intelligence." in prompt
        assert "exactly 1" in prompt


# ---------------------------------------------------------------------------
# extract_json_from_response()
# ---------------------------------------------------------------------------


class TestExtractJsonFromResponse:
    """Tests for extract_json_from_response — JSON extraction strategies."""

    def test_valid_json_string(self):
        raw = '[{"question": "Q?", "options": ["A", "B"], "correct_answer": 0}]'
        result = extract_json_from_response(raw)
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 1

    def test_json_inside_markdown_code_block(self):
        raw = (
            "Here are the questions:\n"
            "```json\n"
            '[{"question": "Q?", "options": ["A", "B"], "correct_answer": 0}]\n'
            "```\n"
            "Hope that helps!"
        )
        result = extract_json_from_response(raw)
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert parsed[0]["question"] == "Q?"

    def test_json_inside_plain_code_block(self):
        raw = (
            "```\n"
            '[{"question": "Q?", "options": ["A", "B"], "correct_answer": 0}]\n'
            "```"
        )
        result = extract_json_from_response(raw)
        parsed = json.loads(result)
        assert isinstance(parsed, list)

    def test_bare_json_array(self):
        raw = (
            "Sure, here are the questions:\n"
            '[{"question": "Q?", "options": ["A", "B"], "correct_answer": 0}]'
            "\nLet me know if you need more."
        )
        result = extract_json_from_response(raw)
        parsed = json.loads(result)
        assert isinstance(parsed, list)

    def test_no_json_raises_value_error(self):
        raw = "I cannot generate quiz questions right now."
        with pytest.raises(ValueError, match="No valid JSON array"):
            extract_json_from_response(raw)

    def test_invalid_json_raises_value_error(self):
        raw = "[{broken json"
        with pytest.raises(ValueError, match="No valid JSON array"):
            extract_json_from_response(raw)


# ---------------------------------------------------------------------------
# parse_and_validate_questions()
# ---------------------------------------------------------------------------


class TestParseAndValidateQuestions:
    """Tests for parse_and_validate_questions — JSON parsing and validation."""

    def test_valid_input(self):
        questions = parse_and_validate_questions(_VALID_LLM_JSON, 2)
        assert len(questions) == 2
        assert questions[0].question == "What organelle is responsible for photosynthesis?"
        assert questions[0].correct_answer == 2
        assert len(questions[0].options) == 4
        assert questions[1].correct_answer == 1

    def test_limits_to_num_questions(self):
        questions = parse_and_validate_questions(_VALID_LLM_JSON, 1)
        assert len(questions) == 1

    def test_missing_required_fields_skipped(self):
        raw = json.dumps([
            {"options": ["A", "B"], "correct_answer": 0},  # missing "question"
            {"question": "Valid?", "options": ["A", "B", "C"], "correct_answer": 1, "explanation": "ok"},
        ])
        questions = parse_and_validate_questions(raw, 5)
        assert len(questions) == 1
        assert questions[0].question == "Valid?"

    def test_out_of_range_correct_answer_falls_back_to_zero(self):
        raw = json.dumps([
            {
                "question": "Q?",
                "options": ["A", "B", "C"],
                "correct_answer": 10,  # out of range
                "explanation": "fallback",
            }
        ])
        questions = parse_and_validate_questions(raw, 5)
        assert len(questions) == 1
        assert questions[0].correct_answer == 0  # fallback

    def test_too_many_options_skipped(self):
        raw = json.dumps([
            {
                "question": "Q?",
                "options": ["A", "B", "C", "D", "E"],  # 5 options, max is 4
                "correct_answer": 0,
                "explanation": "",
            }
        ])
        questions = parse_and_validate_questions(raw, 5)
        assert len(questions) == 0

    def test_too_few_options_skipped(self):
        raw = json.dumps([
            {
                "question": "Q?",
                "options": ["A"],  # only 1 option, min is 2
                "correct_answer": 0,
                "explanation": "",
            }
        ])
        questions = parse_and_validate_questions(raw, 5)
        assert len(questions) == 0

    def test_non_dict_entries_skipped(self):
        raw = json.dumps(["not a dict", {"question": "Q?", "options": ["A", "B"], "correct_answer": 0}])
        questions = parse_and_validate_questions(raw, 5)
        assert len(questions) == 1

    def test_missing_explanation_defaults_to_empty(self):
        raw = json.dumps([
            {"question": "Q?", "options": ["A", "B"], "correct_answer": 0}
        ])
        questions = parse_and_validate_questions(raw, 5)
        assert len(questions) == 1
        assert questions[0].explanation == ""


# ---------------------------------------------------------------------------
# Endpoint validation (422 errors)
# ---------------------------------------------------------------------------


class TestEndpointValidation:
    """Tests for request validation — Pydantic rejects invalid inputs."""

    def test_empty_qa_history_returns_422(self, quiz_client: TestClient):
        resp = quiz_client.post(
            "/api/quiz/generate",
            json={"qa_history": [], "num_questions": 2},
        )
        assert resp.status_code == 422

    def test_num_questions_zero_returns_422(self, quiz_client: TestClient):
        resp = quiz_client.post(
            "/api/quiz/generate",
            json={
                "qa_history": [{"question": "Q?", "answer": "A."}],
                "num_questions": 0,
            },
        )
        assert resp.status_code == 422

    def test_num_questions_too_large_returns_422(self, quiz_client: TestClient):
        resp = quiz_client.post(
            "/api/quiz/generate",
            json={
                "qa_history": [{"question": "Q?", "answer": "A."}],
                "num_questions": 10,
            },
        )
        assert resp.status_code == 422

    def test_missing_qa_history_returns_422(self, quiz_client: TestClient):
        resp = quiz_client.post(
            "/api/quiz/generate",
            json={"num_questions": 2},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Endpoint with mocked LLMService
# ---------------------------------------------------------------------------


class TestEndpointWithMockedLLM:
    """Tests for the /api/quiz/generate endpoint with LLMService mocked."""

    def test_valid_response(self, quiz_client: TestClient):
        """Mocked LLM returns valid JSON — endpoint returns 200 with questions."""
        mock_result = _make_generation_result(_VALID_LLM_JSON)

        with patch("backend.app.quiz_module._llm_service") as mock_llm:
            mock_llm.generate.return_value = mock_result

            resp = quiz_client.post(
                "/api/quiz/generate",
                json={
                    "qa_history": [
                        {"question": "What is photosynthesis?", "answer": "Plants convert sunlight."},
                    ],
                    "num_questions": 2,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "quiz_id" in data
        assert len(data["questions"]) == 2
        assert data["questions"][0]["correct_answer"] == 2
        assert len(data["questions"][0]["options"]) == 4

    def test_unreachable_llm_returns_503(self, quiz_client: TestClient):
        """LLM returns error sentinel — endpoint returns 503."""
        error_answer = "I'm sorry, I was unable to generate a response. Please try again later."
        mock_result = _make_generation_result(error_answer)

        with patch("backend.app.quiz_module._llm_service") as mock_llm:
            mock_llm.generate.return_value = mock_result

            resp = quiz_client.post(
                "/api/quiz/generate",
                json={
                    "qa_history": [
                        {"question": "Q?", "answer": "A."},
                    ],
                    "num_questions": 2,
                },
            )

        assert resp.status_code == 503
        assert "unavailable" in resp.json()["detail"].lower()

    def test_malformed_llm_output_returns_502(self, quiz_client: TestClient):
        """LLM returns unparseable text — endpoint returns 502."""
        mock_result = _make_generation_result("Here are some thoughts but no JSON at all.")

        with patch("backend.app.quiz_module._llm_service") as mock_llm:
            mock_llm.generate.return_value = mock_result

            resp = quiz_client.post(
                "/api/quiz/generate",
                json={
                    "qa_history": [
                        {"question": "Q?", "answer": "A."},
                    ],
                    "num_questions": 2,
                },
            )

        assert resp.status_code == 502
        assert "failed" in resp.json()["detail"].lower()

    def test_llm_returns_valid_but_all_malformed_questions_returns_502(self, quiz_client: TestClient):
        """LLM returns valid JSON but all questions fail validation — endpoint returns 502."""
        bad_json = json.dumps([
            {"options": ["A"], "correct_answer": 0},  # missing question, too few options
        ])
        mock_result = _make_generation_result(bad_json)

        with patch("backend.app.quiz_module._llm_service") as mock_llm:
            mock_llm.generate.return_value = mock_result

            resp = quiz_client.post(
                "/api/quiz/generate",
                json={
                    "qa_history": [
                        {"question": "Q?", "answer": "A."},
                    ],
                    "num_questions": 2,
                },
            )

        assert resp.status_code == 502

    def test_llm_called_with_correct_prompt(self, quiz_client: TestClient):
        """Verify the LLM is called with a prompt containing the Q&A history."""
        mock_result = _make_generation_result(_VALID_LLM_JSON)

        with patch("backend.app.quiz_module._llm_service") as mock_llm:
            mock_llm.generate.return_value = mock_result

            quiz_client.post(
                "/api/quiz/generate",
                json={
                    "qa_history": [
                        {"question": "What is AI?", "answer": "Artificial Intelligence."},
                    ],
                    "num_questions": 2,
                },
            )

        # Verify generate was called and the prompt contains the Q&A pair
        mock_llm.generate.assert_called_once()
        call_args = mock_llm.generate.call_args
        prompt_arg = call_args[0][0]  # first positional arg
        assert "What is AI?" in prompt_arg
        assert "Artificial Intelligence." in prompt_arg
