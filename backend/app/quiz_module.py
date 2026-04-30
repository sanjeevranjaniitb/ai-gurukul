"""Quiz generation module — standalone FastAPI router for session knowledge quizzes.

Accepts session Q&A history and generates multiple-choice questions using the
existing LLM service. Designed as an isolated add-on with no modifications to
existing modules.
"""

from __future__ import annotations

import json
import re
import time
import uuid

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from backend.app.config import load_config
from backend.app.logging_utils import get_logger

logger = get_logger("quiz_module")

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class QAPair(BaseModel):
    """A single question-answer exchange from the session."""

    question: str
    answer: str


class QuizRequest(BaseModel):
    """Request body for session-based quiz generation."""

    qa_history: list[QAPair] = Field(..., min_length=1, max_length=50)
    num_questions: int = Field(default=2, ge=1, le=5)


class DocQuizRequest(BaseModel):
    """Request body for document-based quiz generation."""

    document_id: str
    num_questions: int = Field(default=10, ge=1, le=15)


class QuizOption(BaseModel):
    """A single MCQ option."""

    index: int
    text: str


class QuizQuestion(BaseModel):
    """A generated multiple-choice question."""

    id: int
    question: str
    options: list[QuizOption] = Field(..., min_length=2, max_length=4)
    correct_answer: int  # 0-based index into options
    explanation: str

    @model_validator(mode="after")
    def _validate_correct_answer(self) -> "QuizQuestion":
        if not (0 <= self.correct_answer < len(self.options)):
            raise ValueError(
                f"correct_answer {self.correct_answer} is out of range "
                f"for {len(self.options)} options"
            )
        return self


class QuizResponse(BaseModel):
    """Response containing generated quiz questions."""

    quiz_id: str
    questions: list[QuizQuestion]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def build_quiz_prompt(qa_history: list[QAPair], num_questions: int) -> str:
    """Format all Q&A pairs into a structured LLM prompt requesting JSON MCQs.

    Preconditions:
        - qa_history is non-empty
        - num_questions >= 1

    Postconditions:
        - Returns a prompt string containing every Q&A pair
        - Prompt explicitly requests JSON output format
        - Prompt specifies the exact number of questions to generate
    """
    context_block = ""
    for i, pair in enumerate(qa_history, 1):
        context_block += f"Q{i}: {pair.question}\nA{i}: {pair.answer}\n\n"

    prompt = (
        f"You are a quiz generator. Based ONLY on the answers provided in this session, "
        f"generate {num_questions} multiple-choice questions.\n\n"
        f"STRICT RULES:\n"
        f"- The correct answer MUST come directly from the session answers below\n"
        f"- The 3 wrong options must be clearly WRONG and obviously different from the correct answer\n"
        f"- Do NOT make wrong options similar to the correct answer\n"
        f"- Each question must have exactly 1 correct answer\n"
        f"- Each question should test a different concept\n"
        f"- Return ONLY a valid JSON array, nothing else\n\n"
        f"SESSION:\n{context_block}"
        f'JSON format: [{{"question":"...","options":["correct thing","wrong1","wrong2","wrong3"],"correct_answer":0,"explanation":"short reason"}}]\n\n'
        f"IMPORTANT: correct_answer is the 0-based index of the correct option. Shuffle the position of the correct answer."
    )
    return prompt


def extract_json_from_response(raw_text: str) -> str:
    """Extract a JSON array from LLM response text using multiple strategies.

    Strategy 1: Direct JSON parse
    Strategy 2: Markdown code block extraction
    Strategy 3: Bare JSON array extraction
    Strategy 4: Repair truncated JSON by extracting complete objects

    Raises ValueError if no valid JSON array can be found.
    """
    # Strategy 1: Direct JSON parse (array or single object)
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, list):
            return raw_text.strip()
        if isinstance(parsed, dict) and "question" in parsed:
            return json.dumps([parsed])
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: Markdown code block extraction
    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw_text, re.DOTALL)
    if match:
        candidate = match.group(1)
        try:
            json.loads(candidate)
            return candidate
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: Bare JSON array extraction
    match = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            json.loads(candidate)
            return candidate
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 4: Repair truncated JSON — find complete {...} objects and wrap
    repaired = _repair_truncated_json(raw_text)
    if repaired:
        return repaired

    raise ValueError("No valid JSON array found in LLM response")


def _repair_truncated_json(raw_text: str) -> str | None:
    """Try to salvage complete JSON objects from a truncated response.

    Finds all complete {...} blocks, parses each individually, and wraps
    the valid ones in an array. Returns None if no valid objects found.
    """
    # Find all complete JSON object blocks
    objects: list[dict] = []
    brace_depth = 0
    start = -1

    for i, ch in enumerate(raw_text):
        if ch == "{":
            if brace_depth == 0:
                start = i
            brace_depth += 1
        elif ch == "}":
            brace_depth -= 1
            if brace_depth == 0 and start >= 0:
                candidate = raw_text[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict) and "question" in obj:
                        objects.append(obj)
                except (json.JSONDecodeError, ValueError):
                    pass
                start = -1

    if objects:
        return json.dumps(objects)
    return None


def parse_and_validate_questions(
    json_str: str, num_questions: int
) -> list[QuizQuestion]:
    """Parse a JSON string into validated QuizQuestion objects.

    Skips malformed entries rather than failing the entire request.

    Preconditions:
        - json_str is a valid JSON string containing an array
        - num_questions >= 1

    Postconditions:
        - Returns list of QuizQuestion objects
        - len(result) <= num_questions
        - Each question has 2-4 options with a valid correct_answer index
    """
    raw_questions = json.loads(json_str)
    validated: list[QuizQuestion] = []

    for i, raw in enumerate(raw_questions[:num_questions]):
        try:
            # Validate required fields exist
            if not isinstance(raw, dict):
                continue
            if not all(k in raw for k in ("question", "options", "correct_answer")):
                continue

            options = raw["options"]
            if not isinstance(options, list):
                continue

            correct_idx = int(raw["correct_answer"])

            # Validate option count
            if not (2 <= len(options) <= 4):
                continue

            # Validate correct_answer index
            if not (0 <= correct_idx < len(options)):
                correct_idx = 0  # fallback to first option

            validated.append(
                QuizQuestion(
                    id=i,
                    question=str(raw["question"]),
                    options=[
                        QuizOption(index=j, text=str(opt))
                        for j, opt in enumerate(options)
                    ],
                    correct_answer=correct_idx,
                    explanation=str(raw.get("explanation", "")),
                )
            )
        except (TypeError, ValueError, KeyError) as exc:
            logger.warning("Skipping malformed question at index %d: %s", i, exc)
            continue

    return validated


# ---------------------------------------------------------------------------
# Router & endpoint
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/quiz", tags=["Quiz"])

# Direct Ollama config — bypasses LLMService to avoid the Q&A system prompt
# that conflicts with JSON generation.
_config = load_config()
_OLLAMA_URL = _config.llm_base_url.rstrip("/") + "/api/generate"
_OLLAMA_MODEL = _config.llm_model


def _call_ollama_raw(prompt: str, num_predict: int = 2048) -> str:
    """Call Ollama directly with a raw prompt (no system prompt wrapping).

    Returns the raw response text. Raises HTTPException on failure.
    """
    payload = {
        "model": _OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.6,
            "num_predict": num_predict,
        },
    }
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(_OLLAMA_URL, json=payload)
            resp.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException):
        raise HTTPException(
            status_code=503,
            detail="Quiz generation service unavailable. Please try again later.",
        )
    except httpx.HTTPStatusError:
        raise HTTPException(
            status_code=503,
            detail="Quiz generation service unavailable. Please try again later.",
        )

    data = resp.json()
    return data.get("response", "")


@router.post("/generate", response_model=QuizResponse)
async def generate_quiz(body: QuizRequest) -> QuizResponse:
    """Generate MCQ questions from session Q&A history.

    Returns a QuizResponse with a UUID quiz_id and a list of validated questions.
    - HTTP 503 when the LLM is unreachable or times out
    - HTTP 502 when the LLM output cannot be parsed into valid questions
    """
    # Step 1: Build the quiz generation prompt
    prompt = build_quiz_prompt(body.qa_history, body.num_questions)

    # Step 2: Call Ollama directly (no system prompt interference)
    raw_text = _call_ollama_raw(prompt)

    logger.info("Raw LLM quiz response (%d chars): %s", len(raw_text), raw_text[:300])

    # Step 3: Extract JSON from LLM response
    try:
        json_str = extract_json_from_response(raw_text)
    except ValueError:
        logger.error("Failed to extract JSON from LLM response: %s", raw_text[:200])
        raise HTTPException(
            status_code=502,
            detail="Failed to generate valid quiz questions. Please try again.",
        )

    # Step 4: Parse and validate questions
    questions = parse_and_validate_questions(json_str, body.num_questions)

    if not questions:
        raise HTTPException(
            status_code=502,
            detail="Failed to generate valid quiz questions. Please try again.",
        )

    # Step 5: Return structured response
    return QuizResponse(
        quiz_id=str(uuid.uuid4()),
        questions=questions,
    )


# ---------------------------------------------------------------------------
# Document-based quiz endpoint
# ---------------------------------------------------------------------------

def _get_document_chunks(document_id: str, max_chunks: int = 20) -> list[str]:
    """Retrieve document chunk texts from ChromaDB for a given document_id."""
    import chromadb

    persist_dir = _config.chroma_persist_dir
    client = chromadb.PersistentClient(path=persist_dir)

    try:
        collection = client.get_collection("documents")
    except Exception:
        return []

    # Query all chunks for this document
    results = collection.get(
        where={"document_id": document_id},
        include=["documents"],
        limit=max_chunks,
    )

    if results and results.get("documents"):
        return results["documents"]
    return []


def build_doc_quiz_prompt(doc_text: str, num_questions: int) -> str:
    """Build a prompt for generating quiz questions from document content."""
    return (
        f"You are a quiz generator for educational assessment. "
        f"Based ONLY on the document content below, generate {num_questions} "
        f"multiple-choice questions that test understanding of the key concepts.\n\n"
        f"STRICT RULES:\n"
        f"- Each question must test a DIFFERENT key concept from the document\n"
        f"- The correct answer MUST be factually accurate based on the document\n"
        f"- The 3 wrong options must be clearly WRONG and obviously different\n"
        f"- Questions should progress from basic to advanced concepts\n"
        f"- Return ONLY a valid JSON array, no other text\n"
        f"- Keep options short (under 10 words each)\n"
        f"- Keep explanations short (under 15 words)\n\n"
        f"DOCUMENT CONTENT:\n{doc_text}\n\n"
        f'JSON format: [{{"question":"...","options":["opt1","opt2","opt3","opt4"],'
        f'"correct_answer":0,"explanation":"..."}}]\n\n'
        f"IMPORTANT: correct_answer is the 0-based index. Shuffle correct answer position. "
        f"Generate exactly {num_questions} questions."
    )


@router.post("/generate-from-doc", response_model=QuizResponse)
async def generate_doc_quiz(body: DocQuizRequest) -> QuizResponse:
    """Generate MCQ questions from uploaded document content.

    Retrieves document chunks from ChromaDB and generates quiz questions
    covering the key concepts in the document.
    """
    # Step 1: Get document chunks
    chunks = _get_document_chunks(body.document_id)
    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="Document not found or has no content.",
        )

    # Combine chunks into a single text block (truncate to fit context)
    doc_text = "\n\n".join(chunks)
    if len(doc_text) > 6000:
        doc_text = doc_text[:6000]

    # Step 2: Generate questions in batches to avoid truncation
    # For 10 questions, generate in batches of 3-4
    all_questions: list[QuizQuestion] = []
    remaining = body.num_questions
    batch_num = 0

    while remaining > 0 and batch_num < 5:  # max 5 attempts
        batch_size = min(remaining, 3)
        prompt = build_doc_quiz_prompt(doc_text, batch_size)

        raw_text = _call_ollama_raw(prompt, num_predict=4096)
        logger.info("Doc quiz batch %d response (%d chars): %s",
                     batch_num, len(raw_text), raw_text[:200])

        try:
            json_str = extract_json_from_response(raw_text)
            batch_questions = parse_and_validate_questions(json_str, batch_size)

            # Re-number question IDs
            for q in batch_questions:
                q.id = len(all_questions)
                all_questions.append(q)

            remaining -= len(batch_questions)
        except (ValueError, Exception) as exc:
            logger.warning("Doc quiz batch %d failed: %s", batch_num, exc)

        batch_num += 1

    if not all_questions:
        raise HTTPException(
            status_code=502,
            detail="Failed to generate quiz from document. Please try again.",
        )

    return QuizResponse(
        quiz_id=str(uuid.uuid4()),
        questions=all_questions[:body.num_questions],
    )
