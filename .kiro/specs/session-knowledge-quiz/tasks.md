# Implementation Plan: Session Knowledge Quiz

## Overview

This plan implements a self-contained quiz module that tests users on material covered during a chat session. The backend adds a new FastAPI router (`quiz_module.py`) that generates MCQs via the existing LLM service. The frontend adds a QuizPanel overlay component, a quiz API client, and App-level state for Q&A tracking and quiz triggering. No existing files are structurally modified — changes to `main.py`, `api.ts`, and `App.tsx` are purely additive.

## Tasks

- [x] 1. Implement quiz generation backend module
  - [x] 1.1 Create `backend/app/quiz_module.py` with Pydantic models and FastAPI router
    - Define `QAPair`, `QuizRequest`, `QuizOption`, `QuizQuestion`, and `QuizResponse` Pydantic models with validation constraints (qa_history min 1 / max 50, num_questions 1–5, options 2–4, correct_answer valid index)
    - Implement `build_quiz_prompt(qa_history, num_questions)` that formats all Q&A pairs into a structured LLM prompt requesting JSON-formatted MCQs
    - Implement `extract_json_from_response(raw_text)` with three fallback strategies: direct JSON parse, markdown code block extraction, bare JSON array extraction; raises `ValueError` on failure
    - Implement `parse_and_validate_questions(json_str, num_questions)` that parses JSON into validated `QuizQuestion` objects, skipping malformed entries
    - Implement `POST /api/quiz/generate` endpoint that orchestrates prompt building, LLM call (non-streaming via `LLMService.generate()`), JSON extraction, and validation; returns `QuizResponse` with a UUID `quiz_id`
    - Return HTTP 503 when LLM is unreachable, HTTP 502 when LLM output cannot be parsed
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8_

  - [x] 1.2 Write property tests for quiz generation backend
    - **Property 1: Question count bound** — ∀ quiz response R: `len(R.questions) ≤ request.num_questions`
    - **Validates: Requirements 1.2**
    - **Property 2: Valid correct_answer index** — ∀ question Q in R.questions: `0 ≤ Q.correct_answer < len(Q.options)`
    - **Validates: Requirements 1.2**
    - **Property 3: Option count range** — ∀ question Q in R.questions: `2 ≤ len(Q.options) ≤ 4`
    - **Validates: Requirements 1.2**
    - **Property 5: Empty history rejection** — ∀ quiz request with empty qa_history: returns HTTP 422
    - **Validates: Requirements 1.3**

  - [x] 1.3 Write unit tests for `backend/app/test_quiz_module.py`
    - Test `build_quiz_prompt()` includes all Q&A pairs and specifies correct question count
    - Test `extract_json_from_response()` with: valid JSON string, JSON inside markdown code blocks, bare JSON array, and no JSON (expect `ValueError`)
    - Test `parse_and_validate_questions()` with: valid input, missing required fields, out-of-range `correct_answer`, too many/few options
    - Test endpoint returns 422 for empty `qa_history` and out-of-range `num_questions`
    - Test endpoint with mocked `LLMService` returning valid response, unreachable LLM (503), and malformed output (502)
    - Mock `LLMService` to avoid requiring a running Ollama instance
    - _Requirements: 7.1, 7.2_

- [x] 2. Mount quiz router in the application
  - [x] 2.1 Add quiz router to `backend/app/main.py`
    - Import `router` from `backend.app.quiz_module`
    - Add `app.include_router(router)` alongside the existing avatar router inclusion
    - _Requirements: 2.1, 2.2, 2.3_

- [x] 3. Checkpoint — Backend complete
  - Ensure all backend tests pass (`pytest backend/`), ask the user if questions arise.

- [x] 4. Add quiz API client to frontend
  - [x] 4.1 Add quiz types and `generateQuiz()` function to `frontend/src/api.ts`
    - Export `QuizQuestion` interface with `id`, `question`, `options` (array of `{index, text}`), `correct_answer`, and `explanation`
    - Export `QuizGenerateResponse` interface with `quiz_id` and `questions`
    - Export `generateQuiz(qaHistory, numQuestions?)` async function that POSTs to `/api/quiz/generate` and returns typed `QuizGenerateResponse`
    - Throw a descriptive error on non-OK HTTP status
    - Do not modify any existing exports or functions
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [x] 5. Implement QuizPanel frontend component
  - [x] 5.1 Create `frontend/src/components/QuizPanel.tsx`
    - Accept `qaHistory`, `onClose`, and `onQuizComplete` props as defined in the design
    - On mount, call `generateQuiz()` with the provided Q&A history and show a loading spinner
    - Render each MCQ with question text and radio-button options for answer selection
    - Disable the "Submit" button until all questions have a selected answer
    - On submit, grade answers client-side by comparing selected indices to `correct_answer` values
    - Display score summary (e.g., "2/3") with per-question correct/incorrect indicators and explanations
    - Provide a "Close" button that calls `onClose`; call `onQuizComplete(score, total)` after grading
    - Display an error message with a "Retry" button if the API call fails
    - Ensure keyboard navigability and include appropriate ARIA labels for accessibility
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_

  - [x] 5.2 Write property test for client-side grading
    - **Property 4: Grading score bound** — ∀ grading result G: `G.score ≤ G.total` and `G.total === questions.length`
    - **Validates: Requirements 3.5**
    - **Property 8: Grading idempotency** — Given the same questions and userAnswers, `gradeQuiz()` always returns the same score and results
    - **Validates: Requirements 3.5**

  - [x] 5.3 Write unit tests for `frontend/src/components/QuizPanel.test.tsx`
    - Test loading state renders while waiting for API response
    - Test questions and radio-button options render correctly after API response
    - Test option selection and submit button enable/disable behavior
    - Test grading: all correct, all wrong, mixed results with score display
    - Test error state with retry button on API failure
    - Test `onClose` and `onQuizComplete` callbacks fire correctly
    - _Requirements: 8.1_

- [x] 6. Integrate quiz triggers into App.tsx
  - [x] 6.1 Add Q&A history tracking and quiz state to `App.tsx`
    - Add `qaHistory` state array that accumulates `{question, answer}` pairs from completed chat exchanges
    - Add `questionCount` state that increments on each user question
    - Add `showQuiz` and `showQuizSuggestion` boolean states
    - Wire a callback into `ChatInterface` (via existing `onDone` flow or a new prop) to capture each completed Q&A pair and append it to `qaHistory`
    - _Requirements: 5.1, 5.2, 5.3, 4.3_

  - [x] 6.2 Add quiz trigger button and suggestion banner to `App.tsx`
    - Render a "Test My Knowledge" button when `qaHistory.length > 0` and not currently processing
    - Clicking the button sets `showQuiz = true` to open the QuizPanel overlay
    - Show a suggestion banner when `questionCount >= 3` prompting the user to take a quiz
    - Banner includes a "Take Quiz" button (opens QuizPanel) and a "Later" dismiss button
    - Mount `QuizPanel` when `showQuiz` is true, passing `qaHistory`, `onClose`, and `onQuizComplete`
    - On `onQuizComplete`, reset `questionCount` to 0
    - _Requirements: 4.1, 4.2, 4.4, 4.5, 4.6_

- [x] 7. Checkpoint — Full integration complete
  - Ensure all backend and frontend tests pass, ask the user if questions arise.

- [x] 8. Verify isolation and existing functionality
  - [x] 8.1 Confirm existing tests still pass
    - Run all existing backend tests to verify no regressions from quiz module addition
    - Run all existing frontend tests to verify no regressions from QuizPanel and App.tsx changes
    - Verify existing endpoints (`/api/ask`, `/api/upload/avatar`, `/api/upload/pdf`, `/api/health`, `/api/reset`) are unaffected
    - _Requirements: 2.2, 7.3, 8.2_

- [x] 9. Final checkpoint — All tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The backend uses Python (FastAPI + Pydantic); the frontend uses TypeScript (React)
- No new dependencies are introduced — the module reuses existing `LLMService`, `load_config`, React, and fetch
