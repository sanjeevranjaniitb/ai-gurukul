# Requirements: Session Knowledge Quiz

## Requirement 1: Quiz Generation Backend Module

### Description
A standalone FastAPI router module (`backend/app/quiz_module.py`) that accepts session Q&A history and generates MCQ questions using the existing LLM service, returning structured quiz data.

### Acceptance Criteria
- 1.1 The module exposes a `POST /api/quiz/generate` endpoint that accepts a JSON body with `qa_history` (list of Q&A pairs) and `num_questions` (integer, default 3).
- 1.2 The endpoint returns a JSON response containing a `quiz_id` (UUID string) and a `questions` array, where each question has `id`, `question`, `options` (2–4 items), `correct_answer` (valid 0-based index), and `explanation`.
- 1.3 The endpoint validates that `qa_history` has at least 1 and at most 50 entries, and `num_questions` is between 1 and 5; invalid requests return HTTP 422.
- 1.4 The module reuses the existing `LLMService` and `load_config()` — no new dependencies are introduced.
- 1.5 The quiz prompt includes all Q&A pairs from the history and explicitly instructs the LLM to return JSON-formatted MCQs.
- 1.6 The module extracts JSON from the LLM response using multiple fallback strategies (direct parse, code block extraction, bare array extraction) and raises a descriptive error if all fail.
- 1.7 When the LLM is unreachable or times out, the endpoint returns HTTP 503 with a descriptive error message.
- 1.8 When the LLM returns malformed output that cannot be parsed after all extraction attempts, the endpoint returns HTTP 502 with a descriptive error message.

## Requirement 2: Quiz Router Integration

### Description
The quiz module's FastAPI router is mounted into the existing application without modifying any existing endpoint logic.

### Acceptance Criteria
- 2.1 The quiz router is imported and included in `backend/app/main.py` using `app.include_router()`.
- 2.2 All existing endpoints (`/api/ask`, `/api/upload/avatar`, `/api/upload/pdf`, `/api/health`, `/api/reset`) continue to function identically after the quiz router is added.
- 2.3 The quiz endpoint is accessible at `/api/quiz/generate` and appears in the auto-generated API docs (`/docs`).

## Requirement 3: Quiz Panel Frontend Component

### Description
A React modal/overlay component (`frontend/src/components/QuizPanel.tsx`) that displays generated MCQ questions, collects user answers, grades them client-side, and shows the score.

### Acceptance Criteria
- 3.1 The QuizPanel component accepts `qaHistory`, `onClose`, and `onQuizComplete` props.
- 3.2 On mount, the component calls `POST /api/quiz/generate` with the provided Q&A history and displays a loading indicator while waiting.
- 3.3 Each question is rendered with its text and radio-button options; the user can select one option per question.
- 3.4 A "Submit" button is disabled until all questions have an answer selected.
- 3.5 On submit, the component grades answers client-side by comparing selected indices to `correct_answer` values, then displays the score (e.g., "2/3") with per-question correct/incorrect indicators and explanations.
- 3.6 The component provides a "Close" button that calls `onClose` and a completion callback `onQuizComplete(score, total)` after grading.
- 3.7 If the API call fails, the component displays an error message with a "Retry" button.
- 3.8 The component is keyboard-navigable and includes appropriate ARIA labels for accessibility.

## Requirement 4: Quiz Trigger Mechanisms

### Description
The quiz can be triggered in two ways: manually via a "Test My Knowledge" button, or automatically via a suggestion banner after the user has asked 3 or more questions.

### Acceptance Criteria
- 4.1 A "Test My Knowledge" button is visible in the UI whenever at least one Q&A exchange has been completed and no query is currently processing.
- 4.2 Clicking the button opens the QuizPanel overlay with the current session's Q&A history.
- 4.3 The App component tracks the number of user questions asked in the current session.
- 4.4 After the user has asked 3 or more questions, a suggestion banner appears prompting the user to take a quiz.
- 4.5 The suggestion banner has both a "Take Quiz" button (opens QuizPanel) and a dismiss option ("Later").
- 4.6 After a quiz is completed (via `onQuizComplete`), the question counter resets to 0.

## Requirement 5: Q&A History Tracking

### Description
The frontend accumulates Q&A pairs from completed chat exchanges to provide context for quiz generation.

### Acceptance Criteria
- 5.1 Each completed Q&A exchange (user question + full assistant answer) is appended to a `qaHistory` state array in `App.tsx`.
- 5.2 The Q&A history is passed to the QuizPanel when it opens.
- 5.3 The Q&A history persists across multiple quiz attempts within the same session (it is not cleared when a quiz is completed).

## Requirement 6: Quiz API Client

### Description
A type-safe API function and TypeScript types for the quiz endpoint, added to the existing `frontend/src/api.ts` file.

### Acceptance Criteria
- 6.1 A `generateQuiz(qaHistory, numQuestions?)` function is exported from `api.ts` that calls `POST /api/quiz/generate` and returns a typed `QuizGenerateResponse`.
- 6.2 TypeScript interfaces `QuizQuestion` and `QuizGenerateResponse` are exported from `api.ts`.
- 6.3 The function throws a descriptive error if the API returns a non-OK status.
- 6.4 No existing exports or functions in `api.ts` are modified.

## Requirement 7: Backend Unit Tests

### Description
Comprehensive unit tests for the quiz module covering prompt construction, JSON extraction, validation, and endpoint behavior.

### Acceptance Criteria
- 7.1 Tests exist in `backend/app/test_quiz_module.py` covering: prompt construction, JSON extraction (valid JSON, code blocks, bare arrays, no JSON), question validation (valid input, missing fields, out-of-range indices), and endpoint responses (success, LLM failure, malformed output).
- 7.2 Tests mock the `LLMService` to avoid requiring a running Ollama instance.
- 7.3 All existing tests continue to pass after the quiz module is added.

## Requirement 8: Frontend Unit Tests

### Description
Unit tests for the QuizPanel component covering rendering, interaction, grading, and error states.

### Acceptance Criteria
- 8.1 Tests exist in `frontend/src/components/QuizPanel.test.tsx` covering: loading state, question rendering, option selection, submission and grading, score display, error handling, and close/complete callbacks.
- 8.2 All existing frontend tests continue to pass after the QuizPanel is added.
