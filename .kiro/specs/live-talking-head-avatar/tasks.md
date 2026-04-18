# Implementation Plan: Live Talking Head Avatar

## Overview

This plan implements an end-to-end conversational AI system that transforms a static photograph into an animated talking head avatar, answering questions from uploaded PDF documents using a RAG pipeline, TTS synthesis, and facial animation. The backend is Python/FastAPI, the frontend is React + TypeScript with Vite, and all AI models are open-source and edge-optimized. Tasks are ordered so each step builds on the previous, with no orphaned code.

## Tasks

- [ ] 1. Set up project structure, configuration, and shared data models
  - [x] 1.1 Create the monorepo directory structure with `backend/` (Python/FastAPI) and `frontend/` (React + TypeScript + Vite) directories, plus `data/`, `models/`, and `config/` folders matching the design file storage layout
    - Initialize Python project with `pyproject.toml333223` or `requirements.txt` including FastAPI, uvicorn, PyMuPDF, langchain, sentence-transformers, chromadb, piper-tts, and RAGAS
    - Initialize React + TypeScript project with Vite in `frontend/`
    - _Requirements: 9.5, 10.2_

  - [x] 1.2 Implement the `AppConfig` dataclass and YAML/JSON configuration loader
    - Define all tunable parameters (chunk_size, chunk_overlap, retrieval_top_k, model paths, validation limits) as specified in the design's Configuration Schema
    - Load configuration from a `config.yaml` file with sensible defaults
    - _Requirements: 9.5, 7.3_

  - [x] 1.3 Implement shared data models and types
    - Create all dataclasses: `ParsedDocument`, `PageContent`, `DocumentChunk`, `RetrievalResult`, `GenerationResult`, `AudioResult`, `AvatarProfile`, `VideoResult`, `OrchestratorResponse`, `ProcessingStage`, `EvalResult`
    - _Requirements: 9.5_

  - [x] 1.4 Set up structured logging utility
    - Configure Python structured logging (JSON format) with timestamps, component identifiers, and request correlation IDs
    - _Requirements: 9.1_

- [ ] 2. Implement PDF parsing and chunking pipeline
  - [x] 2.1 Implement `PDFParser` class using PyMuPDF
    - Extract text from all pages, preserve table structure as markdown-formatted text
    - Validate file is not corrupted or password-protected; raise `PDFParseError` with descriptive messages
    - Enforce 50 MB file size limit
    - _Requirements: 2.1, 2.2, 2.3, 2.6, 2.8_

  - [x] 2.2 Implement `ChunkingModule` class using LangChain `RecursiveCharacterTextSplitter`
    - Split text into 512-token chunks with 50-token overlap using tiktoken tokenizer
    - Produce `DocumentChunk` objects with chunk_id, document_id, page_number, token_count, and character offsets
    - _Requirements: 2.4_

  - [x] 2.3 Write unit tests for `PDFParser` and `ChunkingModule`
    - Test text extraction, table preservation, error handling for corrupted/password-protected PDFs, chunk size and overlap correctness
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 9.3_

- [ ] 3. Implement embedding service and vector store
  - [x] 3.1 Implement `EmbeddingStore` class using sentence-transformers and ChromaDB
    - Load `all-MiniLM-L6-v2` model (384-dimensional embeddings)
    - Implement `add_chunks()` to embed and store `DocumentChunk` objects in ChromaDB persistent collection
    - Implement `search()` for top-k semantic similarity retrieval
    - _Requirements: 2.5, 3.1, 3.7_

  - [x] 3.2 Write unit tests for `EmbeddingStore`
    - Test chunk storage, retrieval ranking, and empty-store edge case
    - _Requirements: 2.5, 3.1, 9.3_

- [ ] 4. Implement LLM service and RAG pipeline
  - [x] 4.1 Implement `LLMService` class wrapping Ollama HTTP API
    - Connect to Ollama at configurable base URL, use `llama3.2:3b` model
    - Build prompt template that includes retrieved context chunks and the user question
    - Implement `generate()` for batch response returning `GenerationResult` with answer, token counts, and duration
    - Implement `generate_stream()` that yields tokens as they arrive from Ollama's streaming API
    - Handle connection errors and model failures gracefully
    - _Requirements: 3.2, 3.3, 3.5, 3.6, 10.1_

  - [x] 4.2 Wire the RAG retrieval-generation flow
    - Create a `RAGPipeline` class that calls `EmbeddingStore.search()` then `LLMService.generate()`
    - When no relevant context is found (all scores below threshold), return a "no relevant information" message
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [ ]* 4.3 Write unit tests for `LLMService` and `RAGPipeline`
    - Mock Ollama responses; test prompt construction, no-context fallback, error handling
    - _Requirements: 3.2, 3.3, 3.4, 3.5, 9.3_

- [x] 5. Checkpoint
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. Implement TTS engine
  - [x] 6.1 Implement `TTSEngine` class using Piper TTS
    - Load Piper ONNX model from configurable path
    - Implement `synthesize()` to convert text to WAV audio at ≥22050 Hz sample rate
    - Implement `synthesize_chunk()` for sentence-level synthesis to support pipelined streaming (synthesize individual sentences as they arrive from the LLM stream)
    - Handle unsupported characters by skipping them and continuing
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 10.2_

  - [ ]* 6.2 Write unit tests for `TTSEngine`
    - Test WAV output format, sample rate, and unsupported character handling
    - _Requirements: 4.1, 4.2, 4.5, 9.3_

- [ ] 7. Implement avatar engine
  - [x] 7.1 Implement `AvatarEngine` class using SadTalker
    - Implement `preprocess()` to validate image (PNG/JPG/JPEG, ≥256×256, ≤10 MB), detect face, extract facial landmarks; raise `FaceNotFoundError` if no face detected
    - Implement `animate()` to generate lip-synced MP4 video at ≥25 FPS from avatar profile and audio
    - Implement `animate_chunk()` for incremental animation — generate a video segment from a single audio chunk, enabling pipelined processing
    - On frame generation failure, log error and substitute static avatar image for the failed segment
    - _Requirements: 1.1, 1.2, 1.3, 1.5, 1.6, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 10.3_

  - [ ]* 7.2 Write unit tests for `AvatarEngine`
    - Test image validation, face detection error, video output properties (FPS, format), fallback on failure
    - _Requirements: 1.1, 1.2, 1.3, 1.5, 5.1, 5.3, 5.6, 9.3_

- [ ] 8. Implement the Orchestrator and FastAPI endpoints
  - [x] 8.1 Implement the `Orchestrator` class
    - Wire the streaming pipeline: retrieve → stream generate → chunked synthesize → chunked animate
    - Implement `process_question_stream()` as an async generator that yields `StreamEvent` objects (text_token, audio_chunk, video_chunk, stage_update, done)
    - Accumulate streamed LLM tokens into sentence-level chunks; as each sentence completes, pipeline it through TTS and avatar animation concurrently with continued LLM streaming
    - Implement `upload_avatar()` and `upload_pdf()` async methods
    - Log all request-response cycles with timestamps, durations, and component identifiers
    - Catch unhandled exceptions from any component, log stack traces, and return user-friendly error messages via error StreamEvents
    - Track processing stages with timing for each phase
    - _Requirements: 3.5, 9.1, 9.2, 10.1, 10.5_

  - [x] 8.2 Implement FastAPI API endpoints
    - `POST /api/upload/avatar` — accept multipart file, validate, call `Orchestrator.upload_avatar()`, return `avatar_id`, `preview_url`, `landmarks_ready`
    - `POST /api/upload/pdf` — accept multipart file, validate size, call `Orchestrator.upload_pdf()`, return `document_id`, `name`, `page_count`, `chunk_count`
    - `POST /api/ask` — accept JSON body with `question` and `avatar_id`, return SSE stream (`text/event-stream`) from `Orchestrator.process_question_stream()` yielding text tokens, audio chunk URLs, video chunk URLs, stage updates, and sources progressively
    - `GET /api/health` — return system status, models_loaded flag, memory_usage_mb
    - Serve generated media files (audio chunks, video segments) as static assets
    - _Requirements: 1.4, 2.7, 6.1, 6.2, 9.6, 10.4, 10.7_

  - [ ]* 8.3 Write unit tests for the Orchestrator
    - Mock all sub-components; test full pipeline flow, error propagation, and structured logging output
    - _Requirements: 9.1, 9.2, 9.3_

- [x] 9. Checkpoint
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 10. Implement the React frontend (Chat_Interface)
  - [x] 10.1 Create the main application layout and routing
    - Set up React + TypeScript app with Vite
    - Create responsive layout shell (320px–1920px) with avatar display area, chat area, and file upload area
    - _Requirements: 6.1, 6.6, 6.7_

  - [x] 10.2 Implement avatar image upload component
    - File input accepting PNG/JPG/JPEG, client-side validation for format, size (≤10 MB), and resolution (≥256×256)
    - Call `POST /api/upload/avatar`, display image preview on success, show error messages on validation failure or missing face
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [x] 10.3 Implement PDF document upload component
    - File input accepting PDF, client-side validation for size (≤50 MB)
    - Call `POST /api/upload/pdf`, display document name and page count on success, show error messages on failure
    - _Requirements: 2.6, 2.7, 6.1_

  - [x] 10.4 Implement chat interface with question input and history
    - Text input field for submitting questions
    - Chat history panel showing all user questions and avatar text responses
    - Connect to `POST /api/ask` SSE stream; display text tokens progressively as they arrive (typewriter effect)
    - _Requirements: 6.2, 6.4, 10.1_

  - [x] 10.5 Implement avatar video playback and processing indicators
    - Video player component that plays generated avatar video segments with synchronized audio, queuing and auto-playing chunks as they arrive from the SSE stream
    - Implement a media segment queue: as audio/video chunk URLs arrive via SSE, append them to a playback queue and begin playback of the first segment immediately
    - Loading indicator showing current processing stage (retrieving, generating, synthesizing, animating) updated in real time from SSE stage events
    - _Requirements: 6.3, 6.5, 10.6_

  - [ ]* 10.6 Write unit tests for frontend components
    - Test upload validation, API call integration, chat history rendering, and responsive layout
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

- [ ] 11. Implement RAG evaluation module
  - [x] 11.1 Implement `EvaluationModule` class using RAGAS
    - Implement `evaluate_single()` computing faithfulness, context_relevance, and answer_relevance scores
    - Implement `evaluate_dataset()` to run evaluation against a configurable JSON test dataset
    - Output results in structured JSON format
    - Use Ollama LLM as the local evaluator (no external API calls)
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

  - [x] 11.2 Create a CLI benchmark command for evaluation
    - Add a CLI entry point (e.g., `python -m backend.eval`) that runs `evaluate_dataset()` against a test dataset file and prints JSON results
    - Include a sample `test_dataset.json` with example question-answer pairs
    - _Requirements: 8.6_

  - [ ]* 11.3 Write unit tests for `EvaluationModule`
    - Mock RAGAS scoring; test JSON output format, dataset loading, and metric computation
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 9.3_

- [x] 12. Checkpoint
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 13. Docker deployment and edge optimization
  - [x] 13.1 Create Dockerfiles and docker-compose configuration
    - Multi-stage Dockerfile for the Python backend (slim base image, install dependencies, copy code)
    - Dockerfile for the React frontend (build stage + nginx serve stage)
    - `docker-compose.yml` orchestrating backend, frontend, and Ollama services
    - Support both x86_64 and ARM64 architectures
    - Single-command deployment: `docker compose up`
    - _Requirements: 7.1, 7.5, 7.6_

  - [x] 13.2 Configure edge device resource constraints
    - Set Docker memory limits to stay below 6 GB peak usage
    - Ensure all models use quantized variants (Llama 3.2 Q4_K_M, Piper ONNX, MiniLM)
    - Verify total model disk usage stays below 10 GB
    - Document minimum hardware requirements (8 GB RAM, 4-core CPU)
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [ ] 14. Open-source compliance and API documentation
  - [x] 14.1 Create dependency manifest and license documentation
    - Generate a `LICENSES.md` or `dependencies.json` listing all models and libraries with their license identifiers (Apache 2.0, MIT, BSD, Llama Community License)
    - Verify no paid API keys or cloud services are required for core functionality
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

  - [x] 14.2 Generate API documentation
    - Ensure FastAPI auto-generates OpenAPI/Swagger docs at `/docs`
    - Add docstrings and descriptions to all endpoint definitions
    - _Requirements: 9.6_

- [ ] 15. Integration tests and final wiring
  - [ ]* 15.1 Write integration tests for the end-to-end flow
    - Test the full pipeline from question submission through to avatar video generation (mocking heavy AI models where needed)
    - Verify PDF upload → chunking → embedding → retrieval → generation → TTS → animation → response delivery
    - _Requirements: 9.4_

  - [x] 15.2 Final wiring and smoke test
    - Verify all components are connected: frontend calls backend endpoints, backend orchestrates all services
    - Ensure media files (audio, video) are served correctly and playable in the browser
    - Verify health endpoint reports correct status
    - _Requirements: 6.3, 6.7, 9.1_

- [x] 16. Final checkpoint
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirement acceptance criteria for traceability
- Checkpoints at tasks 5, 9, 12, and 16 ensure incremental validation
- The backend uses Python (FastAPI), the frontend uses TypeScript (React + Vite)
- All AI models are open-source and sized for edge devices with 8 GB RAM
