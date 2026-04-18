"""Shared data models for the Live Talking Head Avatar system."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PageContent:
    page_number: int
    text: str
    tables: list[str]  # Markdown-formatted tables


@dataclass
class ParsedDocument:
    text: str
    pages: list[PageContent]
    page_count: int
    metadata: dict


@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    document_id: str
    page_number: int
    token_count: int
    start_char: int
    end_char: int


@dataclass
class RetrievalResult:
    chunk: DocumentChunk
    score: float
    distance: float


@dataclass
class GenerationResult:
    answer: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    duration_ms: float


@dataclass
class AudioResult:
    file_path: str
    duration_seconds: float
    sample_rate: int
    format: str  # "wav"


@dataclass
class AvatarProfile:
    avatar_id: str
    image_path: str
    landmarks: dict
    preprocessed_at: str


@dataclass
class VideoResult:
    file_path: str
    duration_seconds: float
    fps: int
    resolution: tuple[int, int]
    format: str  # "mp4"


@dataclass
class ProcessingStage:
    name: str
    duration_ms: float
    status: str  # "success" | "error"


@dataclass
class OrchestratorResponse:
    answer: str
    audio_url: str
    video_url: str
    sources: list[RetrievalResult]
    stages: list[ProcessingStage]


@dataclass
class EvalResult:
    question: str
    faithfulness: float
    context_relevance: float
    answer_relevance: float
    metadata: dict


@dataclass
class StreamEvent:
    type: str  # "text_token" | "audio_chunk" | "video_chunk" | "stage_update" | "error" | "done"
    data: dict  # payload varies by type
