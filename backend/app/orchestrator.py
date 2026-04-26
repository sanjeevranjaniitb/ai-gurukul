"""Orchestrator coordinating the streaming pipeline.

Two modes:
- **animated**: Stream text + TTS per sentence → client-side viseme animation (fast)
- **real**: Stream text → collect full answer → single TTS → single Wav2Lip video (high quality)
"""

from __future__ import annotations

import asyncio
import re
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncIterator

from backend.app.avatar_engine import AvatarEngine
from backend.app.chunking import ChunkingModule
from backend.app.config import AppConfig
from backend.app.embedding_store import EmbeddingStore
from backend.app.llm_service import LLMService
from backend.app.logging_utils import get_logger, set_correlation_id
from backend.app.models import AvatarProfile, ProcessingStage, StreamEvent
from backend.app.pdf_parser import PDFParser
from backend.app.tts_engine import TTSEngine
from backend.app.viseme_engine import VisemeEngine, preprocess_avatar_frame

logger = get_logger("orchestrator")

_SENTENCE_END = re.compile(r"(?<=[.!?])(?:\s|$)")


class Orchestrator:

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._executor = ThreadPoolExecutor(max_workers=2)

        self._embedding_store: EmbeddingStore | None = None
        self._llm_service: LLMService | None = None
        self._tts_engine: TTSEngine | None = None
        self._avatar_engine: AvatarEngine | None = None
        self._pdf_parser: PDFParser | None = None
        self._chunking: ChunkingModule | None = None
        self._viseme_engine: VisemeEngine | None = None

        self._avatars: dict[str, AvatarProfile] = {}
        self._avatar_visemes: dict[str, dict[str, str]] = {}
        self._avatar_frames: dict[str, str | None] = {}

    # -- Lazy accessors --

    @property
    def embedding_store(self) -> EmbeddingStore:
        if self._embedding_store is None:
            self._embedding_store = EmbeddingStore(
                collection_name="documents",
                persist_directory=self._config.chroma_persist_dir,
                embedding_model=self._config.embedding_model,
            )
        return self._embedding_store

    @property
    def llm_service(self) -> LLMService:
        if self._llm_service is None:
            self._llm_service = LLMService(config=self._config)
        return self._llm_service

    @property
    def tts_engine(self) -> TTSEngine:
        if self._tts_engine is None:
            self._tts_engine = TTSEngine(config=self._config)
        return self._tts_engine

    @property
    def avatar_engine(self) -> AvatarEngine:
        if self._avatar_engine is None:
            self._avatar_engine = AvatarEngine(config=self._config)
        return self._avatar_engine

    @property
    def pdf_parser(self) -> PDFParser:
        if self._pdf_parser is None:
            self._pdf_parser = PDFParser(config=self._config)
        return self._pdf_parser

    @property
    def chunking(self) -> ChunkingModule:
        if self._chunking is None:
            self._chunking = ChunkingModule(config=self._config)
        return self._chunking

    @property
    def viseme_engine(self) -> VisemeEngine:
        if self._viseme_engine is None:
            self._viseme_engine = VisemeEngine()
        return self._viseme_engine

    # -- Avatar upload --

    async def upload_avatar(self, file_path: str | Path, avatar_id: str | None = None) -> AvatarProfile:
        start = time.monotonic()
        profile = self.avatar_engine.preprocess(str(file_path))
        if avatar_id is not None:
            profile = AvatarProfile(avatar_id=avatar_id, image_path=profile.image_path,
                                    landmarks=profile.landmarks, preprocessed_at=profile.preprocessed_at)
        self._avatars[profile.avatar_id] = profile

        avatar_dir = f"data/avatars/{profile.avatar_id}"
        frame_path = f"{avatar_dir}/frame.jpg"
        preprocessed = preprocess_avatar_frame(profile.image_path, frame_path) or profile.image_path

        viseme_dir = f"{avatar_dir}/visemes"
        viseme_paths = self.viseme_engine.generate_visemes(preprocessed, viseme_dir)

        # Convert viseme images to base64 data URLs for instant client-side swapping
        import base64
        viseme_data_urls = {}
        for name, path in viseme_paths.items():
            try:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                viseme_data_urls[name] = f"data:image/jpeg;base64,{b64}"
            except Exception:
                viseme_data_urls[name] = ""
        self._avatar_visemes[profile.avatar_id] = viseme_data_urls

        try:
            self._avatar_frames[profile.avatar_id] = "/api/data/" + str(Path(frame_path).relative_to("data"))
        except ValueError:
            self._avatar_frames[profile.avatar_id] = None

        logger.info("upload_avatar completed – %s, %.1fms", profile.avatar_id, (time.monotonic() - start) * 1000)
        return profile

    def get_viseme_urls(self, avatar_id: str) -> dict[str, str]:
        return self._avatar_visemes.get(avatar_id, {})

    def get_frame_url(self, avatar_id: str) -> str | None:
        return self._avatar_frames.get(avatar_id)

    # -- PDF upload --

    async def upload_pdf(self, file_path: str | Path, document_id: str | None = None) -> dict:
        start = time.monotonic()
        if document_id is None:
            document_id = uuid.uuid4().hex[:12]
        name = Path(str(file_path)).name
        parsed = self.pdf_parser.parse(str(file_path))
        chunks = self.chunking.chunk(parsed, document_id=document_id)
        self.embedding_store.add_chunks(chunks)
        logger.info("upload_pdf completed – %s, %d pages, %d chunks, %.1fms",
                     document_id, parsed.page_count, len(chunks), (time.monotonic() - start) * 1000)
        return {"document_id": document_id, "name": name,
                "page_count": parsed.page_count, "chunk_count": len(chunks)}

    # ==================================================================
    # MAIN PIPELINE
    # ==================================================================

    async def process_question_stream(
        self, question: str, avatar_id: str, mode: str = "animated"
    ) -> AsyncIterator[StreamEvent]:
        correlation_id = uuid.uuid4().hex[:16]
        set_correlation_id(correlation_id)
        pipeline_start = time.monotonic()

        try:
            profile = self._avatars.get(avatar_id)
            if profile is None:
                yield StreamEvent(type="error", data={"message": f"Avatar '{avatar_id}' not found."})
                return

            # --- Retrieval ---
            yield StreamEvent(type="stage_update", data={"stage": "retrieving", "status": "started"})
            context = self.embedding_store.search(question, top_k=self._config.retrieval_top_k)
            context = [r for r in context if r.score >= 0.35]
            yield StreamEvent(type="stage_update", data={"stage": "retrieving", "status": "completed"})

            # --- Generation (stream tokens) ---
            yield StreamEvent(type="stage_update", data={"stage": "generating", "status": "started"})

            full_answer = ""
            sentence_buffer = ""
            chunk_index = 0

            for token in self.llm_service.generate_stream(question, context):
                if token is None:
                    continue
                full_answer += token
                sentence_buffer += token
                yield StreamEvent(type="text_token", data={"token": token})

                # In animated mode, synthesize audio per sentence as they complete
                if mode == "animated":
                    sentences = _split_sentences(sentence_buffer)
                    if len(sentences) > 1:
                        for s in sentences[:-1]:
                            s = s.strip()
                            if not s:
                                continue
                            async for ev in self._tts_sentence(s, chunk_index, correlation_id):
                                yield ev
                            chunk_index += 1
                        sentence_buffer = sentences[-1]

            # Flush remaining sentence (animated mode)
            if mode == "animated":
                remaining = sentence_buffer.strip()
                if remaining:
                    async for ev in self._tts_sentence(remaining, chunk_index, correlation_id):
                        yield ev

            yield StreamEvent(type="stage_update", data={"stage": "generating", "status": "completed"})

            # --- Real mode: batch TTS + Wav2Lip for the FULL answer ---
            if mode == "real" and full_answer.strip():
                async for ev in self._real_lipsync(full_answer.strip(), profile, correlation_id):
                    yield ev

            # --- Sources ---
            if context:
                yield StreamEvent(type="sources", data={
                    "sources": [{"chunk_text": r.chunk.text[:200], "page": r.chunk.page_number,
                                 "score": round(r.score, 4)} for r in context]
                })

            total_ms = (time.monotonic() - pipeline_start) * 1000
            yield StreamEvent(type="done", data={"total_duration_ms": round(total_ms, 1)})

        except Exception:
            logger.error("Pipeline error:\n%s", traceback.format_exc())
            yield StreamEvent(type="error", data={
                "message": "An internal error occurred. Please try again."})

    # ==================================================================
    # ANIMATED MODE: TTS per sentence (client-side viseme animation)
    # ==================================================================

    async def _tts_sentence(
        self, sentence: str, chunk_index: int, correlation_id: str,
    ) -> AsyncIterator[StreamEvent]:
        loop = asyncio.get_event_loop()
        media_dir = Path(self._config.media_output_dir) / correlation_id
        media_dir.mkdir(parents=True, exist_ok=True)

        yield StreamEvent(type="stage_update", data={
            "stage": "synthesizing", "status": "started", "chunk_index": chunk_index})

        audio_path = str(media_dir / f"audio_{chunk_index}.mp3")
        audio_result = None
        try:
            from backend.app.edge_tts_engine import generate_edge_tts
            audio_result = await loop.run_in_executor(self._executor, generate_edge_tts, sentence, audio_path)
        except Exception:
            pass

        if audio_result is None:
            audio_path = str(media_dir / f"audio_{chunk_index}.wav")
            try:
                audio_result = await loop.run_in_executor(
                    self._executor, self.tts_engine.synthesize_chunk, sentence, audio_path)
            except Exception:
                yield StreamEvent(type="stage_update", data={
                    "stage": "synthesizing", "status": "failed", "chunk_index": chunk_index})
                return

        audio_url = "/api/data/" + str(Path(audio_result.file_path).relative_to("data"))
        yield StreamEvent(type="audio_chunk", data={
            "chunk_url": audio_url, "chunk_index": chunk_index,
            "duration_seconds": audio_result.duration_seconds, "sentence": sentence})
        yield StreamEvent(type="stage_update", data={
            "stage": "synthesizing", "status": "completed", "chunk_index": chunk_index})

    # ==================================================================
    # REAL MODE: Full answer → single TTS → single Wav2Lip video
    # ==================================================================

    async def _real_lipsync(
        self, full_text: str, profile: AvatarProfile, correlation_id: str,
    ) -> AsyncIterator[StreamEvent]:
        """Generate one TTS audio + one Wav2Lip video for the entire answer."""
        loop = asyncio.get_event_loop()
        media_dir = Path(self._config.media_output_dir) / correlation_id
        media_dir.mkdir(parents=True, exist_ok=True)

        # --- Step 1: TTS for full answer ---
        yield StreamEvent(type="stage_update", data={"stage": "synthesizing", "status": "started"})

        audio_mp3 = str(media_dir / "full_audio.mp3")
        audio_result = None
        try:
            from backend.app.edge_tts_engine import generate_edge_tts
            audio_result = await loop.run_in_executor(self._executor, generate_edge_tts, full_text, audio_mp3)
        except Exception as e:
            logger.warning("Edge-TTS failed for full answer: %s", e)

        if audio_result is None:
            audio_wav = str(media_dir / "full_audio.wav")
            try:
                audio_result = await loop.run_in_executor(
                    self._executor, self.tts_engine.synthesize_chunk, full_text, audio_wav)
            except Exception as e:
                logger.error("TTS failed completely: %s", e)
                yield StreamEvent(type="stage_update", data={"stage": "synthesizing", "status": "failed"})
                return

        audio_url = "/api/data/" + str(Path(audio_result.file_path).relative_to("data"))
        yield StreamEvent(type="audio_chunk", data={
            "chunk_url": audio_url, "chunk_index": 0,
            "duration_seconds": audio_result.duration_seconds, "sentence": full_text})
        yield StreamEvent(type="stage_update", data={"stage": "synthesizing", "status": "completed"})

        # --- Step 2: Convert MP3 → WAV for Wav2Lip ---
        wav_path = audio_result.file_path
        if audio_result.file_path.endswith(".mp3"):
            wav_path = str(media_dir / "full_audio_converted.wav")
            try:
                await loop.run_in_executor(self._executor,
                                           lambda: _convert_mp3_to_wav(audio_result.file_path, wav_path))
            except Exception as e:
                logger.error("MP3→WAV conversion failed: %s", e)
                yield StreamEvent(type="stage_update", data={"stage": "animating", "status": "failed"})
                return

        # --- Step 3: Wav2Lip video generation ---
        yield StreamEvent(type="stage_update", data={"stage": "animating", "status": "started"})

        video_path = str(media_dir / "full_video.mp4")
        try:
            video_result = await loop.run_in_executor(
                self._executor,
                lambda: self.avatar_engine.animate_chunk(
                    profile=profile, audio_path=wav_path,
                    output_path=video_path, chunk_index=0))

            video_url = "/api/data/" + str(Path(video_result.file_path).relative_to("data"))
            yield StreamEvent(type="video_chunk", data={
                "chunk_url": video_url, "chunk_index": 0,
                "duration_seconds": video_result.duration_seconds})
            yield StreamEvent(type="stage_update", data={"stage": "animating", "status": "completed"})
            logger.info("Real lip-sync video generated: %.2fs", video_result.duration_seconds)

        except Exception as e:
            logger.error("Wav2Lip failed: %s", e)
            yield StreamEvent(type="stage_update", data={"stage": "animating", "status": "failed"})


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _convert_mp3_to_wav(mp3_path: str, wav_path: str) -> None:
    try:
        from pydub import AudioSegment
        sound = AudioSegment.from_mp3(mp3_path)
        sound.export(wav_path, format="wav")
    except ImportError:
        import subprocess
        subprocess.run(["ffmpeg", "-y", "-i", mp3_path, "-ar", "16000", "-ac", "1", wav_path],
                       capture_output=True, timeout=30)


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_END.split(text)
    if not parts:
        return [text]
    return parts
