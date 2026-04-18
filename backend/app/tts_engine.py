"""Text-to-Speech engine using Piper TTS (ONNX runtime).

Converts text into WAV audio files using the Piper VITS-based model.
Supports full-text synthesis and sentence-level chunked synthesis for
streaming pipelines.
"""

from __future__ import annotations

import re
import wave
from pathlib import Path

from backend.app.config import AppConfig
from backend.app.logging_utils import get_logger
from backend.app.models import AudioResult

logger = get_logger("tts_engine")

# Regex to strip characters that Piper may not handle.
# Keep letters, digits, basic punctuation, and whitespace.
_SUPPORTED_CHARS = re.compile(r"[^\w\s.,!?;:'\"\-\(\)\n]", re.UNICODE)


class TTSEngineError(Exception):
    """Raised when TTS synthesis fails."""


class TTSEngine:
    """Piper TTS wrapper for WAV audio synthesis.

    Parameters
    ----------
    config:
        Application configuration providing ``tts_model_path`` and
        ``tts_sample_rate``.
    """

    def __init__(self, config: AppConfig) -> None:
        self._model_path = config.tts_model_path
        self._sample_rate = config.tts_sample_rate
        self._voice = None

        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the Piper ONNX voice model from disk."""
        model_path = Path(self._model_path)
        if not model_path.is_file():
            raise TTSEngineError(
                f"Piper model not found at '{self._model_path}'. "
                "Download a model from https://github.com/rhasspy/piper and "
                "place the .onnx file at the configured tts_model_path."
            )

        try:
            from piper.voice import PiperVoice  # type: ignore[import-untyped]
        except ImportError as exc:
            raise TTSEngineError(
                "piper-tts is not installed. Install it with: "
                "pip install piper-tts"
            ) from exc

        try:
            self._voice = PiperVoice.load(str(model_path))
            logger.info("Piper model loaded from %s", self._model_path)
        except Exception as exc:
            raise TTSEngineError(
                f"Failed to load Piper model from '{self._model_path}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize(self, text: str, output_path: str) -> AudioResult:
        """Convert *text* to a WAV file at *output_path*.

        Parameters
        ----------
        text:
            The full text to synthesize.
        output_path:
            Destination path for the generated ``.wav`` file.

        Returns
        -------
        AudioResult
            Metadata about the generated audio.
        """
        clean = self._sanitize(text)
        if not clean.strip():
            return self._write_silent_wav(output_path)

        return self._synthesize_to_wav(clean, output_path)

    def synthesize_chunk(self, text: str, output_path: str) -> AudioResult:
        """Synthesize a single sentence / chunk to WAV.

        Intended for pipelined streaming: call this for each sentence as
        it arrives from the LLM stream.

        Parameters
        ----------
        text:
            A single sentence or short text chunk.
        output_path:
            Destination path for the generated ``.wav`` file.

        Returns
        -------
        AudioResult
            Metadata about the generated audio chunk.
        """
        clean = self._sanitize(text)
        if not clean.strip():
            return self._write_silent_wav(output_path)

        return self._synthesize_to_wav(clean, output_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sanitize(self, text: str) -> str:
        """Strip unsupported characters so Piper doesn't choke."""
        return _SUPPORTED_CHARS.sub("", text)

    def _synthesize_to_wav(self, text: str, output_path: str) -> AudioResult:
        """Run Piper synthesis and write a WAV file."""
        if self._voice is None:
            raise TTSEngineError("Piper model is not loaded.")

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        try:
            with wave.open(str(out), "wb") as wav_file:
                # synthesize_wav handles WAV header setup when set_wav_format=True
                self._voice.synthesize_wav(text, wav_file, set_wav_format=True)
        except Exception as exc:
            raise TTSEngineError(
                f"Piper synthesis failed: {exc}"
            ) from exc

        duration = self._wav_duration(str(out))
        logger.info(
            "Synthesized %.2fs of audio -> %s", duration, output_path
        )

        return AudioResult(
            file_path=str(out),
            duration_seconds=duration,
            sample_rate=self._sample_rate,
            format="wav",
        )

    def _write_silent_wav(self, output_path: str) -> AudioResult:
        """Write a minimal silent WAV when there is nothing to synthesize."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        num_frames = self._sample_rate  # 1 second of silence
        with wave.open(str(out), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self._sample_rate)
            wav_file.writeframes(b"\x00\x00" * num_frames)

        return AudioResult(
            file_path=str(out),
            duration_seconds=1.0,
            sample_rate=self._sample_rate,
            format="wav",
        )

    @staticmethod
    def _wav_duration(path: str) -> float:
        """Return the duration in seconds of a WAV file."""
        with wave.open(path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate == 0:
                return 0.0
            return frames / rate
