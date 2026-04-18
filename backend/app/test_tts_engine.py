"""Unit tests for TTSEngine – model loading, synthesis, WAV format, and sanitization.

Piper TTS model files are not available in the test environment, so
PiperVoice is mocked throughout.

Validates: Requirements 4.1, 4.2, 4.5, 9.3
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.app.config import AppConfig
from backend.app.tts_engine import TTSEngine, TTSEngineError, _SUPPORTED_CHARS


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #


def _make_model_file(tmp_path: Path) -> str:
    """Create a dummy .onnx file so the file-existence check passes."""
    model = tmp_path / "model.onnx"
    model.write_bytes(b"\x00" * 64)
    return str(model)


def _config_with_model(model_path: str) -> AppConfig:
    """Return an AppConfig pointing at the given model path."""
    cfg = AppConfig()
    cfg.tts_model_path = model_path
    return cfg


def _build_engine(tmp_path: Path) -> tuple[TTSEngine, MagicMock]:
    """Construct a TTSEngine with a fully mocked PiperVoice.

    Returns the engine and the mock voice instance so callers can
    configure synthesis behaviour.
    """
    model_path = _make_model_file(tmp_path)
    cfg = _config_with_model(model_path)

    mock_voice = MagicMock()

    # By default, synthesize_wav writes a short valid WAV so duration
    # helpers work.  Individual tests can override this.
    def _write_wav(text: str, wav_file, set_wav_format: bool = True):
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(cfg.tts_sample_rate)
        num_frames = cfg.tts_sample_rate  # 1 second of audio
        wav_file.writeframes(b"\x00\x00" * num_frames)

    mock_voice.synthesize_wav.side_effect = _write_wav

    with patch("backend.app.tts_engine.PiperVoice", create=True) as _:
        # We need to patch the import inside _load_model
        with patch.dict(
            "sys.modules",
            {"piper": MagicMock(), "piper.voice": MagicMock()},
        ):
            with patch(
                "backend.app.tts_engine.TTSEngine._load_model"
            ) as mock_load:
                engine = TTSEngine(cfg)
                engine._voice = mock_voice
    return engine, mock_voice


# ------------------------------------------------------------------ #
#  Fixtures                                                            #
# ------------------------------------------------------------------ #


@pytest.fixture()
def engine_and_voice(tmp_path: Path) -> tuple[TTSEngine, MagicMock]:
    return _build_engine(tmp_path)


@pytest.fixture()
def engine(engine_and_voice: tuple[TTSEngine, MagicMock]) -> TTSEngine:
    return engine_and_voice[0]


# ------------------------------------------------------------------ #
#  Model loading tests                                                 #
# ------------------------------------------------------------------ #


class TestModelLoading:
    """TTSEngine._load_model behaviour."""

    def test_missing_model_raises_error(self, tmp_path: Path) -> None:
        """Req 9.3 – TTSEngineError when model file does not exist."""
        cfg = _config_with_model("/nonexistent/model.onnx")
        with pytest.raises(TTSEngineError, match="not found"):
            TTSEngine(cfg)

    def test_load_model_success_with_mock(self, tmp_path: Path) -> None:
        """Model loads successfully when file exists and PiperVoice is available."""
        model_path = _make_model_file(tmp_path)
        cfg = _config_with_model(model_path)

        mock_voice_cls = MagicMock()
        mock_voice_cls.load.return_value = MagicMock()

        with patch.dict(
            "sys.modules",
            {
                "piper": MagicMock(),
                "piper.voice": MagicMock(PiperVoice=mock_voice_cls),
            },
        ):
            engine = TTSEngine(cfg)

        assert engine._voice is not None
        mock_voice_cls.load.assert_called_once_with(model_path)


# ------------------------------------------------------------------ #
#  synthesize() tests                                                  #
# ------------------------------------------------------------------ #


class TestSynthesize:
    """TTSEngine.synthesize() – full text to WAV."""

    def test_produces_wav_file(
        self, engine: TTSEngine, tmp_path: Path
    ) -> None:
        """Req 4.1 – output is a WAV file."""
        out = str(tmp_path / "speech.wav")
        result = engine.synthesize("Hello world", out)

        assert result.format == "wav"
        assert Path(result.file_path).exists()
        assert result.file_path == out

    def test_sample_rate_at_least_22050(
        self, engine: TTSEngine, tmp_path: Path
    ) -> None:
        """Req 4.2 – sample rate ≥ 22050 Hz."""
        out = str(tmp_path / "speech.wav")
        result = engine.synthesize("Test sample rate", out)

        assert result.sample_rate >= 22050

        # Also verify the actual WAV header
        with wave.open(out, "rb") as wf:
            assert wf.getframerate() >= 22050

    def test_wav_is_mono_16bit(
        self, engine: TTSEngine, tmp_path: Path
    ) -> None:
        """WAV output is mono, 16-bit PCM."""
        out = str(tmp_path / "speech.wav")
        engine.synthesize("Check format", out)

        with wave.open(out, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2  # 16-bit

    def test_duration_is_positive(
        self, engine: TTSEngine, tmp_path: Path
    ) -> None:
        """Synthesized audio has a positive duration."""
        out = str(tmp_path / "speech.wav")
        result = engine.synthesize("Some text", out)

        assert result.duration_seconds > 0

    def test_creates_parent_directories(
        self, engine: TTSEngine, tmp_path: Path
    ) -> None:
        """Output directories are created automatically."""
        out = str(tmp_path / "nested" / "dir" / "speech.wav")
        result = engine.synthesize("Hello", out)

        assert Path(result.file_path).exists()

    def test_voice_not_loaded_raises_error(self, tmp_path: Path) -> None:
        """TTSEngineError when voice is None (model not loaded)."""
        engine, _ = _build_engine(tmp_path)
        engine._voice = None

        out = str(tmp_path / "fail.wav")
        with pytest.raises(TTSEngineError, match="not loaded"):
            engine.synthesize("Hello", out)


# ------------------------------------------------------------------ #
#  synthesize_chunk() tests                                            #
# ------------------------------------------------------------------ #


class TestSynthesizeChunk:
    """TTSEngine.synthesize_chunk() – sentence-level synthesis."""

    def test_produces_wav_file(
        self, engine: TTSEngine, tmp_path: Path
    ) -> None:
        """Req 4.1 – chunk synthesis also produces WAV."""
        out = str(tmp_path / "chunk_0.wav")
        result = engine.synthesize_chunk("A single sentence.", out)

        assert result.format == "wav"
        assert Path(result.file_path).exists()

    def test_sample_rate_matches_config(
        self, engine: TTSEngine, tmp_path: Path
    ) -> None:
        """Req 4.2 – chunk audio sample rate matches configured rate."""
        out = str(tmp_path / "chunk_0.wav")
        result = engine.synthesize_chunk("Check rate.", out)

        assert result.sample_rate == engine._sample_rate


# ------------------------------------------------------------------ #
#  Unsupported character sanitization                                  #
# ------------------------------------------------------------------ #


class TestSanitization:
    """Req 4.5 – unsupported characters are stripped, synthesis continues."""

    def test_strips_emoji(self, engine: TTSEngine, tmp_path: Path) -> None:
        out = str(tmp_path / "emoji.wav")
        result = engine.synthesize("Hello 🌍 world!", out)

        assert result.format == "wav"
        assert Path(result.file_path).exists()
        # Voice should have been called with cleaned text (no emoji)
        call_args = engine._voice.synthesize_wav.call_args
        synthesized_text = call_args[0][0]
        assert "🌍" not in synthesized_text

    def test_strips_special_symbols(
        self, engine: TTSEngine, tmp_path: Path
    ) -> None:
        out = str(tmp_path / "symbols.wav")
        result = engine.synthesize("Price is 5€ or £3", out)

        assert result.format == "wav"
        call_args = engine._voice.synthesize_wav.call_args
        synthesized_text = call_args[0][0]
        assert "€" not in synthesized_text
        assert "£" not in synthesized_text

    def test_preserves_basic_punctuation(self) -> None:
        """Basic punctuation should survive sanitization."""
        text = "Hello, world! How are you? Fine; thanks."
        cleaned = _SUPPORTED_CHARS.sub("", text)
        assert cleaned == text

    def test_preserves_alphanumeric(self) -> None:
        """Letters and digits are kept."""
        text = "The answer is 42."
        cleaned = _SUPPORTED_CHARS.sub("", text)
        assert cleaned == text

    def test_sanitize_method_directly(self, engine: TTSEngine) -> None:
        """_sanitize strips unsupported chars and keeps the rest."""
        result = engine._sanitize("Hello™ world© 2024®")
        assert "™" not in result
        assert "©" not in result
        assert "®" not in result
        assert "Hello" in result
        assert "world" in result
        assert "2024" in result


# ------------------------------------------------------------------ #
#  Empty / whitespace-only text → silent WAV                           #
# ------------------------------------------------------------------ #


class TestEmptyText:
    """Empty or whitespace-only input produces a silent WAV."""

    def test_empty_string_produces_silent_wav(
        self, engine: TTSEngine, tmp_path: Path
    ) -> None:
        out = str(tmp_path / "silent.wav")
        result = engine.synthesize("", out)

        assert result.format == "wav"
        assert Path(result.file_path).exists()
        assert result.duration_seconds > 0

        # Voice should NOT have been called
        engine._voice.synthesize_wav.assert_not_called()

        # Verify WAV properties
        with wave.open(out, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() >= 22050

    def test_whitespace_only_produces_silent_wav(
        self, engine: TTSEngine, tmp_path: Path
    ) -> None:
        out = str(tmp_path / "ws.wav")
        result = engine.synthesize("   \n\t  ", out)

        assert result.format == "wav"
        engine._voice.synthesize_wav.assert_not_called()

    def test_only_unsupported_chars_produces_silent_wav(
        self, engine: TTSEngine, tmp_path: Path
    ) -> None:
        """Text that is entirely unsupported chars → silent WAV."""
        out = str(tmp_path / "unsupported.wav")
        result = engine.synthesize("🌍🎉💯", out)

        assert result.format == "wav"
        assert Path(result.file_path).exists()
        engine._voice.synthesize_wav.assert_not_called()

    def test_empty_chunk_produces_silent_wav(
        self, engine: TTSEngine, tmp_path: Path
    ) -> None:
        out = str(tmp_path / "empty_chunk.wav")
        result = engine.synthesize_chunk("", out)

        assert result.format == "wav"
        engine._voice.synthesize_wav.assert_not_called()
