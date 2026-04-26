"""Edge-TTS engine for high-quality speech synthesis.

Uses Microsoft's free Edge TTS API for natural-sounding speech.
Falls back to Piper TTS if Edge-TTS is unavailable.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from backend.app.logging_utils import get_logger
from backend.app.models import AudioResult

logger = get_logger("edge_tts_engine")

VOICE_EN = "en-IN-PrabhatNeural"


async def _generate_audio_async(text: str, output_path: str, voice: str = VOICE_EN) -> None:
    """Generate audio using edge-tts."""
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def generate_edge_tts(text: str, output_path: str, voice: str = VOICE_EN) -> AudioResult | None:
    """Generate MP3 audio from text using Edge-TTS.

    Parameters
    ----------
    text:
        Text to synthesize.
    output_path:
        Path to write the MP3 file.
    voice:
        Edge-TTS voice name.

    Returns
    -------
    AudioResult or None if synthesis fails.
    """
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        asyncio.run(_generate_audio_async(text, output_path, voice))

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            logger.error("Edge-TTS produced empty output")
            return None

        # Get duration using pydub if available, else estimate
        duration = _get_mp3_duration(output_path)

        logger.info("Edge-TTS: %.2fs audio → %s", duration, output_path)
        return AudioResult(
            file_path=output_path,
            duration_seconds=duration,
            sample_rate=24000,
            format="mp3",
        )
    except Exception as e:
        logger.error("Edge-TTS failed: %s", e)
        return None


def _get_mp3_duration(path: str) -> float:
    """Get MP3 duration in seconds."""
    try:
        from pydub import AudioSegment
        sound = AudioSegment.from_mp3(path)
        return sound.duration_seconds
    except Exception:
        # Rough estimate: ~128kbps MP3
        size = os.path.getsize(path)
        return max(size / 16000, 0.5)
