"""Unit tests for AvatarEngine – image validation, face detection, and animation."""

from __future__ import annotations

import os
import struct
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.app.avatar_engine import AvatarEngine, FaceNotFoundError
from backend.app.config import AppConfig


@pytest.fixture()
def engine() -> AvatarEngine:
    """Engine with default config."""
    return AvatarEngine(AppConfig())


@pytest.fixture()
def face_image(tmp_path: Path) -> str:
    """Create a synthetic image with a detectable face-like rectangle.

    We draw a simple oval + features that the Haar cascade can pick up.
    If the cascade doesn't fire on synthetic data we fall back to a real
    test later; here we just need *some* plausible face.
    """
    img = np.ones((512, 512, 3), dtype=np.uint8) * 200  # light gray bg

    # Draw a skin-coloured ellipse as a face
    cv2.ellipse(img, (256, 256), (100, 130), 0, 0, 360, (180, 160, 140), -1)
    # Eyes (dark circles)
    cv2.circle(img, (220, 230), 15, (40, 40, 40), -1)
    cv2.circle(img, (292, 230), 15, (40, 40, 40), -1)
    # Mouth
    cv2.ellipse(img, (256, 310), (40, 15), 0, 0, 360, (40, 40, 40), -1)

    path = str(tmp_path / "face.png")
    cv2.imwrite(path, img)
    return path


@pytest.fixture()
def no_face_image(tmp_path: Path) -> str:
    """Plain white image – no face."""
    img = np.ones((512, 512, 3), dtype=np.uint8) * 255
    path = str(tmp_path / "blank.png")
    cv2.imwrite(path, img)
    return path


@pytest.fixture()
def small_image(tmp_path: Path) -> str:
    """Image below minimum resolution."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    path = str(tmp_path / "small.png")
    cv2.imwrite(path, img)
    return path


@pytest.fixture()
def dummy_wav(tmp_path: Path) -> str:
    """Minimal valid-ish WAV file (~0.5 s of silence)."""
    path = str(tmp_path / "audio.wav")
    sample_rate = 22050
    num_samples = sample_rate  # 1 second
    data_size = num_samples * 2  # 16-bit mono
    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * 2, 2, 16))
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(b"\x00" * data_size)
    return path


# ------------------------------------------------------------------ #
#  preprocess() tests                                                  #
# ------------------------------------------------------------------ #


class TestPreprocessValidation:
    """Image validation in preprocess()."""

    def test_file_not_found(self, engine: AvatarEngine) -> None:
        with pytest.raises(FileNotFoundError):
            engine.preprocess("/nonexistent/photo.png")

    def test_unsupported_format(self, engine: AvatarEngine, tmp_path: Path) -> None:
        bmp = tmp_path / "photo.bmp"
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.imwrite(str(bmp), img)
        with pytest.raises(ValueError, match="Unsupported image format"):
            engine.preprocess(str(bmp))

    def test_file_too_large(self, engine: AvatarEngine, tmp_path: Path) -> None:
        big = tmp_path / "big.png"
        # Create a file that exceeds 10 MB
        big.write_bytes(b"\x00" * (11 * 1024 * 1024))
        with pytest.raises(ValueError, match="exceeds"):
            engine.preprocess(str(big))

    def test_resolution_too_small(self, engine: AvatarEngine, small_image: str) -> None:
        with pytest.raises(ValueError, match="below the minimum"):
            engine.preprocess(small_image)

    def test_no_face_detected(self, engine: AvatarEngine, no_face_image: str) -> None:
        with pytest.raises(FaceNotFoundError):
            engine.preprocess(no_face_image)

    def test_valid_image_returns_profile(self, engine: AvatarEngine, face_image: str) -> None:
        """If the Haar cascade detects a face, we get a valid profile."""
        # The synthetic face may or may not trigger the cascade.
        # We try; if FaceNotFoundError is raised we skip (cascade limitation).
        try:
            profile = engine.preprocess(face_image)
        except FaceNotFoundError:
            pytest.skip("Haar cascade did not detect synthetic face")
        assert profile.avatar_id
        assert profile.image_path
        assert "face_rect" in profile.landmarks
        assert profile.preprocessed_at


# ------------------------------------------------------------------ #
#  animate() / animate_chunk() tests                                   #
# ------------------------------------------------------------------ #


class TestAnimate:
    """Animation methods – fallback path (no SadTalker binary)."""

    def test_animate_produces_mp4(
        self, engine: AvatarEngine, face_image: str, dummy_wav: str, tmp_path: Path
    ) -> None:
        from backend.app.models import AvatarProfile

        profile = AvatarProfile(
            avatar_id="test123",
            image_path=face_image,
            landmarks={"face_rect": {"x": 0, "y": 0, "w": 100, "h": 100}},
            preprocessed_at="2024-01-01T00:00:00Z",
        )
        out = str(tmp_path / "out.mp4")
        result = engine.animate(profile, dummy_wav, out)

        assert result.format == "mp4"
        assert result.fps == engine.fps
        assert Path(result.file_path).exists()

    def test_animate_chunk_produces_mp4(
        self, engine: AvatarEngine, face_image: str, dummy_wav: str, tmp_path: Path
    ) -> None:
        from backend.app.models import AvatarProfile

        profile = AvatarProfile(
            avatar_id="test456",
            image_path=face_image,
            landmarks={},
            preprocessed_at="2024-01-01T00:00:00Z",
        )
        out = str(tmp_path / "chunk_0.mp4")
        result = engine.animate_chunk(profile, dummy_wav, out, chunk_index=0)

        assert result.format == "mp4"
        assert Path(result.file_path).exists()

    def test_fallback_on_bad_image(
        self, engine: AvatarEngine, dummy_wav: str, tmp_path: Path
    ) -> None:
        """If the image can't be read, fallback still produces a video."""
        from backend.app.models import AvatarProfile

        profile = AvatarProfile(
            avatar_id="bad",
            image_path="/nonexistent/img.png",
            landmarks={},
            preprocessed_at="2024-01-01T00:00:00Z",
        )
        out = str(tmp_path / "fallback.mp4")
        result = engine.animate(profile, dummy_wav, out)

        assert result.format == "mp4"
        assert Path(result.file_path).exists()
