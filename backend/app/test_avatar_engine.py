"""Unit tests for AvatarEngine – image validation, face detection, and animation.

Validates: Requirements 1.1, 1.2, 1.3, 1.5, 5.1, 5.3, 5.6, 9.3
"""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from backend.app.avatar_engine import AvatarEngine, FaceNotFoundError
from backend.app.config import AppConfig
from backend.app.models import AvatarProfile


# ------------------------------------------------------------------ #
#  Fixtures                                                            #
# ------------------------------------------------------------------ #


@pytest.fixture()
def engine() -> AvatarEngine:
    """Engine with default config."""
    return AvatarEngine(AppConfig())


@pytest.fixture()
def valid_png(tmp_path: Path) -> str:
    """300×300 white PNG image (valid format, meets resolution)."""
    img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    path = str(tmp_path / "photo.png")
    cv2.imwrite(path, img)
    return path


@pytest.fixture()
def valid_jpg(tmp_path: Path) -> str:
    """300×300 white JPG image."""
    img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    path = str(tmp_path / "photo.jpg")
    cv2.imwrite(path, img)
    return path


@pytest.fixture()
def valid_jpeg(tmp_path: Path) -> str:
    """300×300 white JPEG image."""
    img = np.ones((300, 300, 3), dtype=np.uint8) * 255
    path = str(tmp_path / "photo.jpeg")
    cv2.imwrite(path, img)
    return path


@pytest.fixture()
def face_image(tmp_path: Path) -> str:
    """Synthetic image with a face-like pattern for Haar cascade detection.

    Draws an oval face, dark eye regions, and a mouth to maximise the
    chance of the Haar cascade firing on synthetic data.
    """
    img = np.ones((512, 512, 3), dtype=np.uint8) * 200  # light gray bg

    # Skin-coloured ellipse as a face
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
    """Image below minimum resolution (100×100)."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    path = str(tmp_path / "small.png")
    cv2.imwrite(path, img)
    return path


@pytest.fixture()
def dummy_wav(tmp_path: Path) -> str:
    """Minimal valid WAV file (~1 s of silence at 22050 Hz, 16-bit mono)."""
    path = str(tmp_path / "audio.wav")
    sample_rate = 22050
    num_samples = sample_rate  # 1 second
    data_size = num_samples * 2  # 16-bit mono
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))
        f.write(b"WAVE")
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))
        f.write(struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * 2, 2, 16))
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(b"\x00" * data_size)
    return path


def _make_profile(image_path: str, avatar_id: str = "test-id") -> AvatarProfile:
    """Helper to build an AvatarProfile for animation tests."""
    return AvatarProfile(
        avatar_id=avatar_id,
        image_path=image_path,
        landmarks={"face_rect": {"x": 0, "y": 0, "w": 100, "h": 100}},
        preprocessed_at="2024-01-01T00:00:00Z",
    )


# ------------------------------------------------------------------ #
#  preprocess() – image validation (Req 1.1, 1.2, 1.5)                #
# ------------------------------------------------------------------ #


class TestPreprocessValidation:
    """Image format, size, and resolution validation in preprocess()."""

    def test_accepts_png(self, engine: AvatarEngine, tmp_path: Path) -> None:
        """Req 1.1 – PNG format accepted (face detection mocked)."""
        img = np.ones((300, 300, 3), dtype=np.uint8) * 200
        path = str(tmp_path / "photo.png")
        cv2.imwrite(path, img)
        with patch.object(engine, "_face_cascade") as mock_cascade:
            mock_cascade.detectMultiScale.return_value = np.array([[50, 50, 100, 100]])
            profile = engine.preprocess(path)
        assert profile.avatar_id
        assert profile.image_path == str(Path(path).resolve())

    def test_accepts_jpg(self, engine: AvatarEngine, tmp_path: Path) -> None:
        """Req 1.1 – JPG format accepted."""
        img = np.ones((300, 300, 3), dtype=np.uint8) * 200
        path = str(tmp_path / "photo.jpg")
        cv2.imwrite(path, img)
        with patch.object(engine, "_face_cascade") as mock_cascade:
            mock_cascade.detectMultiScale.return_value = np.array([[50, 50, 100, 100]])
            profile = engine.preprocess(path)
        assert profile.avatar_id

    def test_accepts_jpeg(self, engine: AvatarEngine, tmp_path: Path) -> None:
        """Req 1.1 – JPEG format accepted."""
        img = np.ones((300, 300, 3), dtype=np.uint8) * 200
        path = str(tmp_path / "photo.jpeg")
        cv2.imwrite(path, img)
        with patch.object(engine, "_face_cascade") as mock_cascade:
            mock_cascade.detectMultiScale.return_value = np.array([[50, 50, 100, 100]])
            profile = engine.preprocess(path)
        assert profile.avatar_id

    def test_rejects_unsupported_format(self, engine: AvatarEngine, tmp_path: Path) -> None:
        """Req 1.1 – BMP is not in the allowed list."""
        bmp = tmp_path / "photo.bmp"
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.imwrite(str(bmp), img)
        with pytest.raises(ValueError, match="Unsupported image format"):
            engine.preprocess(str(bmp))

    def test_rejects_gif_format(self, engine: AvatarEngine, tmp_path: Path) -> None:
        """Req 1.1 – GIF is not in the allowed list."""
        gif = tmp_path / "photo.gif"
        gif.write_bytes(b"\x00" * 100)
        with pytest.raises(ValueError, match="Unsupported image format"):
            engine.preprocess(str(gif))

    def test_file_too_large(self, engine: AvatarEngine, tmp_path: Path) -> None:
        """Req 1.5 – Files exceeding 10 MB are rejected."""
        big = tmp_path / "big.png"
        big.write_bytes(b"\x00" * (11 * 1024 * 1024))
        with pytest.raises(ValueError, match="exceeds"):
            engine.preprocess(str(big))

    def test_file_exactly_at_limit(self, engine: AvatarEngine, tmp_path: Path) -> None:
        """Req 1.5 – A file exactly at 10 MB should NOT be rejected by size check.

        (It may fail later at image read, but the size gate should pass.)
        """
        borderline = tmp_path / "borderline.png"
        borderline.write_bytes(b"\x00" * (10 * 1024 * 1024))
        # Should not raise the "exceeds" error – it will fail at cv2.imread instead
        with pytest.raises(ValueError, match="Unable to read"):
            engine.preprocess(str(borderline))

    def test_resolution_too_small(self, engine: AvatarEngine, small_image: str) -> None:
        """Req 1.2 – Images below 256×256 are rejected."""
        with pytest.raises(ValueError, match="below the minimum"):
            engine.preprocess(small_image)

    def test_resolution_width_too_small(self, engine: AvatarEngine, tmp_path: Path) -> None:
        """Req 1.2 – Width below 256 while height is fine."""
        img = np.zeros((300, 200, 3), dtype=np.uint8)
        path = str(tmp_path / "narrow.png")
        cv2.imwrite(path, img)
        with pytest.raises(ValueError, match="below the minimum"):
            engine.preprocess(path)

    def test_resolution_height_too_small(self, engine: AvatarEngine, tmp_path: Path) -> None:
        """Req 1.2 – Height below 256 while width is fine."""
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        path = str(tmp_path / "short.png")
        cv2.imwrite(path, img)
        with pytest.raises(ValueError, match="below the minimum"):
            engine.preprocess(path)


# ------------------------------------------------------------------ #
#  preprocess() – face detection (Req 1.3)                             #
# ------------------------------------------------------------------ #


class TestPreprocessFaceDetection:
    """Face detection behaviour in preprocess()."""

    def test_no_face_raises_error(self, engine: AvatarEngine, no_face_image: str) -> None:
        """Req 1.3 – FaceNotFoundError when no face is detected."""
        with pytest.raises(FaceNotFoundError):
            engine.preprocess(no_face_image)

    def test_valid_face_returns_profile(self, engine: AvatarEngine, face_image: str) -> None:
        """Req 1.3 – A detected face produces a valid AvatarProfile.

        The Haar cascade may not fire on synthetic data, so we mock it
        to guarantee deterministic behaviour.
        """
        with patch.object(engine, "_face_cascade") as mock_cascade:
            mock_cascade.detectMultiScale.return_value = np.array([[100, 80, 200, 250]])
            profile = engine.preprocess(face_image)

        assert profile.avatar_id
        assert profile.image_path
        assert "face_rect" in profile.landmarks
        assert profile.landmarks["face_rect"]["w"] == 200
        assert profile.landmarks["face_rect"]["h"] == 250
        assert "image_size" in profile.landmarks
        assert profile.preprocessed_at

    def test_profile_contains_image_size(self, engine: AvatarEngine, valid_png: str) -> None:
        """Landmarks include the original image dimensions."""
        with patch.object(engine, "_face_cascade") as mock_cascade:
            mock_cascade.detectMultiScale.return_value = np.array([[10, 10, 50, 50]])
            profile = engine.preprocess(valid_png)

        assert profile.landmarks["image_size"]["width"] == 300
        assert profile.landmarks["image_size"]["height"] == 300


# ------------------------------------------------------------------ #
#  preprocess() – edge cases                                           #
# ------------------------------------------------------------------ #


class TestPreprocessEdgeCases:
    """Edge cases: missing file, unreadable image."""

    def test_file_not_found(self, engine: AvatarEngine) -> None:
        """FileNotFoundError for a path that does not exist."""
        with pytest.raises(FileNotFoundError):
            engine.preprocess("/nonexistent/photo.png")

    def test_unreadable_image(self, engine: AvatarEngine, tmp_path: Path) -> None:
        """A file with a valid extension but corrupt content raises ValueError."""
        corrupt = tmp_path / "corrupt.png"
        corrupt.write_bytes(b"not-a-real-image-at-all")
        with pytest.raises(ValueError, match="Unable to read"):
            engine.preprocess(str(corrupt))


# ------------------------------------------------------------------ #
#  animate() / animate_chunk() – fallback path (Req 5.1, 5.3, 5.6)    #
# ------------------------------------------------------------------ #


class TestAnimate:
    """Animation methods – tests the fallback path since Wav2Lip is not available."""

    def test_animate_produces_mp4(
        self, engine: AvatarEngine, valid_png: str, dummy_wav: str, tmp_path: Path,
    ) -> None:
        """Req 5.1 – animate() returns a VideoResult with format 'mp4'."""
        profile = _make_profile(valid_png)
        out = str(tmp_path / "out.mp4")
        result = engine.animate(profile, dummy_wav, out)

        assert result.format == "mp4"
        assert Path(result.file_path).exists()

    def test_animate_fps(
        self, engine: AvatarEngine, valid_png: str, dummy_wav: str, tmp_path: Path,
    ) -> None:
        """Req 5.3 – Video FPS matches the configured value (≥25)."""
        profile = _make_profile(valid_png)
        out = str(tmp_path / "fps_check.mp4")
        result = engine.animate(profile, dummy_wav, out)

        assert result.fps >= 25
        assert result.fps == engine.fps

    def test_animate_resolution(
        self, engine: AvatarEngine, valid_png: str, dummy_wav: str, tmp_path: Path,
    ) -> None:
        """VideoResult resolution matches the configured avatar resolution."""
        profile = _make_profile(valid_png)
        out = str(tmp_path / "res_check.mp4")
        result = engine.animate(profile, dummy_wav, out)

        assert result.resolution == engine.resolution

    def test_animate_chunk_produces_mp4(
        self, engine: AvatarEngine, valid_png: str, dummy_wav: str, tmp_path: Path,
    ) -> None:
        """animate_chunk() also produces an mp4 VideoResult."""
        profile = _make_profile(valid_png, avatar_id="chunk-test")
        out = str(tmp_path / "chunk_0.mp4")
        result = engine.animate_chunk(profile, dummy_wav, out, chunk_index=0)

        assert result.format == "mp4"
        assert Path(result.file_path).exists()

    def test_animate_chunk_different_indices(
        self, engine: AvatarEngine, valid_png: str, dummy_wav: str, tmp_path: Path,
    ) -> None:
        """Multiple chunk indices each produce independent output files."""
        profile = _make_profile(valid_png, avatar_id="multi-chunk")
        results = []
        for i in range(3):
            out = str(tmp_path / f"chunk_{i}.mp4")
            results.append(engine.animate_chunk(profile, dummy_wav, out, chunk_index=i))

        for r in results:
            assert r.format == "mp4"
            assert Path(r.file_path).exists()

    def test_fallback_on_failure(
        self, engine: AvatarEngine, valid_png: str, dummy_wav: str, tmp_path: Path,
    ) -> None:
        """Req 5.6 – On Wav2Lip failure, fallback produces a video from the static image.

        We force the Wav2Lip path to look available but raise an error,
        then verify the fallback still produces a valid video file.
        """
        profile = _make_profile(valid_png)
        out = str(tmp_path / "fallback.mp4")

        # Patch _invoke_wav2lip to simulate failure, and also patch the
        # file checks so the engine thinks Wav2Lip is available.
        with patch.object(engine, "_invoke_wav2lip", side_effect=RuntimeError("boom")):
            with patch("pathlib.Path.is_file", return_value=True):
                result = engine.animate(profile, dummy_wav, out)

        assert result.format == "mp4"
        assert Path(result.file_path).exists()

    def test_fallback_with_unreadable_image(
        self, engine: AvatarEngine, dummy_wav: str, tmp_path: Path,
    ) -> None:
        """Req 5.6 – Even when the source image is unreadable, fallback
        produces a black-frame video (OpenCV path).
        """
        # Use a valid-extension file that cv2.imread can't decode
        bad_img = tmp_path / "bad.png"
        bad_img.write_bytes(b"not-an-image")
        profile = _make_profile(str(bad_img))
        out = str(tmp_path / "fallback_bad.mp4")

        # Force the OpenCV fallback by making ffmpeg appear unavailable
        with patch("subprocess.run", side_effect=FileNotFoundError("no ffmpeg")):
            result = engine.animate(profile, dummy_wav, out)

        assert result.format == "mp4"
        assert Path(result.file_path).exists()
        # The file should have non-zero size (black frames were written)
        assert Path(result.file_path).stat().st_size > 0
