"""Avatar engine for face preprocessing and talking-head video generation.

Uses OpenCV Haar cascade for face detection and Wav2Lip for lip-sync
animation. Wav2Lip is much faster than SadTalker on CPU (~5s vs hours).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from backend.app.config import AppConfig
from backend.app.logging_utils import get_logger
from backend.app.models import AvatarProfile, VideoResult

logger = get_logger("avatar_engine")

# Resolve project root (where models/ lives)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class FaceNotFoundError(Exception):
    """Raised when no human face is detected in the uploaded image."""


class AvatarEngine:
    """Preprocess avatar images and generate lip-synced video via Wav2Lip."""

    def __init__(self, config: AppConfig | None = None) -> None:
        cfg = config or AppConfig()
        self.fps: int = cfg.avatar_fps
        self.resolution: tuple[int, int] = cfg.avatar_resolution
        self.max_image_size_mb: int = cfg.max_image_size_mb
        self.min_image_resolution: tuple[int, int] = cfg.min_image_resolution
        self.allowed_image_formats: list[str] = [
            fmt.lower() for fmt in cfg.allowed_image_formats
        ]
        self.media_output_dir: str = cfg.media_output_dir

        # Wav2Lip paths
        self._wav2lip_dir = _PROJECT_ROOT / "models" / "wav2lip"
        self._checkpoint = self._wav2lip_dir / "checkpoints" / "wav2lip_gan.pth"
        self._python = sys.executable

        # Ensure temp dir exists (Wav2Lip writes intermediate AVI here)
        (_PROJECT_ROOT / "temp").mkdir(exist_ok=True)

        # Load Haar cascade for face detection (ships with OpenCV)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._face_cascade = cv2.CascadeClassifier(cascade_path)

    # ------------------------------------------------------------------ #
    #  Preprocessing / validation                                          #
    # ------------------------------------------------------------------ #

    def preprocess(self, image_path: str) -> AvatarProfile:
        """Validate image, detect face, extract landmarks."""
        path = Path(image_path)

        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        ext = path.suffix.lstrip(".").lower()
        if ext not in self.allowed_image_formats:
            raise ValueError(
                f"Unsupported image format '{ext}'. "
                f"Allowed: {', '.join(self.allowed_image_formats)}"
            )

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > self.max_image_size_mb:
            raise ValueError(
                f"Image size {size_mb:.1f} MB exceeds the "
                f"{self.max_image_size_mb} MB limit"
            )

        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Unable to read image file: {image_path}")

        h, w = img.shape[:2]
        min_h, min_w = self.min_image_resolution
        if w < min_w or h < min_h:
            raise ValueError(
                f"Image resolution {w}×{h} is below the minimum "
                f"{min_w}×{min_h}"
            )

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) == 0:
            raise FaceNotFoundError(
                "No human face detected in the image. "
                "Please upload a photo with a clear, front-facing face."
            )

        x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
        landmarks = {
            "face_rect": {"x": int(x), "y": int(y), "w": int(fw), "h": int(fh)},
            "image_size": {"width": w, "height": h},
        }

        avatar_id = uuid.uuid4().hex[:12]
        logger.info(
            "Preprocessed avatar %s – face at (%d,%d,%d,%d) in %dx%d image",
            avatar_id, x, y, fw, fh, w, h,
        )

        return AvatarProfile(
            avatar_id=avatar_id,
            image_path=str(path.resolve()),
            landmarks=landmarks,
            preprocessed_at=datetime.now(timezone.utc).isoformat(),
        )

    # ------------------------------------------------------------------ #
    #  Animation (Wav2Lip)                                                 #
    # ------------------------------------------------------------------ #

    def animate(
        self, profile: AvatarProfile, audio_path: str, output_path: str,
    ) -> VideoResult:
        """Generate a lip-synced MP4 video using Wav2Lip."""
        return self._run_wav2lip(profile, audio_path, output_path)

    def animate_chunk(
        self, profile: AvatarProfile, audio_path: str, output_path: str,
        chunk_index: int,
    ) -> VideoResult:
        """Generate a single video segment for one audio chunk."""
        logger.info("Animating chunk %d for avatar %s", chunk_index, profile.avatar_id)
        return self._run_wav2lip(profile, audio_path, output_path)

    # ------------------------------------------------------------------ #
    #  Internal: Wav2Lip invocation                                        #
    # ------------------------------------------------------------------ #

    def _run_wav2lip(
        self, profile: AvatarProfile, audio_path: str, output_path: str,
    ) -> VideoResult:
        """Run Wav2Lip inference or fall back to static video."""
        start = time.monotonic()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        inference_script = self._wav2lip_dir / "inference.py"

        try:
            if inference_script.is_file() and self._checkpoint.is_file():
                self._invoke_wav2lip(
                    image_path=profile.image_path,
                    audio_path=str(Path(audio_path).resolve()),
                    output_path=str(out.resolve()),
                )
            else:
                missing = []
                if not inference_script.is_file():
                    missing.append(f"inference.py at {inference_script}")
                if not self._checkpoint.is_file():
                    missing.append(f"checkpoint at {self._checkpoint}")
                logger.warning(
                    "Wav2Lip not available (%s) – generating static fallback",
                    ", ".join(missing),
                )
                self._generate_fallback_video(
                    profile.image_path, audio_path, str(out),
                )
        except Exception:
            logger.exception(
                "Wav2Lip failed for avatar %s – substituting static fallback",
                profile.avatar_id,
            )
            self._generate_fallback_video(
                profile.image_path, audio_path, str(out),
            )

        elapsed = time.monotonic() - start
        duration = self._probe_duration(str(out))
        logger.info("Animation completed in %.1fs, video duration %.2fs", elapsed, duration)

        return VideoResult(
            file_path=str(out),
            duration_seconds=duration,
            fps=self.fps,
            resolution=self.resolution,
            format="mp4",
        )

    def _invoke_wav2lip(
        self, image_path: str, audio_path: str, output_path: str,
    ) -> None:
        """Call Wav2Lip inference.py as a subprocess."""
        wav2lip_dir = str(self._wav2lip_dir.resolve())
        env = os.environ.copy()
        env["PYTHONPATH"] = wav2lip_dir + ":" + env.get("PYTHONPATH", "")

        cmd = [
            self._python,
            str(self._wav2lip_dir / "inference.py"),
            "--checkpoint_path", str(self._checkpoint.resolve()),
            "--face", image_path,
            "--audio", audio_path,
            "--outfile", output_path,
            "--static", "True",
            "--fps", str(self.fps),
            "--resize_factor", "2",
            "--wav2lip_batch_size", "16",
            "--nosmooth",
        ]
        logger.info("Running Wav2Lip: %s", " ".join(cmd))
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            cwd=str(_PROJECT_ROOT), env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Wav2Lip exited with code {result.returncode}: "
                f"{result.stderr[:500]}"
            )
        # Wav2Lip writes to the --outfile path directly
        if not Path(output_path).is_file():
            raise RuntimeError(
                f"Wav2Lip did not produce output at {output_path}"
            )

    # ------------------------------------------------------------------ #
    #  Fallback: static image video                                        #
    # ------------------------------------------------------------------ #

    def _generate_fallback_video(
        self, image_path: str, audio_path: str, output_path: str,
    ) -> None:
        """Create a browser-playable MP4 with ffmpeg from a static image + audio."""
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-loop", "1", "-i", image_path,
                    "-i", audio_path,
                    "-c:v", "libx264", "-tune", "stillimage",
                    "-c:a", "aac", "-b:a", "128k",
                    "-pix_fmt", "yuv420p",
                    "-vf", f"scale={self.resolution[0]}:{self.resolution[1]}:force_original_aspect_ratio=decrease,pad={self.resolution[0]}:{self.resolution[1]}:(ow-iw)/2:(oh-ih)/2",
                    "-shortest",
                    "-movflags", "+faststart",
                    output_path,
                ],
                capture_output=True, timeout=30,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # ffmpeg not available — write raw mp4v with OpenCV
            img = cv2.imread(image_path)
            if img is None:
                img = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
            img = cv2.resize(img, self.resolution)
            duration_s = self._estimate_audio_duration(audio_path)
            total_frames = max(int(duration_s * self.fps), 1)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, self.fps, self.resolution)
            try:
                for _ in range(total_frames):
                    writer.write(img)
            finally:
                writer.release()

    @staticmethod
    def _estimate_audio_duration(audio_path: str) -> float:
        try:
            size = Path(audio_path).stat().st_size
            return max((size - 44) / 44100, 0.5)
        except OSError:
            return 3.0

    @staticmethod
    def _probe_duration(video_path: str) -> float:
        try:
            cap = cv2.VideoCapture(video_path)
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps > 0:
                return round(frames / fps, 2)
        except Exception:
            pass
        return 0.0
