"""Viseme engine: pre-generates mouth shape images for animated lip sync.

Two approaches tried in order:
1. Wav2Lip neural network (in-process) — best quality, uses Poisson seamless cloning
2. OpenCV Gaussian-blended mouth warping — fallback, decent quality

The idle image is always the untouched original frame.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

from backend.app.logging_utils import get_logger

logger = get_logger("viseme_engine")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_WAV2LIP_DIR = _PROJECT_ROOT / "models" / "wav2lip"
_WAV2LIP_CHECKPOINT = _WAV2LIP_DIR / "checkpoints" / "wav2lip_gan.pth"

FRAME_SIZE = 512

CHAR_TO_VISEME = {
    'a': 'a', 'h': 'a', 'r': 'a', 'l': 'a',
    'e': 'e', 'i': 'e', 'y': 'e', 's': 'e', 'z': 'e', 'j': 'e',
    'o': 'o', 'u': 'o', 'w': 'o', 'q': 'o',
    'm': 'm', 'b': 'm', 'p': 'm', 'f': 'm', 'v': 'm',
    't': 'a', 'd': 'a', 'n': 'a', 'k': 'a', 'g': 'a', 'c': 'a', 'x': 'a',
    ' ': 'idle', '.': 'idle', ',': 'idle', '?': 'idle', '!': 'idle',
    '\n': 'idle', '\t': 'idle', ';': 'idle', ':': 'idle',
}


def preprocess_avatar_frame(image_path: str, output_path: str) -> str | None:
    """Crop and center the face into a clean square frame."""
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

    if len(faces) == 0:
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        cropped = img[y0:y0 + side, x0:x0 + side]
    else:
        fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
        cx, cy = fx + fw // 2, fy + fh // 2
        crop_size = int(max(fw, fh) * 2.2)
        x0 = max(0, cx - crop_size // 2)
        y0 = max(0, cy - crop_size // 2)
        x1 = min(w, x0 + crop_size)
        y1 = min(h, y0 + crop_size)
        if x1 - x0 < crop_size: x0 = max(0, x1 - crop_size)
        if y1 - y0 < crop_size: y0 = max(0, y1 - crop_size)
        cropped = img[y0:y1, x0:x1]

    frame = cv2.resize(cropped, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_LANCZOS4)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    logger.info("Preprocessed avatar frame: %dx%d", FRAME_SIZE, FRAME_SIZE)
    return output_path


class VisemeEngine:
    """Generate viseme images from an avatar photo."""

    def __init__(self) -> None:
        self._wav2lip_model = None
        self._face_detector = None
        self._device = 'cpu'
        self._wav2lip_available = False
        self._tried_loading = False

    def _ensure_wav2lip_loaded(self) -> bool:
        """Lazy-load Wav2Lip model on first use."""
        if self._tried_loading:
            return self._wav2lip_available
        self._tried_loading = True

        try:
            import torch

            if not _WAV2LIP_CHECKPOINT.is_file():
                logger.info("Wav2Lip checkpoint not found, using warp fallback")
                return False

            wav2lip_str = str(_WAV2LIP_DIR)
            if wav2lip_str not in sys.path:
                sys.path.insert(0, wav2lip_str)

            from models import Wav2Lip
            import face_detection

            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self._device = 'mps'

            self._face_detector = face_detection.FaceAlignment(
                face_detection.LandmarksType._2D, flip_input=False, device='cpu'
            )

            model = Wav2Lip()
            checkpoint = torch.load(str(_WAV2LIP_CHECKPOINT), map_location='cpu', weights_only=False)
            state = checkpoint["state_dict"]
            new_state = {k.replace('module.', ''): v for k, v in state.items()}
            model.load_state_dict(new_state)
            model = model.to(self._device)
            self._wav2lip_model = model.eval()
            self._wav2lip_available = True
            logger.info("Wav2Lip loaded on %s for neural viseme generation", self._device)
            return True

        except Exception as e:
            logger.warning("Wav2Lip load failed (%s), using warp fallback", e)
            return False

    def generate_visemes(self, image_path: str, output_dir: str) -> dict[str, str]:
        """Generate viseme images. Returns {name → file_path}."""
        os.makedirs(output_dir, exist_ok=True)

        img = cv2.imread(image_path)
        if img is None:
            return self._static_visemes(image_path, output_dir)

        # Idle is always the untouched original
        idle_path = os.path.join(output_dir, "idle.jpg")
        cv2.imwrite(idle_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        result = {"idle": idle_path}

        # Try neural visemes first
        if self._ensure_wav2lip_loaded():
            neural = self._generate_neural(img, output_dir)
            if neural:
                result.update(neural)
                logger.info("Generated %d neural visemes", len(result))
                return result

        # Fallback to warp visemes
        warp = self._generate_warp(img, output_dir)
        result.update(warp)
        logger.info("Generated %d warp visemes (fallback)", len(result))
        return result

    # ------------------------------------------------------------------
    # Neural visemes (Wav2Lip in-process + Poisson seamless cloning)
    # ------------------------------------------------------------------

    def _generate_neural(self, img: np.ndarray, output_dir: str) -> dict[str, str] | None:
        """Use Wav2Lip to generate realistic viseme frames."""
        import torch

        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self._face_detector.get_detections_for_batch(np.array([frame_rgb]))
        if not faces or faces[0] is None:
            logger.warning("Wav2Lip face detector found no face")
            return None

        x1, y1, x2, y2 = faces[0]
        h, w = img.shape[:2]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(w, int(x2))
        y2 = min(h, int(y2) + 10)

        if x2 <= x1 or y2 <= y1:
            return None

        face_roi = img[y1:y2, x1:x2]
        face_96 = cv2.resize(face_roi, (96, 96))

        viseme_mels = {
            'a': np.ones((80, 16), dtype=np.float32) * 2.5,
            'e': self._mel(high_freq=True),
            'o': self._mel(low_freq=True),
            'm': np.ones((80, 16), dtype=np.float32) * -4.0,
        }

        result = {}
        for name, mel_chunk in viseme_mels.items():
            try:
                img_batch = np.asarray([face_96])
                mel_batch = np.asarray([mel_chunk])

                img_masked = img_batch.copy()
                img_masked[:, 48:] = 0

                img_concat = np.concatenate((img_masked, img_batch), axis=3) / 255.0
                mel_batch = mel_batch.reshape(1, 80, 16, 1)

                img_t = torch.FloatTensor(np.transpose(img_concat, (0, 3, 1, 2))).to(self._device)
                mel_t = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self._device)

                with torch.no_grad():
                    pred = self._wav2lip_model(mel_t, img_t)

                pred_np = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0
                pred_bgr = cv2.cvtColor(pred_np[0].astype(np.uint8), cv2.COLOR_RGB2BGR)

                target_h = y2 - y1
                target_w = x2 - x1
                pred_up = cv2.resize(pred_bgr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

                final = self._seamless_blend(img, pred_up, x1, y1, target_w, target_h)

                out_path = os.path.join(output_dir, f"{name}.jpg")
                cv2.imwrite(out_path, final, [cv2.IMWRITE_JPEG_QUALITY, 95])
                result[name] = out_path

            except Exception as e:
                logger.warning("Neural viseme '%s' failed: %s", name, e)
                return None  # Fall back to warp for all

        return result

    def _seamless_blend(self, original, patch, x1, y1, tw, th):
        """Poisson seamless cloning for artifact-free blending."""
        mask = np.zeros(patch.shape, patch.dtype)
        cx, cy = tw // 2, int(th * 0.75)
        ax, ay = int(tw * 0.45), int(th * 0.25)
        cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, (255, 255, 255), -1)

        ih, iw = original.shape[:2]
        pc = (
            max(ax + 1, min(iw - ax - 1, x1 + cx)),
            max(ay + 1, min(ih - ay - 1, y1 + cy)),
        )

        try:
            return cv2.seamlessClone(patch, original, mask, pc, cv2.NORMAL_CLONE)
        except cv2.error:
            result = original.copy()
            alpha = mask.astype(np.float32) / 255.0
            roi = result[y1:y1 + th, x1:x1 + tw]
            if roi.shape == patch.shape:
                result[y1:y1 + th, x1:x1 + tw] = (alpha * patch + (1 - alpha) * roi).astype(np.uint8)
            return result

    @staticmethod
    def _mel(low_freq=False, high_freq=False):
        mel = np.zeros((80, 16), dtype=np.float32)
        if low_freq: mel[:25, :] = 3.0
        if high_freq: mel[55:, :] = 3.0
        return mel

    # ------------------------------------------------------------------
    # Warp visemes (OpenCV fallback with Gaussian blending)
    # ------------------------------------------------------------------

    def _generate_warp(self, img: np.ndarray, output_dir: str) -> dict[str, str]:
        """OpenCV face detection + Gaussian-blended mouth warping."""
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.3, 5)

        result = {}
        if len(faces) == 0:
            for name in ['a', 'e', 'o', 'm']:
                p = os.path.join(output_dir, f"{name}.jpg")
                cv2.imwrite(p, img)
                result[name] = p
            return result

        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        shapes = {'a': (0.35, 0.0), 'e': (0.08, 0.25), 'o': (0.20, -0.20), 'm': (-0.08, 0.0)}

        for name, (open_r, width_r) in shapes.items():
            warped = self._warp_mouth(img, x, y, w, h, open_r, width_r)
            out_path = os.path.join(output_dir, f"{name}.jpg")
            cv2.imwrite(out_path, warped, [cv2.IMWRITE_JPEG_QUALITY, 95])
            result[name] = out_path

        return result

    def _warp_mouth(self, img, x, y, w, h, open_ratio, width_ratio):
        """Warp mouth region with Gaussian-feathered edges.
        
        Face proportions (from Haar cascade bounding box):
        - Eyes: ~35% from top
        - Nose tip: ~65% from top  
        - Mouth center: ~82% from top
        - Chin: ~100% (bottom of box)
        """
        result = img.copy()
        ih, iw = img.shape[:2]

        # Mouth center is at ~82% of face height, mouth spans ~15% of face height
        mouth_center_y = int(y + 0.82 * h)
        mouth_half_h = int(h * 0.08)
        mouth_y = max(0, mouth_center_y - mouth_half_h)
        mouth_h = min(mouth_half_h * 2, ih - mouth_y)
        
        # Mouth width is ~50% of face width, centered
        mouth_x = int(x + w * 0.25)
        mouth_w = int(w * 0.5)
        
        # Clamp
        mouth_x = max(0, min(mouth_x, iw - 2))
        mouth_w = min(mouth_w, iw - mouth_x)

        if mouth_h <= 2 or mouth_w <= 2:
            return result

        jaw_roi = img[mouth_y:mouth_y + mouth_h, mouth_x:mouth_x + mouth_w].copy()
        if jaw_roi.size == 0:
            return result

        shift_y = int(mouth_h * open_ratio * 1.5)  # Amplify the effect
        new_width = max(1, int(mouth_w * (1.0 + width_ratio * 1.5)))

        # Draw dark oral cavity at mouth center
        cv2.ellipse(result, (mouth_x + mouth_w // 2, mouth_center_y),
                     (mouth_w // 4, max(2, int(abs(shift_y) * 1.0) + 1)),
                     0, 0, 360, (15, 10, 25), -1)

        jaw_resized = cv2.resize(jaw_roi, (new_width, mouth_h))
        offset_x = (mouth_w - new_width) // 2
        paste_y = max(0, mouth_y + shift_y)
        paste_x = max(0, mouth_x + offset_x)

        h_src, w_src = jaw_resized.shape[:2]
        if paste_y + h_src > ih: h_src = ih - paste_y
        if paste_x + w_src > iw: w_src = iw - paste_x

        if h_src > 0 and w_src > 0:
            # Gaussian-feathered elliptical mask for smooth blending
            mask = np.zeros((h_src, w_src), dtype=np.float32)
            cv2.ellipse(mask, (w_src // 2, h_src // 2), 
                        (max(1, w_src // 2 - 2), max(1, h_src // 2 - 2)),
                        0, 0, 360, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (11, 11), 4)[:, :, np.newaxis]

            roi = result[paste_y:paste_y + h_src, paste_x:paste_x + w_src].astype(np.float32)
            src = jaw_resized[:h_src, :w_src].astype(np.float32)
            result[paste_y:paste_y + h_src, paste_x:paste_x + w_src] = (
                mask * src + (1 - mask) * roi
            ).astype(np.uint8)

        return result

    # ------------------------------------------------------------------
    # Static fallback
    # ------------------------------------------------------------------

    def _static_visemes(self, image_path: str, output_dir: str) -> dict[str, str]:
        result = {}
        img = cv2.imread(image_path)
        for name in ['idle', 'a', 'e', 'o', 'm']:
            p = os.path.join(output_dir, f"{name}.jpg")
            if img is not None: cv2.imwrite(p, img)
            result[name] = p
        return result
