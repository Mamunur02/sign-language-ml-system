from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

DEFAULT_TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def ensure_hand_landmarker_model(model_path: Path, url: str = DEFAULT_TASK_MODEL_URL) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
        return
    import urllib.request
    urllib.request.urlretrieve(url, str(model_path))


@dataclass(frozen=True)
class LandmarkFeatures:
    x: np.ndarray  # shape (63,)
    bbox_area: float


class HandLandmarkExtractor:
    """
    Extracts 21 hand landmarks (x,y,z) using MediaPipe Tasks HandLandmarker.
    Normalisation:
      - translate so wrist is at origin
      - scale so max distance from origin is 1 (avoids camera distance effects)
    """

    def __init__(
        self,
        task_model_path: str | Path = "artifacts/models/hand_landmarker.task",
        num_hands: int = 2,
    ) -> None:
        self.task_model_path = Path(task_model_path)
        ensure_hand_landmarker_model(self.task_model_path)

        base = mp_python.BaseOptions(model_asset_path=str(self.task_model_path))
        opts = mp_vision.HandLandmarkerOptions(base_options=base, num_hands=num_hands)
        self._landmarker = mp_vision.HandLandmarker.create_from_options(opts)

    def close(self) -> None:
        close_fn = getattr(self._landmarker, "close", None)
        if callable(close_fn):
            close_fn()

    def _select_largest_hand(self, hand_landmarks_list, w: int, h: int) -> Tuple[Optional[list], float]:
        """
        Returns: (hand_landmarks, bbox_area) where hand_landmarks is a list of 21 landmarks.
        If no hands, returns (None, 0.0)
        """
        if not hand_landmarks_list:
            return None, 0.0

        best = None
        best_area = -1.0

        for hand in hand_landmarks_list:
            xs = [lm.x for lm in hand]
            ys = [lm.y for lm in hand]
            x1 = min(xs) * w
            x2 = max(xs) * w
            y1 = min(ys) * h
            y2 = max(ys) * h
            area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
            if area > best_area:
                best_area = area
                best = hand

        return best, float(best_area)

    def extract(self, pil_img: Image.Image) -> Optional[LandmarkFeatures]:
        rgb = np.array(pil_img.convert("RGB"))
        h, w = rgb.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        hand, area = self._select_largest_hand(result.hand_landmarks, w, h)
        if hand is None:
            return None

        # (21,3): x,y,z in normalised coordinates (x,y in [0,1] relative to image; z relative)
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)

        # Normalise: translate wrist to origin
        wrist = pts[0:1, :]  # landmark 0 is wrist
        pts = pts - wrist

        # Scale: max distance from origin to 1 (avoid scale dependence)
        d = np.linalg.norm(pts, axis=1)
        scale = float(np.max(d))
        if scale < 1e-6:
            return None
        pts = pts / scale

        x = pts.reshape(-1)  # (63,)
        return LandmarkFeatures(x=x, bbox_area=area)
