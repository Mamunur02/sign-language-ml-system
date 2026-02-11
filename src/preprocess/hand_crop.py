# src/preprocess/hand_crop.py
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


@dataclass(frozen=True)
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)

    def clamp(self, w: int, h: int) -> "BBox":
        return BBox(
            x1=max(0, min(self.x1, w - 1)),
            y1=max(0, min(self.y1, h - 1)),
            x2=max(1, min(self.x2, w)),
            y2=max(1, min(self.y2, h)),
        )


def _bbox_from_task_landmarks(
    hand_landmarks, image_w: int, image_h: int, pad: float
) -> BBox:
    xs = [lm.x for lm in hand_landmarks]
    ys = [lm.y for lm in hand_landmarks]

    x_min = min(xs) * image_w
    x_max = max(xs) * image_w
    y_min = min(ys) * image_h
    y_max = max(ys) * image_h

    bw = x_max - x_min
    bh = y_max - y_min

    x_min -= pad * bw
    x_max += pad * bw
    y_min -= pad * bh
    y_max += pad * bh

    return BBox(int(x_min), int(y_min), int(x_max), int(y_max)).clamp(image_w, image_h)


def ensure_hand_landmarker_model(model_path: Path, url: str = DEFAULT_TASK_MODEL_URL) -> None:
    """
    Downloads the MediaPipe HandLandmarker task model if it does not exist.
    Keeps the repo usable in Colab/CI without manual steps.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
        return

    import urllib.request

    urllib.request.urlretrieve(url, str(model_path))


class HandCropper:
    """
    MediaPipe Tasks HandLandmarker cropper.
    If multiple hands are detected, chooses the largest bbox.
    """

    def __init__(
        self,
        task_model_path: str | Path = "artifacts/models/hand_landmarker.task",
        num_hands: int = 2,
        pad: float = 0.20,
    ) -> None:
        self.pad = pad
        self.task_model_path = Path(task_model_path)
        ensure_hand_landmarker_model(self.task_model_path)

        base_options = mp_python.BaseOptions(model_asset_path=str(self.task_model_path))
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def close(self) -> None:
        # landmarker has close() in newer versions; safe-guard
        close_fn = getattr(self._landmarker, "close", None)
        if callable(close_fn):
            close_fn()

    def detect_largest_hand_bbox(self, img: Image.Image) -> Optional[BBox]:
        rgb = np.array(img.convert("RGB"))
        h, w = rgb.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return None

        best_bbox: Optional[BBox] = None
        best_area = -1

        for hand in result.hand_landmarks:
            bbox = _bbox_from_task_landmarks(hand, w, h, pad=self.pad)
            area = bbox.area()
            if area > best_area:
                best_area = area
                best_bbox = bbox

        return best_bbox

    def crop_hand(self, img: Image.Image) -> Optional[Image.Image]:
        bbox = self.detect_largest_hand_bbox(img)
        if bbox is None:
            return None
        return img.crop((bbox.x1, bbox.y1, bbox.x2, bbox.y2))
