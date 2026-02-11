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
    x: np.ndarray          # shape (63,)
    bbox_area: float
    handedness: str        # "Left" or "Right" or "Unknown"


def _rotation_matrix(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, -s], [s, c]], dtype=np.float32)


class HandLandmarkExtractor:
    """
    MediaPipe Tasks HandLandmarker -> canonicalised (handedness + rotation) landmark features.

    Canonicalisation:
      1) Choose largest hand by bbox area.
      2) If handedness == Left, mirror x (after wrist-centering) so all hands become "Right-like".
      3) Rotate in XY so the vector wrist -> middle_mcp points upward (+Y).
      4) Scale so max radius is 1.

    Output: 63-dim float32 vector (21 * 3).
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

    def _select_largest_hand(
        self, hand_landmarks_list, handedness_list, w: int, h: int
    ) -> Tuple[Optional[list], float, str]:
        if not hand_landmarks_list:
            return None, 0.0, "Unknown"

        best_hand = None
        best_area = -1.0
        best_handness = "Unknown"

        for i, hand in enumerate(hand_landmarks_list):
            xs = [lm.x for lm in hand]
            ys = [lm.y for lm in hand]
            x1 = min(xs) * w
            x2 = max(xs) * w
            y1 = min(ys) * h
            y2 = max(ys) * h
            area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))

            if area > best_area:
                best_area = area
                best_hand = hand

                # handedness_list[i] is a list of categories; take the top one
                if handedness_list and i < len(handedness_list) and handedness_list[i]:
                    best_handness = handedness_list[i][0].category_name or "Unknown"
                else:
                    best_handness = "Unknown"

        return best_hand, float(best_area), best_handness

    def extract(self, pil_img: Image.Image) -> Optional[LandmarkFeatures]:
        rgb = np.array(pil_img.convert("RGB"))
        h, w = rgb.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        hand, area, handness = self._select_largest_hand(
            result.hand_landmarks, result.handedness, w, h
        )
        if hand is None:
            return None

        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)  # (21,3)

        # 1) wrist-centre
        wrist = pts[0:1, :]
        pts = pts - wrist

        # 2) mirror left -> right canonical
        if handness.lower().startswith("left"):
            pts[:, 0] = -pts[:, 0]

        # 3) rotate XY so wrist->middle_mcp points up
        # middle_mcp landmark index = 9 (standard)
        v = pts[9, 0:2].copy()
        norm = float(np.linalg.norm(v))
        if norm > 1e-6:
            # current angle of v
            angle = float(np.arctan2(v[1], v[0]))
            # we want v to align with +Y, i.e. angle_target = pi/2
            theta = (np.pi / 2.0) - angle
            R = _rotation_matrix(theta)
            pts[:, 0:2] = (pts[:, 0:2] @ R.T)

        # 4) scale by max radius
        d = np.linalg.norm(pts, axis=1)
        scale = float(np.max(d))
        if scale < 1e-6:
            return None
        pts = pts / scale

        x = pts.reshape(-1).astype(np.float32)  # (63,)
        return LandmarkFeatures(x=x, bbox_area=area, handedness=handness)
