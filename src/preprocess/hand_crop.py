# src/preprocess/hand_crop.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image

# MediaPipe + OpenCV (OpenCV only used for array format convenience)
import cv2
import mediapipe as mp


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


def _bbox_from_landmarks(
    hand_landmarks, image_w: int, image_h: int, pad: float = 0.20
) -> BBox:
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]

    x_min = min(xs) * image_w
    x_max = max(xs) * image_w
    y_min = min(ys) * image_h
    y_max = max(ys) * image_h

    bw = x_max - x_min
    bh = y_max - y_min

    # add padding (fraction of bbox size)
    x_min -= pad * bw
    x_max += pad * bw
    y_min -= pad * bh
    y_max += pad * bh

    return BBox(int(x_min), int(y_min), int(x_max), int(y_max)).clamp(image_w, image_h)


class HandCropper:
    """
    Uses MediaPipe Hands to detect hand landmarks and returns a cropped PIL image of the hand.
    Default behaviour: if 2 hands detected, choose the largest bbox (most stable for single-hand signs).
    """

    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        pad: float = 0.20,
    ) -> None:
        self.pad = pad
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        self._hands.close()

    def detect_largest_hand_bbox(self, img: Image.Image) -> Optional[BBox]:
        # PIL -> RGB numpy -> MediaPipe expects RGB
        rgb = np.array(img.convert("RGB"))
        h, w = rgb.shape[:2]

        # MediaPipe API
        results = self._hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None

        # Choose largest bbox
        best_bbox: Optional[BBox] = None
        best_area = -1

        for hand_lms in results.multi_hand_landmarks:
            bbox = _bbox_from_landmarks(hand_lms, w, h, pad=self.pad)
            area = bbox.area()
            if area > best_area:
                best_area = area
                best_bbox = bbox

        return best_bbox

    def crop_hand(self, img: Image.Image) -> Optional[Image.Image]:
        bbox = self.detect_largest_hand_bbox(img)
        if bbox is None:
            return None
        cropped = img.crop((bbox.x1, bbox.y1, bbox.x2, bbox.y2))
        return cropped
