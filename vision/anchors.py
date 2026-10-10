"""
Locating pre-set UI elements by template matching.

Input is images, output is template matches. Used to find buttons and recognize the current window.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# Can edit (conf threshold)
THRESHOLD = 0.8


@dataclass(frozen=True)
class Match:
    label: str
    x: int
    y: int
    w: int
    h: int
    score: float

    @property
    def centre(self) -> tuple[int, int]:
        """Coordinates for where to tap"""
        return self.x + self.w // 2, self.y + self.h // 2


@dataclass(frozen=True)
class Template:
    """An anchor image, and the part of the frame it can appear in."""

    image: np.ndarray
    region: tuple[int, int, int, int] | None = None  # (x0, y0, x1, y1); None searches the whole frame


def peak_scores(frame: np.ndarray, templates: dict[str, Template]) -> dict[str, Match]:
    peaks = {}
    for label, template in templates.items():
        h, w = template.image.shape[:2]
        x0, y0, x1, y1 = template.region or (0, 0, frame.shape[1], frame.shape[0])
        area = frame[y0:y1, x0:x1]
        if area.shape[0] < h or area.shape[1] < w:
            peaks[label] = Match(label, x0, y0, w, h, 0.0)
            continue
        result = cv2.matchTemplate(area, template.image, cv2.TM_CCOEFF_NORMED)
        _, score, _, (x, y) = cv2.minMaxLoc(result)
        peaks[label] = Match(label, x0 + int(x), y0 + int(y), w, h, float(score))

    return peaks


def find_anchors(frame: np.ndarray, templates: dict[str, Template]) -> dict[str, Match]:
    """Find every template that appears in the frame, one match per label."""

    return {label: match for label, match in peak_scores(frame, templates).items() if match.score >= THRESHOLD}
