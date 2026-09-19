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


def peak_scores(frame: np.ndarray, templates: dict[str, np.ndarray]) -> dict[str, Match]:
    """Best correlation for every template, whether or not it clears the threshold.
    """

    peaks = {}
    for label, template in templates.items():
        h, w = template.shape[:2]
        result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, (x, y) = cv2.minMaxLoc(result)
        peaks[label] = Match(label, int(x), int(y), w, h, float(score))

    return peaks


def find_anchors(
    frame: np.ndarray,
    templates: dict[str, np.ndarray],
    threshold: float = THRESHOLD,
) -> list[Match]:
    """Find every template that appears in the frame, one match per label."""

    return [match for match in peak_scores(frame, templates).values() if match.score >= threshold]
