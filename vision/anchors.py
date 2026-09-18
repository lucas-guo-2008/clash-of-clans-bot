"""
Locating pre-set UI elements by template matching.

Image in, findings out. Used to find buttons and recognize the window.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# Set empirically against saved positives and negatives, never eyeballed.
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


def deduplicate(matches: list[Match]) -> list[Match]:
    """Keep only the highest-scoring match per label.

    A template correlates above threshold across a small neighbourhood of its true
    position, so one button yields a cluster of hits. Only the peak matters.
    """

    best: dict[str, Match] = {}
    for match in matches:
        if match.label not in best or match.score > best[match.label].score:
            best[match.label] = match

    return list(best.values())


def find_anchors(
    frame: np.ndarray,
    templates: dict[str, np.ndarray],
    threshold: float = THRESHOLD,
) -> list[Match]:
    """Find every template that appears in the frame, one match per label."""

    matches = []
    for label, template in templates.items():
        h, w = template.shape[:2]
        result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(result >= threshold)
        for x, y in zip(xs, ys):
            matches.append(Match(label, int(x), int(y), w, h, float(result[y, x])))

    return deduplicate(matches)
