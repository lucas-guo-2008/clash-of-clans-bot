"""Drawing what the bot found onto a frame."""

import cv2
import numpy as np

from vision.anchors import Match
from vision.deploy import DeployPoint

GREEN = (80, 220, 80)
GREY = (150, 150, 150)
RED = (60, 60, 240)
MAGENTA = (255, 0, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX
ROW_H = 38


def label(image: np.ndarray, text: str, org: tuple[int, int], color, scale: float = 0.7) -> None:
    """Text on a dark box, readable over any part of the map."""

    (w, h), base = cv2.getTextSize(text, FONT, scale, 2)
    x, y = org
    cv2.rectangle(image, (x - 3, y - h - 5), (x + w + 3, y + base + 2), (0, 0, 0), -1)
    cv2.putText(image, text, (x, y), FONT, scale, color, 2, cv2.LINE_AA)


def draw_anchors(image: np.ndarray, peaks: dict[str, Match], threshold: float) -> None:
    """Box and score for every anchor that clears the threshold."""

    for m in peaks.values():
        if m.score >= threshold:
            cv2.rectangle(image, (m.x, m.y), (m.x + m.w, m.y + m.h), GREEN, 3)
            label(image, f"{m.label} {m.score:.3f}", (m.x, max(24, m.y - 10)), GREEN)


def draw_deploy(image: np.ndarray, lines: np.ndarray, points: list[DeployPoint], card: Match | None) -> None:
    """The outline the rays hit, the deploy points in tap order, and the card to select."""

    image[lines > 0] = MAGENTA
    for i, p in enumerate(points, 1):
        cv2.circle(image, (p.x, p.y), 12, CYAN, 3)
        cv2.putText(image, str(i), (p.x + 14, p.y - 8), FONT, 0.6, CYAN, 2, cv2.LINE_AA)
    if card is not None:
        cv2.rectangle(image, (card.x, card.y), (card.x + card.w, card.y + card.h), GREEN, 3)
        label(image, f"card {card.w}px", (card.x, card.y - 10), GREEN)


def draw_tap(image: np.ndarray, xy: tuple[int, int]) -> None:
    """Crosshair on the spot about to be tapped."""

    x, y = xy
    cv2.circle(image, (x, y), 28, RED, 4)
    cv2.line(image, (x - 40, y), (x + 40, y), RED, 3)
    cv2.line(image, (x, y - 40), (x, y + 40), RED, 3)


def with_header(image: np.ndarray, rows: list[list[tuple[str, tuple]]]) -> np.ndarray:
    """The image under a text panel, so the panel never hides the game. Each row is a list
    of (text, colour) runs drawn left to right."""

    panel = np.zeros((ROW_H * len(rows) + 12, image.shape[1], 3), np.uint8)
    for r, row in enumerate(rows):
        x, y = 12, ROW_H * (r + 1)
        for text, color in row:
            cv2.putText(panel, text, (x, y), FONT, 0.85, color, 2, cv2.LINE_AA)
            x += cv2.getTextSize(text, FONT, 0.85, 2)[0][0] + 24
    return np.vstack([panel, image])
