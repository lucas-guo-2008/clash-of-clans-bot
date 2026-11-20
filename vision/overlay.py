"""Drawing what the bot found onto a frame."""

import cv2
import numpy as np

from vision.anchors import Match
from vision.buildings import Building
from vision.deploy import DeployPoint

GREEN = (80, 220, 80)
GREY = (150, 150, 150)
RED = (60, 60, 240)
MAGENTA = (255, 0, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)
YELLOW = (0, 220, 255)
ORANGE = (0, 140, 255)
PINK = (220, 90, 235)
PURPLE = (200, 70, 150)

# One colour per resource, so a base's loot reads at a glance
BUILDING_COLOURS = {
    "gold_mine": YELLOW, "gold_storage": YELLOW,
    "elixir_collector": PINK, "elixir_storage": PINK,
    "dark_elixir_drill": PURPLE, "dark_elixir_storage": PURPLE,
    "town_hall": ORANGE, "clan_castle": WHITE,
}

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


def draw_deploy(image: np.ndarray, lines: np.ndarray | None, points: list[DeployPoint], slots: list[tuple]) -> None:
    """The outline the rays hit, the deploy points in tap order, and each deck slot with its label, as
    (slot, text) or (slot, text, colour)."""

    if lines is not None:
        image[lines > 0] = MAGENTA
    for i, p in enumerate(points, 1):
        cv2.circle(image, (p.x, p.y), 12, CYAN, 3)
        cv2.putText(image, str(i), (p.x + 14, p.y - 8), FONT, 0.6, CYAN, 2, cv2.LINE_AA)
    for slot, text, *colour in slots:
        colour = colour[0] if colour else GREEN
        cv2.rectangle(image, (slot.x, slot.y), (slot.x + slot.w, slot.y + slot.h), colour, 3)
        label(image, text, (slot.x, slot.y - 10), colour)


def draw_buildings(image: np.ndarray, buildings: list[Building]) -> None:
    """Box, class and confidence for every detected building."""

    for b in buildings:
        colour = BUILDING_COLOURS.get(b.name, GREY)
        cv2.rectangle(image, (b.x0, b.y0), (b.x1, b.y1), colour, 2)
        label(image, f"{b.name} {b.conf:.2f}", (b.x0, max(18, b.y0 - 6)), colour, scale=0.45)


def draw_tap(image: np.ndarray, xy: tuple[int, int]) -> None:
    """Crosshair on the spot about to be tapped."""

    x, y = xy
    cv2.circle(image, (x, y), 28, RED, 4)
    cv2.line(image, (x - 40, y), (x + 40, y), RED, 3)
    cv2.line(image, (x, y - 40), (x, y + 40), RED, 3)


CHART_H = 110  # the loot chart's strip under the text rows
CHART_SECONDS = 60  # the chart's time axis starts this long and grows with the battle


def draw_loot_chart(panel: np.ndarray, top: int, trace: list[list]) -> None:
    """Gold + elixir taken since the deploy, from a battle's Available Loot trace ([s, gold, elixir, dark] per
    change), as a step line on a strip of `panel` starting at row `top`."""

    x0, y0, w, h = 160, top + 12, panel.shape[1] - 200, CHART_H - 30
    cv2.rectangle(panel, (x0, y0), (x0 + w, y0 + h), GREY, 1)
    cv2.putText(panel, "taken", (12, y0 + 24), FONT, 0.7, GREY, 2, cv2.LINE_AA)
    start = trace[0][1] + trace[0][2]
    taken = [(t, start - g - e) for t, g, e, _ in trace]
    seconds = max(CHART_SECONDS, taken[-1][0] * 1.1)
    most = max(max(v for _, v in taken), 1)
    xy = [(x0 + int(w * t / seconds), y0 + h - int(h * v / most)) for t, v in taken]
    steps = [xy[0]] + [q for a, b in zip(xy, xy[1:]) for q in ((b[0], a[1]), b)]
    cv2.polylines(panel, [np.array(steps, np.int32)], False, GREEN, 3, cv2.LINE_AA)
    t, v = taken[-1]
    cv2.putText(panel, f"{v / 1e6:.2f}M at {t:.0f} s", (12, y0 + h), FONT, 0.7, GREEN, 2, cv2.LINE_AA)


def with_header(image: np.ndarray, rows: list[list[tuple[str, tuple]]], trace: list[list] | None = None) -> np.ndarray:
    """The image under a text panel, so the panel never hides the game. Each row is a list
    of (text, colour) runs drawn left to right, then a strip for a battle's loot chart (blank otherwise,
    so the window keeps one size)."""

    text_h = ROW_H * len(rows) + 12
    panel = np.zeros((text_h + CHART_H, image.shape[1], 3), np.uint8)
    for r, row in enumerate(rows):
        x, y = 12, ROW_H * (r + 1)
        for text, color in row:
            cv2.putText(panel, text, (x, y), FONT, 0.85, color, 2, cv2.LINE_AA)
            x += cv2.getTextSize(text, FONT, 0.85, 2)[0][0] + 24
    if trace:
        draw_loot_chart(panel, text_h, trace)
    return np.vstack([panel, image])
