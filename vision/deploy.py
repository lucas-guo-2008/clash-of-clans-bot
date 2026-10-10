"""Finding where troops can be dropped, from the red no-deploy outline on the scout screen.

All constants are in 1920x1080 device pixels, measured from
templates/initial_collection/adb_frame_{5,6,10..17}.png.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from vision.anchors import Match

FRAME_SIZE = (1080, 1920)  # (h, w)

# The outline is a 2 px alpha-blended red line; thinner than kernal drops red decor
RIDGE_KERNEL = 9
RIDGE_MIN = 40

MIN_SPAN = 150

CENTRE = (960, 540)
RAY_STEP_DEG = 2
STEP_OUT = 45  # past the outermost crossing
MAX_POINTS = 24

# A candidate point must also be on grass and clear of any red ridge pixel.
RED_CLEARANCE = 20
GRASS_MIN_G = 90

# Screen furniture: blanked from the mask (the Next button's rim is a long thin red edge) and never tapped. (x0, y0, x1, y1).
UI_RECTS = (
    (0, 0, 450, 320),  # available loot panel
    (840, 0, 1090, 120),  # battle timer
    (1560, 0, 1920, 230),  # resource display
    (0, 740, 900, 870),  # End Battle, Boost Army, Heroes Boosted
    (1560, 680, 1920, 870),  # Next
    (0, 870, 1920, 1080),  # troop deck
)

# Deck slots are found by their outlines, not their art (hero art is dark, spell art is
# white): a card's left and right borders are vertical edges over most of its height, and
# real cards are 130-134 px wide, 139 when raised as the selected card.
DECK_Y = (888, 1072)  # the reported card box; its centre is where a card is tapped
EDGE_Y = (900, 1055)  # inside the card body, clear of the rounded corners
EDGE_STEP = 40  # grey-level change across 2 px that counts as an edge
EDGE_FILL = 0.6  # fraction of EDGE_Y rows that must be edge for a column to be a border
SLOT_W = (125, 140)  # 140 rejects a home-screen pair at 143-144 and a stray edge on frame 7
DECK_LEFT = (120, 170)  # the first card's left border: 134-148 measured
SLOT_GAP = 45  # most blank px between neighbouring cards: 13-31 measured


@dataclass(frozen=True)
class DeployPoint:
    x: int
    y: int


def outline_mask(frame: np.ndarray) -> np.ndarray:
    b, g, r = (frame[..., i].astype(np.int16) for i in range(3))
    redness = np.clip(r - np.maximum(g, b), 0, 255).astype(np.uint8)
    kernel = np.ones((RIDGE_KERNEL, RIDGE_KERNEL), np.uint8)
    ridge = cv2.subtract(redness, cv2.morphologyEx(redness, cv2.MORPH_OPEN, kernel))
    return np.where(ridge > RIDGE_MIN, 255, 0).astype(np.uint8)


def outline_lines(mask: np.ndarray) -> np.ndarray:
    lines = mask.copy()
    for x0, y0, x1, y1 in UI_RECTS:
        lines[y0:y1, x0:x1] = 0

    _, labels, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
    span = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    keep = span >= MIN_SPAN
    keep[0] = False  # background
    return np.where(keep[np.asarray(labels, dtype=np.intp)], 255, 0).astype(np.uint8)


def deploy_points(frame: np.ndarray) -> list[DeployPoint]:
    if frame.shape[:2] != FRAME_SIZE:
        raise ValueError(
            f"expected a {FRAME_SIZE[1]}x{FRAME_SIZE[0]} frame, got {frame.shape[1]}x{frame.shape[0]}; "
            "every constant here is in device pixels"
        )

    mask = outline_mask(frame)
    lines = outline_lines(mask)
    near_red = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RED_CLEARANCE + 1,) * 2))
    h, w = mask.shape
    cx, cy = CENTRE
    reach = np.arange(int(math.hypot(w, h)))

    found = []
    for degrees in range(0, 360, RAY_STEP_DEG):
        dx, dy = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
        xs, ys = (cx + dx * reach).astype(int), (cy + dy * reach).astype(int)
        inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        hits = np.flatnonzero(lines[ys[inside], xs[inside]])
        if hits.size == 0:
            continue

        t = hits[-1] + STEP_OUT
        x, y = int(cx + dx * t), int(cy + dy * t)
        if not (0 <= x < w and 0 <= y < h) or near_red[y, x]:
            continue
        if any(x0 <= x < x1 and y0 <= y < y1 for x0, y0, x1, y1 in UI_RECTS):
            continue
        pb, pg, pr = (int(v) for v in frame[y, x])
        if pg > pb and pg > pr and pg > GRASS_MIN_G:
            found.append(DeployPoint(x, y))

    if len(found) <= MAX_POINTS:
        return found
    return [found[i] for i in np.linspace(0, len(found) - 1, MAX_POINTS).astype(int)]


def deck_slots(frame: np.ndarray) -> list[Match]:
    """Every card in the troop deck, left to right, located by its outline. Empty slots
    (dashed outlines) and screens without a deck give no slots."""

    y0, y1 = EDGE_Y
    grey = cv2.cvtColor(frame[y0:y1], cv2.COLOR_BGR2GRAY).astype(np.int16)
    fill = (np.abs(grey[:, 2:] - grey[:, :-2]) > EDGE_STEP).mean(axis=0)
    borders = [x + 1 for x in range(1, len(fill) - 1)
               if fill[x] > EDGE_FILL and fill[x] >= fill[x - 1] and fill[x] >= fill[x + 1]]

    # Pair each left border with the right border nearest a card's width, left to right;
    # a border with no partner at card width is an edge inside the art, and is skipped.
    lo, hi = SLOT_W
    slots, i = [], 0
    while i < len(borders):
        left = borders[i]
        rights = [r for r in borders[i + 1 :] if lo <= r - left <= hi]
        if not rights:
            i += 1
            continue
        right = min(rights, key=lambda r: abs(r - left - 132))
        # The deck is a chain from the left edge: anything not continuing it is not a card.
        start = slots[-1].x + slots[-1].w if slots else None
        if (start is None and not DECK_LEFT[0] <= left <= DECK_LEFT[1]) or (start is not None and left - start > SLOT_GAP):
            break
        top, bottom = DECK_Y
        score = float(min(fill[left - 1], fill[right - 1]))
        slots.append(Match(f"slot{len(slots) + 1}", left, top, right - left, bottom - top, score))
        i = next((k for k, b in enumerate(borders) if b > right), len(borders))
    return slots
