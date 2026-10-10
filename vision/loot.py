"""Reads the "Available Loot" panel on the attack-search screen.

All constants are in 1920x1080 device pixels, measured from templates/initial_collection/adb_frame_4.png.
The pipeline is shared with any three-row readout drawn in COC's digit font (see vision/result.py);
only the row placement (panel) and font size are different
"""

from dataclasses import dataclass

import cv2
import numpy as np

FRAME_SIZE = (1080, 1920)  # (h, w)

ROW_NAMES = ("gold", "elixir", "dark_elixir")

# Numbers have tint so need tolerance
WHITE_MIN = 150
TINT_TOL = 70


@dataclass(frozen=True)
class Font:
    """One of COC's digit sizes. Templates only match glyphs cut at the same size."""

    min_h: int
    max_h: int
    min_w: int  # "1" is very narrow
    max_w: int
    min_area: int
    canvas: tuple[int, int]  # (h, w) every glyph is pasted into before matching
    space_gap: int  # blank px that separates thousands


@dataclass(frozen=True)
class Panel:
    """Where a gold/elixir/dark readout sits on the frame, and the font it is drawn in."""

    x: tuple[int, int]
    bands: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]  # y-range of each row
    font: Font


LOOT_FONT = Font(min_h=18, max_h=28, min_w=6, max_w=26, min_area=30, canvas=(26, 22), space_gap=7)
LOOT_X = (95, 420)
ROW_BANDS = ((148, 192), (205, 245), (262, 300))  # gold, elixir, dark elixir
LOOT_PANEL = Panel(LOOT_X, ROW_BANDS, LOOT_FONT)

MAX_SHIFT = 2

MIN_SCORE = 0.93
MIN_MARGIN = 0.05

# Need to change this during resource events
PLAUSIBLE_MAX = (2_500_000, 2_500_000, 100_000)


@dataclass(frozen=True)
class Glyph:
    """One segmented digit: its bbox in ROI coordinates plus its isolated mask."""

    x: int
    y: int
    w: int
    h: int
    mask: np.ndarray

    @property
    def right(self) -> int:
        return self.x + self.w


@dataclass(frozen=True)
class RowRead:
    """Outcome of reading one loot row, including per-glyph scores for logging."""

    value: int | None
    scores: tuple[float, ...]
    reason: str | None  # None when the row read cleanly

    @property
    def ok(self) -> bool:
        return self.reason is None


@dataclass(frozen=True)
class LootReading:
    gold: int | None
    elixir: int | None
    dark_elixir: int | None
    ok: bool
    rows: tuple[RowRead, RowRead, RowRead]


def binarize(roi: np.ndarray) -> np.ndarray:
    """Tint-tolerant near-white mask. Returns uint8 0/1."""

    roi = roi.astype(np.int16)
    low = roi.min(axis=2)
    high = roi.max(axis=2)
    return ((low > WHITE_MIN) & ((high - low) < TINT_TOL)).astype(np.uint8)


def segment_glyphs(mask: np.ndarray, font: Font = LOOT_FONT) -> list[Glyph]:
    """Split a binarized row into individual digits, left to right.

    Each glyph carries a black outline in COC's font, so connected components
    separate cleanly without any erosion or morphology first.
    """

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    glyphs = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if area < font.min_area:
            continue
        if not (font.min_h <= h <= font.max_h):
            continue
        if not (font.min_w <= w <= font.max_w):
            continue
        isolated = (labels[y : y + h, x : x + w] == label).astype(np.uint8)
        glyphs.append(Glyph(int(x), int(y), int(w), int(h), isolated))

    glyphs.sort(key=lambda g: g.x)
    return glyphs


def canonicalize(glyph: Glyph, font: Font = LOOT_FONT) -> np.ndarray:
    """Paste a glyph into a fixed canvas: top-aligned, horizontally centred."""

    canvas_h, canvas_w = font.canvas
    canvas = np.zeros((canvas_h, canvas_w), np.uint8)
    h = min(glyph.h, canvas_h)
    w = min(glyph.w, canvas_w)
    x0 = (canvas_w - w) // 2
    canvas[0:h, x0 : x0 + w] = glyph.mask[0:h, 0:w]
    return canvas


def _shift(canvas: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate a canvas by (dy, dx), padding with background."""

    out = np.zeros_like(canvas)
    h, w = canvas.shape
    y0, y1 = max(0, dy), min(h, h + dy)
    x0, x1 = max(0, dx), min(w, w + dx)
    out[y0:y1, x0:x1] = canvas[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
    return out


def best_agreement(canvas: np.ndarray, template: np.ndarray) -> float:
    """Find agreement which allows the glyph to be off by a pixel or two."""

    return max(
        float((_shift(canvas, dy, dx) == template).mean())
        for dy in range(-MAX_SHIFT, MAX_SHIFT + 1)
        for dx in range(-MAX_SHIFT, MAX_SHIFT + 1)
    )


def classify(canvas: np.ndarray, templates: dict[str, np.ndarray]) -> tuple[str, float, float]:
    """Find nearest template by pixel agreement. Returns (digit, score, margin)."""

    scores = sorted(
        ((best_agreement(canvas, tmpl), digit) for digit, tmpl in templates.items()),
        reverse=True,
    )
    best_score, best_digit = scores[0]
    runner_up = scores[1][0] if len(scores) > 1 else 0.0
    return best_digit, best_score, best_score - runner_up


def group_sizes(glyphs: list[Glyph], font: Font = LOOT_FONT) -> list[int]:
    """Digit counts get split on the wide blank spaces (which separate thousands)."""

    sizes = [1]
    for previous, current in zip(glyphs, glyphs[1:]):
        if current.x - previous.right >= font.space_gap:
            sizes.append(1)
        else:
            sizes[-1] += 1
    return sizes


def grouping_is_valid(sizes: list[int]) -> bool:
    """Check if we think we see a ISO 31-0 thousands grouping"""

    if not sizes or not (1 <= sizes[0] <= 3):
        return False
    return all(size == 3 for size in sizes[1:])


def parse_row(row_mask: np.ndarray, templates: dict[str, np.ndarray], font: Font = LOOT_FONT) -> RowRead:
    """Read one binarized loot row."""

    glyphs = segment_glyphs(row_mask, font)
    if not glyphs:
        return RowRead(None, (), "no glyphs")

    digits, scores, margins = "", [], []
    for glyph in glyphs:
        digit, score, margin = classify(canonicalize(glyph, font), templates)
        digits += digit
        scores.append(score)
        margins.append(margin)

    scores = tuple(scores)

    if min(scores) < MIN_SCORE:
        return RowRead(None, scores, f"low score {min(scores):.3f}")
    if min(margins) < MIN_MARGIN:
        return RowRead(None, scores, f"low margin {min(margins):.3f}")

    sizes = group_sizes(glyphs, font)
    if not grouping_is_valid(sizes):
        return RowRead(None, scores, f"bad grouping {sizes}")

    return RowRead(int(digits), scores, None)


def read_loot(frame: np.ndarray, templates: dict[str, np.ndarray]) -> LootReading:
    """Read all three loot rows from a full attack-search frame."""

    return read_panel(frame, templates, LOOT_PANEL)


def read_panel(frame: np.ndarray, templates: dict[str, np.ndarray], panel: Panel) -> LootReading:
    """Read a gold/elixir/dark readout. Templates must be cut at the panel's font size."""

    if frame.shape[:2] != FRAME_SIZE:
        raise ValueError(
            f"expected a {FRAME_SIZE[1]}x{FRAME_SIZE[0]} frame, got {frame.shape[1]}x{frame.shape[0]}; "
            "every constant in this module is in device pixels"
        )

    x0, x1 = panel.x
    rows = tuple(parse_row(binarize(frame[y0:y1, x0:x1]), templates, panel.font) for y0, y1 in panel.bands)

    values = [row.value for row in rows]
    ok = all(row.ok for row in rows)
    for value, ceiling in zip(values, PLAUSIBLE_MAX):
        if value is not None and value > ceiling:
            ok = False

    return LootReading(
        gold=values[0] if ok else None,
        elixir=values[1] if ok else None,
        dark_elixir=values[2] if ok else None,
        ok=ok,
        rows=rows,
    )
