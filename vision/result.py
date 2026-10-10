"""Reads "You got" on the result screen: the loot one battle brought in.

Same pipeline as the loot panel, but it has its own templates.
Constants measured from templates/initial_collection/adb_frame_{8,18..22}.png
glyphs are 29-32 px tall and 12-29 px wide, digits sit 2-4 px apart and thousands 15-17 px apart.
"""

from pathlib import Path

import numpy as np

from vision.glyphs import load_digit_templates
from vision.loot import Font, LootReading, Panel, read_panel

RESULT_FONT = Font(min_h=26, max_h=35, min_w=8, max_w=34, min_area=60, canvas=(32, 30), space_gap=9)
RESULT_PANEL = Panel(x=(700, 1005), bands=((470, 530), (540, 602), (610, 672)), font=RESULT_FONT)

RESULT_DIGITS_DIR = Path(__file__).resolve().parent.parent / "templates" / "result_digits"


def load_result_templates() -> dict[str, np.ndarray]:
    return load_digit_templates(RESULT_DIGITS_DIR, RESULT_FONT)


def read_result(frame: np.ndarray, templates: dict[str, np.ndarray]) -> LootReading:
    """Read the gold/elixir/dark a battle brought in, from its result screen."""

    return read_panel(frame, templates, RESULT_PANEL)
