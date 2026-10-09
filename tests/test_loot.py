"""The loot panel reads exactly what is on screen, and refuses a frame it cannot trust."""

import cv2
import pytest

from vision.loot import read_loot

# (gold, elixir, dark elixir) as shown on each scout frame's Available Loot panel.
LOOT = {
    4: (525_962, 1_333_432, 29_717),
    5: (719_946, 565_118, 3_780),
    6: (395_353, 582_273, 8_004),
    10: (748_298, 624_159, 6_044),
    11: (136_431, 185_623, 1_447),
    12: (300_729, 207_396, 2_763),
    13: (466_957, 787_731, 9_127),
    14: (754_898, 783_766, 7_480),
    15: (592_661, 531_561, 11_244),
    16: (592_661, 531_561, 11_244),
    17: (592_661, 531_561, 11_244),
}


@pytest.mark.parametrize("n, want", LOOT.items())
def test_loot_reads_exactly(frame, digits, n, want):
    reading = read_loot(frame(n), digits)
    assert reading.ok
    assert (reading.gold, reading.elixir, reading.dark_elixir) == want


@pytest.mark.parametrize("n", [1, 3, 8])  # home, army, result: no loot panel
def test_no_panel_is_not_a_reading(frame, digits, n):
    assert not read_loot(frame(n), digits).ok


def test_resized_frame_is_refused(frame, digits):
    with pytest.raises(ValueError):
        read_loot(cv2.resize(frame(5), (960, 540)), digits)
