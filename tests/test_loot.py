"""The loot panel reads exactly what is on screen, and refuses a frame it cannot trust."""

import cv2
import pytest

from vision.loot import read_loot

# (gold, elixir, dark elixir) as shown on each scout frame's Available Loot panel. Frames 26-37 are
# the first bases main.py saved after the 2026-10-29 UI revamp moved the panel up 14 px.
LOOT = {
    24: (742_322, 744_046, 9_140),
    26: (535_269, 397_807, 488),
    27: (601_454, 568_993, 5_614),  # textures still loading: the first base of the session
    28: (263_550, 656_735, 5_357),
    29: (895_480, 895_480, 7_756),  # gold and elixir really are equal
    30: (904_522, 611_816, 6_953),
    31: (609_446, 889_102, 8_513),
    32: (287_352, 331_925, 3_268),
    33: (963_396, 1_339_138, 11_918),
    34: (422_625, 615_209, 4_844),
    35: (718_149, 718_149, 5_243),
    36: (714_386, 456_647, 8_137),
    37: (698_001, 848_390, 9_174),
}

# Scout frames from before the revamp, panel 14 px lower. They read exactly at the legacy bands.
LEGACY = [4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17]


@pytest.mark.parametrize("n, want", LOOT.items())
def test_loot_reads_exactly(frame, digits, n, want):
    reading = read_loot(frame(n), digits)
    assert reading.ok
    assert (reading.gold, reading.elixir, reading.dark_elixir) == want


@pytest.mark.parametrize("n", LEGACY)
def test_legacy_panel_fails_closed(frame, digits, n):
    assert not read_loot(frame(n), digits).ok


@pytest.mark.parametrize("n", [1, 3, 8])  # home, army, result: no loot panel
def test_no_panel_is_not_a_reading(frame, digits, n):
    assert not read_loot(frame(n), digits).ok


def test_resized_frame_is_refused(frame, digits):
    with pytest.raises(ValueError):
        read_loot(cv2.resize(frame(24), (960, 540)), digits)
