""""You got" on the result screen reads exactly, and nothing else reads as a result."""

import cv2
import pytest

from vision.result import read_result

YOU_GOT = {
    8: (3_623, 5_440, 96),  # its dark-row samples were dropped as odd, and it must still read
    18: (358_306, 419_714, 4_272),
    19: (463_831, 398_186, 8_549),
    20: (462_138, 461_718, 3_720),
    21: (639_108, 625_000, 6_491),  # held back
    22: (300_417, 123_312, 1_342),  # held back
}


@pytest.mark.parametrize("n, want", YOU_GOT.items())
def test_you_got_reads_exactly(frame, result_digits, n, want):
    reading = read_result(frame(n), result_digits)
    assert reading.ok
    assert (reading.gold, reading.elixir, reading.dark_elixir) == want


def test_two_row_result_is_unreadable_not_guessed(frame, result_digits):
    # Frame 0 has no dark elixir row, so its two rows sit lower than the three-row bands.
    assert not read_result(frame(0), result_digits).ok


@pytest.mark.parametrize("n", [1, 3, 5, 7, 11])  # home, army, scout, battle, scout
def test_other_screens_have_no_result(frame, result_digits, n):
    assert not read_result(frame(n), result_digits).ok


def test_resized_frame_is_refused(frame, result_digits):
    with pytest.raises(ValueError):
        read_result(cv2.resize(frame(18), (960, 540)), result_digits)
