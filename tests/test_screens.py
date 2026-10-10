"""Every saved frame is named as the right screen, with room to spare around THRESHOLD."""

import cv2
import pytest

from vision.anchors import THRESHOLD, peak_scores
from vision.screens import classify

SCREENS = {
    0: "result", 1: "home", 2: "attack_menu", 3: "army",
    4: "scout", 5: "scout", 6: "scout", 7: "battle", 8: "result",
    9: "battle",  # the scout timer ran out with no troop down: End Battle, no Next
    **{n: "scout" for n in range(10, 18)},
    **{n: "result" for n in range(18, 23)},
    23: "home",  # the Attack! map icon in a different pose from frame 1; the old whole-button template scored 0.763
}

UNRECOGNIZED = [
    ("unknown_20260919_005437_204263.png", "unknown"),
    ("unknown_20260919_005539_938030.png", "unknown"),
    ("unknown_20260919_005548_874261.png", "unknown"),
    ("unknown_20260919_005553_346372.png", "battle"),  # scout fading into clouds; waiting is right
    ("unknown_20260919_005725_732450.png", "unknown"),
    pytest.param(
        "unknown_20260930_222752_292758.png", "unknown",
        marks=pytest.mark.xfail(strict=True, reason=(
            "known gap, left unguarded: the inactivity popup dims the result screen, and "
            "TM_CCOEFF_NORMED cannot see uniform dimming, so Return Home still scores 1.000"
        )),
    ),
]

MARGIN = 0.05


def named(image, buttons) -> str:
    peaks = peak_scores(image, buttons)
    return classify({label: m for label, m in peaks.items() if m.score >= THRESHOLD}).value


@pytest.mark.parametrize("n, want", SCREENS.items())
def test_saved_frame_is_named(frame, buttons, n, want):
    assert named(frame(n), buttons) == want


@pytest.mark.parametrize("name, want", UNRECOGNIZED)
def test_unrecognized_frame_stays_unrecognized(unknown_frame, buttons, name, want):
    assert named(unknown_frame(name), buttons) == want


def test_portrait_frame_is_unknown_not_a_crash(frame, buttons):
    # The emulator sends 1080x1920 when the game is closed (Android's home screen is portrait).
    # The button search regions do not fit; that must read as unknown, so the bot waits, then stops.
    assert named(cv2.rotate(frame(1), cv2.ROTATE_90_CLOCKWISE), buttons) == "unknown"


@pytest.mark.parametrize("n", SCREENS)
def test_no_score_sits_near_the_threshold(frame, buttons, n):
    near = {label: round(m.score, 3) for label, m in peak_scores(frame(n), buttons).items()
            if abs(m.score - THRESHOLD) < MARGIN}
    assert not near
