"""main.py end to end, over saved frames, with a fake device that records taps instead of sending them."""

import sys

import pytest

import main
from policy.navigator import PATIENCE
from vision.deploy import deploy_points, first_card

CLOUD = "unknown_20260919_005725_732450.png"
RETURN_HOME = (961, 925)  # centre of the return-home match on frame 8


class FakeDevice:
    def __init__(self):
        self.taps = []

    def click(self, x, y):
        self.taps.append((x, y))


@pytest.fixture
def run(frame, unknown_frame, monkeypatch):
    """run(sequence) plays frames through main.main(); returns (exit code, taps)."""

    def play(sequence):
        frames = iter([unknown_frame(CLOUD) if s == "cloud" else frame(s) for s in sequence])
        device = FakeDevice()

        def grab(_):
            try:
                return next(frames)
            except StopIteration:
                raise AssertionError("main.py was still running when the frames ran out") from None

        monkeypatch.setattr(main, "connect", lambda: device)
        monkeypatch.setattr(main, "grab_frame", grab)
        monkeypatch.setattr(main.time, "sleep", lambda _: None)
        monkeypatch.setattr(main, "save_unknown", lambda _: main.ROOT / "not-saved-in-tests.png")
        monkeypatch.setattr(sys, "argv", ["main.py"])
        return main.main(), device.taps

    return play


def test_one_full_attack(run, frame):
    # home, army, attack menu, clouds, a rich scout, battle, result (two frames), clouds, home
    code, taps = run([1, 3, 2, "cloud", "cloud", 5, 7, 7, 8, 8, "cloud", 1])

    card = first_card(frame(5)).centre
    points = {(p.x, p.y) for p in deploy_points(frame(5))}
    start = taps.index(card) + 1
    drops = taps[start : start + main.TROOPS]

    assert code == 0
    assert len(drops) == main.TROOPS and set(drops) <= points
    assert taps[start + main.TROOPS :] == [RETURN_HOME, RETURN_HOME]


def test_long_battle_does_not_use_up_patience(run):
    # The unknown frame must come before the result screen: tapping Return Home would reset the count.
    code, _ = run([5, *[7] * (PATIENCE + 2), "cloud", 8, "cloud", 1])
    assert code == 0


def test_no_outline_means_no_taps(run, frame):
    code, taps = run([4, 5, 7, 8, "cloud", 1])
    assert code == 0
    assert taps[0] == first_card(frame(5)).centre != first_card(frame(4)).centre


def test_endless_unknown_aborts_without_tapping(run):
    code, taps = run(["cloud"] * (PATIENCE + 1))
    assert code == 1
    assert taps == []
