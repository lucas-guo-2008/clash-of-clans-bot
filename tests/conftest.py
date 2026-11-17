"""Shared fixtures. Saved frames are local data (gitignored), so a test whose frame is
missing is skipped, not failed -- a fresh clone runs the pure-logic tests only."""

import sys
from pathlib import Path

import cv2
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import main  # noqa: E402
from policy import thresholds  # noqa: E402
from policy.navigator import STAR  # noqa: E402
from vision.buttons import load_button_templates  # noqa: E402
from vision.glyphs import load_digit_templates  # noqa: E402
from vision.result import load_result_templates  # noqa: E402

FRAMES = ROOT / "templates" / "initial_collection"
UNKNOWN = ROOT / "templates" / "unknown"


# What the saved frames were recorded with. main.py and policy/ hold whatever the last live run needed
# (a data run's zero thresholds, tonight's deck, a demo's full battles); the tests must not depend on it.
MIN_GOLD, MIN_ELIXIR, MIN_DARK = 400_000, 400_000, 2_000
DECK = (("troop", 24), ("skip", 0), ("hero", 1), ("hero", 1), ("spell", 11))  # the five cards on frames 9-45
SIX_CARD_DECK = (("troop", 24), ("skip", 0), ("hero", 1), ("hero", 1), ("hero", 1), ("spell", 11))  # frame 46


@pytest.fixture(autouse=True)
def pinned_settings(monkeypatch):
    monkeypatch.setattr(thresholds, "MIN_GOLD", MIN_GOLD)
    monkeypatch.setattr(thresholds, "MIN_ELIXIR", MIN_ELIXIR)
    monkeypatch.setattr(thresholds, "MIN_DARK", MIN_DARK)
    monkeypatch.setattr(main, "DECK", DECK)
    monkeypatch.setattr(main, "ITERATIONS", 1)
    monkeypatch.setattr(main, "ARMS", (STAR,))


def _load(path: Path):
    if not path.exists():
        pytest.skip(f"{path.name} is not on this machine")
    return cv2.imread(str(path))


@pytest.fixture
def frame():
    """frame(n) loads adb_frame_n.png."""
    return lambda n: _load(FRAMES / f"adb_frame_{n}.png")


@pytest.fixture
def unknown_frame():
    """unknown_frame(name) loads a frame main.py saved as unrecognized."""
    return lambda name: _load(UNKNOWN / name)


@pytest.fixture(scope="session")
def buttons():
    return load_button_templates()


@pytest.fixture(scope="session")
def digits():
    return load_digit_templates()


@pytest.fixture(scope="session")
def result_digits():
    return load_result_templates()
