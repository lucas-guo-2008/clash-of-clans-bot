"""Shared fixtures. Saved frames are local data (gitignored), so a test whose frame is
missing is skipped, not failed -- a fresh clone runs the pure-logic tests only."""

import sys
from pathlib import Path

import cv2
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vision.buttons import load_button_templates  # noqa: E402
from vision.glyphs import load_digit_templates  # noqa: E402
from vision.result import load_result_templates  # noqa: E402

FRAMES = ROOT / "templates" / "initial_collection"
UNKNOWN = ROOT / "templates" / "unknown"


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
