"""Loading and building the digit template set.
Templates are stored as binary PNGs (0 or 255) of the glyphs
"""

from pathlib import Path

import cv2
import numpy as np

from vision.loot import LOOT_FONT, Font

DIGITS = "0123456789"
DIGITS_DIR = Path(__file__).resolve().parent.parent / "templates" / "digits"


def load_digit_templates(directory: Path = DIGITS_DIR, font: Font = LOOT_FONT) -> dict[str, np.ndarray]:
    templates = {}
    for digit in DIGITS:
        path = directory / f"{digit}.png"
        if not path.exists():
            continue
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"could not read template {path}")
        if image.shape != font.canvas:
            raise ValueError(
                f"{path} is {image.shape}, expected {font.canvas} -- "
                "re-cut it with tools/extract_digits.py"
            )
        templates[digit] = (image > 127).astype(np.uint8)

    missing = [d for d in DIGITS if d not in templates]
    if missing:
        raise FileNotFoundError(
            f"no template for digit(s) {''.join(missing)} in {directory}. "
            "Capture more frames containing them and run tools/extract_digits.py"
        )
    return templates


def majority_vote(samples: list[np.ndarray]) -> np.ndarray:
    """Combine several samples of one digit into a template, because 5 and 9 are too similar."""

    stack = np.stack(samples).astype(np.uint16)
    return (stack.sum(axis=0) * 2 > len(samples)).astype(np.uint8)


def save_template(canvas: np.ndarray, digit: str, directory: Path = DIGITS_DIR) -> Path:
    """Write one glyph out as a binary PNG."""

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digit}.png"
    cv2.imwrite(str(path), canvas.astype(np.uint8) * 255)
    return path
