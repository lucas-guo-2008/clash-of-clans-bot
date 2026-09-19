"""Loading the button anchor template set."""

from pathlib import Path

import cv2
import numpy as np

BUTTONS = ("attack", "attack-button", "find-match", "next-button", "end-battle")
DEFAULT_DIR = Path(__file__).resolve().parent.parent / "templates" / "buttons"


def load_button_templates(directory: Path = DEFAULT_DIR) -> dict[str, np.ndarray]:
    """Load every anchor template as a BGR image, keyed by label."""

    templates = {}
    for name in BUTTONS:
        path = directory / f"{name}.png"
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(
                f"could not read button template {path}. Crop it from a saved frame, "
                "keeping clear of any shine sweep -- the animation breaks correlation."
            )
        templates[name] = image
    return templates
