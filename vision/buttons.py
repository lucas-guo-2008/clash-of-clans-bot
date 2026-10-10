"""Loading the button anchor template set."""

from pathlib import Path

import cv2

from vision.anchors import Template

BUTTONS_DIR = Path(__file__).resolve().parent.parent / "templates" / "buttons"
FRAME_W, FRAME_H = 1920, 1080

BUTTON_AT = {
    "attack": (32, 900),
    "attack-button": (1516, 926),
    "find-match": (112, 734),
    "next-button": (1599, 703),
    "end-battle": (30, 773),
    "surrender": (50, 788),
    "return-home": (895, 895),
}
BUTTONS = tuple(BUTTON_AT)
SEARCH_MARGIN = 40


def load_button_templates() -> dict[str, Template]:
    templates = {}
    for name, (x, y) in BUTTON_AT.items():
        path = BUTTONS_DIR / f"{name}.png"
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(
                f"could not read button template {path}. Crop it from a saved frame, "
                "keeping clear of any shine sweep -- the animation breaks correlation."
            )
        h, w = image.shape[:2]
        region = (
            max(0, x - SEARCH_MARGIN),
            max(0, y - SEARCH_MARGIN),
            min(FRAME_W, x + w + SEARCH_MARGIN),
            min(FRAME_H, y + h + SEARCH_MARGIN),
        )
        templates[name] = Template(image, region)
    return templates
