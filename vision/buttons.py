"""Loading the button anchor template set."""

from pathlib import Path

import cv2

from vision.anchors import Template

BUTTONS_DIR = Path(__file__).resolve().parent.parent / "templates" / "buttons"
FRAME_W, FRAME_H = 1920, 1080

BUTTON_AT = {
    "battle": (66, 1011),  # "Battle" lettering only (home UI revamp, frame 25)
    "attack-button": (1516, 926),
    "find-match": (112, 734),
    "next-button": (1599, 703),
    "end-battle": (30, 773),
    "surrender": (50, 788),
    "return-home": (895, 895),
    "okay": (1099, 672),  # "Okay" lettering on the End Battle? confirmation (frame 39)
    # Reward screens, 2026-11-20. Worst score on any other saved frame (1,567 checked) in brackets.
    "claim-reward": (870, 908),  # "Claim Reward" lettering where Return Home sits, on the result
    # screen after the day's star bonus progress earns a crate (unknown_20261116_211314) [0.407]
    "crate": (1050, 868),  # the rightmost rolling pin under a reward crate (frame 48); it turns
    # dark on the last tap the crate needs [0.485]
    "continue": (835, 900),  # "Continue" lettering under an opened crate's reward (frame 52) [0.387]
    "claim": (942, 825),  # "Claim" lettering on the Star Bonus Received! popup (frame 53) [0.411]
}
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
