"""Naming the screen the bot is looking at, from the anchors found on it."""

from enum import Enum

import numpy as np

from vision.anchors import Match, find_anchors


class Screen(Enum):
    HOME = "home"
    ATTACK_MENU = "attack_menu"
    ARMY = "army"
    SCOUT = "scout"
    UNKNOWN = "unknown"


# First rule whose anchors are all present wins, so the order is the tie-breaker.
RULES: tuple[tuple[Screen, frozenset[str]], ...] = (
    (Screen.SCOUT, frozenset({"next-button"})),
    (Screen.ATTACK_MENU, frozenset({"find-match"})),
    (Screen.ARMY, frozenset({"attack-button"})),
    (Screen.HOME, frozenset({"attack"})),
)


def classify(matches: list[Match]) -> Screen:
    """Name the screen these anchors belong to."""

    found = {match.label for match in matches}
    for screen, required in RULES:
        if required <= found:
            return screen
    return Screen.UNKNOWN


def identify(frame: np.ndarray, templates: dict[str, np.ndarray]) -> tuple[Screen, list[Match]]:
    """Locate every anchor in a frame and name the screen."""

    matches = find_anchors(frame, templates)
    return classify(matches), matches
