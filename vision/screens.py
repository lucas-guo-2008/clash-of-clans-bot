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


# First rule whose anchor is present wins, so the order is the tie-breaker.
RULES: tuple[tuple[Screen, str], ...] = (
    (Screen.SCOUT, "next-button"),
    (Screen.ATTACK_MENU, "find-match"),
    (Screen.ARMY, "attack-button"),
    (Screen.HOME, "attack"),
)


def classify(matches: dict[str, Match]) -> Screen:
    """Name the screen these anchors belong to."""

    for screen, anchor in RULES:
        if anchor in matches:
            return screen
    return Screen.UNKNOWN


def identify(frame: np.ndarray, templates: dict[str, np.ndarray]) -> tuple[Screen, dict[str, Match]]:
    """Locate every anchor in a frame and name the screen."""

    matches = find_anchors(frame, templates)
    return classify(matches), matches
