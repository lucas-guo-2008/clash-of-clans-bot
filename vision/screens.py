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
#
# SCOUT leads because `end-battle` shows there as well as in a live battle, and HOME trails
# because the home village stays visible behind the army and attack-menu panels.
RULES: tuple[tuple[Screen, frozenset[str]], ...] = (
    (Screen.SCOUT, frozenset({"next-button"})),
    (Screen.ATTACK_MENU, frozenset({"find-match"})),
    (Screen.ARMY, frozenset({"attack-button"})),
    (Screen.HOME, frozenset({"attack"})),
)


def classify(matches: list[Match]) -> Screen:
    """Name the screen these anchors belong to.

    Anything matching no rule is UNKNOWN: the clouds between bases, a popup, or a screen
    nobody has looked at yet. The clouds carry no buttons at all, so they are indistinguishable
    from a popup here -- telling them apart is the caller's job, and it does it by waiting.
    Clouds clear on their own; a popup does not.
    """

    found = {match.label for match in matches}
    for screen, required in RULES:
        if required <= found:
            return screen
    return Screen.UNKNOWN


def identify(frame: np.ndarray, templates: dict[str, np.ndarray]) -> tuple[Screen, list[Match]]:
    """Locate every anchor in a frame and name the screen. Returns both.

    The matches come back alongside the name because the caller needs them to tap: control
    takes a Match something actually found, never a coordinate it guessed.
    """

    matches = find_anchors(frame, templates)
    return classify(matches), matches
