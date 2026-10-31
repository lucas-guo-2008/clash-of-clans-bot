"""Naming the screen the bot is looking at, from the anchors found on it."""

from enum import Enum

from vision.anchors import Match


class Screen(Enum):
    HOME = "home"
    ATTACK_MENU = "attack_menu"
    ARMY = "army"
    SCOUT = "scout"
    BATTLE = "battle"
    RESULT = "result"
    END_CONFIRM = "end_confirm"
    UNKNOWN = "unknown"


# First rule whose anchor is present wins, so the order is the tie-breaker.
RULES: tuple[tuple[Screen, str], ...] = (
    # The End Battle? modal dims the battle behind it, and end-battle still scores 1.000 there.
    (Screen.END_CONFIRM, "okay"),
    (Screen.SCOUT, "next-button"),
    (Screen.BATTLE, "end-battle"),
    (Screen.RESULT, "return-home"),
    (Screen.BATTLE, "surrender"),
    (Screen.ATTACK_MENU, "find-match"),
    (Screen.ARMY, "attack-button"),
    (Screen.HOME, "battle"),
)


def classify(matches: dict[str, Match]) -> Screen:
    """Name the screen these anchors belong to."""

    for screen, anchor in RULES:
        if anchor in matches:
            return screen
    return Screen.UNKNOWN

