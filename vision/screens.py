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
    CRATE = "crate"  # a reward crate waiting for taps
    REWARD = "reward"  # what the crate held, with Continue
    STAR_BONUS = "star_bonus"  # the Star Bonus Received! popup over the home village
    UNKNOWN = "unknown"


# First rule whose anchor is present wins, so the order is the tie-breaker.
RULES: tuple[tuple[Screen, str], ...] = (
    # The End Battle? modal dims the battle behind it, and end-battle still scores 1.000 there.
    (Screen.END_CONFIRM, "okay"),
    (Screen.SCOUT, "next-button"),
    (Screen.BATTLE, "end-battle"),
    (Screen.RESULT, "return-home"),
    (Screen.RESULT, "claim-reward"),  # the result screen when the battle earned a reward crate
    (Screen.BATTLE, "surrender"),
    (Screen.ATTACK_MENU, "find-match"),
    (Screen.ARMY, "attack-button"),
    (Screen.CRATE, "crate"),
    (Screen.REWARD, "continue"),
    # The popup covers the home village, and battle still scores 1.000 behind it (frame 53).
    (Screen.STAR_BONUS, "claim"),
    (Screen.HOME, "battle"),
)


def classify(matches: dict[str, Match]) -> Screen:
    """Name the screen these anchors belong to."""

    for screen, anchor in RULES:
        if anchor in matches:
            return screen
    return Screen.UNKNOWN

