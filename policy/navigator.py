"""Hmmm...what to do? Uses the current screen from vision."""

from dataclasses import dataclass

from policy.thresholds import should_attack
from vision.loot import LootReading
from vision.screens import Screen

# How many unrecognized windows to allow before exiting
PATIENCE = 4

# Random popups may require backs so we allow up to this many
MAX_BACKS = 3

TAP = "tap"
BACK = "back"
WAIT = "wait"
DEPLOY = "deploy"
ABORT = "abort"


@dataclass(frozen=True)
class Action:
    kind: str
    anchor: str | None = None
    why: str = ""


def decide(screen: Screen, reading: LootReading | None, unknown_streak: int) -> Action:
    """Making bad decisions. Given a screen, the loot showing on it, and how long we have
    been lost, return an action."""

    if screen is Screen.HOME:
        return Action(TAP, "attack", "open the army screen")

    if screen is Screen.ARMY:
        return Action(TAP, "attack-button", "open the attack menu")

    if screen is Screen.ATTACK_MENU:
        return Action(TAP, "find-match", "spend 900 to find a base")

    if screen is Screen.SCOUT:
        # No reading, or one we could not trust, skips the base: a wasted 900 is far
        # cheaper than an army spent on loot that was guessed at.
        if reading is not None and should_attack(reading):
            return Action(DEPLOY, why="loot clears the thresholds")
        return Action(TAP, "next-button", "spend 900 on the next base")

    if unknown_streak < PATIENCE:
        waited = f"{unknown_streak + 1}/{PATIENCE}"
        return Action(WAIT, why=f"unrecognized, waiting for clouds to clear ({waited})")
    if unknown_streak < PATIENCE + MAX_BACKS:
        return Action(BACK, why="still unrecognized, pressing back")
    return Action(ABORT, why=f"unrecognized for {unknown_streak} frames, giving up")
