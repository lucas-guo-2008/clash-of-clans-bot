"""Hmmm...what to do? Uses the current screen from vision."""

from dataclasses import dataclass

from policy.thresholds import should_attack
from vision.loot import LootReading
from vision.screens import Screen

# How many unrecognized frames in a row to wait out before exiting. Waiting is the only
# recovery: clouds clear on their own, and a back press mid-search or mid-battle does harm.
# Counted in loop steps, not seconds -- the longest cloud seen live lasted 4.
PATIENCE = 8

TAP = "tap"
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

    if screen is Screen.BATTLE:
        # Never end a battle early: the game ends it when the last troop dies or time runs out.
        return Action(WAIT, why="battle in progress")

    if screen is Screen.RESULT:
        return Action(TAP, "return-home", "battle over, return home")

    if unknown_streak < PATIENCE:
        waited = f"{unknown_streak + 1}/{PATIENCE}"
        return Action(WAIT, why=f"unrecognized, waiting for clouds to clear ({waited})")
    return Action(ABORT, why=f"unrecognized for {unknown_streak} frames, giving up")
