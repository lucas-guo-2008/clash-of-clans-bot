"""Hmmm...what to do? Uses the current screen from vision."""

from dataclasses import dataclass

from policy.thresholds import should_attack
from vision.loot import LootReading
from vision.screens import Screen

PATIENCE_SECONDS = 20.0

GATE_TIMEOUT = 4.0  # s; after this the tap is assumed missed and may be repeated
GATE_STABLE_FRAMES = 2  # a new screen must hold this many frames: a one-frame misread can't release it

TAP = "tap"
WAIT = "wait"
DEPLOY = "deploy"
ABORT = "abort"


@dataclass(frozen=True)
class Action:
    kind: str
    anchor: str | None = None
    why: str = ""


def decide(screen: Screen, reading: LootReading | None, unknown_seconds: float) -> Action:
    """Making bad decisions. Given a screen, the loot showing on it, and how long we have
    been lost, return an action."""

    if screen is Screen.HOME:
        return Action(TAP, "attack", "open the army screen")

    if screen is Screen.ARMY:
        return Action(TAP, "attack-button", "open the attack menu")

    if screen is Screen.ATTACK_MENU:
        return Action(TAP, "find-match", "spend 900 to find a base")

    if screen is Screen.SCOUT:
        if reading is None:
            return Action(WAIT, why="waiting for the loot panel to settle")
        if should_attack(reading):
            return Action(DEPLOY, why="loot clears the thresholds")
        return Action(TAP, "next-button", "spend 900 on the next base")

    if screen is Screen.BATTLE:
        return Action(WAIT, why="battle in progress")

    if screen is Screen.RESULT:
        if reading is None:
            return Action(WAIT, why="waiting for the loot count-up to settle")
        return Action(TAP, "return-home", "battle over, return home")

    if unknown_seconds < PATIENCE_SECONDS:
        waited = f"{unknown_seconds:.0f}/{PATIENCE_SECONDS:.0f} s"
        return Action(WAIT, why=f"unrecognized, waiting for clouds to clear ({waited})")
    return Action(ABORT, why=f"unrecognized for {unknown_seconds:.0f} s, giving up")


def gate_open(screen: Screen, tapped_on: Screen | None, seconds_since_tap: float, frames_on_screen: int) -> bool:
    """May the bot act on this frame? Yes if nothing is pending, once the screen tapped on has
    given way to another for GATE_STABLE_FRAMES frames, or once GATE_TIMEOUT has passed."""

    if tapped_on is None or seconds_since_tap >= GATE_TIMEOUT:
        return True
    return screen is not tapped_on and frames_on_screen >= GATE_STABLE_FRAMES
