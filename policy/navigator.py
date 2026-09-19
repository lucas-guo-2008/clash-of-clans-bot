"""Hmmm...what to do? Uses the current screen from vision."""

from dataclasses import dataclass, replace

from vision.screens import Screen

# How many unrecognized windows to allow before exiting
PATIENCE = 4

# Random popups may require backs so we allow up to this many
MAX_BACKS = 3

TAP = "tap"
BACK = "back"
WAIT = "wait"
DONE = "done"
ABORT = "abort"


@dataclass(frozen=True)
class Action:
    kind: str
    anchor: str | None = None
    why: str = ""


@dataclass(frozen=True)
class Progress:
    max_nexts: int
    nexts_used: int = 0
    unknown_streak: int = 0
    searched: bool = False  # a Find a Match has been paid for this run

    @property
    def budget_spent(self) -> bool:
        return self.nexts_used >= self.max_nexts


def decide(screen: Screen, progress: Progress) -> Action:
    """Making bad decisions. Given a screen and progress, return an action."""
    if screen is Screen.HOME:
        if progress.searched:
            return Action(DONE, why="home again, run complete")
        return Action(TAP, "attack", "open the army screen")

    if screen is Screen.ARMY:
        return Action(TAP, "attack-button", "open the attack menu")

    if screen is Screen.ATTACK_MENU:
        return Action(TAP, "find-match", "spend 900 to find a base")

    if screen is Screen.SCOUT:
        if progress.budget_spent:
            return Action(TAP, "end-battle", "budget spent, leaving without attacking")
        spend = f"{progress.nexts_used + 1}/{progress.max_nexts}"
        return Action(TAP, "next-button", f"spend 900 on the next base ({spend})")

    if progress.unknown_streak < PATIENCE:
        waited = f"{progress.unknown_streak + 1}/{PATIENCE}"
        return Action(WAIT, why=f"unrecognized, waiting for clouds to clear ({waited})")
    if progress.unknown_streak < PATIENCE + MAX_BACKS:
        return Action(BACK, why="still unrecognized, pressing back")
    return Action(ABORT, why=f"unrecognized for {progress.unknown_streak} frames, giving up")


def advance(progress: Progress, action: Action) -> Progress:
    """More bad decisions. Given the previous progress and an action taken, return an updated progress object."""
    if action.kind in (WAIT, BACK):
        return replace(progress, unknown_streak=progress.unknown_streak + 1)
    if action.anchor == "find-match":
        return replace(progress, searched=True, unknown_streak=0)
    if action.anchor == "next-button":
        return replace(progress, nexts_used=progress.nexts_used + 1, unknown_streak=0)
    return replace(progress, unknown_streak=0)
