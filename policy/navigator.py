"""What to do next, given the screen vision named."""

from dataclasses import dataclass

from policy.thresholds import should_attack
from vision.loot import LootReading
from vision.screens import Screen

PATIENCE_SECONDS = 20.0

GATE_TIMEOUT = 4.0  # s; after this the tap is assumed missed and may be repeated
GATE_STABLE_FRAMES = 2  # a new screen must hold this many frames: a one-frame misread can't release it
STALL_SECONDS = 30.0  # Available Loot unchanged this long ends the battle (longest live mid-battle gap: 17.6 s)
STAR_STABLE_FRAMES = 2  # End Battle must hold this many frames after Surrender: a one-frame misread can't end a battle

STAR = "star"
STALL = "stall"
END_WHY = {STAR: "first star won, end the battle", STALL: "loot has stopped coming, end the battle"}
ARMS = (STAR, STALL)

TAP = "tap"
WAIT = "wait"
DEPLOY = "deploy"
ABORT = "abort"


@dataclass(frozen=True)
class Action:
    kind: str
    anchor: str | None = None
    why: str = ""


def decide(screen: Screen, reading: LootReading | None, unknown_seconds: float,
           end_with: str | None = None, end_rule: str = STALL) -> Action:
    """Given a screen, the loot showing on it, and how long we have
    been lost, return an action. `end_with` is the button
    that ends this battle (End Battle or Surrender), once main.py has judged it over by `end_rule`."""

    if screen is Screen.HOME:
        return Action(TAP, "battle", "open the attack menu")

    if screen is Screen.ARMY:
        return Action(TAP, "attack-button", "start the search")

    if screen is Screen.ATTACK_MENU:
        return Action(TAP, "find-match", "spend 900 to find a base")

    if screen is Screen.SCOUT:
        if reading is None:
            return Action(WAIT, why="waiting for the loot panel to settle")
        if should_attack(reading):
            return Action(DEPLOY, why="loot clears the thresholds")
        return Action(TAP, "next-button", "spend 900 on the next base")

    if screen is Screen.BATTLE:
        if end_with is not None:
            return Action(TAP, end_with, END_WHY[end_rule])
        return Action(WAIT, why="battle in progress")

    if screen is Screen.END_CONFIRM:
        return Action(TAP, "okay", "confirm ending the battle")

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


def battle_stalled(trace: list[list], since_deploy: float) -> bool:
    """Has this battle stopped paying? Asked only once every card is down and every hero ability
    used. `trace` is the Available Loot each time it changed, as [seconds since deploy, gold,
    elixir, dark elixir]; empty if the panel never read, and then the deploy is the last change.
    End Battle vs Surrender says nothing here: the button flips at the first star, not when the
    army is dead."""

    last_change = trace[-1][0] if trace else 0.0
    return since_deploy - last_change >= STALL_SECONDS


def star_won(surrender_seen: bool, end_battle_frames: int) -> bool:
    """Has this battle earned its first star? The button reads End Battle before the first troop
    lands, Surrender once one has, and End Battle again at the first star (Town Hall down or 50%),
    so End Battle alone is not the signal: it counts only after Surrender was seen in this battle.
    `end_battle_frames` is how many battle frames in a row have shown End Battle since then."""

    return surrender_seen and end_battle_frames >= STAR_STABLE_FRAMES
