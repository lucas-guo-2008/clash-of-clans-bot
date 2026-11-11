"""decide() and should_attack() are pure, so these need no frames at all."""

import pytest

from policy.navigator import (ABORT, DEPLOY, GATE_STABLE_FRAMES, GATE_TIMEOUT, PATIENCE_SECONDS, STALL_SECONDS,
                              STAR_STABLE_FRAMES, TAP, WAIT, battle_stalled, decide, gate_open, star_won)
from policy.thresholds import MIN_DARK, MIN_ELIXIR, MIN_GOLD, should_attack
from vision.loot import LootReading
from vision.screens import Screen


def loot(gold, elixir, dark, ok=True) -> LootReading:
    return LootReading(gold, elixir, dark, ok, rows=())  # rows only matter for logging


RICH = loot(MIN_GOLD, MIN_ELIXIR, MIN_DARK)


@pytest.mark.parametrize("screen, kind, anchor", [
    (Screen.HOME, TAP, "battle"),
    (Screen.ARMY, TAP, "attack-button"),
    (Screen.ATTACK_MENU, TAP, "find-match"),
    (Screen.BATTLE, WAIT, None),
    (Screen.END_CONFIRM, TAP, "okay"),
])
def test_each_screen_has_its_action(screen, kind, anchor):
    action = decide(screen, None, 0)
    assert (action.kind, action.anchor) == (kind, anchor)


def test_result_waits_until_settled_then_returns_home():
    assert decide(Screen.RESULT, None, 0).kind == WAIT
    action = decide(Screen.RESULT, RICH, 0)
    assert (action.kind, action.anchor) == (TAP, "return-home")


def test_scout_waits_until_the_loot_panel_settles():
    assert decide(Screen.SCOUT, None, 0).kind == WAIT


def test_rich_base_is_attacked():
    assert decide(Screen.SCOUT, RICH, 0).kind == DEPLOY


@pytest.mark.parametrize("reading", [
    loot(MIN_GOLD - 1, MIN_ELIXIR, MIN_DARK),
    loot(MIN_GOLD, MIN_ELIXIR - 1, MIN_DARK),
    loot(MIN_GOLD, MIN_ELIXIR, MIN_DARK - 1),
    loot(None, None, None, ok=False),
])
def test_poor_or_unread_base_is_skipped(reading):
    action = decide(Screen.SCOUT, reading, 0)
    assert (action.kind, action.anchor) == (TAP, "next-button")


def test_thresholds_are_inclusive():
    assert should_attack(RICH)


def test_unknown_waits_out_its_patience_then_aborts():
    kinds = [decide(Screen.UNKNOWN, None, s).kind for s in (0.0, PATIENCE_SECONDS - 0.1, PATIENCE_SECONDS)]
    assert kinds == [WAIT, WAIT, ABORT]


@pytest.mark.parametrize("screen, tapped_on, since, frames, want", [
    (Screen.HOME, None, 0.0, 1, True),  # nothing pending
    (Screen.HOME, Screen.HOME, 1.0, 5, False),  # same screen: the tap hasn't taken effect yet
    (Screen.ARMY, Screen.HOME, 0.2, GATE_STABLE_FRAMES - 1, False),  # new screen, not held long enough
    (Screen.ARMY, Screen.HOME, 0.4, GATE_STABLE_FRAMES, True),  # new screen, held
    (Screen.HOME, Screen.HOME, GATE_TIMEOUT, 99, True),  # timed out: the tap was missed
])
def test_gate(screen, tapped_on, since, frames, want):
    assert gate_open(screen, tapped_on, since, frames) is want


@pytest.mark.parametrize("anchor", ["end-battle", "surrender"])
def test_stalled_battle_is_ended_with_the_button_showing(anchor):
    action = decide(Screen.BATTLE, None, 0, end_with=anchor)
    assert (action.kind, action.anchor) == (TAP, anchor)


TRACE = [[0.4, 600_000, 600_000, 6_000], [20.0, 300_000, 300_000, 3_000]]


def test_stall_counts_from_the_last_loot_change():
    assert not battle_stalled(TRACE, 20.0 + STALL_SECONDS - 0.1)
    assert battle_stalled(TRACE, 20.0 + STALL_SECONDS)


def test_unread_panel_counts_from_the_deploy():
    assert not battle_stalled([], STALL_SECONDS - 0.1)
    assert battle_stalled([], STALL_SECONDS)


def test_star_needs_surrender_first_and_end_battle_to_hold():
    assert star_won(True, STAR_STABLE_FRAMES)
    assert not star_won(True, STAR_STABLE_FRAMES - 1)  # one frame can be a misread
    assert not star_won(False, STAR_STABLE_FRAMES + 50)  # End Battle before any troop landed
