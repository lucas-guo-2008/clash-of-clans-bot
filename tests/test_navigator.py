"""decide() and should_attack() are pure, so these need no frames at all."""

import pytest

from policy.navigator import ABORT, DEPLOY, PATIENCE, TAP, WAIT, decide
from policy.thresholds import MIN_DARK, MIN_ELIXIR, MIN_GOLD, should_attack
from vision.loot import LootReading
from vision.screens import Screen


def loot(gold, elixir, dark, ok=True) -> LootReading:
    return LootReading(gold, elixir, dark, ok, rows=())  # rows only matter for logging


RICH = loot(MIN_GOLD, MIN_ELIXIR, MIN_DARK)


@pytest.mark.parametrize("screen, kind, anchor", [
    (Screen.HOME, TAP, "attack"),
    (Screen.ARMY, TAP, "attack-button"),
    (Screen.ATTACK_MENU, TAP, "find-match"),
    (Screen.BATTLE, WAIT, None),
])
def test_each_screen_has_its_action(screen, kind, anchor):
    action = decide(screen, None, 0)
    assert (action.kind, action.anchor) == (kind, anchor)


def test_result_waits_until_settled_then_returns_home():
    assert decide(Screen.RESULT, None, 0).kind == WAIT
    action = decide(Screen.RESULT, RICH, 0)
    assert (action.kind, action.anchor) == (TAP, "return-home")


def test_rich_base_is_attacked():
    assert decide(Screen.SCOUT, RICH, 0).kind == DEPLOY


@pytest.mark.parametrize("reading", [
    None,
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
    kinds = [decide(Screen.UNKNOWN, None, streak).kind for streak in range(PATIENCE + 1)]
    assert kinds == [WAIT] * PATIENCE + [ABORT]
