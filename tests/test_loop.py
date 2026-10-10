"""main.py end to end, over saved frames, with a fake device that records taps instead of sending them."""

import math
import sys

import numpy as np
import pytest

import main
from policy.navigator import GATE_STABLE_FRAMES, GATE_TIMEOUT, PATIENCE_SECONDS
from vision.deploy import deck_slots, deploy_points

CLOUD = "unknown_20260919_005725_732450.png"
ATTACK = (125, 1021)
ATTACK_BUTTON = (1696, 963)
RETURN_HOME = (961, 925)  # centre of the return-home match

STEP = 0.2
HOLD = GATE_STABLE_FRAMES + 1
SETTLE = math.ceil(main.RESULT_SETTLE_SECONDS / STEP) + 1
SCOUT = math.ceil(main.SCOUT_SETTLE_SECONDS / STEP) + 2  # scout frames until a readable base is judged
NEXT = (1749, 771)  # centre of the Next button
PATIENCE_FRAMES = math.ceil(PATIENCE_SECONDS / STEP) + 1  # unknown frames until the bot gives up


class FakeDevice:
    def __init__(self):
        self.taps = []

    def click(self, x, y):
        self.taps.append((x, y))


class OutOfFrames(Exception):
    pass


@pytest.fixture
def run(frame, unknown_frame, monkeypatch):
    def play(sequence):
        cache = {}

        def load(s):
            if isinstance(s, np.ndarray):
                return s
            if s not in cache:
                cache[s] = unknown_frame(CLOUD) if s == "cloud" else frame(s)
            return cache[s]

        expanded = [e for item in sequence for e in ([item[0]] * item[1] if isinstance(item, tuple) else [item])]
        frames = iter([load(s) for s in expanded])
        device = FakeDevice()
        count = [0]

        def grab(_):
            count[0] += 1
            try:
                return next(frames)
            except StopIteration:
                raise OutOfFrames from None

        monkeypatch.setattr(main, "connect", lambda: device)
        monkeypatch.setattr(main, "grab_frame", grab)
        monkeypatch.setattr(main, "clock", lambda: count[0] * STEP)
        monkeypatch.setattr(main, "save_unknown", lambda _: main.ROOT / "not-saved-in-tests.png")
        monkeypatch.setattr(sys, "argv", ["main.py"])
        try:
            return main.main(), device.taps
        except OutOfFrames:
            return None, device.taps

    return play


def expected_deploy(frame):
    taps = []
    for slot, (kind, count) in zip(deck_slots(frame), main.DECK):
        if kind in ("troop", "hero"):
            taps += [slot.centre] + ["point"] * count
    return taps


def as_planned(taps, frame):
    points = {(p.x, p.y) for p in deploy_points(frame)}
    return ["point" if t in points else t for t in taps]


def test_one_full_attack(run, frame):
    code, taps = run([(1, HOLD), (3, HOLD), (2, HOLD), ("cloud", HOLD), (5, SCOUT), (7, HOLD),
                      (8, SETTLE + 2), ("cloud", HOLD), 1])
    deploy = expected_deploy(frame(5))

    assert code == 0
    assert as_planned(taps[3 : 3 + len(deploy)], frame(5)) == deploy  # after the 3 navigation taps
    assert taps[3 + len(deploy) :] == [RETURN_HOME]  # once: the gate holds the later result frames


def test_hero_abilities_fire_once_in_battle(run, frame, monkeypatch):
    monkeypatch.setattr(main, "HERO_ABILITY_DELAY", 0.0)
    code, taps = run([(5, SCOUT), (7, HOLD), (8, SETTLE + 2), ("cloud", HOLD), 1])
    deploy = expected_deploy(frame(5))
    heroes = [s.centre for s, (kind, _) in zip(deck_slots(frame(5)), main.DECK) if kind == "hero"]

    assert code == 0
    assert taps[len(deploy) :] == [*heroes, RETURN_HOME]  # first battle frame; abilities are never gated


def test_deck_that_does_not_match_DECK_aborts_without_tapping(run, monkeypatch):
    monkeypatch.setattr(main, "DECK", main.DECK[:-1])
    code, taps = run([(5, SCOUT)])
    assert code == 1
    assert taps == []


def test_long_battle_does_not_use_up_patience(run):
    code, _ = run([(5, SCOUT), (7, PATIENCE_FRAMES + 2), "cloud", (8, SETTLE + 2), ("cloud", HOLD), 1])
    assert code == 0


def test_no_outline_means_no_taps(run, frame):
    code, taps = run([(4, SCOUT), (5, SCOUT), (7, HOLD), (8, SETTLE + 2), ("cloud", HOLD), 1])
    assert code == 0
    assert taps[0] == deck_slots(frame(5))[0].centre != deck_slots(frame(4))[0].centre


def test_result_is_counted_only_once_it_settles(run, capsys):
    code, taps = run([(5, SCOUT), (7, HOLD), (18, HOLD), (19, SETTLE + 2), ("cloud", HOLD), 1])
    out = capsys.readouterr().out

    assert code == 0
    assert taps.count(RETURN_HOME) == 1
    assert "battle 1 got 463831/398186/8549" in out
    assert "358306" not in out.split("battle 1 got")[1]  # the unsettled reading was never counted


def faded(scout):
    """A scout frame with its loot panel whited out, as clouds still fading over it leave it:
    still a scout screen, but an unreadable panel."""

    image = scout.copy()
    image[140:310, 90:430] = 255
    return image


def test_cloud_fade_over_the_loot_panel_is_waited_out(run, frame):
    code, taps = run([(faded(frame(5)), 5), (5, SCOUT)])
    assert taps and taps[0] == deck_slots(frame(5))[0].centre
    assert NEXT not in taps


def test_panel_that_stays_unreadable_is_skipped(run, frame):
    code, taps = run([(faded(frame(5)), math.ceil(main.SCOUT_UNREADABLE_SECONDS / STEP) + 2)])
    assert taps == [NEXT]


def test_endless_unknown_aborts_without_tapping(run):
    code, taps = run([("cloud", PATIENCE_FRAMES + 1)])
    assert code == 1
    assert taps == []


# The tap gate: after a tap, act again only once another screen has held GATE_STABLE_FRAMES, or GATE_TIMEOUT has passed (the tap was missed).

def test_no_second_tap_while_the_screen_has_not_changed(run):
    _, taps = run([(1, int(GATE_TIMEOUT / STEP) - 2)])
    assert taps == [ATTACK]


def test_missed_tap_is_retried_after_the_timeout(run):
    _, taps = run([(1, math.ceil(GATE_TIMEOUT / STEP) + 2)])
    assert taps == [ATTACK, ATTACK]


def test_one_frame_flicker_does_not_release_the_gate(run):
    _, taps = run([(1, 2), "cloud", (1, 3)])
    assert taps == [ATTACK]


def test_a_new_screen_held_for_two_frames_releases_the_gate(run):
    _, taps = run([1, (3, GATE_STABLE_FRAMES)])
    assert taps == [ATTACK, ATTACK_BUTTON]
