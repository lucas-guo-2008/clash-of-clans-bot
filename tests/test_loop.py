"""main.py end to end, over saved frames, with a fake device that records taps instead of sending them."""

import csv
import math
import sys

import numpy as np
import pytest

import main
from policy.navigator import GATE_STABLE_FRAMES, GATE_TIMEOUT, PATIENCE_SECONDS
from vision.deploy import UI_RECTS, deck_slots, deploy_points

CLOUD = "unknown_20260919_005725_732450.png"
BATTLE = (126, 1026)  # home, revamped UI (frame 25)
FIND_MATCH = (329, 802)
RETURN_HOME = (961, 925)  # centre of the return-home match

STEP = 0.2
HOLD = GATE_STABLE_FRAMES + 1
SETTLE = math.ceil(main.RESULT_SETTLE_SECONDS / STEP) + 1
SCOUT = math.ceil(main.SCOUT_SETTLE_SECONDS / STEP) + 2  # scout frames until a readable base is judged
NEXT = (1749, 771)  # centre of the Next button
SCOUT_AT = 24  # a readable base with an outline, in the revamped UI
PATIENCE_FRAMES = math.ceil(PATIENCE_SECONDS / STEP) + 1  # unknown frames until the bot gives up


class FakeDevice:
    def __init__(self):
        self.taps = []

    def click(self, x, y):
        self.taps.append((x, y))


class OutOfFrames(Exception):
    pass


@pytest.fixture
def run(frame, unknown_frame, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "load_detector", lambda: None)
    monkeypatch.setattr(main, "detect_buildings", lambda frame, model: [])

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
        monkeypatch.setattr(main, "SCOUT_DIR", tmp_path / "scout")
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
    code, taps = run([(25, HOLD), (2, HOLD), (3, HOLD), ("cloud", HOLD), (SCOUT_AT, SCOUT), (7, HOLD),
                      (8, SETTLE + 2), ("cloud", HOLD), 25])
    deploy = expected_deploy(frame(SCOUT_AT))

    assert code == 0
    assert as_planned(taps[3 : 3 + len(deploy)], frame(SCOUT_AT)) == deploy  # after the 3 navigation taps
    assert taps[3 + len(deploy) :] == [RETURN_HOME]  # once: the gate holds the later result frames


def test_hero_abilities_fire_once_in_battle(run, frame, monkeypatch):
    monkeypatch.setattr(main, "HERO_ABILITY_DELAY", 0.0)
    code, taps = run([(SCOUT_AT, SCOUT), (7, HOLD), (8, SETTLE + 2), ("cloud", HOLD), 25])
    deploy = expected_deploy(frame(SCOUT_AT))
    heroes = [s.centre for s, (kind, _) in zip(deck_slots(frame(SCOUT_AT)), main.DECK) if kind == "hero"]

    assert code == 0
    assert taps[len(deploy) :] == [*heroes, RETURN_HOME]  # first battle frame; abilities are never gated


def test_deck_that_does_not_match_DECK_aborts_without_tapping(run, monkeypatch):
    monkeypatch.setattr(main, "DECK", main.DECK[:-1])
    code, taps = run([(SCOUT_AT, SCOUT)])
    assert code == 1
    assert taps == []


def test_long_battle_does_not_use_up_patience(run):
    code, _ = run([(SCOUT_AT, SCOUT), (7, PATIENCE_FRAMES + 2), "cloud", (8, SETTLE + 2), ("cloud", HOLD), 25])
    assert code == 0


def no_outline(scout):
    """A scout frame with the map painted flat grass outside the UI, as before the red outline is
    drawn: readable loot, but no deploy points."""

    image = scout.copy()
    keep = np.zeros(image.shape[:2], bool)
    for x0, y0, x1, y1 in UI_RECTS:
        keep[y0:y1, x0:x1] = True
    image[~keep] = (60, 160, 90)
    return image


def test_no_outline_means_no_taps(run, frame):
    blank = no_outline(frame(SCOUT_AT))
    assert not deploy_points(blank)
    code, taps = run([(blank, SCOUT), (SCOUT_AT, SCOUT), (7, HOLD), (8, SETTLE + 2), ("cloud", HOLD), 25])
    assert code == 0
    assert taps[0] == deck_slots(frame(SCOUT_AT))[0].centre  # deployed only once the outline appeared
    assert len(taps) == len(expected_deploy(frame(SCOUT_AT))) + 1  # one deploy, then Return Home


def test_result_is_counted_only_once_it_settles(run, capsys):
    code, taps = run([(SCOUT_AT, SCOUT), (7, HOLD), (18, HOLD), (19, SETTLE + 2), ("cloud", HOLD), 25])
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
    code, taps = run([(faded(frame(SCOUT_AT)), 5), (SCOUT_AT, SCOUT)])
    assert taps and taps[0] == deck_slots(frame(SCOUT_AT))[0].centre
    assert NEXT not in taps


def test_panel_that_stays_unreadable_is_skipped(run, frame):
    code, taps = run([(faded(frame(SCOUT_AT)), math.ceil(main.SCOUT_UNREADABLE_SECONDS / STEP) + 2)])
    assert taps == [NEXT]


def test_each_judged_base_is_saved_once(run, frame, tmp_path):
    skipped = math.ceil(main.SCOUT_UNREADABLE_SECONDS / STEP) + 2
    run([(faded(frame(SCOUT_AT)), skipped + HOLD), ("cloud", HOLD), (SCOUT_AT, SCOUT + HOLD)])
    scout = tmp_path / "scout"
    rows = list(csv.DictReader((scout / "index.csv").open()))
    loot = main.read_loot(frame(SCOUT_AT), main.load_digit_templates())

    assert [r["decision"] for r in rows] == ["next", "attack"]  # not again while the gate holds each base
    assert rows[0]["gold"] == "" and rows[1]["gold"] == str(loot.gold)
    assert sorted(p.name for p in scout.glob("*.png")) == sorted(r["file"] for r in rows)


def test_each_judged_base_is_detected_once(run, frame, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(main, "detect_buildings", lambda f, model: calls.append(f) or
                        [main.Building("gold_mine", 0, 0, 10, 10, 0.9)] * 2)
    skipped = math.ceil(main.SCOUT_UNREADABLE_SECONDS / STEP) + 2
    run([(faded(frame(SCOUT_AT)), skipped + HOLD), ("cloud", HOLD), (SCOUT_AT, SCOUT + HOLD)])

    assert len(calls) == 2  # the skipped base and the attacked one, not every scout frame
    assert np.array_equal(calls[1], frame(SCOUT_AT))
    assert capsys.readouterr().out.count("buildings gold_mine:2") == 2


def test_endless_unknown_aborts_without_tapping(run):
    code, taps = run([("cloud", PATIENCE_FRAMES + 1)])
    assert code == 1
    assert taps == []


# The tap gate: after a tap, act again only once another screen has held GATE_STABLE_FRAMES, or GATE_TIMEOUT has passed (the tap was missed).

def test_no_second_tap_while_the_screen_has_not_changed(run):
    _, taps = run([(25, int(GATE_TIMEOUT / STEP) - 2)])
    assert taps == [BATTLE]


def test_missed_tap_is_retried_after_the_timeout(run):
    _, taps = run([(25, math.ceil(GATE_TIMEOUT / STEP) + 2)])
    assert taps == [BATTLE, BATTLE]


def test_one_frame_flicker_does_not_release_the_gate(run):
    _, taps = run([(25, 2), "cloud", (25, 3)])
    assert taps == [BATTLE]


def test_a_new_screen_held_for_two_frames_releases_the_gate(run):
    _, taps = run([25, (2, GATE_STABLE_FRAMES)])
    assert taps == [BATTLE, FIND_MATCH]
