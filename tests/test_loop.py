"""main.py end to end, over saved frames, with a fake device that records taps instead of sending them."""

import json
import math
import sys

import cv2
import numpy as np
import pytest

import main
from policy.navigator import GATE_STABLE_FRAMES, GATE_TIMEOUT, PATIENCE_SECONDS, STALL_SECONDS
from vision.deploy import UI_RECTS, deck_slots, deploy_points

CLOUD = "unknown_20260919_005725_732450.png"
BATTLE = (126, 1026)  # home, revamped UI (frame 25)
FIND_MATCH = (329, 802)
RETURN_HOME = (961, 925)  # centre of the return-home match
END_BATTLE = (137, 807)  # centre of the end-battle match
OKAY = (1166, 692)  # centre of the okay match on the End Battle? popup

STEP = 0.2
HOLD = GATE_STABLE_FRAMES + 1
SETTLE = math.ceil(main.RESULT_SETTLE_SECONDS / STEP) + 1
SCOUT = math.ceil(main.SCOUT_SETTLE_SECONDS / STEP) + 2  # scout frames until a readable base is judged
NEXT = (1749, 771)  # centre of the Next button
SCOUT_AT = 24  # a readable base with an outline, in the revamped UI
PATIENCE_FRAMES = math.ceil(PATIENCE_SECONDS / STEP) + 1  # unknown frames until the bot gives up
STALL = math.ceil(STALL_SECONDS / STEP) + 5  # battle frames with unchanged loot until the bot ends it


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
    monkeypatch.setattr(main, "save_unknown", lambda _: main.ROOT / "not-saved-in-tests.png")

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
            taps += [slot.centre] + ["point"] * (count if kind == "troop" else 1)
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


def test_each_judged_base_is_logged_once_and_only_attacked_frames_kept(run, frame, tmp_path):
    skipped = math.ceil(main.SCOUT_UNREADABLE_SECONDS / STEP) + 2
    run([(faded(frame(SCOUT_AT)), skipped + HOLD), ("cloud", HOLD), (SCOUT_AT, SCOUT + HOLD)])
    scout = tmp_path / "scout"
    lines = (scout / main.BASES_LOG).read_text().splitlines()
    records = [json.loads(line) for line in lines]
    loot = main.read_loot(frame(SCOUT_AT), main.load_digit_templates())

    assert [r["decision"] for r in records] == ["next", "attack"]  # not again while the gate holds each base
    assert str(loot.gold) not in lines[0] and str(loot.gold) in lines[1]
    assert sorted(p.name for p in scout.glob("*.png")) == sorted(r["frame"] for r in records)  # skipped ones too
    assert len({r["id"] for r in records}) == 2


def read_log(tmp_path, log):
    return [json.loads(line) for line in (tmp_path / "scout" / log).read_text().splitlines()]


def test_battle_is_logged_against_the_base_it_attacked(run, frame, tmp_path):
    sequence = [(SCOUT_AT, SCOUT), (7, HOLD), (8, SETTLE + 2), ("cloud", HOLD), 25]
    code, _ = run(sequence)
    [base] = read_log(tmp_path, main.BASES_LOG)
    [battle] = read_log(tmp_path, main.BATTLES_LOG)
    got = main.read_result(frame(8), main.load_result_templates())

    assert code == 0
    assert battle["base"] == base["id"] and base["decision"] == "attack"
    assert battle["got"] == {"gold": got.gold, "elixir": got.elixir, "dark_elixir": got.dark_elixir}
    assert (tmp_path / "scout" / battle["frame"]).exists()
    assert battle["ended_early"] is False
    frames = sum(n for _, n in sequence[:3])
    assert main.RESULT_SETTLE_SECONDS < battle["battle_seconds"] < frames * STEP  # deploy to settled result


def battle_showing(battle, scout):
    """A battle frame showing another base's loot panel, so a battle can show two readings. Frame 7
    predates the revamp (its own panel reads unreadable); frame 38 proves the positions match."""

    image = battle.copy()
    image[0:320, 0:450] = scout[0:320, 0:450]
    return image


def test_battle_loot_is_traced_on_each_change(run, frame, tmp_path):
    before, after = battle_showing(frame(7), frame(SCOUT_AT)), battle_showing(frame(7), frame(26))
    run([(SCOUT_AT, SCOUT), (before, HOLD), (after, HOLD), (frame(7), 2), (8, SETTLE + 2), ("cloud", HOLD), 25])
    [battle] = read_log(tmp_path, main.BATTLES_LOG)
    digits = main.load_digit_templates()
    left = [main.read_loot(f, digits) for f in (frame(SCOUT_AT), frame(26))]

    # One entry per change: repeats and the unreadable frames (7 as saved) add nothing.
    assert [t[1:] for t in battle["loot_trace"]] == [[r.gold, r.elixir, r.dark_elixir] for r in left]
    assert battle["loot_trace"][0][0] < battle["loot_trace"][1][0]  # seconds since deploy


def test_result_without_a_deploy_is_logged_with_no_base(run, tmp_path):
    code, _ = run([(8, SETTLE + 2), ("cloud", HOLD), 25])  # started mid-battle
    [battle] = read_log(tmp_path, main.BATTLES_LOG)
    assert code == 0
    assert battle["base"] is None and battle["battle_seconds"] is None


def test_result_shown_again_after_return_home_is_counted_once(run, tmp_path, monkeypatch, capsys):
    # Live, the result screen came back through the clouds after Return Home was tapped.
    monkeypatch.setattr(main, "ITERATIONS", 2)
    _, taps = run([(SCOUT_AT, SCOUT), (7, HOLD), (8, SETTLE + 2), ("cloud", HOLD), (8, SETTLE + 2)])
    out = capsys.readouterr().out

    assert len(read_log(tmp_path, main.BATTLES_LOG)) == 1
    assert "battle 1 got" in out and "battle 2" not in out
    assert taps.count(RETURN_HOME) == 2  # still leaves the screen it came back to


def test_next_battle_after_going_home_is_counted(run, frame, monkeypatch, capsys):
    monkeypatch.setattr(main, "ITERATIONS", 2)
    battle = [(SCOUT_AT, SCOUT), (7, HOLD), (8, SETTLE + 2), ("cloud", HOLD)]
    code, _ = run([*battle, (25, HOLD), (2, HOLD), (3, HOLD), ("cloud", HOLD), *battle, 25])
    assert code == 0
    assert "battle 2 got" in capsys.readouterr().out


def test_each_judged_base_is_detected_once(run, frame, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(main, "detect_buildings", lambda f, model: calls.append(f) or
                        [main.Building("gold_mine", 0, 0, 10, 10, 0.9)] * 2)
    skipped = math.ceil(main.SCOUT_UNREADABLE_SECONDS / STEP) + 2
    run([(faded(frame(SCOUT_AT)), skipped + HOLD), ("cloud", HOLD), (SCOUT_AT, SCOUT + HOLD)])

    assert len(calls) == 2  # the skipped base and the attacked one, not every scout frame
    assert np.array_equal(calls[1], frame(SCOUT_AT))
    assert capsys.readouterr().out.count("buildings gold_mine:2") == 2


@pytest.fixture
def saved_unknowns(monkeypatch):
    saved = []
    monkeypatch.setattr(main, "save_unknown", lambda f: saved.append(f) or main.ROOT / "not-saved-in-tests.png")
    return saved


def test_clouds_that_clear_are_not_saved(run, saved_unknowns):
    cloud = int(main.UNKNOWN_SAVE_SECONDS / STEP)  # the last of these is just under the limit
    run([25, ("cloud", cloud), (2, HOLD), ("cloud", cloud), (2, HOLD)])
    assert saved_unknowns == []


def test_unknown_that_lasts_is_saved_once_then_at_abort(run, saved_unknowns):
    code, _ = run([("cloud", PATIENCE_FRAMES + 1)])
    assert code == 1
    assert len(saved_unknowns) == 2  # once past UNKNOWN_SAVE_SECONDS, once on giving up


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


def test_stalled_battle_is_ended_and_logged(run, frame, tmp_path):
    code, taps = run([(SCOUT_AT, SCOUT), (38, STALL), (39, HOLD), (40, SETTLE + 2), ("cloud", HOLD), 25])
    deploy = expected_deploy(frame(SCOUT_AT))
    heroes = [s.centre for s, (kind, _) in zip(deck_slots(frame(SCOUT_AT)), main.DECK) if kind == "hero"]
    [battle] = read_log(tmp_path, main.BATTLES_LOG)

    assert code == 0
    assert taps[len(deploy) :] == [*heroes, END_BATTLE, OKAY, RETURN_HOME]  # End Battle once, then confirm
    assert battle["ended_early"] is True
    assert [t[1:] for t in battle["loot_trace"]] == [[179_550, 129_210, 500]]  # the popup's dimmed panel adds nothing


def test_battle_still_paying_is_not_ended(run, frame):
    # The loot changes just before the window runs out, which restarts it.
    paying = battle_showing(frame(38), frame(SCOUT_AT))
    code, taps = run([(SCOUT_AT, SCOUT), (38, STALL - 10), (paying, STALL - 10), (40, SETTLE + 2), ("cloud", HOLD), 25])
    assert code == 0
    assert END_BATTLE not in taps


def test_run_settings_are_logged_once_and_every_record_points_at_them(run, tmp_path):
    run([(SCOUT_AT, SCOUT), (38, HOLD), (40, SETTLE + 2), ("cloud", HOLD), 25])
    [run_record] = read_log(tmp_path, main.RUNS_LOG)
    [base] = read_log(tmp_path, main.BASES_LOG)
    [battle] = read_log(tmp_path, main.BATTLES_LOG)

    assert base["run"] == battle["run"] == run_record["id"]
    assert run_record["stall_seconds"] == STALL_SECONDS and run_record["deck"] == [list(s) for s in main.DECK]
    assert len(run_record["commit"]) == 40


def test_search_time_runs_from_the_search_tap_to_the_judged_base(run, tmp_path):
    run([(3, HOLD), ("cloud", 10), (SCOUT_AT, SCOUT)])
    [base] = read_log(tmp_path, main.BASES_LOG)
    assert (10 + SCOUT - 1) * STEP <= base["search_seconds"] <= (HOLD + 10 + SCOUT) * STEP


def test_battle_frames_are_kept_every_few_seconds_and_at_the_end(run, tmp_path):
    every = round(main.BATTLE_FRAME_SECONDS / STEP)
    run([(SCOUT_AT, SCOUT), (38, 2 * every + 1), (40, SETTLE + 2), ("cloud", HOLD), 25])
    [battle] = read_log(tmp_path, main.BATTLES_LOG)
    times = [t for t, _ in battle["battle_frames"]]

    # On a grid from the deploy: the first battle frame, then the first frame past each 5 s mark.
    assert [k * main.BATTLE_FRAME_SECONDS <= t < k * main.BATTLE_FRAME_SECONDS + 1 for k, t in enumerate(times)] == [True] * 3
    files = [f for _, f in battle["battle_frames"]] + [battle["final_battle_frame"]]
    assert all((tmp_path / "scout" / f).exists() for f in files)
    assert battle["final_battle_frame"].startswith(f"battle_{battle['base']}_")


OUTLINE = math.ceil(main.OUTLINE_SECONDS / STEP) + 2  # scout frames until a base with no outline is skipped


def test_base_with_no_outline_is_skipped_not_guessed(run, frame, tmp_path):
    blank = no_outline(frame(SCOUT_AT))
    code, taps = run([(blank, OUTLINE)])
    [base] = read_log(tmp_path, main.BASES_LOG)
    assert taps == [NEXT]  # no card tapped, no drop
    assert base["decision"] == "attack" and base["deploy_points"] == []  # the loot rule said attack


def test_battle_that_never_got_its_army_is_not_linked_to_the_base(run, frame, tmp_path):
    # The scout timer ran out while the outline was still missing (frame 9: End Battle, no Next).
    blank = no_outline(frame(SCOUT_AT))
    run([(blank, SCOUT), (9, HOLD), (8, SETTLE + 2), ("cloud", HOLD), 25])
    [battle] = read_log(tmp_path, main.BATTLES_LOG)
    assert battle["base"] is None and battle["deploy_points"] == [] and battle["battle_seconds"] is None


def test_battle_logs_the_points_the_army_went_down_on(run, frame, tmp_path):
    run([(SCOUT_AT, SCOUT), (38, HOLD), (40, SETTLE + 2), ("cloud", HOLD), 25])
    [battle] = read_log(tmp_path, main.BATTLES_LOG)
    assert battle["deploy_points"] == [[p.x, p.y] for p in deploy_points(frame(SCOUT_AT))]


def test_final_battle_frame_is_the_last_one_whose_loot_reads(run, frame, tmp_path):
    # The end-of-battle banner dims the battle screen: still a battle, but the panel stops reading.
    banner = (frame(38) * 0.4).astype(np.uint8)
    run([(SCOUT_AT, SCOUT), (38, HOLD), (banner, HOLD), (40, SETTLE + 2), ("cloud", HOLD), 25])
    [battle] = read_log(tmp_path, main.BATTLES_LOG)
    final = cv2.imread(str(tmp_path / "scout" / battle["final_battle_frame"]))
    assert np.array_equal(final, frame(38))


CHECK = math.ceil(main.CARD_CHECK_SECONDS / STEP) + 2


def test_refused_troop_is_dropped_again_until_the_card_is_spent(run, frame, tmp_path, monkeypatch):
    monkeypatch.setattr(main, "HERO_ABILITY_DELAY", 60.0)
    code, taps = run([(SCOUT_AT, SCOUT), (45, CHECK), (44, 2 * CHECK), (40, SETTLE + 2), ("cloud", HOLD), 25])
    deploy = expected_deploy(frame(SCOUT_AT))
    points = deploy_points(frame(SCOUT_AT))
    dragons = deck_slots(frame(SCOUT_AT))[0].centre
    opposite = points[len(points) // 2]

    assert code == 0
    assert taps[len(deploy) :] == [dragons, (opposite.x, opposite.y), RETURN_HOME]
    [battle] = read_log(tmp_path, main.BATTLES_LOG)
    assert [r[1:] for r in battle["redeployed"]] == [[opposite.x, opposite.y]]


def test_spent_card_gets_no_more_drops(run, frame, monkeypatch):
    monkeypatch.setattr(main, "HERO_ABILITY_DELAY", 60.0)
    code, taps = run([(SCOUT_AT, SCOUT), (44, 3 * CHECK)])
    assert as_planned(taps, frame(SCOUT_AT)) == expected_deploy(frame(SCOUT_AT))
