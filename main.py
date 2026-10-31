"""
Farm: home -> attack menu -> army -> scout -> Next until a base clears the thresholds ->
drop the army around it -> wait out the battle -> Return Home, ITERATIONS times.

  python main.py --dry-run
  python main.py
  python main.py --show      # live window: screen, scores, buildings, deploy points, next tap

Stops back home after ITERATIONS battles. Prints the buildings detected on each judged base
(nothing decides on them yet) and logs every judged base to templates/scout/bases.jsonl, with
the frame for attacked ones. After each battle it prints what "You got" showed and a running
summary of the run, and logs the result to templates/scout/battles.jsonl, pointing at its base.
"""

import argparse
import itertools
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2

from capture.adb import connect, grab_frame
from control.tap import tap_match, tap_point
from policy.navigator import (ABORT, DEPLOY, GATE_TIMEOUT, PATIENCE_SECONDS, TAP, WAIT, Action, battle_stalled, decide,
                              gate_open)
from vision.anchors import THRESHOLD, Match, peak_scores
from vision.buildings import MODEL_PATH, Building, detect_buildings, load_detector
from vision.buttons import load_button_templates
from vision.deploy import DeployPoint, deck_slots, deploy_points, outline_lines, outline_mask
from vision.glyphs import load_digit_templates
from vision.loot import LootReading, read_loot
from vision.overlay import (CYAN, GREEN, GREY, RED, WHITE, YELLOW, draw_anchors, draw_buildings, draw_deploy,
                            draw_tap, with_header)
from vision.result import load_result_templates, read_result
from vision.screens import Screen, classify

ROOT = Path(__file__).resolve().parent
RESULT_SETTLE_SECONDS = 1.0  # "You got" must read the same for this long before it counts
SCOUT_SETTLE_SECONDS = 0.6  # a readable loot panel must hold this long before the base is judged
SCOUT_UNREADABLE_SECONDS = 2.0  # an unreadable panel is skipped only this long after the base appeared
UNKNOWN_SAVE_SECONDS = 5.0  # an unrecognized stretch is saved only once it outlasts a cloud (live: under 1 s)
clock = time.monotonic  # every timing reads this, so tests can drive time
UNKNOWN_DIR = ROOT / "templates" / "unknown"
SCOUT_DIR = ROOT / "templates" / "scout"
BASES_LOG = "bases.jsonl"  # in SCOUT_DIR: one JSON record per judged base
BATTLES_LOG = "battles.jsonl"  # in SCOUT_DIR: one per settled result, pointing at its base
WINDOW = "bot"

ITERATIONS = 1
# The army, slot by slot left to right, as (kind, count). Slots are found on screen; only
# what is in them is declared here, so reordering the army means editing this line.
#   troop: tap the card, then `count` drops cycling the deploy points
#   hero:  tap the card, one drop; the ability fires HERO_ABILITY_DELAY s later (card tapped again)
#   spell: located but not cast -- aiming waits for building detection (Phase 4/5)
DECK = (("troop", 24), ("skip", 0), ("hero", 1), ("hero", 1), ("spell", 11))
HERO_ABILITY_DELAY = 10.0


def describe(matches) -> str:
    if not matches:
        return "(no anchors)"
    return " ".join(f"{m.label}:{m.score:.3f}" for m in sorted(matches.values(), key=lambda m: -m.score))


def describe_loot(reading: LootReading) -> str:
    if not reading.ok:
        reasons = ",".join(row.reason for row in reading.rows if row.reason) or "?"
        return f"loot=UNREADABLE({reasons})"
    return f"loot={reading.gold}/{reading.elixir}/{reading.dark_elixir}"


def describe_buildings(buildings: list[Building]) -> str:
    counts = Counter(b.name for b in buildings)
    return " ".join(f"{name}:{n}" for name, n in sorted(counts.items())) or "(none)"


def run_summary(looted: tuple[int, int, int], searches: int, minutes: float) -> str:
    return f"run: looted {looted[0]} gold, {looted[1]} elixir, {looted[2]} dark elixir across {searches} search{'es' if searches != 1 else ''} in {minutes:.1f} min"


def deploy_army(device, slots: list[Match], points: list[DeployPoint]) -> list[Match]:
    """Drop every troop and hero card left to right, cycling the deploy points.
    Returns the hero cards, for the ability tap later."""

    drops = itertools.cycle(points)
    heroes = []
    for slot, (kind, count) in zip(slots, DECK):
        if kind not in ("troop", "hero"):
            continue
        tap_match(device, slot)
        for _ in range(count):
            tap_point(device, next(drops))
        if kind == "hero":
            heroes.append(slot)
    return heroes


def save_unknown(frame) -> Path:
    UNKNOWN_DIR.mkdir(parents=True, exist_ok=True)
    out = UNKNOWN_DIR / f"unknown_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
    cv2.imwrite(str(out), frame)
    return out


def loot_fields(reading: LootReading, key: str) -> dict:
    """A reading as record fields: the numbers only if every row read, else why each row did not."""

    if reading.ok:
        return {key: {"gold": reading.gold, "elixir": reading.elixir, "dark_elixir": reading.dark_elixir},
                "unreadable": None}
    return {key: None, "unreadable": [row.reason for row in reading.rows]}


def append_record(log: str, record: dict) -> None:
    SCOUT_DIR.mkdir(parents=True, exist_ok=True)
    with (SCOUT_DIR / log).open("a") as f:
        f.write(json.dumps(record) + "\n")


def base_record(stamp: str, reading: LootReading, decision: str, buildings: list[Building],
                points: list[DeployPoint], frame_file: str | None) -> dict:
    """One judged base as Phase 5 will read it back: enough to compute features for a base whose
    frame was not kept. A battle record points back at it by `id`."""

    return {
        "id": stamp,
        "decision": decision,
        "frame": frame_file,
        **loot_fields(reading, "loot"),
        # Detections depend on the model: a retrained one would count differently.
        "detector": MODEL_PATH.name,
        "buildings": [{**asdict(b), "conf": round(b.conf, 3)} for b in buildings],
        "deploy_points": [[p.x, p.y] for p in points],
    }


def save_base(frame, reading: LootReading, decision: str, buildings: list[Building],
              points: list[DeployPoint]) -> str:
    """Log a judged base to bases.jsonl and return its id. The frame is kept only for a base that
    is attacked: those are the ones a battle result labels. (index.csv is the frozen Phase 4 log.)"""

    SCOUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S_%f}"
    frame_file = None
    if decision == "attack":
        frame_file = f"scout_{stamp}.png"
        cv2.imwrite(str(SCOUT_DIR / frame_file), frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    append_record(BASES_LOG, base_record(stamp, reading, decision, buildings, points, frame_file))
    return stamp


def save_battle(frame, result: LootReading, base_id: str | None, battle_seconds: float | None,
                trace: list[list], ended_early: bool) -> None:
    """Log a settled result to battles.jsonl with its frame, so a misread "You got" can be read
    again. `base` is the attacked base's id in bases.jsonl; None (and no duration) when this run
    did not deploy the army, e.g. it was started mid-battle. `trace` is the battle screen's
    Available Loot each time it changed, as [seconds since deploy, gold, elixir, dark elixir].
    `ended_early` is True when the bot ended the battle on a loot stall rather than the timer."""

    stamp = f"{datetime.now():%Y%m%d_%H%M%S_%f}"
    frame_file = f"result_{stamp}.png"
    SCOUT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(SCOUT_DIR / frame_file), frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    append_record(BATTLES_LOG, {
        "id": stamp,
        "base": base_id,
        **loot_fields(result, "got"),
        # From the army going down to "You got" settling: includes the count-up and RESULT_SETTLE_SECONDS.
        "battle_seconds": None if battle_seconds is None else round(battle_seconds, 1),
        "loot_trace": trace,
        "ended_early": ended_early,  # a stall the rule misjudged shows as loot cut short here
        "frame": frame_file,
    })


def show(
    frame,
    step: int,
    screen: Screen,
    action: Action,
    peaks: dict[str, Match],
    reading: LootReading | None,
    slots: list[Match],
    points: list[DeployPoint],
    buildings: list[Building] | None,
    tap: tuple[int, int] | None,
    unknown_seconds: float,
    battles: int,
) -> None:
    """Draw everything the bot found and decided this step, and put it in the window."""

    image = frame.copy()
    if screen is Screen.SCOUT:
        draw_buildings(image, buildings or [])
        labels = [(slot, f"{slot.label} {kind} x{count}") for slot, (kind, count) in zip(slots, DECK)]
        draw_deploy(image, outline_lines(outline_mask(frame)), points, labels)
    draw_anchors(image, peaks, THRESHOLD)
    if tap is not None:
        draw_tap(image, tap)

    target = f" {action.anchor}" if action.anchor else ""
    status = [(f"battles {battles}/{ITERATIONS}", WHITE)]
    if screen is Screen.UNKNOWN:
        status.append((f"unknown {unknown_seconds:.0f}/{PATIENCE_SECONDS:.0f} s", RED))
    if reading is not None:
        status.append((describe_loot(reading), GREEN if reading.ok else RED))
    if screen is Screen.SCOUT:
        status.append((f"{len(points)} deploy points, {len(slots)}/{len(DECK)} cards", CYAN))
        if buildings is not None:
            status.append((f"{len(buildings)} buildings", YELLOW))

    rows = [
        [(f"step {step}", WHITE), (screen.value.upper(), RED if screen is Screen.UNKNOWN else GREEN),
         (f"-> {action.kind}{target}  ({action.why})", WHITE)],
        [(f"{label} {m.score:.3f}", GREEN if m.score >= THRESHOLD else GREY)
         for label, m in sorted(peaks.items(), key=lambda kv: -kv[1].score)],
        status,
    ]
    cv2.imshow(WINDOW, with_header(image, rows))


def window_alive() -> bool:
    """Let the window repaint; False means q or Esc was pressed."""

    return cv2.waitKey(1) & 0xFF not in (ord("q"), 27)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="classify one screen and report, but do not tap")
    parser.add_argument("--show", action="store_true", help="show what the bot sees and decides, live")
    args = parser.parse_args()

    buttons = load_button_templates()
    digits = load_digit_templates()
    result_digits = load_result_templates()
    detector = load_detector()
    device = connect()
    if args.show:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 960, 600)

    unknown_since = None
    unknown_saved = False  # this unrecognized stretch already has a frame in UNKNOWN_DIR
    battles = 0
    pending, pending_since = None, 0.0
    settled = None  # this result screen's reading, once it held for RESULT_SETTLE_SECONDS
    result_counted = False  # this battle's result is counted; the screen can come back through clouds after Return Home
    scout_values, scout_since, scout_seen = None, 0.0, None  # this base's latest loot reading, since when, first seen
    scout_saved = False  # this base is logged in BASES_LOG
    buildings = None  # this base's detections, found once when it is judged
    heroes: list[Match] = []  # deployed hero cards whose ability has not been used yet
    ability_at = 0.0
    attacked_id, deployed_at = None, None  # the base under attack (bases.jsonl id), and when its army went down
    trace: list[list] = []  # this battle's Available Loot readings, one per change (Phase 5f's stall window)
    ended_early = False  # the bot tapped End Battle / Surrender on this battle
    looted = (0, 0, 0)  # gold, elixir, dark elixir over every readable result
    searches = 0
    started = clock()
    tapped_on, tapped_at = None, 0.0  # the screen of the last gated tap, and when
    last_screen, frames_on_screen = None, 0
    last_logged = None

    for step in itertools.count(1):
        frame = grab_frame(device)
        peaks = peak_scores(frame, buttons)
        matches = {label: m for label, m in peaks.items() if m.score >= THRESHOLD}
        screen = classify(matches)
        reading = read_loot(frame, digits) if screen is Screen.SCOUT else None

        # Available Loot counts down as it is taken. Unreadable frames are skipped, not guessed.
        if screen is Screen.BATTLE and deployed_at is not None:
            left = read_loot(frame, digits)
            values = [left.gold, left.elixir, left.dark_elixir]
            if left.ok and (not trace or trace[-1][1:] != values):
                trace.append([round(clock() - deployed_at, 1), *values])

        frames_on_screen = frames_on_screen + 1 if screen is last_screen else 1
        last_screen = screen

        now = clock()
        if screen is Screen.UNKNOWN:
            unknown_since = now if unknown_since is None else unknown_since
        else:
            unknown_since, unknown_saved = None, False
        unknown_seconds = now - unknown_since if unknown_since is not None else 0.0

        result = None
        if screen is Screen.RESULT:
            if settled is None:
                read = read_result(frame, result_digits)
                values = (read.gold, read.elixir, read.dark_elixir)
                if values != pending:
                    pending, pending_since = values, now
                elif now - pending_since >= RESULT_SETTLE_SECONDS:
                    settled = read  # Return Home is tapped on every settled result, counted or not
                    if not result_counted:
                        result, result_counted = read, True
                        battles += 1
                        if read.ok:
                            looted = (looted[0] + read.gold, looted[1] + read.elixir, looted[2] + read.dark_elixir)
        else:
            pending, settled = None, None
            if screen is not Screen.UNKNOWN:  # left the result screen for real, not just into clouds
                result_counted = False

        scout_settled = None
        if screen is Screen.SCOUT:
            values = (reading.gold, reading.elixir, reading.dark_elixir)
            if values != scout_values:
                scout_values, scout_since = values, now
            scout_seen = now if scout_seen is None else scout_seen
            if reading.ok and now - scout_since >= SCOUT_SETTLE_SECONDS:
                scout_settled = reading
            elif not reading.ok and now - scout_seen >= SCOUT_UNREADABLE_SECONDS:
                scout_settled = reading
        else:
            scout_values, scout_seen, scout_saved, buildings = None, None, False, None

        line = f"step {step:>3}  {screen.value:<12} {describe(matches)}"
        if reading is not None:
            line += f"  {describe_loot(reading)}"

        if screen is Screen.HOME and battles >= ITERATIONS:
            print(f"{line}\n\nhome after {battles} battle(s) -- done")
            print(run_summary(looted, searches, (clock() - started) / 60))
            return 0

        # Phase 5f: once the whole army is down and every ability used, end a battle that stopped paying.
        end_with = None
        if screen is Screen.BATTLE and deployed_at is not None and not heroes:
            if battle_stalled(trace, now - deployed_at):
                end_with = "end-battle" if "end-battle" in matches else "surrender"  # End Battle once a star is won

        action = decide(screen, settled if screen is Screen.RESULT else scout_settled, unknown_seconds, end_with)
        judged = scout_settled is not None and not scout_saved
        if judged:
            buildings = detect_buildings(frame, detector)
            decision = "attack" if action.kind == DEPLOY else "next"
            base_id = save_base(frame, scout_settled, decision, buildings, deploy_points(frame))
            if decision == "attack":
                attacked_id = base_id
            scout_saved = True

        if tapped_on is not None and gate_open(screen, tapped_on, now - tapped_at, frames_on_screen):
            tapped_on = None
        if tapped_on is not None and action.kind in (TAP, DEPLOY):
            waited = f"{now - tapped_at:.1f}/{GATE_TIMEOUT:.0f} s"
            action = Action(WAIT, why=f"holding until the screen changes after the last tap ({waited})")

        # Frames come ~5 per second: log when what the bot sees or means to do changes.
        target = f" {action.anchor}" if action.anchor else ""
        logged = (screen, action.kind, action.anchor)
        if logged != last_logged:
            print(f"{line}  -> {action.kind}{target}  ({action.why})")
            last_logged = logged
        if judged:
            print(f"        buildings {describe_buildings(buildings)}")

        if result is not None:
            got = f"{result.gold}/{result.elixir}/{result.dark_elixir}" if result.ok else "UNREADABLE (not counted)"
            print(f"        battle {battles} got {got}")
            print(f"        {run_summary(looted, searches, (clock() - started) / 60)}")
            save_battle(frame, result, attacked_id, None if deployed_at is None else now - deployed_at, trace,
                        ended_early)
            attacked_id, deployed_at, trace, ended_early = None, None, [], False

        # One frame per unrecognized stretch that outlasts a cloud, plus the one it gives up on
        # (usually a popup). Clouds between screens clear on their own and are not worth keeping.
        outlasted = unknown_seconds >= UNKNOWN_SAVE_SECONDS and not unknown_saved
        if outlasted or action.kind == ABORT:
            print(f"        saved {save_unknown(frame).relative_to(ROOT)}")
            unknown_saved = True

        slots, points = [], []
        if screen is Screen.SCOUT and (args.show or action.kind == DEPLOY):
            slots, points = deck_slots(frame), deploy_points(frame)
        if action.kind == DEPLOY:
            print(f"        {len(slots)}/{len(DECK)} deck cards, {len(points)} deploy points")

        tap = None
        if action.kind == TAP:
            tap = matches[action.anchor].centre
        elif action.kind == DEPLOY and slots and points:
            tap = slots[0].centre

        if args.show:
            show(frame, step, screen, action, peaks, reading, slots, points, buildings, tap, unknown_seconds, battles)

        if args.dry_run:
            if tap is not None:
                print(f"        would tap {action.anchor or 'card'} at {tap}")
            if args.show:
                cv2.waitKey(0)
            return 0

        if action.kind == ABORT:
            print(f"\ngiving up: {action.why}", file=sys.stderr)
            return 1

        if action.kind == TAP:
            tap_match(device, matches[action.anchor])
            tapped_on, tapped_at = screen, clock()
            if action.anchor in ("find-match", "next-button"):
                searches += 1
            if action.anchor in ("end-battle", "surrender"):
                ended_early = True
        elif action.kind == DEPLOY:
            if not points:
                print("        no deploy points on this frame, retrying")
            elif len(slots) != len(DECK):
                print(f"\ndeck shows {len(slots)} cards but DECK lists {len(DECK)} -- update DECK in main.py",
                      file=sys.stderr)
                return 1
            else:
                deployed_at, trace = clock(), []
                heroes = deploy_army(device, slots, points)
                ability_at = clock() + HERO_ABILITY_DELAY
                tapped_on, tapped_at = screen, clock()

        if screen is Screen.BATTLE and heroes and clock() >= ability_at:
            for hero in heroes:
                tap_match(device, hero)
            print(f"        hero abilities: tapped {len(heroes)} hero card(s)")
            heroes = []
        if screen is Screen.RESULT:
            heroes = []

        if args.show and not window_alive():
            print("\nstopped from the window")
            return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nstopped")
        sys.exit(0)
