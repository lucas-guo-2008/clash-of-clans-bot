"""
Farm: home -> army -> attack menu -> scout -> Next until a base clears the thresholds ->
drop the first card's troops around it -> wait out the battle -> Return Home, ITERATIONS times.

  python main.py --dry-run
  python main.py
  python main.py --show      # live window: screen, scores, deploy points, next tap

Stops back home after ITERATIONS battles, or on Ctrl-C (or q in the --show window). After each
battle it prints what "You got" showed, and a running summary of the run.
"""

import argparse
import itertools
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

from capture.adb import connect, grab_frame
from control.tap import tap_match, tap_point
from policy.navigator import ABORT, DEPLOY, GATE_TIMEOUT, PATIENCE_SECONDS, TAP, WAIT, Action, decide, gate_open
from vision.anchors import THRESHOLD, Match, peak_scores
from vision.buttons import load_button_templates
from vision.deploy import DeployPoint, deck_slots, deploy_points, outline_lines, outline_mask
from vision.glyphs import load_digit_templates
from vision.loot import LootReading, read_loot
from vision.overlay import CYAN, GREEN, GREY, RED, WHITE, draw_anchors, draw_deploy, draw_tap, with_header
from vision.result import load_result_templates, read_result
from vision.screens import Screen, classify

ROOT = Path(__file__).resolve().parent
RESULT_SETTLE_SECONDS = 1.0  # "You got" must read the same for this long before it counts
SCOUT_SETTLE_SECONDS = 0.6  # a readable loot panel must hold this long before the base is judged
SCOUT_UNREADABLE_SECONDS = 2.0  # an unreadable panel is skipped only this long after the base appeared
clock = time.monotonic  # every timing reads this, so tests can drive time
UNKNOWN_DIR = ROOT / "templates" / "unknown"
WINDOW = "bot"

ITERATIONS = 1
# The army, slot by slot left to right, as (kind, count). Slots are found on screen; only
# what is in them is declared here, so reordering the army means editing this line.
#   troop: tap the card, then `count` drops cycling the deploy points
#   hero:  tap the card, one drop; the ability fires HERO_ABILITY_DELAY s later (card tapped again)
#   spell: located but not cast -- aiming waits for building detection (Phase 4/5)
#   skip:  not deployed (the Balloon card)
DECK = (("troop", 24), ("skip", 0), ("hero", 1), ("hero", 1), ("spell", 11))
HERO_ABILITY_DELAY = 10.0
SEARCH_COST = 900


def describe(matches) -> str:
    if not matches:
        return "(no anchors)"
    return " ".join(f"{m.label}:{m.score:.3f}" for m in sorted(matches.values(), key=lambda m: -m.score))


def describe_loot(reading: LootReading) -> str:
    if not reading.ok:
        reasons = ",".join(row.reason for row in reading.rows if row.reason) or "?"
        return f"loot=UNREADABLE({reasons})"
    return f"loot={reading.gold}/{reading.elixir}/{reading.dark_elixir}"


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


def show(
    frame,
    step: int,
    screen: Screen,
    action: Action,
    peaks: dict[str, Match],
    reading: LootReading | None,
    slots: list[Match],
    points: list[DeployPoint],
    tap: tuple[int, int] | None,
    unknown_seconds: float,
    battles: int,
) -> None:
    """Draw everything the bot found and decided this step, and put it in the window."""

    image = frame.copy()
    if screen is Screen.SCOUT:
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
    device = connect()
    if args.show:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 960, 600)

    unknown_since = None
    battles = 0
    pending, pending_since = None, 0.0
    settled = None  # this result screen's reading, once it held for RESULT_SETTLE_SECONDS
    scout_values, scout_since, scout_seen = None, 0.0, None  # this base's latest loot reading, since when, first seen
    heroes: list[Match] = []  # deployed hero cards whose ability has not been used yet
    ability_at = 0.0
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

        frames_on_screen = frames_on_screen + 1 if screen is last_screen else 1
        last_screen = screen

        now = clock()
        new_unknown = screen is Screen.UNKNOWN and unknown_since is None
        if screen is Screen.UNKNOWN:
            unknown_since = now if unknown_since is None else unknown_since
        else:
            unknown_since = None
        unknown_seconds = now - unknown_since if unknown_since is not None else 0.0

        result = None
        if screen is Screen.RESULT:
            if settled is None:
                read = read_result(frame, result_digits)
                values = (read.gold, read.elixir, read.dark_elixir)
                if values != pending:
                    pending, pending_since = values, now
                elif now - pending_since >= RESULT_SETTLE_SECONDS:
                    settled = result = read
                    battles += 1
                    if read.ok:
                        looted = (looted[0] + read.gold, looted[1] + read.elixir, looted[2] + read.dark_elixir)
        else:
            pending, settled = None, None

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
            scout_values, scout_seen = None, None

        line = f"step {step:>3}  {screen.value:<12} {describe(matches)}"
        if reading is not None:
            line += f"  {describe_loot(reading)}"

        if screen is Screen.HOME and battles >= ITERATIONS:
            print(f"{line}\n\nhome after {battles} battle(s) -- done")
            print(run_summary(looted, searches, (clock() - started) / 60))
            return 0

        action = decide(screen, settled if screen is Screen.RESULT else scout_settled, unknown_seconds)
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

        if result is not None:
            got = f"{result.gold}/{result.elixir}/{result.dark_elixir}" if result.ok else "UNREADABLE (not counted)"
            print(f"        battle {battles} got {got}")
            print(f"        {run_summary(looted, searches, (clock() - started) / 60)}")

        # One frame per unrecognized stretch, plus the one it gives up on (usually a popup).
        if new_unknown or action.kind == ABORT:
            print(f"        saved {save_unknown(frame).relative_to(ROOT)}")

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
            show(frame, step, screen, action, peaks, reading, slots, points, tap, unknown_seconds, battles)

        if args.dry_run:
            if tap is not None:
                print(f"        would tap {action.anchor or 'card'} at {tap}")
            if args.show:
                cv2.waitKey(0)  # hold the one frame until a key is pressed
            return 0

        if action.kind == ABORT:
            print(f"\ngiving up: {action.why}", file=sys.stderr)
            return 1

        if action.kind == TAP:
            tap_match(device, matches[action.anchor])
            tapped_on, tapped_at = screen, clock()
            if action.anchor in ("find-match", "next-button"):
                searches += 1
        elif action.kind == DEPLOY:
            if not points:
                print("        no deploy points on this frame, retrying")
            elif len(slots) != len(DECK):
                print(f"\ndeck shows {len(slots)} cards but DECK lists {len(DECK)} -- update DECK in main.py",
                      file=sys.stderr)
                return 1
            else:
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
