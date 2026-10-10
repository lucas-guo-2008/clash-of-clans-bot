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
from policy.navigator import ABORT, DEPLOY, PATIENCE, TAP, Action, decide
from vision.anchors import THRESHOLD, Match, peak_scores
from vision.buttons import load_button_templates
from vision.deploy import DeployPoint, deck_slots, deploy_points, outline_lines, outline_mask
from vision.glyphs import load_digit_templates
from vision.loot import LootReading, read_loot
from vision.overlay import CYAN, GREEN, GREY, RED, WHITE, draw_anchors, draw_deploy, draw_tap, with_header
from vision.result import load_result_templates, read_result
from vision.screens import Screen, classify

ROOT = Path(__file__).resolve().parent
SETTLE_SECONDS = 1.5
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
    unknown_streak: int,
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
        status.append((f"unknown {unknown_streak + 1}/{PATIENCE}", RED))
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


def pause(seconds: float, window: bool) -> bool:
    """Wait between steps. With a window, keep it responsive; False means q or Esc was pressed."""

    if not window:
        time.sleep(seconds)
        return True
    return cv2.waitKey(int(seconds * 1000)) & 0xFF not in (ord("q"), 27)


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

    unknown_streak = 0
    battles = 0
    previous_result = None  # what "You got" read on the previous frame of this result screen
    settled = None  # this result screen's reading, once two frames in a row agreed
    heroes: list[Match] = []  # deployed hero cards whose ability has not been used yet
    ability_at = 0.0
    looted = (0, 0, 0)  # gold, elixir, dark elixir over every readable result
    searches = 0
    started = time.monotonic()

    for step in itertools.count(1):
        frame = grab_frame(device)
        peaks = peak_scores(frame, buttons)
        matches = {label: m for label, m in peaks.items() if m.score >= THRESHOLD}
        screen = classify(matches)
        reading = read_loot(frame, digits) if screen is Screen.SCOUT else None

        # "You got" counts up from 0 as the result screen appears, and a part-way number is a
        # valid reading. Trust it only once two frames in a row agree, then count the battle once.
        result = None
        if screen is Screen.RESULT:
            if settled is None:
                now = read_result(frame, result_digits)
                values = (now.gold, now.elixir, now.dark_elixir)
                if values == previous_result:
                    settled = result = now
                    battles += 1
                    if now.ok:
                        looted = (looted[0] + now.gold, looted[1] + now.elixir, looted[2] + now.dark_elixir)
                previous_result = values
        else:
            previous_result, settled = None, None

        line = f"step {step:>3}  {screen.value:<12} {describe(matches)}"
        if reading is not None:
            line += f"  {describe_loot(reading)}"

        if screen is Screen.HOME and battles >= ITERATIONS:
            print(f"{line}\n\nhome after {battles} battle(s) -- done")
            print(run_summary(looted, searches, (time.monotonic() - started) / 60))
            return 0

        action = decide(screen, settled if screen is Screen.RESULT else reading, unknown_streak)
        target = f" {action.anchor}" if action.anchor else ""
        print(f"{line}  -> {action.kind}{target}  ({action.why})")

        if result is not None:
            got = f"{result.gold}/{result.elixir}/{result.dark_elixir}" if result.ok else "UNREADABLE (not counted)"
            print(f"        battle {battles} got {got}")
            print(f"        {run_summary(looted, searches, (time.monotonic() - started) / 60)}")

        if screen is Screen.UNKNOWN:
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
            show(frame, step, screen, action, peaks, reading, slots, points, tap, unknown_streak, battles)

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
            if action.anchor in ("find-match", "next-button"):
                searches += 1
        elif action.kind == DEPLOY:
            if not points:
                print("        no deploy points on this frame, retrying")
            elif len(slots) != len(DECK):
                # Retrying would not change the deck: DECK no longer describes the army.
                print(f"\ndeck shows {len(slots)} cards but DECK lists {len(DECK)} -- update DECK in main.py",
                      file=sys.stderr)
                return 1
            else:
                heroes = deploy_army(device, slots, points)
                ability_at = time.monotonic() + HERO_ABILITY_DELAY

        if screen is Screen.BATTLE and heroes and time.monotonic() >= ability_at:
            for hero in heroes:
                tap_match(device, hero)
            print(f"        hero abilities: tapped {len(heroes)} hero card(s)")
            heroes = []
        if screen is Screen.RESULT:
            heroes = []  # battle over; an unused ability never carries into the next one

        unknown_streak = unknown_streak + 1 if screen is Screen.UNKNOWN else 0
        if not pause(SETTLE_SECONDS, args.show):
            print("\nstopped from the window")
            return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nstopped")
        sys.exit(0)
