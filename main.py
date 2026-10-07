"""
Farm: home -> army -> attack menu -> scout -> Next until a base clears the thresholds ->
drop the first card's troops around it -> wait out the battle -> Return Home, ITERATIONS times.

  python main.py --dry-run
  python main.py
  python main.py --show      # live window: screen, scores, deploy points, next tap

Stops back home after ITERATIONS battles, or on Ctrl-C (or q in the --show window).
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
from vision.deploy import DeployPoint, deploy_points, first_card, outline_lines, outline_mask
from vision.glyphs import load_digit_templates
from vision.loot import LootReading, read_loot
from vision.overlay import CYAN, GREEN, GREY, RED, WHITE, draw_anchors, draw_deploy, draw_tap, with_header
from vision.screens import Screen, classify

ROOT = Path(__file__).resolve().parent
SETTLE_SECONDS = 1.5
UNKNOWN_DIR = ROOT / "templates" / "unknown"
WINDOW = "bot"

ITERATIONS = 1
TROOPS = 24  # in the first card; slot and count detection come later


def describe(matches) -> str:
    if not matches:
        return "(no anchors)"
    return " ".join(f"{m.label}:{m.score:.3f}" for m in sorted(matches.values(), key=lambda m: -m.score))


def describe_loot(reading: LootReading) -> str:
    if not reading.ok:
        reasons = ",".join(row.reason for row in reading.rows if row.reason) or "?"
        return f"loot=UNREADABLE({reasons})"
    return f"loot={reading.gold}/{reading.elixir}/{reading.dark_elixir}"


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
    card: Match | None,
    points: list[DeployPoint],
    tap: tuple[int, int] | None,
    unknown_streak: int,
    battles: int,
) -> None:
    """Draw everything the bot found and decided this step, and put it in the window."""

    image = frame.copy()
    if screen is Screen.SCOUT:
        draw_deploy(image, outline_lines(outline_mask(frame)), points, card)
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
        status.append((f"{len(points)} deploy points, {'card' if card else 'NO card'}", CYAN))

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
    device = connect()
    if args.show:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, 960, 600)

    unknown_streak = 0
    battles = 0
    previous = None

    for step in itertools.count(1):
        frame = grab_frame(device)
        peaks = peak_scores(frame, buttons)
        matches = {label: m for label, m in peaks.items() if m.score >= THRESHOLD}
        screen = classify(matches)
        reading = read_loot(frame, digits) if screen is Screen.SCOUT else None

        if screen is Screen.RESULT and previous is not Screen.RESULT:
            battles += 1
        previous = screen

        line = f"step {step:>3}  {screen.value:<12} {describe(matches)}"
        if reading is not None:
            line += f"  {describe_loot(reading)}"

        if screen is Screen.HOME and battles >= ITERATIONS:
            print(f"{line}\n\nhome after {battles} battle(s) -- done")
            return 0

        action = decide(screen, reading, unknown_streak)
        target = f" {action.anchor}" if action.anchor else ""
        print(f"{line}  -> {action.kind}{target}  ({action.why})")

        if screen is Screen.UNKNOWN:
            print(f"        saved {save_unknown(frame).relative_to(ROOT)}")

        card, points = None, []
        if screen is Screen.SCOUT and (args.show or action.kind == DEPLOY):
            card, points = first_card(frame), deploy_points(frame)
        if action.kind == DEPLOY:
            found = f"card at {card.centre}" if card else "no card"
            print(f"        {found}, {len(points)} deploy points")

        tap = None
        if action.kind == TAP:
            tap = matches[action.anchor].centre
        elif action.kind == DEPLOY and card is not None and points:
            tap = card.centre

        if args.show:
            show(frame, step, screen, action, peaks, reading, card, points, tap, unknown_streak, battles)

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
        elif action.kind == DEPLOY:
            if card is None or not points:
                print("        nothing to tap on this frame, retrying")
            else:
                tap_match(device, card)
                for point in itertools.islice(itertools.cycle(points), TROOPS):
                    tap_point(device, point)

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
