"""
Drive the navigation loop: home -> army -> attack menu -> scout -> Next... -> home.

  python -m tools.navigate --max-nexts 2 --dry-run
  python -m tools.navigate --max-nexts 2

--dry-run tag reports decisions but does not send the adb input.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from capture.adb import connect, grab_frame  # noqa: E402
from control.tap import back, tap_match  # noqa: E402
from policy.navigator import ABORT, BACK, DONE, TAP, WAIT, Progress, advance, decide  # noqa: E402
from vision.buttons import load_button_templates  # noqa: E402
from vision.glyphs import load_digit_templates  # noqa: E402
from vision.loot import read_loot  # noqa: E402
from vision.screens import Screen, identify  # noqa: E402

SETTLE_SECONDS = 1.5
MAX_STEPS = 60  # a 2-Next run takes roughly 20 steps
UNKNOWN_DIR = ROOT / "templates" / "unknown"


def describe(matches) -> str:
    if not matches:
        return "(no anchors)"
    return " ".join(f"{m.label}:{m.score:.3f}" for m in sorted(matches, key=lambda m: -m.score))


def describe_loot(frame, digit_templates) -> str:
    reading = read_loot(frame, digit_templates)
    if not reading.ok:
        reasons = ",".join(row.reason for row in reading.rows if row.reason) or "?"
        return f"loot=UNREADABLE({reasons})"
    return f"loot={reading.gold}/{reading.elixir}/{reading.dark_elixir}"


def save_unknown(frame) -> Path:
    UNKNOWN_DIR.mkdir(parents=True, exist_ok=True)
    out = UNKNOWN_DIR / f"unknown_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
    cv2.imwrite(str(out), frame)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--max-nexts",
        type=int,
        required=True,
        help="how many 900-gold Next taps this run may spend (required, no default)",
    )
    parser.add_argument("--dry-run", action="store_true", help="classify one screen and report, but do not tap")
    parser.add_argument(
        "--settle", type=float, default=SETTLE_SECONDS, help=f"seconds to wait after acting (default {SETTLE_SECONDS})"
    )
    parser.add_argument(
        "--max-steps", type=int, default=MAX_STEPS, help=f"give up after this many steps (default {MAX_STEPS})"
    )
    parser.add_argument(
        "--no-save-unknown",
        action="store_true",
        help=f"do not write unrecognized frames to {UNKNOWN_DIR.relative_to(ROOT)}/",
    )
    args = parser.parse_args()

    if args.max_nexts < 0:
        raise SystemExit("--max-nexts cannot be negative")

    buttons = load_button_templates()
    digits = load_digit_templates()
    device = connect()

    cost = 900 * (1 + args.max_nexts)
    plan = f"1 Find a Match + {args.max_nexts} Next"
    print(f"{'(dry run) ' if args.dry_run else ''}budget: {plan} = {cost} gold\n")

    progress = Progress(max_nexts=args.max_nexts)

    for step in range(1, args.max_steps + 1):
        frame = grab_frame(device)
        screen, matches = identify(frame, buttons)

        line = f"step {step:>3}  {screen.value:<12} {describe(matches)}"
        if screen is Screen.SCOUT:
            line += f"  {describe_loot(frame, digits)}"

        action = decide(screen, progress)
        target = f" {action.anchor}" if action.anchor else ""
        print(f"{line}  -> {action.kind}{target}  ({action.why})")

        if screen is Screen.UNKNOWN and not args.no_save_unknown:
            print(f"        saved {save_unknown(frame).relative_to(ROOT)}")

        if args.dry_run:
            if action.kind != TAP:
                return 0
            match = next((m for m in matches if m.label == action.anchor), None)
            if match is None:
                print(f"        {action.anchor} is NOT on screen -- a live run would stop here", file=sys.stderr)
                return 1
            print(f"        would tap {action.anchor} at {match.centre}")
            return 0

        if action.kind == DONE:
            print(f"\ndone: {progress.nexts_used} Next(s) spent, {cost} gold")
            return 0
        if action.kind == ABORT:
            print(f"\ngiving up: {action.why}", file=sys.stderr)
            return 1

        if action.kind == TAP:
            match = next((m for m in matches if m.label == action.anchor), None)
            if match is None:
                print(f"\n{action.anchor} is not on screen -- nothing tapped", file=sys.stderr)
                return 1
            tap_match(device, match)
        elif action.kind == BACK:
            back(device)

        progress = advance(progress, action)
        time.sleep(args.settle)

    print(f"\ngiving up: {args.max_steps} steps without finishing", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
