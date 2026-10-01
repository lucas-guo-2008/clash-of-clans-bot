"""
Farm: home -> army -> attack menu -> scout -> Next until a base clears the thresholds.

  python main.py --dry-run
  python main.py

The loop does not stop on its own -- Ctrl-C ends a run. Every search and every Next
costs 900 gold, so an unattended run against a strict threshold can spend a lot of it.
"""

import argparse
import itertools
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

from capture.adb import connect, grab_frame
from control.tap import tap_match
from policy.navigator import ABORT, DEPLOY, TAP, decide
from vision.buttons import load_button_templates
from vision.glyphs import load_digit_templates
from vision.loot import LootReading, read_loot
from vision.screens import Screen, identify

ROOT = Path(__file__).resolve().parent
SETTLE_SECONDS = 1.5
UNKNOWN_DIR = ROOT / "templates" / "unknown"


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="classify one screen and report, but do not tap")
    args = parser.parse_args()

    buttons = load_button_templates()
    digits = load_digit_templates()
    device = connect()

    unknown_streak = 0

    for step in itertools.count(1):
        frame = grab_frame(device)
        screen, matches = identify(frame, buttons)
        reading = read_loot(frame, digits) if screen is Screen.SCOUT else None

        line = f"step {step:>3}  {screen.value:<12} {describe(matches)}"
        if reading is not None:
            line += f"  {describe_loot(reading)}"

        action = decide(screen, reading, unknown_streak)
        target = f" {action.anchor}" if action.anchor else ""
        print(f"{line}  -> {action.kind}{target}  ({action.why})")

        if screen is Screen.UNKNOWN:
            print(f"        saved {save_unknown(frame).relative_to(ROOT)}")

        if args.dry_run:
            if action.kind == TAP:
                print(f"        would tap {action.anchor} at {matches[action.anchor].centre}")
            return 0

        if action.kind == DEPLOY:
            # Phase 3 drops the army here. Until it lands, stop rather than Next past a
            # base the bot just decided was worth attacking.
            print("\nthis base clears the thresholds, but deploying is not built yet -- stopping")
            return 0
        if action.kind == ABORT:
            print(f"\ngiving up: {action.why}", file=sys.stderr)
            return 1

        if action.kind == TAP:
            # Always present: every TAP targets the anchor that named the screen.
            tap_match(device, matches[action.anchor])

        # Count what was seen, not what was done: a recognized screen that answers WAIT
        # must not use up the patience reserved for unrecognized ones.
        unknown_streak = unknown_streak + 1 if screen is Screen.UNKNOWN else 0
        time.sleep(SETTLE_SECONDS)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nstopped")
        sys.exit(0)
