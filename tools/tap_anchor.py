"""Tap a named UI anchor in BlueStacks, and show what changed.

Involves full capture -> vision -> control path in one command.

  python -m tools.tap_anchor attack --dry-run
  python -m tools.tap_anchor attack

Options: attack, find-match, attack-button, next-button, end-battle
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from capture.adb import connect, grab_frame  # noqa: E402
from control.tap import tap_match  # noqa: E402
from vision.anchors import find_anchors  # noqa: E402
from vision.buttons import load_button_templates  # noqa: E402

SETTLE_SECONDS = 1.5


def describe(matches) -> str:
    if not matches:
        return "(none)"
    return ", ".join(f"{m.label}@{m.centre} {m.score:.3f}" for m in sorted(matches, key=lambda m: -m.score))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", help="anchor to tap, e.g. attack, find-match, next-button")
    parser.add_argument("--dry-run", action="store_true", help="locate and report, but do not tap")
    parser.add_argument(
        "--settle",
        type=float,
        default=SETTLE_SECONDS,
        help=f"seconds to wait after tapping before re-capturing (default {SETTLE_SECONDS})",
    )
    args = parser.parse_args()

    templates = load_button_templates()
    if args.label not in templates:
        raise SystemExit(f"unknown anchor '{args.label}'; known: {', '.join(sorted(templates))}")

    device = connect()
    before = find_anchors(grab_frame(device), templates)
    print(f"before: {describe(before)}")

    target = next((m for m in before if m.label == args.label), None)
    if target is None:
        print(f"\n'{args.label}' is not on screen -- nothing tapped.", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print(f"\nwould tap {target.label} at {target.centre} (score {target.score:.3f})")
        return

    x, y = tap_match(device, target)
    print(f"tapped  {target.label} at ({x}, {y}) score {target.score:.3f}")

    time.sleep(args.settle)
    after = find_anchors(grab_frame(device), templates)
    print(f"after:  {describe(after)}")

    changed = {m.label for m in before} != {m.label for m in after}
    print(f"\nscreen {'changed' if changed else 'did NOT change'}")
    sys.exit(0 if changed else 2)


if __name__ == "__main__":
    main()
