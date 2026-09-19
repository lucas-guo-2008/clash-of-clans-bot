"""
Name the screen in saved frames, and show how much room THRESHOLD has because we want it between lowest score accepted and highest score rejected.

  python -m tools.classify_screens templates/initial_collection/adb_frame_*.png
  python -m tools.classify_screens templates/initial_collection/adb_frame_*.png --expect home attack_menu army scout
"""

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vision.anchors import THRESHOLD, peak_scores  # noqa: E402
from vision.buttons import load_button_templates  # noqa: E402
from vision.screens import Screen, classify  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", nargs="+", type=Path)
    parser.add_argument(
        "--expect",
        nargs="+",
        metavar="SCREEN",
        help="one expected screen name per frame",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help=f"override the match threshold for this run (default {THRESHOLD})",
    )
    args = parser.parse_args()

    if args.expect and len(args.expect) != len(args.frames):
        raise SystemExit(f"--expect takes one name per frame: {len(args.frames)} frames, {len(args.expect)} names")

    try:
        expected = [Screen(name) for name in args.expect] if args.expect else [None] * len(args.frames)
    except ValueError as err:
        raise SystemExit(f"{err}; known screens: {', '.join(s.value for s in Screen)}") from None

    templates = load_button_templates()

    failures = 0
    accepted: list[tuple[float, str]] = []  # scores at or above threshold
    rejected: list[tuple[float, str]] = []  # scores below it

    for path, want in zip(args.frames, expected):
        frame = cv2.imread(str(path))
        if frame is None:
            raise SystemExit(f"could not read {path}")

        peaks = peak_scores(frame, templates)
        matches = [m for m in peaks.values() if m.score >= args.threshold]
        screen = classify(matches)

        verdict = ""
        if want is not None:
            if screen is want:
                verdict = "  OK"
            else:
                verdict = f"  MISMATCH (expected {want.value})"
                failures += 1

        print(f"{path}  ->  {screen.value}{verdict}")
        for label, match in sorted(peaks.items(), key=lambda kv: -kv[1].score):
            hit = match.score >= args.threshold
            (accepted if hit else rejected).append((match.score, f"{label} on {path.name}"))
            print(f"    {label:<14} {match.score:.3f}  {'match' if hit else ''}")

    print(f"\nthreshold {args.threshold}")
    if accepted:
        score, where = min(accepted)
        print(f"  lowest accepted   {score:.3f}   {where}")
    if rejected:
        score, where = max(rejected)
        print(f"  highest rejected  {score:.3f}   {where}")
    if accepted and rejected:
        gap = min(accepted)[0] - max(rejected)[0]
        print(f"  gap               {gap:.3f}   {'fine' if gap > 0 else 'OVERLAP -- threshold cannot separate these'}")

    if failures:
        print(f"\n{failures} frame(s) classified wrongly", file=sys.stderr)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
