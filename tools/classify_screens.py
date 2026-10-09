"""
Name the screen in saved frames, and show how much room THRESHOLD has because we want it between lowest score accepted and highest score rejected.

Expected screens live in tests/test_screens.py.

  python -m tools.classify_screens templates/initial_collection/adb_frame_*.png
"""

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vision.anchors import THRESHOLD, peak_scores  # noqa: E402
from vision.buttons import load_button_templates  # noqa: E402
from vision.screens import classify  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", nargs="+", type=Path)
    args = parser.parse_args()

    templates = load_button_templates()

    accepted: list[tuple[float, str]] = []  # scores at or above threshold
    rejected: list[tuple[float, str]] = []  # scores below it

    for path in args.frames:
        frame = cv2.imread(str(path))
        if frame is None:
            raise SystemExit(f"could not read {path}")

        peaks = peak_scores(frame, templates)
        matches = {label: m for label, m in peaks.items() if m.score >= THRESHOLD}
        screen = classify(matches)

        print(f"{path}  ->  {screen.value}")
        for label, match in sorted(peaks.items(), key=lambda kv: -kv[1].score):
            hit = match.score >= THRESHOLD
            (accepted if hit else rejected).append((match.score, f"{label} on {path.name}"))
            print(f"    {label:<14} {match.score:.3f}  {'match' if hit else ''}")

    print(f"\nthreshold {THRESHOLD}")
    if accepted:
        score, where = min(accepted)
        print(f"  lowest accepted   {score:.3f}   {where}")
    if rejected:
        score, where = max(rejected)
        print(f"  highest rejected  {score:.3f}   {where}")
    if accepted and rejected:
        gap = min(accepted)[0] - max(rejected)[0]
        print(f"  gap               {gap:.3f}   {'fine' if gap > 0 else 'OVERLAP -- threshold cannot separate these'}")


if __name__ == "__main__":
    main()
