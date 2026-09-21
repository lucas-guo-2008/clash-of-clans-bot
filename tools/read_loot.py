"""
Read the loot panel from saved frames, offline.
Pass --expect to compare to values you read off the screen yourself.

  python -m tools.read_loot templates/initial_collection/adb_frame_4.png
  python -m tools.read_loot templates/initial_collection/adb_frame_4.png --expect 525962 1333432 29717
"""

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vision.glyphs import load_digit_templates  # noqa: E402
from vision.loot import ROW_NAMES, read_loot  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", nargs="+", type=Path)
    parser.add_argument(
        "--expect",
        nargs=3,
        metavar=("GOLD", "ELIXIR", "DARK"),
        help="assert the reading matches these values (single frame only)",
    )
    args = parser.parse_args()

    templates = load_digit_templates()

    failures = 0
    for path in args.frames:
        frame = cv2.imread(str(path))
        if frame is None:
            raise SystemExit(f"could not read {path}")

        reading = read_loot(frame, templates)
        status = "ok" if reading.ok else "REJECTED"
        print(f"{path} [{status}]")
        for name, row in zip(ROW_NAMES, reading.rows):
            scores = f"min score {round(min(row.scores), 3)}" if row.scores else "-"
            detail = row.reason or scores
            print(f"   {name:<12} {str(row.value):>10}   {detail}")

        if args.expect:
            want = tuple(int(v) for v in args.expect)
            got = (reading.gold, reading.elixir, reading.dark_elixir)
            if got == want:
                print(f"   OK: {got}")
            else:
                print(f"   FAILED: expected {want}, got {got}")
                failures += 1

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
