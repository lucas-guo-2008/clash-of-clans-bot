"""
Read the loot panel from saved frames, offline. Expected values live in tests/test_loot.py.

  python -m tools.read_loot templates/initial_collection/adb_frame_4.png
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
    args = parser.parse_args()

    templates = load_digit_templates()

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


if __name__ == "__main__":
    main()
