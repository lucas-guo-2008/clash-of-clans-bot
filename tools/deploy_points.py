"""
Find deploy points on saved scout frames, offline. Pass --show to see the outline lines the
rays hit and the points drawn over each frame (any key for the next one).

  python -m tools.deploy_points templates/initial_collection/adb_frame_*.png --show
"""

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vision.deploy import UI_RECTS, deploy_points, outline_lines, outline_mask  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("frames", nargs="+", type=Path)
    parser.add_argument("--show", action="store_true", help="display an overlay for each frame")
    args = parser.parse_args()

    for path in args.frames:
        frame = cv2.imread(str(path))
        if frame is None:
            raise SystemExit(f"could not read {path}")

        points = deploy_points(frame)
        print(f"{path}  {len(points):>2} deploy points")

        if args.show:
            overlay = frame.copy()
            overlay[outline_lines(outline_mask(frame)) > 0] = (255, 0, 255)
            for x0, y0, x1, y1 in UI_RECTS:
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (128, 128, 128), 2)
            for p in points:
                cv2.circle(overlay, (p.x, p.y), 12, (255, 255, 0), 3)
            cv2.namedWindow("deploy points", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("deploy points", 960, 540)
            cv2.imshow("deploy points", overlay)
            if cv2.waitKey(0) & 0xFF in (ord("q"), 27):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
