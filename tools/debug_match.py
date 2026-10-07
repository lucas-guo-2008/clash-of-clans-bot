"""Live overlay showing which UI anchors are matching, and the score.

    python -m tools.debug_match

Press 'q' or ESC to quit, as always
"""

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from capture.adb import connect, grab_frame  # noqa: E402
from vision.anchors import THRESHOLD, find_anchors  # noqa: E402
from vision.buttons import load_button_templates  # noqa: E402
from vision.overlay import draw_anchors  # noqa: E402


def main() -> None:
    templates = load_button_templates()
    device = connect()

    cv2.namedWindow("anchors", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("anchors", 960, 540)
    print(f"Matching {len(templates)} templates. Press 'q' to quit.")

    while True:
        frame = grab_frame(device)
        overlay = frame.copy()

        draw_anchors(overlay, find_anchors(frame, templates), THRESHOLD)

        cv2.imshow("anchors", overlay)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
