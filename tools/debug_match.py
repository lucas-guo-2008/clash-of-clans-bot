"""Live overlay showing which UI anchors are matching, and how well.

    python -m tools.debug_match

Press 'q' or ESC to quit, as always
"""

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from capture.adb import connect, grab_frame  # noqa: E402
from vision.anchors import find_anchors  # noqa: E402

TEMPLATE_DIR = ROOT / "templates" / "initial_collection"
TEMPLATE_NAMES = ("attack", "attack-button", "find-match", "end-battle", "next-button")


def load_templates() -> dict:
    templates = {}
    for name in TEMPLATE_NAMES:
        path = TEMPLATE_DIR / f"{name}.png"
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"could not read template {path}")
        templates[name] = image
    return templates


def main() -> None:
    templates = load_templates()
    device = connect()

    cv2.namedWindow("anchors", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("anchors", 960, 540)
    print(f"Matching {len(templates)} templates. Press 'q' to quit.")

    while True:
        frame = grab_frame(device)
        overlay = frame.copy()

        for match in find_anchors(frame, templates):
            cv2.rectangle(
                overlay, (match.x, match.y), (match.x + match.w, match.y + match.h), (0, 255, 0), 2
            )
            cv2.putText(
                overlay,
                f"{match.label} {match.score:.3f}",
                (match.x, max(20, match.y - 10)),
                cv2.FONT_HERSHEY_COMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        cv2.imshow("anchors", overlay)
        if cv2.waitKey(250) & 0xFF in (ord("q"), 27):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
