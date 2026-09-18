"""
Useful for grabbing frames using ADB for template files.
Open BlueStacks, run this file, enter a file prefix, and use 's' for save, 'q' to quit.
Frames land in templates/initial_collection/<prefix>_<index>.png.
"""

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from capture.adb import connect, grab_frame  # noqa: E402

OUT_DIR = ROOT / "templates" / "initial_collection"


if __name__ == "__main__":
    device = connect()

    cv2.namedWindow("ADB capture", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ADB capture", 960, 540)

    print("Streaming... Press 's' to save frame, 'q' to quit.")

    index = 0
    file_prefix = input("File name?\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Loop continuously to create a live display feed
    while True:
        frame = grab_frame(device)

        # Display the frame to the OpenCV window
        cv2.imshow("ADB capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            # Skip indices already on disk so a rerun does not overwrite earlier frames
            while (OUT_DIR / f"{file_prefix}_{index}.png").exists():
                index += 1
            out = OUT_DIR / f"{file_prefix}_{index}.png"
            cv2.imwrite(str(out), frame)
            print(f"Frame saved to {out.relative_to(ROOT)}")
            index += 1
        elif key == ord("q") or key == 27:  # 'q' or ESC to exit
            break

    cv2.destroyAllWindows()
