import cv2 
import numpy as np
import struct
import adbutils

SERIAL = "127.0.0.1:5555"
HEADER_SIZE = 16

def grab_frame(device: adbutils.AdbDevice) -> np.ndarray:
    """Capture the device screen as a BGR frame in 1920x1080"""

    raw = device.shell("screencap", encoding=None)
    width, height, format, _colorspace = struct.unpack_from("<IIII", raw)

    rgba = np.frombuffer(raw, np.uint8, offset=HEADER_SIZE).reshape(height, width, 4)
    # OpenCV needs BGR channel order
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def deduplicate(boxes: list) -> list:
    """Keep only the highest-scoring box per label."""

    best = {}
    for box in boxes:
        label, score = box[4], box[5]
        if label not in best or score > best[label][5]:
            best[label] = box

    return list(best.values())


if __name__ == "__main__":
    adb = adbutils.AdbClient(host="127.0.0.1", port=5037)
    adb.connect(SERIAL)
    device = adb.device(SERIAL)

    cv2.namedWindow("ADB capture", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ADB capture", 960, 540)

    templates = {
        "attack_template": cv2.imread("templates/attack.png"),
        "find_match_template": cv2.imread("templates/find-match.png"),
        "attack_button_template": cv2.imread("templates/attack-button.png"),
        "end_battle_template": cv2.imread("templates/end-battle.png"),
        "next_button_template": cv2.imread("templates/next-button.png")
    }

    THRESHOLD = 0.8

    while "Screen Capturing":
        all_boxes = []
        frame = grab_frame(device)

        img2 = frame.copy()

        for label, template in templates.items():
            h, w = template.shape[:2]
            res = cv2.matchTemplate(img2, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= THRESHOLD)
            for pt in zip(*loc[::-1]):  # Convert (y, x) to (x, y)
                all_boxes.append([pt[0], pt[1], pt[0] + w, pt[1] + h, label, float(res[pt[1], pt[0]])])
        
        for (x1, y1, x2, y2, label, score) in deduplicate(all_boxes):
            cv2.rectangle(img2, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img2, f"{label}\nconf: {round(score, 4)}", (x1, y1-40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

        cv2.imshow("Multi-Template Matching", img2)

        if cv2.waitKey(2000) == ord("q"):
            cv2.destroyAllWindows()
            break
