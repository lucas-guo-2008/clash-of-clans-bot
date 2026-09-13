import cv2 
import numpy as np
import struct
import adbutils

SERIAL = "127.0.0.1:5555"
HEADER_SIZE = 16

def grab_frame(device: adbutils.AdbDevice) -> np.ndarray:
    """Capture the device screen as a BGR frame in device pixels (e.g. 1920x1080)."""

    raw = device.shell("screencap", encoding=None)
    width, height, format, _colorspace = struct.unpack_from("<IIII", raw)

    rgba = np.frombuffer(raw, np.uint8, offset=HEADER_SIZE).reshape(height, width, 4)
    # OpenCV needs BGR channel order
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


if __name__ == "__main__":
    adb = adbutils.AdbClient(host="127.0.0.1", port=5037)
    adb.connect(SERIAL)
    device = adb.device(SERIAL)

    cv2.namedWindow("ADB capture", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ADB capture", 960, 540)

    template = cv2.imread("template matching/th-template.png")
    h, w, c = template.shape

    while "Screen Capturing":
        frame = grab_frame(device)

        img2 = frame.copy()
        result = cv2.matchTemplate(img2, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        location = max_loc

        cv2.rectangle(img2, location, (location[0]+w, location[1]+h), 255, 5)
        cv2.putText(img2, f"match {max_val}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        cv2.imshow("ADB capture", img2)

        if cv2.waitKey(2000) == ord("q"):
                cv2.destroyAllWindows()
