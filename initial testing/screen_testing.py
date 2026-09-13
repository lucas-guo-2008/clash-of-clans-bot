import struct
import adbutils
import cv2
import numpy as np

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

    while "Screen Capturing":
        frame = grab_frame(device)
        cv2.imshow("ADB capture", frame)

        if cv2.waitKey(25) == ord("q"):
            cv2.destroyAllWindows()
            break
