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

    print("Streaming... Press 's' to save frame, 'q' to quit.")

    index = 1

    # Loop continuously to create a live display feed
    while True:
        frame = grab_frame(device)
        
        # Display the frame to the OpenCV window
        cv2.imshow("ADB capture", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            cv2.imwrite(f"template matching/adb_frame_{index}.png", frame)
            print(f"Frame saved to template matching/adb_frame_{index}.png")
            index = index+1
        elif key == ord("q") or key == 27:  # 'q' or ESC to exit
            break

    cv2.destroyAllWindows()