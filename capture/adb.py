"""Screen capture over ADB."""

import struct

import adbutils
import cv2
import numpy as np

SERIAL = "127.0.0.1:5555"
HEADER_SIZE = 16
CHUNK_SIZE = 1 << 20


def connect(serial: str = SERIAL) -> adbutils.AdbDevice:
    """Open a persistent connection to the emulator."""

    client = adbutils.AdbClient(host="127.0.0.1", port=5037)
    client.connect(serial)
    return client.device(serial)


def grab_frame(device: adbutils.AdbDevice) -> np.ndarray:
    """Capture the device screen as a BGR frame in device pixels (e.g. 1920x1080).

    Reads the socket directly rather than using the buffered `shell(encoding=None)`
    form, which measures ~540 ms per frame against ~176 ms here for byte-identical
    output -- a 3x difference on an 8 MB transfer. `adb exec-out screencap` is a hair
    faster still (~164 ms) but spawns a process per frame, which would give up the
    persistent connection for about 7%.
    """

    conn = device.shell("screencap", stream=True)
    try:
        chunks = []
        while True:
            chunk = conn.read(CHUNK_SIZE)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        conn.close()

    raw = b"".join(chunks)
    width, height, _format, _colorspace = struct.unpack_from("<IIII", raw)

    rgba = np.frombuffer(raw, np.uint8, offset=HEADER_SIZE).reshape(height, width, 4)
    # OpenCV needs BGR channel order
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
