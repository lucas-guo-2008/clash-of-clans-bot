import adbutils
import pywinctl as pwc
import cv2
import numpy as np
from mss import MSS

# finding Bluestacks window
bluestacks_window = pwc.getWindowsWithTitle('BlueStacks Air')[0]
if not bluestacks_window:
    raise RuntimeError("BlueStacks window not found.")

# moving window
bluestacks_window.activate()
bluestacks_window.moveTo(0, 0)
bluestacks_window.resizeTo(1920, 1080)

# grab adb
adb = adbutils.AdbClient(host="127.0.0.1", port=5037)

d = adb.device()
d.click(50, 50)