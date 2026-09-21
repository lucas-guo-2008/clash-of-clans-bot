"""Functions for actions on the device (tap, back)."""

import adbutils

from vision.anchors import Match

KEYCODE_BACK = 4


def tap_match(device: adbutils.AdbDevice, match: Match) -> None:
    """Tap the centre of a located UI element."""

    x, y = match.centre
    device.click(x, y)


def back(device: adbutils.AdbDevice) -> None:
    """Press the Android back button."""

    device.keyevent(KEYCODE_BACK)
