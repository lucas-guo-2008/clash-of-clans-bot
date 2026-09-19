"""Functions for actions on the device (tap, swipe, back)."""

import adbutils

from vision.anchors import Match

BACK = 4  # Android KEYCODE_BACK


def tap(device: adbutils.AdbDevice, x: int, y: int) -> None:
    """Tap a point in device pixels."""

    device.click(x, y)


def tap_match(device: adbutils.AdbDevice, match: Match) -> tuple[int, int]:
    """Tap the centre of a located UI element. Returns the point tapped."""

    x, y = match.centre
    device.click(x, y)
    return x, y


def swipe(
    device: adbutils.AdbDevice,
    sx: int,
    sy: int,
    ex: int,
    ey: int,
    duration: float = 0.3,
) -> None:
    """Drag from one point to another over `duration` seconds."""

    device.swipe(sx, sy, ex, ey, duration)


def back(device: adbutils.AdbDevice) -> None:
    """Press the Android back button."""

    device.keyevent(BACK)
