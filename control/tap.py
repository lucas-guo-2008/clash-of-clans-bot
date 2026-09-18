"""Actions on the device.

adbutils drives input over the persistent connection, so nothing here shells out to
`adb shell input`. Coordinates are device pixels, the same space capture returns and
the same space every constant in the vision layer is measured in.

The layer fails closed the way vision does: `tap_match` takes a Match that something
actually found, never coordinates a caller guessed at. A caller holding no match taps
nothing. Tapping a screen the bot has not identified is how a bot ends up spending gems
or surrendering a battle.
"""

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
    """Press the Android back button.

    The recovery primitive: an unrecognized screen gets a back press and a re-classify,
    rather than a handler per popup.
    """

    device.keyevent(BACK)
