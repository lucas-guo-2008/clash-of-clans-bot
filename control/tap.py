"""Functions for actions on the device (tap)."""

import adbutils

from vision.anchors import Match


def tap_match(device: adbutils.AdbDevice, match: Match) -> None:
    """Tap the centre of a located UI element."""

    x, y = match.centre
    device.click(x, y)
