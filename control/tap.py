"""Functions for actions on the device (tap)."""

import adbutils

from vision.anchors import Match
from vision.deploy import DeployPoint


def tap_match(device: adbutils.AdbDevice, match: Match) -> None:
    """Tap the centre of a located UI element."""

    x, y = match.centre
    device.click(x, y)


def tap_point(device: adbutils.AdbDevice, point: DeployPoint) -> None:
    """Drop a troop on a deploy point found on a frame."""

    device.click(point.x, point.y)
