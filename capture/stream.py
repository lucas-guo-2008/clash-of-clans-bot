"""Live video of the device screen, for the --show window only.

H.264 from the device's `screenrecord`, decoded by ffmpeg: 60 fps, ~100 ms behind screencap (measured
2026-11-17, from the first second on), against `screencap`'s ~3.6 fps loop. The bot never
decides on these frames: every template and digit threshold was set on lossless screencap frames.

BlueStacks is throttled by macOS while its window is covered or minimized: measured 2026-11-17, screencap
195 -> 480 ms and the stream 60 -> 25 fps, ~1 s behind. Keep the emulator window visible.
"""

import subprocess
import threading
import time

import numpy as np

from capture.adb import SERIAL

RECORD_SIZE = (1920, 1080)  # (w, h) encoded on the device. 1280x720 cost ffmpeg 6% of a core instead of 14%,
# but upscaled to device pixels it looked soft on a Retina screen
TIME_LIMIT = 180  # screenrecord's own maximum on Android 13
OVERLAP = 5.0  # the next recording starts this long before the current one ends, so the picture never stops

RECORD = ["adb", "-s", SERIAL, "exec-out", "screenrecord", "--output-format=h264", f"--time-limit={TIME_LIMIT}",
          f"--size={RECORD_SIZE[0]}x{RECORD_SIZE[1]}", "-"]
# One decoder thread: ffmpeg's default frame threading held the stream at ~30 fps and ~1 s behind.
# Probing off, so the first frame comes out at once (~0.6 s after the start); passthrough, so frames are
# not resampled to 25 fps. The scale only matters for a portrait screen (game closed), squashed to fit.
DECODE = ["ffmpeg", "-v", "error", "-threads", "1", "-probesize", "32", "-analyzeduration", "0",
          "-f", "h264", "-i", "pipe:0", "-fps_mode", "passthrough", "-vf", f"scale={RECORD_SIZE[0]}:{RECORD_SIZE[1]}",
          "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]


class Feed:
    """One screenrecord -> ffmpeg pipeline, up to TIME_LIMIT long, and the latest frame it decoded."""

    def __init__(self) -> None:
        self.latest: np.ndarray | None = None
        self.first = threading.Event()  # a frame has arrived
        self.ended = threading.Event()  # the time limit, the device gone, or kill()
        self.started = time.monotonic()
        self._record = subprocess.Popen(RECORD, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._decode = subprocess.Popen(DECODE, stdin=self._record.stdout, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL)
        self._record.stdout.close()  # ffmpeg owns it now
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        w, h = RECORD_SIZE
        size = w * h * 3
        while True:
            buf = self._decode.stdout.read(size)
            if len(buf) < size:
                break
            self.latest = np.frombuffer(buf, np.uint8).reshape(h, w, 3)  # scaled up only when shown
            self.first.set()
        self.ended.set()

    def kill(self) -> None:
        self._record.kill()
        self._decode.kill()


class Stream:
    """The latest frame of the live video, kept current by a background thread. `latest` is None until
    the first frame arrives, and stays None if ffmpeg or adb cannot be started."""

    def __init__(self) -> None:
        self._feed: Feed | None = None
        self._closed = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    @property
    def latest(self) -> np.ndarray | None:
        feed = self._feed
        return None if feed is None else feed.latest

    def _run(self) -> None:
        try:
            self._feed = Feed()
        except FileNotFoundError:  # no ffmpeg or adb on PATH: the window falls back to the bot's frames
            return
        while not self._closed.is_set():
            feed = self._feed
            # Until the current recording is about to end (or ended early: the device went away)
            feed.ended.wait(max(0.0, feed.started + TIME_LIMIT - OVERLAP - time.monotonic()))
            if self._closed.is_set():
                break
            if feed.ended.is_set() and time.monotonic() - feed.started < OVERLAP:
                self._closed.wait(1.0)  # failing at once: retry calmly
            following = Feed()
            while not (following.first.is_set() or following.ended.is_set() or self._closed.is_set()):
                following.first.wait(0.1)
            self._feed = following  # the switch: one assignment, the window reads either feed whole
            feed.kill()
        self._feed.kill()

    def close(self) -> None:
        """Stop the stream. Killing the local adb leaves `screenrecord` running on the device until its
        time limit, so it is killed there too."""

        self._closed.set()
        if self._feed is not None:
            self._feed.kill()
        subprocess.run(["adb", "-s", SERIAL, "shell", "pkill", "screenrecord"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
