"""Small, local Trackmania bridge for the Aimap experiment.

Uses the existing Openplanet telemetry plugin, TMRL's screenshot-derived LIDAR,
and the installed virtual gamepad. It never moves or focuses the game window.
All controls are released on exit or failure.
"""

import ctypes
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import select
import socket
import time

import numpy as np
import vgamepad
import win32gui
from tmrl.custom.tm.utils.control_keyboard import DEL, PressKey, ReleaseKey
from tmrl.custom.tm.utils.tools import Lidar
from tmrl.custom.tm.utils.window import WindowInterface

from probe import FIELDS, PACKET


ROOT = Path(__file__).resolve().parent
MAP = Path(os.environ.get("TRACKMANIA_MAP_PATH",
                          Path.home() / "Documents" / "Trackmania" / "Maps" /
                          "My Maps" / "Aimap.Map.Gbx"))
FROZEN_MAP = ROOT / "data" / "Aimap.Map.Gbx"
EXPECTED_MAP_SHA256 = "1477f55a124739f26b9d7a9b16633e6ffd3eb93deebc02fa7c7ab0e2c8e1e6c3"
START = (400.0, 368.0)


def check_map() -> str:
    original = hashlib.sha256(MAP.read_bytes()).hexdigest()
    if original != EXPECTED_MAP_SHA256:
        raise RuntimeError("Aimap has changed since the human benchmark")
    if FROZEN_MAP.exists() and hashlib.sha256(FROZEN_MAP.read_bytes()).hexdigest() != original:
        raise RuntimeError("The optional frozen Aimap copy differs from the installed map")
    return original


class Telemetry:
    """Return the newest complete packet, discarding queued older packets."""

    def __init__(self):
        self.socket = socket.create_connection(("127.0.0.1", 9000), timeout=3)
        self.socket.setblocking(False)
        self.buffer = bytearray()

    def latest(self, timeout: float = 0.25) -> dict[str, float]:
        deadline = time.monotonic() + timeout
        newest = None
        while True:
            wait = max(0.0, deadline - time.monotonic()) if newest is None else 0.0
            readable, _, _ = select.select([self.socket], [], [], wait)
            if not readable:
                if newest is None:
                    raise TimeoutError("Openplanet telemetry stopped")
                return newest
            chunk = self.socket.recv(65536)
            if not chunk:
                raise ConnectionError("Openplanet closed its telemetry connection")
            self.buffer.extend(chunk)
            complete = len(self.buffer) // PACKET.size
            if complete:
                raw = self.buffer[(complete - 1) * PACKET.size:complete * PACKET.size]
                del self.buffer[:complete * PACKET.size]
                values = PACKET.unpack(raw)
                if not all(math.isfinite(value) for value in values):
                    raise ValueError("Non-finite game telemetry")
                newest = dict(zip(FIELDS, values))

    def close(self):
        self.socket.close()


@dataclass
class Sample:
    telemetry: dict[str, float]
    lidar: np.ndarray
    timestamp: float


class GameBridge:
    def __init__(self):
        check_map()
        ctypes.windll.user32.SetProcessDPIAware()
        self.hwnd = win32gui.FindWindow(None, "Trackmania")
        if not self.hwnd:
            raise RuntimeError("Trackmania window not found")
        self.check_focus()
        self.window = WindowInterface("Trackmania")
        self.telemetry = Telemetry()
        self.pad = vgamepad.VX360Gamepad()
        self.lidar = None
        self.frame_shape = None
        self.release()

    def check_focus(self):
        if (win32gui.IsIconic(self.hwnd) or
                win32gui.GetForegroundWindow() != self.hwnd):
            raise RuntimeError("Trackmania lost foreground focus")

    def release(self):
        self.pad.reset()
        self.pad.update()

    def apply(self, steering: float, drive: float):
        """Steering is continuous; drive selects full gas, coast or full brake.

        Half trigger did not register as gas in the earlier game probe, so the
        initial policy uses a documented three-way drive mapping.
        """
        self.check_focus()
        if not all(math.isfinite(value) and -1 <= value <= 1
                   for value in (steering, drive)):
            raise ValueError("actions must be finite and within [-1, 1]")
        self.pad.right_trigger_float(1.0 if drive > 0.2 else 0.0)
        self.pad.left_trigger_float(1.0 if drive < -0.2 else 0.0)
        self.pad.left_joystick_float(steering, 0.0)
        self.pad.update()

    def sample(self) -> Sample:
        self.check_focus()
        frame = self.window.screenshot()[:, :, :3]
        shape = frame.shape
        if self.frame_shape is None:
            self.frame_shape = shape
            self.lidar = Lidar(frame)
            if min(shape[:2]) < 100 or np.std(frame[::32, ::32]) < 2:
                raise RuntimeError("Game capture appears empty")
        elif shape != self.frame_shape:
            raise RuntimeError("Game window size changed during a run")
        rays = self.lidar.lidar_20(frame, show=False)
        data = self.telemetry.latest()
        return Sample(data, rays, time.monotonic())

    def at_start(self, data: dict[str, float]) -> bool:
        return (math.hypot(data["x"] - START[0], data["z"] - START[1]) < 2
                and abs(data["speed"]) < 1 and data["finished"] < 0.5)

    def ensure_start(self):
        if not self.at_start(self.telemetry.latest()):
            raise RuntimeError("Reset to Aimap's start before launching the run")

    def reset(self, timeout: float = 7.0):
        self.check_focus()
        self.release()
        # Gamepad B did not return a mid-route car to the start in our probe.
        # Delete did, so use the verified full restart binding here.
        PressKey(DEL)
        try:
            time.sleep(0.12)
        finally:
            ReleaseKey(DEL)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.check_focus()
            if self.at_start(self.telemetry.latest()):
                time.sleep(0.25)
                return
        raise TimeoutError("Reset did not return the car to Aimap's start")

    def close(self):
        try:
            self.release()
        finally:
            self.telemetry.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
