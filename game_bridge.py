"""Small, local Trackmania bridge for the test-map experiment.

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

import cv2
import dxcam
import numpy as np
import vgamepad
import win32api
import win32con
import win32gui
from tmrl.custom.tm.utils.control_keyboard import DEL, PressKey, ReleaseKey
from tmrl.custom.tm.utils.tools import Lidar

from probe import FIELDS, PACKET


ROOT = Path(__file__).resolve().parent
MAP = Path(os.environ.get("TRACKMANIA_MAP_PATH",
                          Path.home() / "Documents" / "Trackmania" / "Maps" /
                          "My Maps" / "Aimap.Map.Gbx"))
FROZEN_MAP = ROOT / "data" / "Aimap.Map.Gbx"
EXPECTED_MAP_SHA256 = "1477f55a124739f26b9d7a9b16633e6ffd3eb93deebc02fa7c7ab0e2c8e1e6c3"
START = (400.0, 368.0)
CAPTURE_WIDTH = 960


class ScaledWindowCapture:
    """Capture fresh rendered frames from the primary display, then resize."""

    def __init__(self, hwnd: int):
        self.hwnd = hwnd
        rect = win32gui.GetClientRect(hwnd)
        self.source_size = (rect[2], rect[3])
        if min(self.source_size) < 100:
            raise RuntimeError("Game client area is too small to capture")
        width = min(CAPTURE_WIDTH, self.source_size[0])
        height = round(width * self.source_size[1] / self.source_size[0])
        self.output_size = (width, height)
        self.camera = dxcam.create(output_idx=0, output_color="BGR")

    def screenshot(self) -> np.ndarray:
        left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
        monitor = win32api.GetMonitorInfo(
            win32api.MonitorFromWindow(self.hwnd, win32con.MONITOR_DEFAULTTONEAREST))
        monitor_left, monitor_top, monitor_right, monitor_bottom = monitor["Monitor"]
        if (left < monitor_left or top < monitor_top or
                right > monitor_right or bottom > monitor_bottom):
            raise RuntimeError("Trackmania must fit entirely on one monitor "
                               "for reliable image capture")
        if (not monitor["Flags"] & 1 or
                (monitor_right - monitor_left, monitor_bottom - monitor_top) !=
                (self.camera.width, self.camera.height)):
            raise RuntimeError("Trackmania must be on the primary display")
        rect = win32gui.GetClientRect(self.hwnd)
        if (rect[2], rect[3]) != self.source_size:
            raise RuntimeError(f"Game window size changed from {self.source_size} "
                               f"to {(rect[2], rect[3])}")
        client_left, client_top = win32gui.ClientToScreen(self.hwnd, (0, 0))
        region = (client_left - monitor_left, client_top - monitor_top,
                  client_left - monitor_left + self.source_size[0],
                  client_top - monitor_top + self.source_size[1])
        deadline = time.monotonic() + 0.1
        frame = self.camera.grab(region=region, new_frame_only=True)
        while frame is None and time.monotonic() < deadline:
            time.sleep(0.002)
            frame = self.camera.grab(region=region, new_frame_only=True)
        if frame is None:
            raise TimeoutError("No newly rendered Trackmania frame")
        return cv2.resize(frame, self.output_size, interpolation=cv2.INTER_AREA)

    def close(self):
        self.camera.release()


def extract_road_view(frame: np.ndarray) -> np.ndarray:
    """Cheap, lightly averaged 16x8 view of the lower road scene."""
    height, width = frame.shape[:2]
    road = frame[int(height * 0.40):int(height * 0.85),
                 int(width * 0.05):int(width * 0.95)]
    coarse = cv2.resize(road, (64, 32), interpolation=cv2.INTER_LINEAR)
    small = cv2.resize(coarse, (16, 8), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


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
    road_view: np.ndarray
    timestamp: float


class GameBridge:
    def __init__(self):
        check_map()
        ctypes.windll.user32.SetProcessDPIAware()
        self.hwnd = win32gui.FindWindow(None, "Trackmania")
        if not self.hwnd:
            raise RuntimeError("Trackmania window not found")
        self.check_focus()
        self.window = ScaledWindowCapture(self.hwnd)
        try:
            self.telemetry = Telemetry()
            try:
                self.pad = vgamepad.VX360Gamepad()
                self.lidar = None
                self.frame_shape = None
                self.release()
            except BaseException:
                self.telemetry.close()
                raise
        except BaseException:
            self.window.close()
            raise

    def check_focus(self):
        if (win32gui.IsIconic(self.hwnd) or
                win32gui.GetForegroundWindow() != self.hwnd):
            raise RuntimeError("Trackmania lost foreground focus")

    def release(self):
        self.pad.reset()
        self.pad.update()

    def apply(self, steering: float, drive: float):
        """Steering is continuous; drive selects full gas or full brake.

        Half trigger did not register as gas in the earlier game probe, so the
        policy defaults to gas and brakes only on a strong negative request.
        """
        self.check_focus()
        if not all(math.isfinite(value) and -1 <= value <= 1
                   for value in (steering, drive)):
            raise ValueError("actions must be finite and within [-1, 1]")
        braking = drive < -0.5
        self.pad.right_trigger_float(0.0 if braking else 1.0)
        self.pad.left_trigger_float(1.0 if braking else 0.0)
        self.pad.left_joystick_float(steering, 0.0)
        self.pad.update()

    def sample(self) -> Sample:
        self.check_focus()
        started = time.perf_counter()
        frame = self.window.screenshot()
        captured = time.perf_counter()
        shape = frame.shape
        if self.frame_shape is None:
            self.frame_shape = shape
            self.lidar = Lidar(frame)
            if min(shape[:2]) < 100 or np.std(frame[::32, ::32]) < 2:
                raise RuntimeError("Game capture appears empty")
        elif shape != self.frame_shape:
            raise RuntimeError("Game window size changed during a run")
        rays = self.lidar.lidar_20(frame, show=False)
        ranged = time.perf_counter()
        road_view = extract_road_view(frame)
        viewed = time.perf_counter()
        data = self.telemetry.latest()
        received = time.perf_counter()
        self.last_sample_timing = (captured - started, ranged - captured,
                                   viewed - ranged, received - viewed)
        return Sample(data, rays, road_view, time.monotonic())

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
        data = self.telemetry.latest()
        if data["x"] == 0.0 and data["z"] == 0.0:
            raise RuntimeError("Trackmania left driving mode; return to the test-map start")
        raise TimeoutError("Reset did not return the car to the test-map start")

    def close(self):
        try:
            self.release()
        finally:
            try:
                self.telemetry.close()
            finally:
                self.window.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
