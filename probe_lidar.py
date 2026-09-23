"""Capture one TMRL observation and save a diagnostic overlay; no controls."""
import ctypes
ctypes.windll.user32.SetProcessDPIAware()
import json
from pathlib import Path
import time
import cv2
import numpy as np
import win32gui
from tmrl.custom.tm.utils.window import WindowInterface
from tmrl.custom.tm.utils.tools import Lidar

hwnd = win32gui.FindWindow(None, 'Trackmania')
if not hwnd or win32gui.IsIconic(hwnd):
    raise RuntimeError('Trackmania must be open and not minimised')
if win32gui.GetForegroundWindow() != hwnd:
    raise RuntimeError('Leave Trackmania in the foreground before capture')
frame = WindowInterface('Trackmania').screenshot()[:, :, :3].copy()
if min(frame.shape[:2]) < 100 or np.std(frame) < 2:
    raise RuntimeError('Capture is empty or nearly uniform')
lidar = Lidar(frame)
distances = lidar.lidar_20(frame)
overlay = frame.copy()
hits = []
origin = (lidar.road_point[1], lidar.road_point[0])
for xs, ys, value in zip(lidar.list_axis_x, lidar.list_axis_y, distances):
    i = int(value)
    endpoint = (int(ys[i]), int(xs[i]))
    hit = bool(np.all(frame[xs[i], ys[i]] < lidar.black_threshold))
    hits.append(hit)
    color = (0, 220, 0) if hit else (0, 160, 255)
    cv2.line(overlay, origin, endpoint, color, 2)
    cv2.circle(overlay, endpoint, 4, (0, 0, 255), -1)
out = Path(__file__).resolve().parent / 'runs' / f'lidar-{time.time_ns()}'
out.mkdir(parents=True)
assert cv2.imwrite(str(out / 'frame.png'), frame)
assert cv2.imwrite(str(out / 'overlay.png'), overlay)
result = dict(width=frame.shape[1], height=frame.shape[0],
              black_threshold=np.asarray(lidar.black_threshold).tolist(),
              distances=distances.tolist(), dark_pixel_hits=hits,
              note='Pixel sample indices, not metres. Orange rays reached image edge.')
(out / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result))
print(out / 'overlay.png')
