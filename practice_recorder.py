"""Record one bounded human-practice session without sending game inputs.

--smoke runs for three seconds and is never a benchmark. --practice starts the
30-minute clock at process startup. Official lap times must be read in game.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time

from probe import FIELDS, read_packet

ROOT = Path(__file__).resolve().parent
MAP = Path(os.environ.get('TRACKMANIA_MAP_PATH',
                          Path.home() / 'Documents' / 'Trackmania' / 'Maps' /
                          'My Maps' / 'Aimap.Map.Gbx'))
FROZEN = ROOT / 'data' / 'Aimap.Map.Gbx'
EXPECTED_MAP_SHA256 = '1477f55a124739f26b9d7a9b16633e6ffd3eb93deebc02fa7c7ab0e2c8e1e6c3'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def capture_image(directory, index):
    """Save the actual detector input and its 19 distances when the game is visible."""
    import cv2
    import numpy as np
    import win32gui
    from tmrl.custom.tm.utils.window import WindowInterface
    from tmrl.custom.tm.utils.tools import Lidar

    hwnd = win32gui.FindWindow(None, 'Trackmania')
    if not hwnd or win32gui.IsIconic(hwnd) or win32gui.GetForegroundWindow() != hwnd:
        return None
    image = WindowInterface('Trackmania').screenshot()[:, :, :3].copy()
    if min(image.shape[:2]) < 100 or np.std(image) < 2:
        return None
    lidar = Lidar(image)
    distances = lidar.lidar_20(image)
    filename = f'frame-{index:04d}.jpg'
    if not cv2.imwrite(str(directory / filename), image, [cv2.IMWRITE_JPEG_QUALITY, 75]):
        raise OSError(f'Could not save {filename}')
    return filename, image.shape[1], image.shape[0], *[float(v) for v in distances]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke', action='store_true', help='3-second check, not practice')
    mode.add_argument('--practice', action='store_true', help='start the 30-minute practice clock')
    args = parser.parse_args()
    seconds = 3 if args.smoke else 1800
    map_hash = sha256(MAP)
    if map_hash != EXPECTED_MAP_SHA256:
        raise RuntimeError('Map differs from benchmark map; resolve before practice')
    if FROZEN.exists() and map_hash != sha256(FROZEN):
        raise RuntimeError('The optional frozen copy differs from the installed map')

    # Finish all imports and connect before starting the practice clock.
    import cv2  # noqa: F401
    import win32gui  # noqa: F401
    from tmrl.custom.tm.utils.window import WindowInterface  # noqa: F401
    with socket.create_connection(('127.0.0.1', 9000), timeout=3) as sock:
        try:
            initial = read_packet(sock, time.monotonic() + 5)
        except TimeoutError as exc:
            raise RuntimeError('No live car telemetry. Open Aimap in driving mode before starting.') from exc
        if args.practice:
            if initial['finished'] > 0.5 or abs(initial['speed']) > 1:
                raise RuntimeError('Reset and leave the car stationary at the Aimap start before practice.')
            start_error = math.dist((initial['x'], initial['y'], initial['z']),
                                    (400.0, 10.01, 368.0))
            if start_error > 2:
                raise RuntimeError(f'Car is {start_error:.1f} units from the frozen map start; reset first.')
        start = time.monotonic()
        stop = start + seconds
        start_utc = datetime.now(timezone.utc)
        name = ('smoke-' if args.smoke else 'practice-') + start_utc.strftime('%Y%m%dT%H%M%SZ')
        directory = ROOT / 'runs' / name
        directory.mkdir(parents=True, exist_ok=False)
        metadata = dict(mode='smoke' if args.smoke else 'practice',
                        start_utc=start_utc.isoformat(), duration_limit_seconds=seconds,
                        map_sha256=map_hash, telemetry_fields=FIELDS,
                        official_times='Read from Trackmania; telemetry has no official timer',
                        image_interval_seconds=5)
        (directory / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print(f'RECORDING STARTED {start_utc.isoformat()} ({seconds} seconds)', flush=True)
        print(f'Output: {directory}', flush=True)
        frames = finishes = images = 0
        last_finished = False
        next_image = start
        error = None
        try:
            with (directory / 'telemetry.csv').open('w', newline='', encoding='utf-8') as fp, \
                 (directory / 'vision.csv').open('w', newline='', encoding='utf-8') as vp:
                telemetry = csv.writer(fp)
                vision = csv.writer(vp)
                telemetry.writerow(('elapsed_seconds',) + FIELDS)
                vision.writerow(('elapsed_seconds', 'image', 'width', 'height') +
                                tuple(f'beam_{i:02d}' for i in range(19)))
                while time.monotonic() < stop:
                    try:
                        packet = read_packet(sock, min(stop, time.monotonic() + 5))
                    except TimeoutError:
                        if time.monotonic() < stop - 0.1:
                            raise RuntimeError('Telemetry stopped for five seconds')
                        break
                    elapsed = time.monotonic() - start
                    telemetry.writerow((f'{elapsed:.4f}',) + tuple(packet[k] for k in FIELDS))
                    frames += 1
                    finished = packet['finished'] > 0.5
                    just_finished = finished and not last_finished
                    if just_finished:
                        finishes += 1
                        print(f'FINISH FLAG {finishes} at recording +{elapsed:.1f}s', flush=True)
                    last_finished = finished
                    if elapsed >= next_image - start or just_finished:
                        try:
                            result = capture_image(directory, images)
                            if result:
                                vision.writerow((f'{elapsed:.4f}',) + result)
                                images += 1
                        except Exception as exc:
                            print(f'Image skipped at +{elapsed:.1f}s: {exc}', flush=True)
                        next_image = time.monotonic() + 5
                    if frames % 1000 == 0:
                        fp.flush()
                        vp.flush()
        except Exception as exc:
            error = str(exc)
            print(f'RECORDER ERROR: {error}', flush=True)
            raise
        finally:
            end_utc = datetime.now(timezone.utc)
            result = dict(**metadata, end_utc=end_utc.isoformat(), frames=frames,
                          finish_flag_rises=finishes, images=images, error=error,
                          actual_elapsed_seconds=time.monotonic() - start)
            (directory / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
            print(f'RECORDING ENDED: {frames} frames, {finishes} finishes, {images} images', flush=True)
            print(directory / 'result.json', flush=True)


if __name__ == '__main__':
    main()
