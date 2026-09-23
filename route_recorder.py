"""Save the first complete human lap after an Aimap start/reset.

Read-only. Stores positions and a finish marker at approximately 20 Hz.
No controls, lap times, images, or whole-session data are recorded.
"""

import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time

from probe import read_packet

ROOT = Path(__file__).resolve().parent
ORIGINAL = Path(os.environ.get('TRACKMANIA_MAP_PATH',
                               Path.home() / 'Documents' / 'Trackmania' / 'Maps' /
                               'My Maps' / 'Aimap.Map.Gbx'))
FROZEN = ROOT / 'data' / 'Aimap.Map.Gbx'
EXPECTED_MAP_SHA256 = '1477f55a124739f26b9d7a9b16633e6ffd3eb93deebc02fa7c7ab0e2c8e1e6c3'
START = (400.0, 10.01, 368.0)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def position(frame):
    return frame['x'], frame['y'], frame['z']


def at_start(frame):
    return math.dist(position(frame), START) < 2 and abs(frame['speed']) < 1


def main():
    map_hash = file_hash(ORIGINAL)
    if map_hash != EXPECTED_MAP_SHA256:
        raise RuntimeError('Aimap differs from the benchmark map')
    if FROZEN.exists() and map_hash != file_hash(FROZEN):
        raise RuntimeError('The optional frozen copy differs from the installed map')

    samples = []
    attempt_start = None
    last_sample = 0.0
    last_position = None
    first_started = None
    deadline = time.monotonic() + 20 * 60
    print('Waiting for a start/reset on Aimap; then recording one complete lap.', flush=True)
    with socket.create_connection(('127.0.0.1', 9000), timeout=3) as sock:
        while time.monotonic() < deadline:
            frame = read_packet(sock, min(deadline, time.monotonic() + 5))
            now = time.monotonic()
            here = position(frame)
            if at_start(frame):
                was_away = last_position is not None and math.dist(last_position, START) > 5
                if attempt_start is None or was_away:
                    samples = []
                    attempt_start = now
                    last_sample = 0.0
                    first_started = datetime.now(timezone.utc)
                    print('Start detected; recording route.', flush=True)
            last_position = here
            if attempt_start is None:
                continue
            elapsed = now - attempt_start
            if elapsed - last_sample >= 0.05 or not samples or frame['finished'] > 0.5:
                samples.append((round(elapsed, 3), *here, int(frame['finished'] > 0.5)))
                last_sample = elapsed
            if frame['finished'] > 0.5 and math.dist(here, START) > 50:
                finished_at = datetime.now(timezone.utc)
                out = ROOT / 'data' / ('route-' + finished_at.strftime('%Y%m%dT%H%M%SZ'))
                out.mkdir(parents=True, exist_ok=False)
                with (out / 'positions.csv').open('w', newline='', encoding='utf-8') as fp:
                    writer = csv.writer(fp)
                    writer.writerow(('seconds_from_start', 'x', 'y', 'z', 'finished'))
                    writer.writerows(samples)
                metadata = dict(map_sha256=map_hash, attempt_started_utc=first_started.isoformat(),
                                finish_observed_utc=finished_at.isoformat(), samples=len(samples),
                                note='Position route only. The elapsed seconds are not an official lap time.')
                (out / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
                print(f'ROUTE COMPLETE: {len(samples)} samples in {out}', flush=True)
                return
    raise TimeoutError('No complete lap was captured within 20 minutes; no route file was written')


if __name__ == '__main__':
    main()
