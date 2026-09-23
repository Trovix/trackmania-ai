"""One real-time driving attempt and its observation contract."""

from collections import deque
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import random
import time

import numpy as np

from game_bridge import GameBridge, Sample
from route_progress import ProgressTracker, Route


ROOT = Path(__file__).resolve().parent
ROUTE_CSV = ROOT / "data" / "route-20260923T131307Z" / "positions.csv"
OBSERVATION_SIZE = 1 + 1 + 4 * 19 + 2
ACTION_SIZE = 2
STEP_SECONDS = 0.05


class Observations:
    """[speed, route progress, 4 recent LIDAR vectors, previous action].

    Speed is divided by 300, pixel ray indices by game width, and progress by
    route length. All values are float32. History is reset at every episode.
    """

    def __init__(self, route_length: float, game_width: int):
        self.route_length = route_length
        self.game_width = game_width
        self.rays = deque(maxlen=4)

    def encode(self, sample: Sample, progress: float, previous_action) -> np.ndarray:
        rays = np.clip(sample.lidar / self.game_width, 0.0, 1.0).astype(np.float32)
        if not self.rays:
            self.rays.extend([rays.copy() for _ in range(4)])
        else:
            self.rays.append(rays)
        result = np.concatenate((
            np.array([np.clip(sample.telemetry["speed"] / 300.0, 0.0, 1.0),
                      np.clip(progress / self.route_length, 0.0, 1.0)], dtype=np.float32),
            np.array(self.rays, dtype=np.float32).reshape(-1),
            np.asarray(previous_action, dtype=np.float32),
        ))
        if result.shape != (OBSERVATION_SIZE,) or not np.isfinite(result).all():
            raise RuntimeError("Bad observation shape or value")
        return result


class ExploratoryDriver:
    """Full throttle with slowly changing random steering; not a learned AI."""

    def __init__(self, seed: int = 1):
        self.random = random.Random(seed)
        self.remaining = 0
        self.steer = 0.0

    def act(self, _observation: np.ndarray) -> np.ndarray:
        if self.remaining <= 0:
            self.steer = self.random.uniform(-0.45, 0.45)
            self.remaining = self.random.randint(5, 12)
        self.remaining -= 1
        return np.array([self.steer, 1.0], dtype=np.float32)


@dataclass
class EpisodeResult:
    steps: int
    seconds: float
    progress: float
    progress_fraction: float
    reward: float
    max_speed: float
    finished: bool
    reason: str
    median_step_seconds: float
    p95_step_seconds: float

    def as_json(self):
        return asdict(self)


class EpisodeRunner:
    def __init__(self, game: GameBridge, route: Route):
        self.game = game
        self.route = route

    def run(self, policy, *, buffer=None, max_seconds: float = 100.0,
            stuck_seconds: float = 8.0) -> EpisodeResult:
        self.game.ensure_start()
        tracker = ProgressTracker(self.route)
        initial = self.game.sample()
        first = tracker.update(initial.telemetry["x"], initial.telemetry["z"])
        if not first.accepted:
            raise RuntimeError(f"Invalid initial route position: {first.reason}")
        encoder = Observations(self.route.length, self.game.frame_shape[1])
        state = encoder.encode(initial, first.current, (0.0, 0.0))
        start = time.monotonic()
        last_progress_time = start
        tick = start
        max_speed = 0.0
        reward_total = 0.0
        step_durations = []
        try:
            step = 0
            while True:
                action = np.asarray(policy.act(state), dtype=np.float32)
                if action.shape != (ACTION_SIZE,) or not np.isfinite(action).all():
                    raise RuntimeError("Policy returned an invalid action")
                action = np.clip(action, -1.0, 1.0)
                self.game.apply(float(action[0]), float(action[1]))
                tick += STEP_SECONDS
                time.sleep(max(0.0, tick - time.monotonic()))

                sample = self.game.sample()
                update = tracker.update(sample.telemetry["x"], sample.telemetry["z"])
                if not update.accepted and update.reason != "outside_corridor":
                    raise RuntimeError(f"Invalid route projection: {update.reason}")
                next_state = encoder.encode(sample, update.current, action)
                elapsed = time.monotonic() - start
                finished = sample.telemetry["finished"] > 0.5
                if update.gained > 0.05:
                    last_progress_time = time.monotonic()
                max_speed = max(max_speed, sample.telemetry["speed"])
                reward = update.reward + (10.0 if finished else 0.0)
                if finished:
                    reason = "finish"
                elif update.reason == "outside_corridor":
                    reason = "off_route"
                elif elapsed > 4.0 and time.monotonic() - last_progress_time > stuck_seconds:
                    reason = "stuck"
                elif elapsed >= max_seconds:
                    reason = "time_limit"
                else:
                    reason = "running"
                terminated = reason in ("finish", "off_route", "stuck")
                truncated = reason == "time_limit"
                if buffer is not None:
                    buffer.add(state, action, reward, next_state, terminated, truncated)
                state = next_state
                step += 1
                reward_total += reward
                step_durations.append(sample.timestamp - initial.timestamp)
                initial = sample
                if reason != "running":
                    break
        finally:
            self.game.release()

        return EpisodeResult(step, round(elapsed, 3), round(tracker.best, 3),
                             round(tracker.best / self.route.length, 4),
                             round(reward_total, 3), round(max_speed, 3),
                             finished, reason,
                             round(float(np.median(step_durations)), 4),
                             round(float(np.percentile(step_durations, 95)), 4))


def save_episode(result: EpisodeResult, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as file:
        file.write(json.dumps(result.as_json()) + "\n")
