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
OBSERVATION_SIZE = 1 + 1 + 10 * 19 + 8 * 16 + 2
ACTION_SIZE = 2
STEP_SECONDS = 0.02


class Observations:
    """[speed, route progress, 10 recent LIDAR vectors, road view, previous action].

    Speed is divided by 300, pixel ray indices by game width, the 16x8
    grayscale road view by 255, and progress by route length. All values are
    float32. History is reset at every episode.
    """

    def __init__(self, route_length: float, game_width: int):
        self.route_length = route_length
        self.game_width = game_width
        self.rays = deque(maxlen=10)

    def encode(self, sample: Sample, progress: float, previous_action) -> np.ndarray:
        rays = np.clip(sample.lidar / self.game_width, 0.0, 1.0).astype(np.float32)
        if not self.rays:
            self.rays.extend([rays.copy() for _ in range(10)])
        else:
            self.rays.append(rays)
        road_view = np.asarray(sample.road_view, dtype=np.float32)
        if road_view.shape != (8, 16):
            raise RuntimeError("Bad road-view shape")
        result = np.concatenate((
            np.array([np.clip(sample.telemetry["speed"] / 300.0, 0.0, 1.0),
                      np.clip(progress / self.route_length, 0.0, 1.0)], dtype=np.float32),
            np.array(self.rays, dtype=np.float32).reshape(-1),
            (road_view / 255.0).reshape(-1),
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
            self.remaining = self.random.randint(12, 30)
        self.remaining -= 1
        return np.array([self.steer, 1.0], dtype=np.float32)


class ImpactDetector:
    """Conservative speed-loss proxy for a barrier impact, not a contact flag."""

    def __init__(self):
        self.last_impact = float("-inf")

    def update(self, before: Sample, after: Sample, drive: float,
               lateral_error: float | None) -> bool:
        gap = after.timestamp - before.timestamp
        if (not 0 < gap <= 0.2 or
                after.timestamp - self.last_impact < 0.75 or
                drive < -0.5 or
                lateral_error is None or lateral_error < 10.0 or
                before.telemetry["speed"] < 20.0 or
                before.telemetry["speed"] - after.telemetry["speed"] < 8.0):
            return False
        self.last_impact = after.timestamp
        return True


def likely_wall_scrape(sample: Sample, lateral_error: float | None) -> bool:
    """The car is near the recorded route edge on the dark shoulder."""
    return (lateral_error is not None and lateral_error > 15.0 and
            np.count_nonzero(sample.lidar <= 1.0) >= 16)


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
    likely_impacts: int
    impact_penalty: float
    wall_scrape_frames: int
    wall_scrape_penalty: float
    no_progress_penalty: float
    missed_deadlines: int

    def as_json(self):
        return asdict(self)


class EpisodeRunner:
    def __init__(self, game: GameBridge, route: Route):
        self.game = game
        self.route = route

    def run(self, policy, *, buffer=None, max_seconds: float = 100.0,
            stuck_seconds: float = 8.0, observer=None) -> EpisodeResult:
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
        likely_impacts = 0
        wall_scrape_frames = 0
        wall_scrape_cost = 0.0
        no_progress_cost = 0.0
        missed_deadlines = 0
        impact_detector = ImpactDetector()
        step_durations = []
        try:
            step = 0
            while True:
                now = time.monotonic()
                if now - tick > STEP_SECONDS:
                    missed_deadlines += int((now - tick) // STEP_SECONDS)
                    tick = now
                action = np.asarray(policy.act(state), dtype=np.float32)
                if action.shape != (ACTION_SIZE,) or not np.isfinite(action).all():
                    raise RuntimeError("Policy returned an invalid action")
                action = np.clip(action, -1.0, 1.0)
                self.game.apply(float(action[0]), float(action[1]))
                tick += STEP_SECONDS
                requested_sleep = max(0.0, tick - time.monotonic())
                sleep_start = time.perf_counter()
                time.sleep(requested_sleep)
                self.game.last_sleep_timing = (requested_sleep,
                                               time.perf_counter() - sleep_start)

                sample = self.game.sample()
                sample_interval = max(0.0, min(sample.timestamp - initial.timestamp, 0.2))
                update = tracker.update(sample.telemetry["x"], sample.telemetry["z"])
                impact = impact_detector.update(initial, sample, float(action[1]),
                                                update.lateral_error)
                scraping = likely_wall_scrape(sample, update.lateral_error)
                likely_impacts += int(impact)
                wall_scrape_frames += int(scraping)
                if observer is not None:
                    observer(step, action, sample, update, impact, scraping)
                if not update.accepted and update.reason != "outside_corridor":
                    raise RuntimeError(f"Invalid route projection: {update.reason}")
                next_state = encoder.encode(sample, update.current, action)
                elapsed = time.monotonic() - start
                finished = sample.telemetry["finished"] > 0.5
                no_progress = elapsed > 4.0 and update.gained < 0.05
                scrape_cost = 0.4 * sample_interval if scraping else 0.0
                idle_cost = 0.1 * sample_interval if no_progress else 0.0
                wall_scrape_cost += scrape_cost
                no_progress_cost += idle_cost
                if update.gained > 0.05:
                    last_progress_time = time.monotonic()
                max_speed = max(max_speed, sample.telemetry["speed"])
                reward = (update.reward + (10.0 if finished else 0.0) -
                          (0.5 if impact else 0.0) - scrape_cost - idle_cost)
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
                             round(float(np.percentile(step_durations, 95)), 4),
                             likely_impacts, round(-0.5 * likely_impacts, 3),
                             wall_scrape_frames, round(-wall_scrape_cost, 3),
                             round(-no_progress_cost, 3), missed_deadlines)


def save_episode(result: EpisodeResult, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as file:
        file.write(json.dumps(result.as_json()) + "\n")
