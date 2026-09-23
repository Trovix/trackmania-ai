"""Capture one exploratory attempt around the early route bottleneck."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np

from episode import EpisodeRunner, ExploratoryDriver, OBSERVATION_SIZE, ROUTE_CSV
from game_bridge import GameBridge
from route_progress import Route
from sac import SAC


class DeterministicActor:
    def __init__(self, checkpoint):
        self.agent = SAC.load(checkpoint)
        if self.agent.actor.net[0].in_features != OBSERVATION_SIZE:
            raise ValueError("Checkpoint observation size differs from current capture")

    def act(self, observation):
        return self.agent.act(observation, deterministic=True)


class DiagnosticObserver:
    def __init__(self, game, directory):
        self.game = game
        self.directory = directory
        self.rows = []
        self.thresholds = iter((0, 50, 100, 125, 150, 175, 190, 200, 225))
        self.next_threshold = next(self.thresholds, None)
        self.saved_off_route = False

    def __call__(self, step, action, sample, update, impact, scraping):
        data = sample.telemetry
        self.rows.append(dict(step=step, x=data["x"], z=data["z"],
                              speed=data["speed"], steer_action=float(action[0]),
                              drive_action=float(action[1]),
                              route_progress=update.current, best_progress=update.best,
                              lateral_error=update.lateral_error,
                              likely_impact=impact,
                              likely_wall_scrape=scraping,
                              accepted=update.accepted, reason=update.reason,
                              sample_timing_ms=" ".join(
                                  f"{1000 * value:.2f}" for value in
                                  self.game.last_sample_timing),
                              sleep_timing_ms=" ".join(
                                  f"{1000 * value:.2f}" for value in
                                  self.game.last_sleep_timing),
                              lidar=" ".join(str(int(value)) for value in sample.lidar)))
        if self.next_threshold is not None and update.best >= self.next_threshold:
            label = f"progress-{self.next_threshold:03d}"
            self.save_image(label)
            self.next_threshold = next(self.thresholds, None)
        if update.reason == "outside_corridor" and not self.saved_off_route:
            self.save_image("off-route")
            self.saved_off_route = True

    def save_image(self, label):
        frame = self.game.window.screenshot()[:, :, :3].copy()
        lidar = self.game.lidar
        values = lidar.lidar_20(frame)
        overlay = frame.copy()
        origin = (lidar.road_point[1], lidar.road_point[0])
        for xs, ys, value in zip(lidar.list_axis_x, lidar.list_axis_y, values):
            index = int(value)
            endpoint = (int(ys[index]), int(xs[index]))
            hit = bool(np.all(frame[xs[index], ys[index]] < lidar.black_threshold))
            cv2.line(overlay, origin, endpoint,
                     (0, 220, 0) if hit else (0, 160, 255), 2)
            cv2.circle(overlay, endpoint, 4, (0, 0, 255), -1)
        if not cv2.imwrite(str(self.directory / f"{label}.jpg"), overlay):
            raise RuntimeError("Failed to save diagnostic frame")

    def save_trace(self):
        with (self.directory / "trace.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=25.0)
    parser.add_argument("--checkpoint", type=Path,
                        help="drive with a saved actor instead of random exploration")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 60:
        parser.error("seconds must be in [1, 60]")
    route = Route.from_csv(ROUTE_CSV)
    directory = (Path(__file__).resolve().parent / "runs" /
                 datetime.now(timezone.utc).strftime("diagnose-%Y%m%dT%H%M%SZ"))
    directory.mkdir(parents=True, exist_ok=False)
    with GameBridge() as game:
        observer = DiagnosticObserver(game, directory)
        try:
            policy = (DeterministicActor(args.checkpoint) if args.checkpoint
                      else ExploratoryDriver(args.seed))
            result = EpisodeRunner(game, route).run(policy,
                                                     max_seconds=args.seconds,
                                                     observer=observer)
            (directory / "result.json").write_text(json.dumps(result.as_json(), indent=2),
                                                    encoding="utf-8")
            print(result.as_json(), flush=True)
            game.reset()
        finally:
            if observer.rows:
                observer.save_trace()
    print(directory)


if __name__ == "__main__":
    main()
