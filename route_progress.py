"""Offline route progress for Aimap; no game connection or learner code.

The recorded route supplies positions, not driving actions. Progress is measured
along its horizontal (x, z) path. Only new progress within an episode earns a
reward; this module never decides whether a lap has finished.
"""

from bisect import bisect_left, bisect_right
import csv
from dataclasses import dataclass
from math import hypot, isfinite
from pathlib import Path


@dataclass(frozen=True)
class Route:
    points: tuple[tuple[float, float], ...]
    distances: tuple[float, ...]

    @property
    def length(self) -> float:
        return self.distances[-1]

    @classmethod
    def from_csv(cls, path: str | Path, spacing: float = 2.0) -> "Route":
        """Discard stationary samples and interpolate points at even distances."""
        if not isfinite(spacing) or spacing <= 0:
            raise ValueError("spacing must be positive and finite")

        with Path(path).open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if not rows or rows[-1]["finished"] != "1":
            raise ValueError("route must end with the game's finish flag")

        raw: list[tuple[float, float]] = []
        cumulative = [0.0]
        for row in rows:
            point = (float(row["x"]), float(row["z"]))
            if not all(isfinite(value) for value in point):
                raise ValueError("route contains a non-finite position")
            if not raw:
                raw.append(point)
                continue
            step = hypot(point[0] - raw[-1][0], point[1] - raw[-1][1])
            if step < 0.01:  # stationary telemetry/noise, including the wait at start
                continue
            raw.append(point)
            cumulative.append(cumulative[-1] + step)

        if len(raw) < 2:
            raise ValueError("route contains no movement")

        length = cumulative[-1]
        distances = [i * spacing for i in range(int(length // spacing) + 1)]
        if length - distances[-1] > 1e-6:
            distances.append(length)
        else:
            distances[-1] = length

        points = []
        for distance in distances:
            segment = min(bisect_right(cumulative, distance) - 1, len(raw) - 2)
            fraction = ((distance - cumulative[segment]) /
                        (cumulative[segment + 1] - cumulative[segment]))
            start, end = raw[segment], raw[segment + 1]
            points.append((start[0] + fraction * (end[0] - start[0]),
                           start[1] + fraction * (end[1] - start[1])))
        return cls(tuple(points), tuple(distances))

    def project_near(self, x: float, z: float, current: float,
                     search_window: float) -> tuple[float, float]:
        """Return (distance along route, distance from route) near current."""
        first = max(0, bisect_left(self.distances, current - search_window) - 1)
        last = min(len(self.points) - 2,
                   bisect_right(self.distances, current + search_window))
        nearest_squared = float("inf")
        nearest_progress = current
        for index in range(first, last + 1):
            ax, az = self.points[index]
            bx, bz = self.points[index + 1]
            dx, dz = bx - ax, bz - az
            segment_squared = dx * dx + dz * dz
            fraction = max(0.0, min(1.0,
                                    ((x - ax) * dx + (z - az) * dz) /
                                    segment_squared))
            projected_x, projected_z = ax + fraction * dx, az + fraction * dz
            error_squared = (x - projected_x) ** 2 + (z - projected_z) ** 2
            if error_squared < nearest_squared:
                nearest_squared = error_squared
                nearest_progress = (self.distances[index] + fraction *
                                    (self.distances[index + 1] - self.distances[index]))
        return nearest_progress, nearest_squared ** 0.5


@dataclass(frozen=True)
class ProgressUpdate:
    current: float
    best: float
    gained: float
    reward: float
    lateral_error: float | None
    accepted: bool
    reason: str | None = None


class ProgressTracker:
    """Track one attempt. Call reset() before each new attempt.

    Limits are provisional for Aimap and must be checked against live driving.
    A rejected jump requires the caller to investigate/reset the episode.
    """

    def __init__(self, route: Route, *, search_window: float = 25.0,
                 corridor: float = 30.0, max_step: float = 12.0):
        if not all(isfinite(v) and v > 0 for v in
                   (search_window, corridor, max_step)):
            raise ValueError("limits must be positive and finite")
        self.route = route
        self.search_window = search_window
        self.corridor = corridor
        self.max_step = max_step
        self.reset()

    def reset(self) -> None:
        self.current = 0.0
        self.best = 0.0
        self.last_position = self.route.points[0]

    def update(self, x: float, z: float) -> ProgressUpdate:
        if not isfinite(x) or not isfinite(z):
            raise ValueError("position must be finite")
        movement = hypot(x - self.last_position[0], z - self.last_position[1])
        if movement > self.max_step:
            return ProgressUpdate(self.current, self.best, 0.0, 0.0,
                                  None, False, "position_jump")

        progress, error = self.route.project_near(x, z, self.current,
                                                   self.search_window)
        if error > self.corridor:
            return ProgressUpdate(self.current, self.best, 0.0, 0.0,
                                  error, False, "outside_corridor")
        if abs(progress - self.current) > 2.0 * movement + 2.0:
            return ProgressUpdate(self.current, self.best, 0.0, 0.0,
                                  error, False, "implausible_progress")

        self.last_position = (x, z)
        self.current = progress
        gained = max(0.0, progress - self.best)
        self.best = max(self.best, progress)
        return ProgressUpdate(self.current, self.best, gained,
                              10.0 * gained / self.route.length,
                              error, True)


if __name__ == "__main__":
    path = (Path(__file__).resolve().parent / "data" /
            "route-20260923T131307Z" / "positions.csv")
    route = Route.from_csv(path)
    tracker = ProgressTracker(route)
    with path.open(newline="", encoding="utf-8") as file:
        updates = [tracker.update(float(row["x"]), float(row["z"]))
                   for row in csv.DictReader(file)]
    print(f"Route: {len(route.points)} points, {route.length:.2f} distance units")
    print(f"Recorded lap: {sum(u.accepted for u in updates)}/{len(updates)} accepted, "
          f"{updates[-1].best:.2f} best progress, "
          f"{sum(u.reward for u in updates):.2f} progress reward")
