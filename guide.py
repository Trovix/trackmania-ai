"""Training-only route guide. Its position access is never used by evaluation."""

import math

import numpy as np

from route_progress import Route


class RouteGeometry:
    """Two route-relative features derived from live position and motion."""

    def __init__(self, route: Route, *, lookahead_seconds: float = 0.5):
        if lookahead_seconds <= 0:
            raise ValueError("lookahead must be positive")
        self.route = route
        self.lookahead_seconds = lookahead_seconds
        self.previous_position = None
        start = route.point_at(0.0)
        ahead = route.point_at(5.0)
        self.heading = self._unit(ahead[0] - start[0], ahead[1] - start[1])
        self.position = start
        self.progress = 0.0
        self.speed = 0.0

    @staticmethod
    def _unit(x: float, z: float) -> tuple[float, float]:
        norm = math.hypot(x, z)
        return (x / norm, z / norm) if norm > 1e-6 else (1.0, 0.0)

    def observe(self, sample, progress: float) -> None:
        x, z = sample.telemetry["x"], sample.telemetry["z"]
        self.speed = sample.telemetry["speed"]
        if self.previous_position is not None and self.speed > 2.0:
            dx, dz = x - self.previous_position[0], z - self.previous_position[1]
            if math.hypot(dx, dz) > 0.1:
                travel = self._unit(dx, dz)
                self.heading = self._unit(0.7 * self.heading[0] + 0.3 * travel[0],
                                          0.7 * self.heading[1] + 0.3 * travel[1])
        self.previous_position = (x, z)
        self.position = (x, z)
        self.progress = progress

    def features(self) -> tuple[float, float]:
        distance = max(12.0, min(45.0,
                                 8.0 + self.lookahead_seconds * self.speed))
        target = self.route.point_at(self.progress + distance)
        aim = self._unit(target[0] - self.position[0],
                         target[1] - self.position[1])
        cross = self.heading[0] * aim[1] - self.heading[1] * aim[0]
        dot = self.heading[0] * aim[0] + self.heading[1] * aim[1]
        target_angle = math.atan2(cross, dot)
        route_point = self.route.point_at(self.progress)
        before = self.route.point_at(self.progress - 2.0)
        after = self.route.point_at(self.progress + 2.0)
        tangent = self._unit(after[0] - before[0], after[1] - before[1])
        offset_x = self.position[0] - route_point[0]
        offset_z = self.position[1] - route_point[1]
        signed_offset = tangent[0] * offset_z - tangent[1] * offset_x
        return target_angle, signed_offset


class RouteGuide:
    """Steer toward a point ahead on the recorded route at a modest speed."""

    def __init__(self, route: Route, *, speed_cap: float = 35.0,
                 lookahead_seconds: float = 0.5, steering_gain: float = 2.0):
        if min(speed_cap, lookahead_seconds, steering_gain) <= 0:
            raise ValueError("guide settings must be positive")
        self.geometry = RouteGeometry(route, lookahead_seconds=lookahead_seconds)
        self.speed_cap = speed_cap
        self.steering_gain = steering_gain

    def observe(self, sample, progress: float) -> None:
        self.geometry.observe(sample, progress)

    def act(self, _observation) -> np.ndarray:
        target_angle, _ = self.geometry.features()
        # A guided moving-car trace at the opening bend showed positive gamepad
        # steering turning toward +z when travelling along +x.
        steer = float(np.clip(self.steering_gain * target_angle,
                              -0.9, 0.9))
        drive = -1.0 if self.geometry.speed > self.speed_cap else 1.0
        return np.array([steer, drive], dtype=np.float32)
