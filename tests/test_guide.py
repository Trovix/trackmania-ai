"""Geometry checks for the privileged training-only route guide."""

import unittest

import numpy as np

from diagnose import SpeedGovernedActor
from game_bridge import Sample
from guide import RouteGeometry, RouteGuide
from route_progress import Route


class GuideTests(unittest.TestCase):
    def test_aiming_toward_positive_z_uses_positive_steering(self):
        route = Route(((0.0, 0.0), (10.0, 0.0), (20.0, 10.0)),
                      (0.0, 10.0, 24.1421356))
        guide = RouteGuide(route, speed_cap=35.0)
        sample = Sample({"x": 7.0, "z": 0.0, "speed": 20.0},
                        np.zeros(19), np.zeros((8, 16)), 0.0)
        guide.observe(sample, 7.0)
        action = guide.act(None)
        self.assertGreater(action[0], 0.0)
        self.assertEqual(action[1], 1.0)
        guide.observe(Sample({"x": 7.5, "z": 0.0, "speed": 40.0},
                             np.zeros(19), np.zeros((8, 16)), 0.02), 7.5)
        self.assertEqual(guide.act(None)[1], -1.0)

    def test_route_point_clamps_to_finish(self):
        route = Route(((0.0, 0.0), (10.0, 0.0)), (0.0, 10.0))
        self.assertEqual(route.point_at(20.0), (10.0, 0.0))

    def test_route_features_expose_turn_and_signed_offset(self):
        route = Route(((0.0, 0.0), (10.0, 0.0), (20.0, 10.0)),
                      (0.0, 10.0, 24.1421356))
        geometry = RouteGeometry(route)
        geometry.observe(Sample({"x": 7.0, "z": 2.0, "speed": 20.0},
                                np.zeros(19), np.zeros((8, 16)), 0.0), 7.0)
        angle, offset = geometry.features()
        self.assertGreater(angle, 0.0)
        self.assertGreater(offset, 0.0)

    def test_speed_zone_only_applies_near_the_bend(self):
        class FakeActor:
            def act(self, _observation):
                return np.array([0.3, 1.0], dtype=np.float32)

        policy = SpeedGovernedActor(FakeActor(), 50.0, 1900.0,
                                    [(1400.0, 1660.0, 40.0)])
        state = np.zeros(324, dtype=np.float32)
        state[0] = 0.45
        state[1] = 1500 / 1900
        self.assertAlmostEqual(float(policy.act(state)[0]), 0.3)
        self.assertEqual(policy.act(state)[1], -1.0)
        state[1] = 1200 / 1900
        self.assertEqual(policy.act(state)[1], 1.0)


if __name__ == "__main__":
    unittest.main()
