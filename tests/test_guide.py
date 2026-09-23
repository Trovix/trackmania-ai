"""Geometry checks for the privileged training-only route guide."""

import unittest

import numpy as np

from game_bridge import Sample
from guide import RouteGuide
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


if __name__ == "__main__":
    unittest.main()
