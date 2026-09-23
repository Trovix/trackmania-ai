"""The road view must remain available when the black-border rays disappear."""

import unittest

import numpy as np

from episode import (OBSERVATION_SIZE, ImpactDetector, Observations,
                     likely_wall_scrape)
from game_bridge import Sample, extract_road_view


class ObservationTests(unittest.TestCase):
    def test_fast_road_view_keeps_bright_lane_and_dark_edge(self):
        frame = np.full((800, 1200, 3), 25, dtype=np.uint8)
        frame[320:680, 600:] = 150
        view = extract_road_view(frame)
        self.assertEqual(view.shape, (8, 16))
        self.assertLess(float(view[:, :6].mean()), 40)
        self.assertGreater(float(view[:, 10:].mean()), 130)

    def test_dark_road_does_not_blank_observation(self):
        view = np.tile(np.linspace(20, 180, 16, dtype=np.uint8), (8, 1))
        sample = Sample({"speed": 30.0}, np.zeros(19, dtype=np.float32),
                        view, 0.0)
        encoded = Observations(1902.0, 1400).encode(sample, 100.0, (0.0, 1.0))
        self.assertEqual(encoded.shape, (OBSERVATION_SIZE,))
        self.assertTrue(np.isfinite(encoded).all())
        self.assertEqual(np.count_nonzero(encoded[2:192]), 0)
        self.assertGreater(np.ptp(encoded[192:-2]), 0.5)

    def test_likely_impact_requires_fast_loss_without_braking(self):
        view = np.zeros((8, 16), dtype=np.uint8)
        rays = np.zeros(19, dtype=np.float32)

        def frame(speed, stamp):
            return Sample({"speed": speed}, rays, view, stamp)

        detector = ImpactDetector()
        self.assertFalse(detector.update(frame(45, 0), frame(35, 0.05), -1, 18))
        self.assertFalse(detector.update(frame(45, 0), frame(35, 0.05), 1, 3))
        self.assertTrue(detector.update(frame(45, 0), frame(35, 0.05), 1, 18))
        self.assertFalse(detector.update(frame(35, 0.05), frame(25, 0.10), 1, 18))
        self.assertTrue(detector.update(frame(45, 1.00), frame(34, 1.05), 1, 18))

    def test_wall_scrape_needs_both_edge_and_dark_rays(self):
        view = np.zeros((8, 16), dtype=np.uint8)
        dark = Sample({"speed": 25.0}, np.zeros(19), view, 0.0)
        clear = Sample({"speed": 25.0}, np.full(19, 200), view, 0.0)
        self.assertTrue(likely_wall_scrape(dark, 18.0))
        self.assertFalse(likely_wall_scrape(dark, 5.0))
        self.assertFalse(likely_wall_scrape(clear, 18.0))



if __name__ == "__main__":
    unittest.main()
