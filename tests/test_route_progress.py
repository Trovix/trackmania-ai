"""Checks for the map-specific progress signal using the frozen route."""

from bisect import bisect_right
import csv
from pathlib import Path
import unittest

from route_progress import ProgressTracker, Route


ROUTE_CSV = (Path(__file__).resolve().parents[1] / "data" /
             "route-20260923T131307Z" / "positions.csv")


def recorded_positions():
    with ROUTE_CSV.open(newline="", encoding="utf-8") as file:
        return [(float(row["x"]), float(row["z"]))
                for row in csv.DictReader(file)]


class RouteProgressTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.route = Route.from_csv(ROUTE_CSV)
        cls.recording = recorded_positions()

    def test_recorded_lap_reaches_end_once(self):
        tracker = ProgressTracker(self.route)
        updates = [tracker.update(*point) for point in self.recording]
        self.assertTrue(all(update.accepted for update in updates))
        self.assertAlmostEqual(updates[-1].best, self.route.length, places=5)
        self.assertAlmostEqual(sum(update.reward for update in updates), 10.0)
        self.assertAlmostEqual(sum(update.gained for update in updates),
                               self.route.length)

    def test_stationary_and_reversing_do_not_earn_repeat_reward(self):
        tracker = ProgressTracker(self.route)
        updates = [tracker.update(*point) for point in self.recording[:600]]
        self.assertTrue(all(update.gained == 0 for update in updates[:20]))
        previous_best = tracker.best

        floor = bisect_right(self.route.distances, tracker.current) - 1
        reverse_and_return = (list(reversed(self.route.points[floor - 4:floor])) +
                              list(self.route.points[floor - 4:floor + 1]))
        repeated = [tracker.update(*point) for point in reverse_and_return]
        self.assertTrue(all(update.accepted for update in repeated))
        self.assertTrue(all(update.gained == 0 for update in repeated))
        self.assertAlmostEqual(tracker.best, previous_best)

    def test_far_jump_is_rejected_without_reward(self):
        tracker = ProgressTracker(self.route)
        for point in self.recording[:640]:
            self.assertTrue(tracker.update(*point).accepted)
        previous_best = tracker.best
        jump = tracker.update(*self.recording[846])
        self.assertFalse(jump.accepted)
        self.assertEqual(jump.reason, "position_jump")
        self.assertEqual(jump.gained, 0)
        self.assertEqual(tracker.best, previous_best)

    def test_nearby_parallel_section_cannot_skip_a_turn(self):
        # The last point lies 1 unit from the start, but 21 route units ahead.
        hairpin = Route(((0.0, 0.0), (10.0, 0.0), (10.0, 1.0), (0.0, 1.0)),
                        (0.0, 10.0, 11.0, 21.0))
        tracker = ProgressTracker(hairpin)
        jump = tracker.update(0.0, 1.0)
        self.assertFalse(jump.accepted)
        self.assertEqual(jump.reason, "implausible_progress")
        self.assertEqual(jump.reward, 0)

    def test_reset_starts_new_attempt_at_zero(self):
        tracker = ProgressTracker(self.route)
        for point in self.recording[:600]:
            tracker.update(*point)
        self.assertGreater(tracker.best, 0)
        tracker.reset()
        self.assertEqual(tracker.best, 0)
        self.assertEqual(tracker.current, 0)
        self.assertTrue(tracker.update(*self.recording[0]).accepted)


if __name__ == "__main__":
    unittest.main()
