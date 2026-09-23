"""Small correctness gates before the real-time trainer uses these modules."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from replay import ReplayBuffer
from sac import SAC


class LearningTests(unittest.TestCase):
    def test_replay_wraparound_and_flags(self):
        replay = ReplayBuffer(3, 4, 2)
        for i in range(5):
            replay.add(np.full(4, i), np.zeros(2), float(i),
                       np.full(4, i + 1), i == 4, i == 3)
        self.assertEqual(len(replay), 3)
        self.assertEqual(replay.position, 2)
        self.assertEqual(set(replay.rewards[:3, 0]), {2.0, 3.0, 4.0})
        batch = replay.sample(3, torch.device("cpu"))
        self.assertEqual(batch[0].shape, (3, 4))
        self.assertEqual(batch[1].shape, (3, 2))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.npz"
            replay.save(path)
            loaded = ReplayBuffer.load(path)
            self.assertEqual(loaded.position, replay.position)
            np.testing.assert_array_equal(loaded.states[:3], replay.states[:3])

    def test_sac_update_and_checkpoint(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(3)
        replay = ReplayBuffer(256, 4, 2)
        for i in range(256):
            replay.add(rng.normal(size=4), rng.uniform(-1, 1, size=2),
                       float(rng.normal()), rng.normal(size=4),
                       i % 11 == 0, i % 17 == 0 and i % 11 != 0)
        agent = SAC(4, 2, device="cpu")
        self.assertTrue(np.all(np.abs(agent.act(np.zeros(4, dtype=np.float32))) <= 1))
        metrics = agent.update(replay, 64)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        self.assertEqual(agent.updates, 1)
        observation = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float32)
        before = agent.act(observation, deterministic=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            agent.save(path)
            loaded = SAC.load(path, device="cpu")
            np.testing.assert_allclose(loaded.act(observation, deterministic=True),
                                       before, atol=1e-7)
            self.assertEqual(loaded.updates, agent.updates)


if __name__ == "__main__":
    unittest.main()
