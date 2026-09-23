"""Imitation must not react to stadium graphics it has only seen once."""

import unittest

import numpy as np
import torch

from bootstrap import imitate
from episode import ACTION_SIZE, OBSERVATION_SIZE
from replay import ReplayBuffer
from sac import SAC


class BootstrapTests(unittest.TestCase):
    def test_initial_actor_ignores_visual_channels(self):
        replay = ReplayBuffer(32, OBSERVATION_SIZE, ACTION_SIZE)
        rng = np.random.default_rng(1)
        for _ in range(32):
            state = rng.uniform(0, 1, OBSERVATION_SIZE).astype(np.float32)
            action = np.array([state[2], 1.0], dtype=np.float32)
            replay.add(state, action, 0.0, state, False, False)
        agent = SAC(OBSERVATION_SIZE, ACTION_SIZE, device="cpu")
        imitate(agent, replay, updates=3, batch_size=16, seed=1)
        first = replay.states[0].copy()
        changed = first.copy()
        changed[4:-2] = 1.0 - changed[4:-2]
        with torch.no_grad():
            predicted = agent.actor.deterministic(
                torch.as_tensor(np.stack((first, changed))))
        self.assertTrue(torch.allclose(predicted[0], predicted[1]))


if __name__ == "__main__":
    unittest.main()
