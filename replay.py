"""Fixed-capacity CPU replay buffer for off-policy learning."""

from pathlib import Path

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, observation_size: int, action_size: int,
                 seed: int = 1):
        if min(capacity, observation_size, action_size) <= 0:
            raise ValueError("replay dimensions must be positive")
        self.capacity = capacity
        self.observation_size = observation_size
        self.action_size = action_size
        self.states = np.empty((capacity, observation_size), dtype=np.float32)
        self.actions = np.empty((capacity, action_size), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.next_states = np.empty((capacity, observation_size), dtype=np.float32)
        self.terminated = np.empty((capacity, 1), dtype=np.float32)
        self.truncated = np.empty((capacity, 1), dtype=np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def add(self, state, action, reward, next_state, terminated, truncated):
        state = np.asarray(state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        if state.shape != (self.observation_size,) or next_state.shape != state.shape:
            raise ValueError("wrong observation shape")
        if action.shape != (self.action_size,):
            raise ValueError("wrong action shape")
        if not (np.isfinite(state).all() and np.isfinite(next_state).all() and
                np.isfinite(action).all() and np.isfinite(reward)):
            raise ValueError("non-finite transition")
        if terminated and truncated:
            raise ValueError("a transition cannot be terminal and truncated")
        index = self.position
        self.states[index] = state
        self.actions[index] = action
        self.rewards[index, 0] = reward
        self.next_states[index] = next_state
        self.terminated[index, 0] = terminated
        self.truncated[index, 0] = truncated
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        if self.size < batch_size:
            raise ValueError("not enough transitions")
        indices = self.rng.integers(self.size, size=batch_size)
        return tuple(torch.as_tensor(values[indices], device=device)
                     for values in (self.states, self.actions, self.rewards,
                                    self.next_states, self.terminated,
                                    self.truncated))

    def save(self, path: str | Path):
        # Save only populated rows, not uninitialised capacity.
        np.savez_compressed(path, states=self.states[:self.size],
                            actions=self.actions[:self.size],
                            rewards=self.rewards[:self.size],
                            next_states=self.next_states[:self.size],
                            terminated=self.terminated[:self.size],
                            truncated=self.truncated[:self.size],
                            position=self.position, size=self.size,
                            capacity=self.capacity,
                            rng_state=np.array([self.rng.bit_generator.state], dtype=object))

    @classmethod
    def load(cls, path: str | Path) -> "ReplayBuffer":
        with np.load(path, allow_pickle=True) as saved:
            capacity = int(saved["capacity"])
            states = saved["states"]
            actions = saved["actions"]
            buffer = cls(capacity, states.shape[1], actions.shape[1])
            buffer.size = int(saved["size"])
            buffer.position = int(saved["position"])
            for name in ("states", "actions", "rewards", "next_states",
                         "terminated", "truncated"):
                getattr(buffer, name)[:buffer.size] = saved[name]
            buffer.rng.bit_generator.state = saved["rng_state"].item()
            return buffer
