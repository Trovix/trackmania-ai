"""Small Soft Actor-Critic learner for two continuous policy outputs."""

from copy import deepcopy
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


def mlp(input_size: int, output_size: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(),
                         nn.Linear(128, 128), nn.ReLU(),
                         nn.Linear(128, output_size))


class Actor(nn.Module):
    def __init__(self, observation_size: int, action_size: int):
        super().__init__()
        self.net = mlp(observation_size, action_size * 2)
        self.action_size = action_size

    def forward(self, state):
        mean, log_std = self.net(state).split(self.action_size, dim=-1)
        return mean, log_std.clamp(-5.0, 2.0)

    def sample(self, state):
        mean, log_std = self(state)
        distribution = Normal(mean, log_std.exp())
        raw = distribution.rsample()
        action = torch.tanh(raw)
        # Change-of-variables term for tanh-squashed Gaussian actions.
        log_probability = (distribution.log_prob(raw) -
                           torch.log(1.0 - action.square() + 1e-6)).sum(-1, keepdim=True)
        return action, log_probability

    def deterministic(self, state):
        return torch.tanh(self(state)[0])


class Critic(nn.Module):
    def __init__(self, observation_size: int, action_size: int):
        super().__init__()
        self.net = mlp(observation_size + action_size, 1)

    def forward(self, state, action):
        return self.net(torch.cat((state, action), dim=-1))


class SAC:
    def __init__(self, observation_size: int, action_size: int,
                 device: str = "cuda", seed: int = 1,
                 gamma: float = 0.995, tau: float = 0.005):
        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.actor = Actor(observation_size, action_size).to(self.device)
        self.critic1 = Critic(observation_size, action_size).to(self.device)
        self.critic2 = Critic(observation_size, action_size).to(self.device)
        self.target1 = deepcopy(self.critic1)
        self.target2 = deepcopy(self.critic2)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=3e-4)
        self.log_alpha = torch.tensor(math.log(0.01), device=self.device,
                                      requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.target_entropy = -float(action_size)
        self.gamma = gamma
        self.tau = tau
        self.updates = 0

    def act(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        state = torch.as_tensor(observation, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = (self.actor.deterministic(state) if deterministic
                      else self.actor.sample(state)[0])
        return action.squeeze(0).cpu().numpy()

    def update(self, replay, batch_size: int = 256) -> dict[str, float]:
        state, action, reward, next_state, terminated, _truncated = replay.sample(
            batch_size, self.device)
        with torch.no_grad():
            next_action, next_logp = self.actor.sample(next_state)
            next_q = torch.minimum(self.target1(next_state, next_action),
                                   self.target2(next_state, next_action))
            target = reward + self.gamma * (1.0 - terminated) * (
                next_q - self.log_alpha.exp() * next_logp)

        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)
        critic_loss = (nn.functional.mse_loss(q1, target) +
                       nn.functional.mse_loss(q2, target))
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        for critic in (self.critic1, self.critic2):
            critic.requires_grad_(False)
        chosen_action, logp = self.actor.sample(state)
        chosen_q = torch.minimum(self.critic1(state, chosen_action),
                                 self.critic2(state, chosen_action))
        actor_loss = (self.log_alpha.exp().detach() * logp - chosen_q).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        for critic in (self.critic1, self.critic2):
            critic.requires_grad_(True)

        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            for target_net, source_net in ((self.target1, self.critic1),
                                           (self.target2, self.critic2)):
                for target_param, source_param in zip(target_net.parameters(),
                                                      source_net.parameters()):
                    target_param.lerp_(source_param, self.tau)
        self.updates += 1
        metrics = dict(critic_loss=float(critic_loss.item()),
                       actor_loss=float(actor_loss.item()),
                       alpha=float(self.log_alpha.exp().item()),
                       mean_q=float(torch.minimum(q1, q2).mean().item()))
        if not all(np.isfinite(value) for value in metrics.values()):
            raise RuntimeError("Non-finite SAC update")
        return metrics

    def save(self, path: str | Path):
        torch.save(dict(actor=self.actor.state_dict(),
                        critic1=self.critic1.state_dict(), critic2=self.critic2.state_dict(),
                        target1=self.target1.state_dict(), target2=self.target2.state_dict(),
                        actor_optimizer=self.actor_optimizer.state_dict(),
                        critic_optimizer=self.critic_optimizer.state_dict(),
                        log_alpha=self.log_alpha.detach(),
                        alpha_optimizer=self.alpha_optimizer.state_dict(),
                        updates=self.updates, gamma=self.gamma, tau=self.tau,
                        observation_size=self.actor.net[0].in_features,
                        action_size=self.actor.action_size,
                        torch_rng=torch.get_rng_state(),
                        cuda_rng=(torch.cuda.get_rng_state_all()
                                  if self.device.type == "cuda" else [])), path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cuda") -> "SAC":
        saved = torch.load(path, map_location=device, weights_only=True)
        agent = cls(saved["observation_size"], saved["action_size"], device=device,
                    gamma=saved["gamma"], tau=saved["tau"])
        for name in ("actor", "critic1", "critic2", "target1", "target2"):
            getattr(agent, name).load_state_dict(saved[name])
        agent.actor_optimizer.load_state_dict(saved["actor_optimizer"])
        agent.critic_optimizer.load_state_dict(saved["critic_optimizer"])
        with torch.no_grad():
            agent.log_alpha.copy_(saved["log_alpha"])
        agent.alpha_optimizer.load_state_dict(saved["alpha_optimizer"])
        agent.updates = saved["updates"]
        torch.set_rng_state(saved["torch_rng"].cpu())
        if saved.get("cuda_rng") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
        return agent
