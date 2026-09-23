"""Collect clean guided laps, imitate them, and test the unguided actor."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from episode import ACTION_SIZE, OBSERVATION_SIZE, EpisodeRunner, ROUTE_CSV, STEP_SECONDS
from game_bridge import GameBridge, check_map
from guide import RouteGuide
from replay import ReplayBuffer
from route_progress import Route
from sac import SAC
from train import REWARD_VERSION


ROOT = Path(__file__).resolve().parent


def copy_transitions(source: ReplayBuffer, target: ReplayBuffer) -> None:
    for index in range(len(source)):
        target.add(source.states[index], source.actions[index],
                   float(source.rewards[index, 0]), source.next_states[index],
                   bool(source.terminated[index, 0]),
                   bool(source.truncated[index, 0]))


def imitate(agent: SAC, replay: ReplayBuffer, updates: int, batch_size: int,
            seed: int) -> dict[str, float]:
    """Fit route-geometry controls first; SAC may learn visual weights later."""
    rng = np.random.default_rng(seed)
    agent.actor.train()
    visual_slice = slice(4, -2)
    with torch.no_grad():
        agent.actor.net[0].weight[:, visual_slice].zero_()
    for _ in range(updates):
        indices = rng.integers(len(replay), size=batch_size)
        state = torch.as_tensor(replay.states[indices].copy(), device=agent.device)
        state[:, visual_slice] = 0.0
        action = torch.as_tensor(replay.actions[indices], device=agent.device)
        mean, _ = agent.actor(state)
        steer_loss = nn.functional.mse_loss(torch.tanh(mean[:, 0]), action[:, 0])
        brake_target = (action[:, 1] < -0.5).float()
        brake_loss = nn.functional.binary_cross_entropy_with_logits(
            -mean[:, 1] - 0.55, brake_target)
        loss = steer_loss + 0.25 * brake_loss
        agent.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        agent.actor_optimizer.step()

    with torch.no_grad():
        agent.actor.net[0].weight[:, visual_slice].zero_()
    with torch.no_grad():
        steer_errors = []
        brake_correct = 0
        for start in range(0, len(replay), 1024):
            states = torch.as_tensor(replay.states[start:start + 1024],
                                     device=agent.device)
            actions = torch.as_tensor(replay.actions[start:start + 1024],
                                      device=agent.device)
            predicted = agent.actor.deterministic(states)
            steer_errors.append(torch.abs(predicted[:, 0] - actions[:, 0]).cpu())
            brake_correct += int(((predicted[:, 1] < -0.5) ==
                                  (actions[:, 1] < -0.5)).sum().item())
        return dict(steer_mae=float(torch.cat(steer_errors).mean().item()),
                    brake_accuracy=brake_correct / len(replay),
                    examples=len(replay), updates=updates)


class DeterministicActor:
    def __init__(self, agent: SAC):
        self.agent = agent

    def act(self, observation):
        return self.agent.act(observation, deterministic=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guide-attempts", type=int, default=1)
    parser.add_argument("--actor-attempts", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=90.0)
    parser.add_argument("--speed-cap", type=float, default=35.0)
    parser.add_argument("--lookahead-seconds", type=float, default=0.5)
    parser.add_argument("--imitation-updates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if (not 1 <= args.guide_attempts <= 10 or
            not 0 <= args.actor_attempts <= 10 or
            not 1 <= args.seconds <= 120 or
            not 1 <= args.imitation_updates <= 20_000):
        parser.error("attempt or duration settings are outside safe bounds")

    route = Route.from_csv(ROUTE_CSV)
    map_hash = check_map()
    run = ROOT / "runs" / datetime.now(timezone.utc).strftime("bootstrap-%Y%m%dT%H%M%SZ")
    run.mkdir(parents=True, exist_ok=False)
    (run / "config.json").write_text(json.dumps(vars(args) | dict(
        map_sha256=map_hash,
        route_sha256=hashlib.sha256(ROUTE_CSV.read_bytes()).hexdigest(),
        observation_size=OBSERVATION_SIZE, action_size=ACTION_SIZE,
        reward_version=REWARD_VERSION, step_seconds=STEP_SECONDS,
        source="guided route positions for training only",
        imitation_visual_weights_masked=True), indent=2),
        encoding="utf-8")
    replay = ReplayBuffer(200_000, OBSERVATION_SIZE, ACTION_SIZE, args.seed)
    agent = SAC(OBSERVATION_SIZE, ACTION_SIZE, seed=args.seed)
    guide_successes = 0
    try:
        with GameBridge() as game:
            game.ensure_start()
            config_path = run / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config.update(capture_source_size=list(game.window.source_size),
                          capture_output_size=list(game.window.output_size))
            config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            runner = EpisodeRunner(game, route)
            for index in range(args.guide_attempts):
                game.reset()  # start the game clock immediately before driving
                trial = ReplayBuffer(10_000, OBSERVATION_SIZE, ACTION_SIZE,
                                     args.seed + index)
                guide = RouteGuide(route, speed_cap=args.speed_cap,
                                   lookahead_seconds=args.lookahead_seconds)
                result = runner.run(guide, buffer=trial, max_seconds=args.seconds)
                accepted = result.finished and result.wall_scrape_frames == 0
                if accepted:
                    copy_transitions(trial, replay)
                    guide_successes += 1
                record = result.as_json() | dict(policy="route_guide", accepted=accepted)
                if result.finished:
                    cv2.imwrite(str(run / f"guide-finish-{index + 1}.jpg"),
                                game.window.screenshot())
                with (run / "episodes.jsonl").open("a", encoding="utf-8") as file:
                    file.write(json.dumps(record) + "\n")
                print(f"Guide {index + 1}: {record}", flush=True)
                # Leave the post-finish screen immediately; it stops accepting
                # the driving reset key after the offline fit has run.
                game.reset()
            if not guide_successes:
                print("No clean guided lap; actor training skipped", flush=True)
                return

            metrics = imitate(agent, replay, args.imitation_updates, 256, args.seed)
            (run / "imitation.json").write_text(json.dumps(metrics, indent=2),
                                                encoding="utf-8")
            agent.save(run / "checkpoint.pt")
            replay.save(run / "replay.npz")
            print(f"Imitation: {metrics}", flush=True)
            for index in range(args.actor_attempts):
                game.reset()  # fitting may have left the clock running at start
                result = runner.run(DeterministicActor(agent), max_seconds=args.seconds)
                record = result.as_json() | dict(policy="imitated_actor")
                if result.finished:
                    cv2.imwrite(str(run / f"actor-finish-{index + 1}.jpg"),
                                game.window.screenshot())
                with (run / "episodes.jsonl").open("a", encoding="utf-8") as file:
                    file.write(json.dumps(record) + "\n")
                print(f"Actor {index + 1}: {record}", flush=True)
                game.reset()
    finally:
        if len(replay):
            replay.save(run / "replay.npz")
        print(f"Run saved in {run}", flush=True)


if __name__ == "__main__":
    main()
