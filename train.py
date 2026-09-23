"""Collect Aimap attempts and train a small SAC agent between episodes."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch

from episode import (ACTION_SIZE, OBSERVATION_SIZE, EpisodeRunner,
                     ExploratoryDriver, ROUTE_CSV)
from game_bridge import GameBridge, check_map
from replay import ReplayBuffer
from route_progress import Route
from sac import SAC


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--warmup-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--updates-per-step", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume-run", type=Path,
                        help="continue this run for --episodes additional attempts")
    args = parser.parse_args()
    if not 1 <= args.episodes <= 500 or not 1 <= args.seconds <= 120:
        parser.error("episodes must be 1-500 and seconds must be 1-120")
    if args.warmup_steps < args.batch_size or args.batch_size < 2:
        parser.error("warmup steps must be at least batch size, which must be >= 2")
    if not 0 < args.updates_per_step <= 5:
        parser.error("updates-per-step must be in (0, 5]")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")

    route = Route.from_csv(ROUTE_CSV)
    map_hash = check_map()
    route_hash = hashlib.sha256(ROUTE_CSV.read_bytes()).hexdigest()
    if args.resume_run:
        run = args.resume_run.resolve()
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        if (config["map_sha256"] != map_hash or
                config["route_sha256"] != route_hash or
                config["observation_size"] != OBSERVATION_SIZE or
                config["action_size"] != ACTION_SIZE):
            raise RuntimeError("Run does not match the frozen map, route or observation contract")
        for name in ("seconds", "warmup_steps", "batch_size", "updates_per_step",
                     "seed", "device"):
            setattr(args, name, config[name])
        replay = ReplayBuffer.load(run / "replay.npz")
        agent = SAC.load(run / "checkpoint.pt", device=args.device)
        history = (run / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        last = json.loads(history[-1])
        start_index, total_steps = last["episode"], last["total_steps"]
        if len(replay) < total_steps and total_steps < replay.capacity:
            raise RuntimeError("Replay snapshot is older than the episode log")
        explorer = ExploratoryDriver(args.seed + start_index)
        with (run / "resumes.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(dict(started_utc=datetime.now(timezone.utc).isoformat(),
                                       from_episode=start_index,
                                       additional_episodes=args.episodes)) + "\n")
    else:
        label = datetime.now(timezone.utc).strftime("train-%Y%m%dT%H%M%SZ")
        run = ROOT / "runs" / label
        run.mkdir(parents=True, exist_ok=False)
        config = {key: value for key, value in vars(args).items() if key != "resume_run"}
        config.update(map_sha256=map_hash, route_sha256=route_hash,
                      observation_size=OBSERVATION_SIZE, action_size=ACTION_SIZE,
                      torch=torch.__version__)
        (run / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        replay = ReplayBuffer(200_000, OBSERVATION_SIZE, ACTION_SIZE, args.seed)
        agent = SAC(OBSERVATION_SIZE, ACTION_SIZE, device=args.device, seed=args.seed)
        explorer = ExploratoryDriver(args.seed)
        start_index, total_steps = 0, 0
    try:
        with GameBridge() as game:
            runner = EpisodeRunner(game, route)
            for index in range(start_index, start_index + args.episodes):
                if index > start_index:
                    game.reset()
                using_agent = total_steps >= args.warmup_steps and len(replay) >= args.batch_size
                policy = agent if using_agent else explorer
                print(f"Attempt {index + 1}/{start_index + args.episodes}: "
                      f"{'SAC actor' if using_agent else 'exploratory driver'}", flush=True)
                result = runner.run(policy, buffer=replay, max_seconds=args.seconds)
                total_steps += result.steps
                record = result.as_json() | dict(episode=index + 1, total_steps=total_steps,
                                                 policy="sac" if using_agent else "explore")
                with (run / "episodes.jsonl").open("a", encoding="utf-8") as file:
                    file.write(json.dumps(record) + "\n")
                print(f"  {record}", flush=True)

                if len(replay) >= max(args.batch_size, args.warmup_steps):
                    count = max(1, round(result.steps * args.updates_per_step))
                    last_metrics = None
                    for _ in range(count):
                        last_metrics = agent.update(replay, args.batch_size)
                    with (run / "updates.jsonl").open("a", encoding="utf-8") as file:
                        file.write(json.dumps(dict(episode=index + 1, count=count,
                                                   updates=agent.updates,
                                                   **last_metrics)) + "\n")
                    print(f"  {count} SAC updates; {last_metrics}", flush=True)
                agent.save(run / "checkpoint.pt")
                replay.save(run / "replay.npz")
            game.reset()
    finally:
        # Keep the latest recoverable model and transitions if a later episode fails.
        agent.save(run / "checkpoint.pt")
        replay.save(run / "replay.npz")
        print(f"Run saved in {run}", flush=True)


if __name__ == "__main__":
    main()
