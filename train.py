"""Collect Aimap attempts and train a small SAC agent between episodes."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch

from episode import (ACTION_SIZE, OBSERVATION_SIZE, EpisodeRunner,
                     ExploratoryDriver, ROUTE_CSV, STEP_SECONDS)
from game_bridge import GameBridge, check_map
from replay import ReplayBuffer
from route_progress import Route
from sac import SAC


ROOT = Path(__file__).resolve().parent
REWARD_VERSION = "wall-scrape-v8-route-geometry"


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
    parser.add_argument("--bootstrap-run", type=Path,
                        help="start a new SAC run from clean guided examples and actor")
    args = parser.parse_args()
    if args.resume_run and args.bootstrap_run:
        parser.error("choose a resumed run or a bootstrap source")
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
                config.get("reward_version") != REWARD_VERSION or
                config.get("step_seconds") != STEP_SECONDS or
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
        start_index, logged_steps = last["episode"], last["total_steps"]
        if len(replay) < logged_steps and logged_steps < replay.capacity:
            raise RuntimeError("Replay snapshot is older than the episode log")
        # A live interruption can leave valid transitions from an unfinished
        # attempt in the replay snapshot but no corresponding episode record.
        interrupted_steps = max(0, len(replay) - logged_steps)
        total_steps = logged_steps + interrupted_steps
        explorer = ExploratoryDriver(args.seed + start_index)
        with (run / "resumes.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(dict(started_utc=datetime.now(timezone.utc).isoformat(),
                                       from_episode=start_index,
                                       interrupted_steps=interrupted_steps,
                                       additional_episodes=args.episodes)) + "\n")
    else:
        source = args.bootstrap_run.resolve() if args.bootstrap_run else None
        if source:
            source_config = json.loads((source / "config.json").read_text(
                encoding="utf-8"))
            if (source_config["map_sha256"] != map_hash or
                    source_config["route_sha256"] != route_hash or
                    source_config["observation_size"] != OBSERVATION_SIZE or
                    source_config["action_size"] != ACTION_SIZE or
                    source_config.get("reward_version", REWARD_VERSION) != REWARD_VERSION):
                raise RuntimeError("Bootstrap source does not match this experiment")
        label = datetime.now(timezone.utc).strftime("train-%Y%m%dT%H%M%SZ")
        run = ROOT / "runs" / label
        run.mkdir(parents=True, exist_ok=False)
        config = {key: value for key, value in vars(args).items()
                  if key not in ("resume_run", "bootstrap_run")}
        config.update(map_sha256=map_hash, route_sha256=route_hash,
                      reward_version=REWARD_VERSION,
                      step_seconds=STEP_SECONDS,
                      observation_size=OBSERVATION_SIZE, action_size=ACTION_SIZE,
                      torch=torch.__version__,
                      bootstrap_run=str(source) if source else None)
        (run / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        if source:
            replay = ReplayBuffer.load(source / "replay.npz")
            agent = SAC.load(source / "checkpoint.pt", device=args.device)
            # Imitation fits deterministic action means. Start SAC near those
            # actions instead of sampling from its still-untrained variance head.
            with torch.no_grad():
                output = agent.actor.net[-1]
                output.weight[ACTION_SIZE:].zero_()
                output.bias[ACTION_SIZE:].fill_(-2.0)
            config["bootstrap_examples"] = len(replay)
            config["bootstrap_log_std"] = -2.0
        else:
            replay = ReplayBuffer(200_000, OBSERVATION_SIZE, ACTION_SIZE, args.seed)
            agent = SAC(OBSERVATION_SIZE, ACTION_SIZE, device=args.device,
                        seed=args.seed)
        explorer = ExploratoryDriver(args.seed)
        start_index, total_steps = 0, len(replay)
    try:
        with GameBridge() as game:
            if args.resume_run:
                game.reset()
            game.ensure_start()
            game.sample()
            source_size = list(game.window.source_size)
            output_size = list(game.window.output_size)
            if args.resume_run:
                if (config.get("capture_source_size") != source_size or
                        config.get("capture_output_size") != output_size):
                    raise RuntimeError("Capture dimensions differ from the saved run")
            else:
                config.update(capture_source_size=source_size,
                              capture_output_size=output_size)
                if source and source_config.get("capture_output_size") not in (None, output_size):
                    raise RuntimeError("Bootstrap capture dimensions differ from this run")
                (run / "config.json").write_text(json.dumps(config, indent=2),
                                                 encoding="utf-8")
            runner = EpisodeRunner(game, route)
            for index in range(start_index, start_index + args.episodes):
                game.reset()  # zero the game clock just before each attempt
                using_agent = total_steps >= args.warmup_steps and len(replay) >= args.batch_size
                policy = agent if using_agent else explorer
                print(f"Attempt {index + 1}/{start_index + args.episodes}: "
                      f"{'SAC actor' if using_agent else 'exploratory driver'}", flush=True)
                result = runner.run(policy, buffer=replay, max_seconds=args.seconds)
                game.reset()  # leave the finish screen before offline updates
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
    finally:
        # Keep the latest recoverable model and transitions if a later episode fails.
        agent.save(run / "checkpoint.pt")
        replay.save(run / "replay.npz")
        print(f"Run saved in {run}", flush=True)


if __name__ == "__main__":
    main()
