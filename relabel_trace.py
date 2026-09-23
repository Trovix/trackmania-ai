"""Correct a learned actor using guide labels at its own visited positions."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from bootstrap import imitate
from episode import ACTION_SIZE, OBSERVATION_SIZE, Observations, ROUTE_CSV
from game_bridge import Sample, check_map
from guide import RouteGuide
from replay import ReplayBuffer
from route_progress import Route
from sac import SAC


class Examples:
    def __init__(self, states, actions):
        self.states = np.asarray(states, dtype=np.float32)
        self.actions = np.asarray(actions, dtype=np.float32)

    def __len__(self):
        return len(self.states)


def reconstruct_labels(trace: Path, route: Route, speed_cap: float) -> Examples:
    with trace.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("empty trace")
    encoder = Observations(route, 960)
    guide = RouteGuide(route, speed_cap=speed_cap)
    states, actions = [], []
    for index, row in enumerate(rows):
        telemetry = {name: float(row[name]) for name in ("x", "z", "speed")}
        sample = Sample(telemetry, np.zeros(19, dtype=np.float32),
                        np.zeros((8, 16), dtype=np.uint8), index * 0.02)
        progress = float(row["route_progress"])
        previous_action = (float(row["steer_action"]), float(row["drive_action"]))
        state = encoder.encode(sample, progress, previous_action)
        guide.observe(sample, progress)
        if telemetry["speed"] > 3.0 and float(row["lateral_error"]) < 10.0:
            states.append(state)
            actions.append(guide.act(state))
    if len(states) < 100:
        raise ValueError("trace has too few moving states to relabel")
    return Examples(states, actions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-run", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--speed-cap", type=float, default=35.0)
    parser.add_argument("--updates", type=int, default=2500)
    args = parser.parse_args()
    if args.speed_cap <= 0 or not 1 <= args.updates <= 20_000:
        parser.error("invalid speed cap or update count")
    map_hash = check_map()
    source = args.bootstrap_run.resolve()
    source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if (source_config["map_sha256"] != map_hash or
            source_config["route_sha256"] != hashlib.sha256(ROUTE_CSV.read_bytes()).hexdigest() or
            source_config["observation_size"] != OBSERVATION_SIZE or
            source_config["action_size"] != ACTION_SIZE):
        raise ValueError("bootstrap observation contract differs")
    trace_config = json.loads((args.trace.parent / "config.json").read_text(
        encoding="utf-8"))
    if trace_config.get("actor_speed_cap") != args.speed_cap:
        raise ValueError("trace speed cap differs from relabel request")
    route = Route.from_csv(ROUTE_CSV)
    replay = ReplayBuffer.load(source / "replay.npz")
    correction = reconstruct_labels(args.trace, route, args.speed_cap)
    # Give visited-state corrections at least as much weight as the tidy lap.
    repeats = max(1, round(len(replay) / len(correction)))
    examples = Examples(np.concatenate((replay.states[:len(replay)],
                                        np.tile(correction.states, (repeats, 1)))),
                        np.concatenate((replay.actions[:len(replay)],
                                        np.tile(correction.actions, (repeats, 1)))))
    agent = SAC.load(source / "checkpoint.pt")
    metrics = imitate(agent, examples, args.updates, 256, seed=2)
    run = Path(__file__).resolve().parent / "runs" / datetime.now(timezone.utc).strftime(
        "relabel-%Y%m%dT%H%M%SZ")
    run.mkdir(parents=True, exist_ok=False)
    agent.save(run / "checkpoint.pt")
    np.savez_compressed(run / "corrections.npz", states=correction.states,
                        actions=correction.actions)
    (run / "result.json").write_text(json.dumps(dict(
        bootstrap_run=str(source), actor_trace=str(args.trace.resolve()),
        speed_cap=args.speed_cap, moving_corrections=len(correction),
        correction_repeats=repeats, **metrics), indent=2), encoding="utf-8")
    print(json.dumps(dict(run=str(run), moving_corrections=len(correction),
                          **metrics)), flush=True)


if __name__ == "__main__":
    main()
