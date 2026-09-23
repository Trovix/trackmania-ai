"""Run a few short, automatic exploratory attempts on the frozen Aimap."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from episode import EpisodeRunner, ExploratoryDriver, ROUTE_CSV, save_episode
from game_bridge import GameBridge
from route_progress import Route


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.episodes <= 20 or not 1 <= args.seconds <= 120:
        parser.error("episodes must be 1-20 and seconds must be 1-120")

    route = Route.from_csv(ROUTE_CSV)
    label = datetime.now(timezone.utc).strftime("explore-%Y%m%dT%H%M%SZ")
    output = Path(__file__).resolve().parent / "runs" / label / "episodes.jsonl"
    with GameBridge() as game:
        runner = EpisodeRunner(game, route)
        driver = ExploratoryDriver(args.seed)
        for index in range(args.episodes):
            if index:
                game.reset()
            result = runner.run(driver, max_seconds=args.seconds)
            save_episode(result, output)
            print(f"Attempt {index + 1}: {result.as_json()}", flush=True)
        game.reset()
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
