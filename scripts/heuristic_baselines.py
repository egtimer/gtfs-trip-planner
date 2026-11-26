"""Model-free difficulty check: how often simple, familiar planning strategies already reach
the optimum. The direct-trip baseline is 0% by construction; the others show how much of the
task needs walking changes or three-plus legs.

Run: uv run python scripts/heuristic_baselines.py --split eval --difficulty hard --tasks-per-region 50
"""

import argparse
from collections import Counter

from gtfs_trip_planner.taskset import TripPlannerConfig, TripPlannerTaskset, _network


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="eval")
    ap.add_argument("--difficulty", default="standard", choices=["standard", "hard"])
    ap.add_argument("--tasks-per-region", type=int, default=50)
    args = ap.parse_args()
    config = TripPlannerConfig(split=args.split, difficulty=args.difficulty, tasks_per_region=args.tasks_per_region)
    tasks = TripPlannerTaskset(config).load()
    strategies = {
        "direct trip only": (1, False),
        "at most 2 legs, no walking": (2, False),
        "at most 2 legs, walking": (2, True),
        "at most 3 legs, no walking": (3, False),
        "at most 3 legs, walking": (3, True),
    }
    hits = Counter()
    for task in tasks:
        d = task.data
        net = _network(d.region, d.service_date)
        for name, (legs, walking) in strategies.items():
            arr = net.best_within_legs(d.origin_id, d.destination_id, d.depart, legs, walking)
            hits[name] += arr is not None and arr <= d.optimal_arrival
    fewest = dict(sorted(Counter(t.data.min_legs for t in tasks).items()))
    print(f"{len(tasks)} {args.split}/{args.difficulty} tasks; fewest legs that reach the optimum: {fewest}")
    for name in strategies:
        print(f"  {name:<28} reaches the optimum in {hits[name] / len(tasks):6.1%}")


if __name__ == "__main__":
    main()
