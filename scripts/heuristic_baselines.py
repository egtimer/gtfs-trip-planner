"""Model-free difficulty check: how often simple, familiar planning strategies already reach
the optimum. The direct-trip baseline is 0% by construction; the others show how much of the
task needs walking changes or three-plus legs.

Run: uv run python scripts/heuristic_baselines.py --split eval --tasks-per-region 50
"""

import argparse
from collections import Counter

from gtfs_trip_planner.network import CHANGE_S, INF, Network
from gtfs_trip_planner.taskset import TripPlannerConfig, TripPlannerTaskset, _network


def best_with_rules(net: Network, origin: str, dest: str, depart: int, max_legs: int, walking: bool) -> int | None:
    """Round-based search (RAPTOR-like over departures) limited to `max_legs` and optionally
    without walking changes."""
    o, d = net.index[origin], net.index[dest]
    ready = {o: depart}
    best = INF
    for _round in range(max_legs):
        nxt: dict[int, int] = {}
        for u, r in ready.items():
            for dep, tid, k in net.departures[u]:
                if dep < r:
                    continue
                if dep >= best:
                    break
                for v, arr, _, _, drop in net.trips[tid][2][k + 1 :]:
                    if not drop:
                        continue
                    if v == d:
                        best = min(best, arr)
                    cands = [(v, arr + CHANGE_S)] + ([(w, arr + ws) for w, ws in net.footpaths[v]] if walking else [])
                    for w, t in cands:
                        if t < nxt.get(w, INF) and t < ready.get(w, INF):
                            nxt[w] = t
        ready = {**ready, **{k: min(v, ready.get(k, INF)) for k, v in nxt.items()}}
    return None if best >= INF else best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="eval")
    ap.add_argument("--tasks-per-region", type=int, default=50)
    args = ap.parse_args()
    tasks = TripPlannerTaskset(TripPlannerConfig(split=args.split, tasks_per_region=args.tasks_per_region)).load()
    strategies = {
        "direct trip only": (1, False),
        "at most 2 legs, no walking": (2, False),
        "at most 2 legs, walking": (2, True),
        "at most 3 legs, no walking": (3, False),
    }
    hits = Counter()
    for task in tasks:
        d = task.data
        net = _network(d.region, d.service_date)
        for name, (legs, walking) in strategies.items():
            arr = best_with_rules(net, d.origin_id, d.destination_id, d.depart, legs, walking)
            hits[name] += arr is not None and arr <= d.optimal_arrival
    print(
        f"{len(tasks)} {args.split} tasks; legs in the optimum: {dict(Counter(t.data.reference_legs for t in tasks))}"
    )
    for name in strategies:
        print(f"  {name:<28} reaches the optimum in {hits[name] / len(tasks):6.1%}")


if __name__ == "__main__":
    main()
