"""Summarise a vf-run output directory: reward with its standard error, valid itineraries,
reward by the fewest legs the optimum needs and by region, timetable lookups and errors.

Run: uv run python scripts/summarize_run.py outputs/<run dir> [outputs/<run dir> ...]
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def mean_se(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)


def rollouts(run_dir: Path) -> list[dict]:
    rows = []
    for line in (run_dir / "traces.jsonl").read_text().splitlines():
        episode = json.loads(line)
        for trace in episode.get("traces") or []:
            data = trace["task"]["data"]
            rows.append(
                {
                    "model": trace["calls"][0]["model"] if trace.get("calls") else "?",
                    "region": data["region"],
                    "difficulty": data.get("difficulty", "standard"),
                    "legs": data.get("min_legs") or data["reference_legs"],
                    "reward": float(trace.get("reward") or 0.0),
                    "valid": float(trace["metrics"].get("valid_itinerary", 0.0)),
                    "lookups": float(trace["metrics"].get("timetable_lookups", 0.0)),
                    "error": bool(trace.get("has_error")) or not trace.get("ok", True),
                    "timeout": bool(trace.get("is_timeout")),
                }
            )
    return rows


def summarize(run_dir: Path) -> str:
    rows = rollouts(run_dir)
    if not rows:
        return f"{run_dir.name}: no rollouts"
    reward, se = mean_se([r["reward"] for r in rows])
    by_legs: dict[int, list[float]] = defaultdict(list)
    by_region: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_legs[r["legs"]].append(r["reward"])
        by_region[r["region"]].append(r["reward"])
    legs = ", ".join(f"{k} legs {sum(v) / len(v):.2f} (n={len(v)})" for k, v in sorted(by_legs.items()))
    regions = ", ".join(f"{k} {sum(v) / len(v):.2f}" for k, v in sorted(by_region.items()))
    return "\n".join(
        [
            f"{run_dir.name}",
            f"  model {rows[0]['model']}, difficulty {rows[0]['difficulty']}, {len(rows)} rollouts",
            f"  reward {reward:.2f} ± {se:.2f} (s.e.), valid itinerary {sum(r['valid'] for r in rows) / len(rows):.2f}",
            f"  by fewest legs to the optimum: {legs}",
            f"  by region: {regions}",
            (
                f"  mean timetable lookups {sum(r['lookups'] for r in rows) / len(rows):.1f}, "
                f"errors {sum(r['error'] for r in rows)}, timeouts {sum(r['timeout'] for r in rows)}"
            ),
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    args = ap.parse_args()
    for run_dir in args.run_dirs:
        print(summarize(run_dir))


if __name__ == "__main__":
    main()
