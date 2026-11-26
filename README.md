# gtfs-trip-planner

A multi-turn, tool-using RL environment for [verifiers](https://github.com/PrimeIntellect-ai/verifiers): plan the public-transport journey that **arrives earliest**, using only timetable tools over real GTFS data.

The submitted legs are checked against the timetable and compared with the optimum from a Connection Scan router, so the reward does not depend on what the model writes.

### Overview
- **Environment ID**: `gtfs-trip-planner`
- **Tags**: tool-use, agentic, multi-turn, planning
- **Data**: Entur's national Norwegian GTFS (Parquet mirror [`knuthp/GTFS_Entur`](https://huggingface.co/datasets/knuthp/GTFS_Entur), pinned to revision `be37f77`). Service day: Wednesday 2026-04-15. Data © Entur AS, licensed under [NLOD 2.0](https://data.norge.no/nlod/en/2.0).

### Task
Each task gives a region, an origin stop, a destination stop and an earliest departure time. The agent reads the timetable through tools and submits one itinerary: an ordered list of legs `(trip_id, board_stop_id, alight_stop_id)`.

Rules, identical in the prompt, the router and the verifier:
- the first leg boards at the origin no earlier than the departure time; the last leg alights at the destination;
- a change at the same stop needs at least 2 minutes;
- or walk once between legs to a stop within 400 m (walking time = 60 s + distance at 1.2 m/s);
- boarding needs a pickup and alighting a drop-off at that stop; no walking at the start or end.

Two difficulties (`--taskset.difficulty`):

| Difficulty | Optimal trip | Kept only if |
| --- | --- | --- |
| `standard` (default) | 20 to 150 min | **no single trip reaches the optimum**, so every task needs at least one change |
| `hard` | 30 to 180 min | **no journey with one or two legs reaches the optimum**, so every task needs at least two changes |

Each task stores `min_legs`, the fewest legs of any journey that reaches the optimum, from a leg-bounded search.

### Splits
Regions are whole operators that share no stops or trips, so the eval split measures planning on unseen networks rather than memorised timetables.

| Split | Regions |
| --- | --- |
| `train` | Skyss (Bergen), Kolumbus (Stavanger), AKT (Kristiansand), Brakar (Drammen), Ostfold |
| `eval` | AtB (Trondheim), Troms fylkestrafikk (Tromso), Ruter (Oslo, 3,725 stops and 18,594 trips on the service day) |

### Tools
| Tool | What it returns |
| --- | --- |
| `transit_search_stops(query)` | up to 10 stops whose name contains the query, with ids and coordinates |
| `transit_departures(stop_id, after, limit)` | trips you can board at a stop at or after `HH:MM` (max 20) |
| `transit_trip_stops(trip_id)` | every stop of a trip with times and pickup/drop-off restrictions |
| `transit_nearby_stops(stop_id)` | stops within 400 m and the walking time to each |
| `transit_submit_itinerary(legs)` | records the final itinerary (once) and ends the rollout |

Tools never reveal the optimum. Scoring reads the legs from rollout state, never from the model's text.

### Reward and metrics
| Name | Weight | Meaning |
| --- | --- | --- |
| `earliest_arrival` | 1.0 | 1 if the itinerary is valid and arrives at the optimum, else 0 |
| `valid_itinerary` | metric | 1 if the itinerary follows every rule |
| `minutes_late` | metric | minutes after the optimum (valid itineraries), -1 otherwise |
| `submitted_legs` | metric | number of legs submitted |
| `timetable_lookups` | metric | timetable tool calls |

### Quickstart
```bash
uv pip install -e .
uv run vf-validate gtfs-trip-planner --runtime.type subprocess --taskset.tasks-per-region 5   # gold = 1, untouched = 0
uv run vf-run gtfs-trip-planner -n 5 -r 3 -m <model>
```

Configuration (`--taskset.*`): `split` (`eval` by default), `difficulty` (`standard` or `hard`), `tasks_per_region` (default 200 per train region, so 1,000 train tasks, and 50 per eval region, so 150 eval tasks), `seed` (0). Generated task lists are cached next to the networks. The first load downloads the pinned snapshot (~75 MB) and caches one compact network per region in `~/.cache/gtfs_trip_planner` (override with `GTFS_TRIP_PLANNER_CACHE`).

### How it is checked
- `tests/test_env.py`: the gold journey scores 1 on every task; no submission, a made-up trip, a reversed leg, stopping short, skipping a leg, a too-tight change and walking too far all score 0; a valid but later journey is marked valid and scores 0; the tools do not leak the answer; generation is deterministic.
- The Connection Scan router is cross-checked against an independent label-setting search (`tests/reference_router.py`): 0 mismatches on 180 random queries across three regions.
- `uv run vf-validate`: gold 10/10 valid, untouched 10/10 scores 0.
- A leg-bounded search (`Network.best_within_legs`) gives the same optimum as the router on every task when its leg limit is high, and sets `min_legs`.
- `scripts/heuristic_baselines.py` measures how often simple strategies already reach the optimum (eval split, 50 tasks per region, 150 tasks per difficulty):

| Strategy | `standard` | `hard` |
| --- | --- | --- |
| direct trip only | 0% (by construction) | 0% |
| at most 2 legs, no walking | 36.7% | 0% (by construction) |
| at most 2 legs, with walking | 44.7% | 0% (by construction) |
| at most 3 legs, no walking | 60.0% | 38.7% |
| at most 3 legs, with walking | 76.7% | 69.3% |

Model baselines from v0.1 (eval split before Oslo was added: Trondheim and Tromso, `--env.taskset.tasks-per-region 10`, 20 tasks, 1 rollout each, `null` harness so the model only has the transit tools):

| Model | Reward (± s.e.) | Valid itinerary | Reward on 2 / 3 / 4-leg optima | Mean timetable lookups |
| --- | --- | --- | --- | --- |
| claude-haiku-5-5 | 0.80 ± 0.09 | 0.95 | 1.00 / 0.70 / 0.50 | 27.3 |
| claude-sonnet-5-5 | 0.95 ± 0.05 | 1.00 | 1.00 / 1.00 / 0.50 | 16.8 |

Every miss was a valid journey that arrived 2 to 26 minutes late, except one invalid Haiku itinerary. Reward falls as the optimum needs more legs, which is why v0.2 adds the Oslo network and the `hard` difficulty.

Model baselines on v0.2 `hard` (eval split, 30 tasks per region, 90 tasks, 1 rollout each, `null` harness, at most 60 turns, models served by Prime Inference; `.github/workflows/baselines.yml`):

| Model | Reward (± s.e.) | Valid itinerary | Reward when the optimum needs 3 / 4 / 5 legs (n = 60 / 22 / 8) |
| --- | --- | --- | --- |
| deepseek/deepseek-v4.1-flash-1017 | 0.58 ± 0.05 | 0.97 | 0.70 / 0.32 / 0.38 |
| google/gemma-4-26b-a4b-it-flex | 0.00 | 0.00 | 0 / 0 / 0 |

DeepSeek's misses are almost all valid journeys that arrive later than the optimum (median 15 minutes late), so the failures are planning failures rather than rule violations. Gemma submitted an itinerary in 62 of 90 tasks and hit the turn limit in the rest, but none of its itineraries was valid; this may be a tool-calling format problem rather than a planning one and has not been investigated, so treat that row as unverified. A smaller earlier run on the same split (10 tasks per region) gave qwen/qwen3.5-9b 0 of 30 and qwen/qwen3.6-35b-a3b 3 correct out of 14 completed tasks. With per-call cached input, a `hard` task cost about 2 US cents for these models.
