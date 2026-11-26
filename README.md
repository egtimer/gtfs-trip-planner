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

Tasks are kept only if the optimal journey takes 20 to 150 minutes and **no single trip reaches the optimum**, so every task needs at least one change.

### Splits
Regions are whole operators that share no stops or trips, so the eval split measures planning on unseen networks rather than memorised timetables.

| Split | Regions |
| --- | --- |
| `train` | Skyss (Bergen), Kolumbus (Stavanger), AKT (Kristiansand), Brakar (Drammen), Ostfold |
| `eval` | AtB (Trondheim), Troms fylkestrafikk (Tromso) |

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

Configuration (`--taskset.*`): `split` (`eval` by default), `tasks_per_region` (50), `seed` (0). The first load downloads the pinned snapshot (~75 MB) and caches one compact network per region in `~/.cache/gtfs_trip_planner` (override with `GTFS_TRIP_PLANNER_CACHE`).

### How it is checked
- `tests/test_env.py`: the gold journey scores 1 on every task; no submission, a made-up trip, a reversed leg, stopping short, skipping a leg, a too-tight change and walking too far all score 0; a valid but later journey is marked valid and scores 0; the tools do not leak the answer; generation is deterministic.
- The Connection Scan router is cross-checked against an independent label-setting search (`tests/reference_router.py`): 0 mismatches on 180 random queries across three regions.
- `uv run vf-validate`: gold 10/10 valid, untouched 10/10 scores 0.
- `scripts/heuristic_baselines.py` measures how often simple strategies already reach the optimum (eval split, 100 tasks):

| Strategy | Reaches the optimum |
| --- | --- |
| direct trip only | 0% (by construction) |
| at most 2 legs, no walking | 53% |
| at most 2 legs, with walking | 62% |
| at most 3 legs, no walking | 82% |

Model baselines: not run yet.
