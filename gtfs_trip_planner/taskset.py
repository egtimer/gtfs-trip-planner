"""gtfs-trip-planner: plan the earliest-arriving public-transport journey from real timetables.

Each task is a (region, origin, destination, departure time) query on one service day of a
real Norwegian network (Entur national GTFS). The agent only sees the timetable through tools
(stop search, departures, trip stops, nearby stops) and submits an itinerary as legs of
(trip, boarding stop, alighting stop). The task re-checks the legs against the timetable and
rewards 1.0 only when the itinerary is valid AND arrives as early as the exact optimum
computed by a Connection Scan router. Tasks are kept only when no single trip reaches the
optimum, so every task needs at least one change; the `hard` difficulty keeps only journeys
whose optimum needs at least three legs. Train and eval regions share no stops or trips, so
the eval split measures planning on unseen networks, not memorised timetables.
"""

import random
from collections.abc import Iterator
from typing import Literal

import verifiers.v1 as vf

from gtfs_trip_planner.build import REGIONS, SERVICE_DATE, build_networks, network_path
from gtfs_trip_planner.network import (
    CHANGE_S,
    WALK_RADIUS_M,
    Leg,
    Network,
    fmt_time,
    load_network,
)
from gtfs_trip_planner.servers.transit import TransitState, TransitToolset, TransitToolsetConfig

MIN_STOP_DEPARTURES = 20
MAX_COUNTED_LEGS = 8

# difficulty -> (shortest optimal trip, longest optimal trip, fewest legs that reach the optimum)
DIFFICULTIES: dict[str, tuple[int, int, int]] = {
    "standard": (20 * 60, 150 * 60, 2),
    "hard": (30 * 60, 180 * 60, 3),
}

INSTRUCTIONS = f"""You are a public-transport journey planner. Use the transit tools to read the timetable and find the itinerary that ARRIVES EARLIEST at the destination.

Rules (the itinerary is checked against the timetable):
- An itinerary is an ordered list of legs. Each leg rides one trip from a boarding stop to a later stop of the same trip where you get off.
- The first leg must board at the origin stop, no earlier than the departure time. The last leg must get off at the destination stop.
- To change at the same stop, board the next trip at least {CHANGE_S // 60} minutes after you arrive.
- You may instead walk once between legs to another stop within {int(WALK_RADIUS_M)} m; board there no earlier than your arrival plus the walking time shown by transit_nearby_stops.
- No walking at the start or the end of the journey.
- Times use the service-day clock (25:10 means 01:10 the next morning).

When you are confident you have the earliest arrival, call transit_submit_itinerary once with the legs."""


class TripData(vf.TaskData):
    region: str
    service_date: str
    origin_id: str
    destination_id: str
    depart: int
    """Earliest departure, seconds after midnight of the service day."""
    optimal_arrival: int
    """Exact earliest arrival (seconds), from the Connection Scan router."""
    best_direct_arrival: int | None
    """Earliest arrival with a single trip, None if no direct trip exists."""
    reference_legs: int
    """Number of legs in the router's optimal journey."""
    min_legs: int = 0
    """Fewest legs of any journey that reaches the optimum (counted up to 8; 0 if not computed)."""
    difficulty: str = "standard"


class TripTaskConfig(vf.TaskConfig):
    tools: TransitToolsetConfig = TransitToolsetConfig()


def _network(region: str, date: str) -> Network:
    return load_network(str(network_path(region, date)))


def _legs(raw: list[dict] | None) -> list[Leg] | None:
    if raw is None:
        return None
    return [Leg(str(r["trip_id"]), str(r["board_stop_id"]), str(r["alight_stop_id"])) for r in raw]


class TripTask(vf.Task[TripData, TransitState, TripTaskConfig]):
    @classmethod
    def toolsets(cls, config: TripTaskConfig) -> list[vf.Toolset]:
        return [TransitToolset(config.tools)]

    def _check(self, legs: list[Leg] | None):
        if legs is None:
            return None
        d = self.data
        return _network(d.region, d.service_date).check(d.origin_id, d.destination_id, d.depart, legs)

    @vf.stop
    async def submitted(self, trace: vf.Trace) -> bool:
        return trace.state.submitted is not None

    @vf.reward(weight=1.0)
    async def earliest_arrival(self, trace: vf.Trace) -> float:
        check = self._check(_legs(trace.state.submitted))
        return float(bool(check and check.ok and check.arrival <= self.data.optimal_arrival))

    @vf.metric
    async def valid_itinerary(self, trace: vf.Trace) -> float:
        check = self._check(_legs(trace.state.submitted))
        return float(bool(check and check.ok))

    @vf.metric
    async def minutes_late(self, trace: vf.Trace) -> float:
        """Minutes after the optimum for a valid itinerary; -1 when invalid or not submitted."""
        check = self._check(_legs(trace.state.submitted))
        if not (check and check.ok):
            return -1.0
        return (check.arrival - self.data.optimal_arrival) / 60

    @vf.metric
    async def submitted_legs(self, trace: vf.Trace) -> float:
        return float(len(trace.state.submitted or []))

    @vf.metric
    async def timetable_lookups(self, trace: vf.Trace) -> float:
        return float(trace.state.lookups)

    async def validate(self, runtime: vf.Runtime) -> bool:
        """Gold check: the router's own journey must verify and reach the stored optimum."""
        d = self.data
        net = _network(d.region, d.service_date)
        result = net.earliest_arrival(d.origin_id, d.destination_id, d.depart)
        if result is None:
            return False
        arrival, legs = result
        check = net.check(d.origin_id, d.destination_id, d.depart, legs)
        return check.ok and check.arrival == arrival == d.optimal_arrival


def make_prompt(net: Network, o: int, dest: int, depart: int, date: str) -> str:
    return (
        f"{INSTRUCTIONS}\n\n"
        f"Network: {net.meta['region_name']}, service day {date} (a Wednesday).\n"
        f"Origin: {net.stop_names[o]} (stop id {net.stop_ids[o]})\n"
        f"Destination: {net.stop_names[dest]} (stop id {net.stop_ids[dest]})\n"
        f"Leave no earlier than {fmt_time(depart)}."
    )


def min_legs(net: Network, origin: str, dest: str, depart: int, optimum: int, start: int = 1) -> int:
    """Fewest legs that still reach `optimum`, trying `start`, `start + 1`, ... up to MAX_COUNTED_LEGS."""
    for k in range(start, MAX_COUNTED_LEGS + 1):
        best = net.best_within_legs(origin, dest, depart, k)
        if best is not None and best <= optimum:
            return k
    return MAX_COUNTED_LEGS + 1


def generate(
    region: str, net: Network, count: int, seed: int, date: str, difficulty: str = "standard"
) -> Iterator[TripData]:
    """Deterministic task stream for one region: random stop pairs and departure times, kept
    only when the optimal trip length is in range and no journey with fewer legs than the
    difficulty's minimum reaches the optimum (standard: 20-150 min, at least 2 legs; hard:
    30-180 min, at least 3 legs)."""
    min_s, max_s, need_legs = DIFFICULTIES[difficulty]
    rng = random.Random(f"{seed}:{region}" if difficulty == "standard" else f"{seed}:{region}:{difficulty}")
    busy = [i for i, deps in enumerate(net.departures) if len(deps) >= MIN_STOP_DEPARTURES]
    made = 0
    for _ in range(count * 200):
        if made >= count:
            return
        o, dest = rng.sample(busy, 2)
        depart = rng.randrange(6 * 3600 + 30 * 60, 19 * 3600 + 30 * 60, 300)
        result = net.earliest_arrival(net.stop_ids[o], net.stop_ids[dest], depart)
        if result is None:
            continue
        arrival, legs = result
        if not (min_s <= arrival - depart <= max_s) or len(legs) < 2:
            continue
        direct = net.best_direct(net.stop_ids[o], net.stop_ids[dest], depart)
        if direct is not None and direct <= arrival:
            continue
        fewest = min_legs(net, net.stop_ids[o], net.stop_ids[dest], depart, arrival, start=2)
        if fewest < need_legs:
            continue
        yield TripData(
            id=f"{region}-{made:04d}" if difficulty == "standard" else f"{region}-{difficulty}-{made:04d}",
            name=f"{region}: {net.stop_names[o]} -> {net.stop_names[dest]} after {fmt_time(depart)}",
            prompt=make_prompt(net, o, dest, depart, date),
            region=region,
            service_date=date,
            origin_id=net.stop_ids[o],
            destination_id=net.stop_ids[dest],
            depart=depart,
            optimal_arrival=arrival,
            best_direct_arrival=direct,
            reference_legs=len(legs),
            min_legs=fewest,
            difficulty=difficulty,
        )
        made += 1


class TripPlannerConfig(vf.TasksetConfig):
    split: Literal["train", "eval"] = "eval"
    """Train regions: Bergen, Stavanger, Kristiansand, Drammen, Ostfold. Eval regions
    (held out, no shared stops or trips): Trondheim, Tromso, Oslo."""
    difficulty: Literal["standard", "hard"] = "standard"
    """standard: optimal trips of 20-150 min that need at least 2 legs. hard: 30-180 min and
    no journey with fewer than 3 legs reaches the optimum."""
    tasks_per_region: int = 50
    seed: int = 0
    task: TripTaskConfig = TripTaskConfig()


class TripPlannerTaskset(vf.Taskset[TripTask, TripPlannerConfig]):
    def load(self) -> list[TripTask]:
        regions = [r for r, spec in REGIONS.items() if spec["split"] == self.config.split]
        paths = build_networks(regions, SERVICE_DATE)
        tasks = []
        for region in regions:
            net = load_network(str(paths[region]))
            for data in generate(
                region, net, self.config.tasks_per_region, self.config.seed, SERVICE_DATE, self.config.difficulty
            ):
                tasks.append(TripTask(data, self.config.task))
        return tasks
