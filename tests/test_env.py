"""Controls for the verifier: gold scores 1, every shortcut or broken itinerary scores 0."""

import asyncio
import dataclasses
from types import SimpleNamespace

import pytest
from reference_router import reference_earliest_arrival

from gtfs_trip_planner.network import CHANGE_S, Leg, Network, walk_seconds
from gtfs_trip_planner.servers.transit import TransitState, TransitToolset, TransitToolsetConfig
from gtfs_trip_planner.taskset import TripPlannerConfig, TripPlannerTaskset, _network


@pytest.fixture(scope="module")
def tasks():
    return TripPlannerTaskset(TripPlannerConfig(split="eval", tasks_per_region=10)).load()


def score(task, legs):
    trace = SimpleNamespace(
        state=TransitState(submitted=None if legs is None else [dataclasses.asdict(x) for x in legs])
    )
    reward = asyncio.run(task.earliest_arrival(trace))
    valid = asyncio.run(task.valid_itinerary(trace))
    return reward, valid


def gold(task):
    d = task.data
    arrival, legs = _network(d.region, d.service_date).earliest_arrival(d.origin_id, d.destination_id, d.depart)
    return arrival, legs


def test_tasks_need_a_change_and_have_a_valid_gold(tasks):
    assert len(tasks) == 20
    for task in tasks:
        assert asyncio.run(task.validate(None))
        assert task.data.reference_legs >= 2
        direct = task.data.best_direct_arrival
        assert direct is None or direct > task.data.optimal_arrival
        assert score(task, gold(task)[1]) == (1.0, 1.0)


def test_router_matches_independent_search(tasks):
    for task in tasks[:8]:
        d = task.data
        net = _network(d.region, d.service_date)
        assert reference_earliest_arrival(net, d.origin_id, d.destination_id, d.depart) == d.optimal_arrival


def test_no_submission_scores_zero(tasks):
    assert score(tasks[0], None) == (0.0, 0.0)


def test_fabricated_trip_scores_zero(tasks):
    legs = gold(tasks[0])[1]
    fake = [dataclasses.replace(legs[0], trip_id="RUT:ServiceJourney:made-up")] + legs[1:]
    assert score(tasks[0], fake) == (0.0, 0.0)


def test_reversed_leg_scores_zero(tasks):
    legs = gold(tasks[1])[1]
    flipped = [Leg(legs[0].trip_id, legs[0].alight_stop_id, legs[0].board_stop_id)] + legs[1:]
    assert score(tasks[1], flipped) == (0.0, 0.0)


def test_stopping_short_scores_zero(tasks):
    legs = gold(tasks[2])[1]
    assert score(tasks[2], legs[:-1]) == (0.0, 0.0)


def test_skipping_a_middle_leg_scores_zero(tasks):
    for task in tasks:
        legs = gold(task)[1]
        if len(legs) >= 3:
            assert score(task, [legs[0], legs[2], *legs[3:]])[1] == 0.0
            return
    pytest.skip("no task with three legs")


def test_valid_but_later_journey_is_valid_and_scores_zero(tasks):
    checked = 0
    for task in tasks:
        d = task.data
        net = _network(d.region, d.service_date)
        later = net.earliest_arrival(d.origin_id, d.destination_id, d.depart + 45 * 60)
        if later is None or later[0] <= d.optimal_arrival:
            continue
        assert score(task, later[1]) == (0.0, 1.0)
        checked += 1
    assert checked >= 3


def tiny_network() -> Network:
    # Stops A, B, C on one line; D is 100 m from B; E is 1 km from B.
    stops = [
        ["A", "Alpha", 60.0, 10.0],
        ["B", "Bravo", 60.01, 10.0],
        ["C", "Charlie", 60.02, 10.0],
        ["D", "Delta", 60.0109, 10.0],
        ["E", "Echo", 60.019, 10.01],
    ]
    t = lambda h, m: h * 3600 + m * 60
    trips = {
        "T1": ["1", "Bravo", [[0, t(8, 0), t(8, 0), True, True], [1, t(8, 10), t(8, 10), True, True]]],
        # Leaves B one minute after T1 arrives: too tight for a change.
        "T2": ["2", "Charlie", [[1, t(8, 11), t(8, 11), True, True], [2, t(8, 20), t(8, 20), True, True]]],
        "T3": ["3", "Charlie", [[1, t(8, 12), t(8, 12), True, True], [2, t(8, 25), t(8, 25), True, True]]],
        "T4": ["4", "Charlie", [[3, t(8, 13), t(8, 13), True, True], [2, t(8, 22), t(8, 22), True, True]]],
        "T5": ["5", "Charlie", [[4, t(8, 15), t(8, 15), True, True], [2, t(8, 18), t(8, 18), True, True]]],
    }
    return Network({"region": "tiny", "region_name": "Tiny", "date": "2026-04-15"}, stops, trips)


def test_change_and_walking_rules():
    net = tiny_network()
    t8 = 8 * 3600
    assert abs(net._walk[(1, 3)] - walk_seconds(100)) <= 2
    assert (1, 4) not in net._walk  # E is beyond the walking radius
    # Too tight: 1 minute change.
    assert not net.check("A", "C", t8, [Leg("T1", "A", "B"), Leg("T2", "B", "C")]).ok
    # Exactly the minimum change time is fine.
    assert CHANGE_S == 120
    assert net.check("A", "C", t8, [Leg("T1", "A", "B"), Leg("T3", "B", "C")]).arrival == t8 + 25 * 60
    # Walk B -> D takes about 2.4 min, so the 08:13 departure from D is reachable.
    assert net.check("A", "C", t8, [Leg("T1", "A", "B"), Leg("T4", "D", "C")]).arrival == t8 + 22 * 60
    # E is too far to walk to, even though T5 would be fastest.
    assert not net.check("A", "C", t8, [Leg("T1", "A", "B"), Leg("T5", "E", "C")]).ok
    # The router finds the walking journey and agrees with the verifier.
    arrival, legs = net.earliest_arrival("A", "C", t8)
    assert arrival == t8 + 22 * 60
    assert [x.trip_id for x in legs] == ["T1", "T4"]


def test_tools_show_timetable_without_revealing_the_answer(tasks):
    task = tasks[0]
    tools = TransitToolset(TransitToolsetConfig())
    asyncio.run(tools.setup_task(task.data))
    out = tools.departures(task.data.origin_id, "06:00", 5)
    assert "trip " in out
    assert "Unknown stop id" in tools.departures("nope", "06:00")
    assert "HH:MM" in tools.departures(task.data.origin_id, "6pm")
    trip_id = out.splitlines()[0].split("trip ")[-1]
    assert task.data.origin_id in tools.trip_stops(trip_id)
    assert tools.search_stops(tools.net.stop_names[0])
    assert tools.state.submitted is None
    tools.submit_itinerary([])
    assert tools.submit_itinerary([]) == "An itinerary was already submitted."


def test_generation_is_deterministic():
    a = TripPlannerTaskset(TripPlannerConfig(split="eval", tasks_per_region=3, seed=7)).load()
    b = TripPlannerTaskset(TripPlannerConfig(split="eval", tasks_per_region=3, seed=7)).load()
    assert [x.data.model_dump() for x in a] == [x.data.model_dump() for x in b]
