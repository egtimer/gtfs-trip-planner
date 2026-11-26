"""The journey planner's tool server: timetable lookups plus one final itinerary submission.

Scoring never parses tool output or the model's prose: the submitted legs are stored on the
rollout state and re-checked against the timetable by the task."""

import verifiers.v1 as vf
from pydantic import BaseModel, Field

from gtfs_trip_planner.build import build_networks
from gtfs_trip_planner.network import Network, fmt_time, load_network, parse_time

MAX_DEPARTURES = 20


class LegInput(BaseModel):
    trip_id: str = Field(description="Trip id from transit_departures or transit_trip_stops")
    board_stop_id: str = Field(description="Stop id where you board this trip")
    alight_stop_id: str = Field(description="Stop id where you get off this trip")


class TransitToolsetConfig(vf.ToolsetConfig):
    pass


class TransitState(vf.State):
    """Per-rollout state shared with the task's scoring."""

    submitted: list[dict] | None = None
    """Legs of the submitted itinerary (None until submitted)."""
    lookups: int = 0
    """Timetable tool calls (search, departures, trip stops, nearby stops)."""


class TransitToolset(vf.Toolset[TransitToolsetConfig, TransitState]):
    TOOL_PREFIX = "transit"

    async def setup_task(self, task) -> None:
        path = build_networks([task.region], task.service_date)[task.region]
        self.net: Network = load_network(str(path))

    def _unknown(self, stop_id: str) -> str | None:
        if stop_id in self.net.index:
            return None
        return f"Unknown stop id {stop_id!r}. Use transit_search_stops to find stop ids."

    @vf.tool
    def search_stops(self, query: str) -> str:
        """Find stops whose name contains the query (case-insensitive). Returns up to 10 stops
        with their ids and coordinates."""
        self.state.lookups += 1
        hits = self.net.search(query)
        if not hits:
            return f"No stop matches {query!r}."
        net = self.net
        return "\n".join(
            f"{net.stop_ids[i]} | {net.stop_names[i]} | lat {net.stop_lat[i]:.5f}, lon {net.stop_lon[i]:.5f}"
            for i in hits
        )

    @vf.tool
    def departures(self, stop_id: str, after: str, limit: int = 10) -> str:
        """List trips that you can board at a stop at or after a time (HH:MM, service day clock,
        so 25:10 is 01:10 the next morning). Returns up to `limit` (max 20) departures with the
        trip id, route, headsign and departure time."""
        self.state.lookups += 1
        if err := self._unknown(stop_id):
            return err
        s = self.net.index[stop_id]
        try:
            t = parse_time(after)
        except ValueError as e:
            return str(e)
        limit = max(1, min(int(limit), MAX_DEPARTURES))
        rows = []
        for dep, tid, _k in self.net.departures[s]:
            if dep < t:
                continue
            route, headsign, _ = self.net.trips[tid]
            rows.append(f"{fmt_time(dep)} | route {route} | to {headsign or '?'} | trip {tid}")
            if len(rows) >= limit:
                break
        if not rows:
            return f"No departures from {stop_id} at or after {after}."
        return "\n".join(rows)

    @vf.tool
    def trip_stops(self, trip_id: str) -> str:
        """Show every stop of a trip in order, with arrival and departure times and whether
        boarding or alighting is allowed there."""
        self.state.lookups += 1
        if trip_id not in self.net.trips:
            return f"Unknown trip id {trip_id!r}."
        route, headsign, rows = self.net.trips[trip_id]
        lines = [f"Trip {trip_id}: route {route} to {headsign or '?'}"]
        for s, arr, dep, pick, drop in rows:
            flags = ("" if pick else " (no boarding)") + ("" if drop else " (no alighting)")
            lines.append(f"{fmt_time(arr)}-{fmt_time(dep)} | {self.net.stop_ids[s]} | {self.net.stop_names[s]}{flags}")
        return "\n".join(lines)

    @vf.tool
    def nearby_stops(self, stop_id: str) -> str:
        """List other stops within walking distance of a stop (at most 400 m) and the walking
        time you must allow between alighting at this stop and boarding at the other one."""
        self.state.lookups += 1
        if err := self._unknown(stop_id):
            return err
        s = self.net.index[stop_id]
        paths = self.net.footpaths[s]
        if not paths:
            return f"No other stop within walking distance of {stop_id}."
        return "\n".join(
            f"{self.net.stop_ids[j]} | {self.net.stop_names[j]} | walk {w // 60} min {w % 60:02d} s" for j, w in paths
        )

    @vf.tool
    def submit_itinerary(self, legs: list[LegInput]) -> str:
        """Submit your final itinerary as an ordered list of legs. You can submit only once and
        the conversation ends after submitting."""
        if self.state.submitted is not None:
            return "An itinerary was already submitted."
        self.state.submitted = [leg.model_dump() for leg in legs]
        return f"Itinerary with {len(legs)} leg(s) submitted."


if __name__ == "__main__":
    TransitToolset.run()
