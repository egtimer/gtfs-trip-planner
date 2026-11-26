"""A transit network for one region on one service day, with an exact router and the
itinerary verifier. Standard library only: the tool server, the task's scoring and the
tests all load the same compact cache file built by `gtfs_trip_planner.build`.

Rules shared by the router and the verifier (they are the task's contract):
- An itinerary is a list of legs; each leg rides one trip from a boarding stop to a later
  alighting stop on that trip. Boarding needs a pickup and alighting a drop-off there.
- The first leg boards at the origin, no earlier than the departure time. The last leg
  alights at the destination.
- Between legs you either change at the same stop, boarding at least CHANGE_S seconds after
  alighting, or walk once to a stop within WALK_RADIUS_M, boarding no earlier than alighting
  plus that stop pair's walking time. No walking at the start or the end.
"""

from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

CHANGE_S = 120
WALK_RADIUS_M = 400.0
WALK_SPEED_MPS = 1.2
WALK_BUFFER_S = 60
INF = 1 << 60


def walk_seconds(meters: float) -> int:
    return WALK_BUFFER_S + math.ceil(meters / WALK_SPEED_MPS)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def fmt_time(seconds: int) -> str:
    h, rem = divmod(int(seconds), 3600)
    return f"{h:02d}:{rem // 60:02d}"


def parse_time(text: str) -> int:
    parts = text.strip().split(":")
    if len(parts) not in (2, 3) or not all(p.isdigit() for p in parts):
        raise ValueError(f"time must look like HH:MM, got {text!r}")
    h, m = int(parts[0]), int(parts[1])
    s = int(parts[2]) if len(parts) == 3 else 0
    if m > 59 or s > 59:
        raise ValueError(f"time must look like HH:MM, got {text!r}")
    return h * 3600 + m * 60 + s


@dataclass(frozen=True)
class Leg:
    trip_id: str
    board_stop_id: str
    alight_stop_id: str


@dataclass(frozen=True)
class Check:
    ok: bool
    arrival: int | None
    reason: str


class Network:
    """Stops are StopPlaces (platforms merged). Each trip is a list of
    (stop index, arrival s, departure s, pickup ok, drop-off ok) rows in travel order."""

    def __init__(self, meta: dict, stops: list[list], trips: dict[str, list]):
        self.meta = meta
        self.stop_ids: list[str] = [s[0] for s in stops]
        self.stop_names: list[str] = [s[1] for s in stops]
        self.stop_lat: list[float] = [float(s[2]) for s in stops]
        self.stop_lon: list[float] = [float(s[3]) for s in stops]
        self.index = {sid: i for i, sid in enumerate(self.stop_ids)}
        # trip_id -> (route name, headsign, rows)
        self.trips: dict[str, tuple[str, str, list[tuple[int, int, int, bool, bool]]]] = {
            tid: (t[0], t[1], [tuple(r) for r in t[2]]) for tid, t in trips.items()
        }
        self._build_indexes()

    @classmethod
    def load(cls, path: str | Path) -> Network:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            raw = json.load(f)
        return cls(raw["meta"], raw["stops"], raw["trips"])

    def save(self, path: str | Path) -> None:
        raw = {
            "meta": self.meta,
            "stops": [
                [sid, name, lat, lon]
                for sid, name, lat, lon in zip(self.stop_ids, self.stop_names, self.stop_lat, self.stop_lon)
            ],
            "trips": {tid: [r, h, [list(x) for x in rows]] for tid, (r, h, rows) in self.trips.items()},
        }
        tmp = Path(f"{path}.part")
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            json.dump(raw, f, separators=(",", ":"))
        tmp.rename(path)

    def _build_indexes(self) -> None:
        n = len(self.stop_ids)
        self.departures: list[list[tuple[int, str, int]]] = [[] for _ in range(n)]
        conns: list[tuple[int, int, int, int, str, bool, bool]] = []
        for tid, (_, _, rows) in self.trips.items():
            for k, (s, _arr, dep, pick, _drop) in enumerate(rows):
                if pick and k + 1 < len(rows):
                    self.departures[s].append((dep, tid, k))
            for k in range(len(rows) - 1):
                s, _, dep, pick, _ = rows[k]
                s2, arr2, _, _, drop2 = rows[k + 1]
                conns.append((dep, arr2, s, s2, tid, pick, drop2))
        for d in self.departures:
            d.sort()
        conns.sort(key=lambda c: (c[0], c[1]))
        self.connections = conns
        # Footpaths between distinct stops within the walking radius (grid bucketing).
        cell = 0.005
        grid: dict[tuple[int, int], list[int]] = {}
        for i in range(n):
            grid.setdefault((int(self.stop_lat[i] / cell), int(self.stop_lon[i] / (cell * 2))), []).append(i)
        self.footpaths: list[list[tuple[int, int]]] = [[] for _ in range(n)]
        for (gy, gx), members in grid.items():
            near = [j for dy in (-1, 0, 1) for dx in (-1, 0, 1) for j in grid.get((gy + dy, gx + dx), [])]
            for i in members:
                for j in near:
                    if i == j:
                        continue
                    d = haversine_m(self.stop_lat[i], self.stop_lon[i], self.stop_lat[j], self.stop_lon[j])
                    if d <= WALK_RADIUS_M:
                        self.footpaths[i].append((j, walk_seconds(d)))
        for f in self.footpaths:
            f.sort(key=lambda x: (x[1], x[0]))
        self._walk = {(i, j): w for i, fp in enumerate(self.footpaths) for j, w in fp}

    def earliest_arrival(self, origin: str, dest: str, depart: int) -> tuple[int, list[Leg]] | None:
        """Connection Scan with change and walking rules; returns (arrival, legs) or None."""
        o, d = self.index[origin], self.index[dest]
        n = len(self.stop_ids)
        ready = [INF] * n  # earliest time one can board at the stop
        ready_src: list[tuple] = [()] * n  # ("origin",) | ("alight", stop) | ("walk", from_stop)
        arrived = [INF] * n  # earliest alighting arrival at the stop
        arrived_by: list[tuple] = [()] * n  # (trip_id, board_stop, board_row, alight_row)
        boarded: dict[str, tuple[int, int]] = {}  # trip -> (board stop, board row)
        row_of: dict[str, int] = {}
        ready[o], ready_src[o] = depart, ("origin",)
        for dep, arr, s, s2, tid, pick, drop in self.connections:
            if dep < depart:
                continue
            if arrived[d] <= dep:
                break
            row = row_of.get(tid)
            if row is None:
                rows = self.trips[tid][2]
                row = next(k for k in range(len(rows)) if rows[k][0] == s and rows[k][2] == dep)
            if tid not in boarded and pick and ready[s] <= dep:
                boarded[tid] = (s, row)
            row_of[tid] = row + 1
            if tid in boarded and drop and arr < arrived[s2]:
                arrived[s2] = arr
                arrived_by[s2] = (tid, boarded[tid][0], boarded[tid][1], row + 1)
                if arr + CHANGE_S < ready[s2]:
                    ready[s2], ready_src[s2] = arr + CHANGE_S, ("alight", s2)
                for j, w in self.footpaths[s2]:
                    if arr + w < ready[j]:
                        ready[j], ready_src[j] = arr + w, ("walk", s2)
        if arrived[d] >= INF:
            return None
        legs: list[Leg] = []
        stop = d
        for _ in range(64):
            tid, b, _brow, _arow = arrived_by[stop]
            legs.append(Leg(tid, self.stop_ids[b], self.stop_ids[stop]))
            src = ready_src[b]
            if src[0] == "origin":
                break
            stop = src[1]
        else:
            raise RuntimeError("journey reconstruction did not terminate")
        legs.reverse()
        return arrived[d], legs

    def best_within_legs(self, origin: str, dest: str, depart: int, max_legs: int, walking: bool = True) -> int | None:
        """Earliest arrival using at most `max_legs` legs (round-based, RAPTOR-like over
        departures); with `walking=False`, changes only happen at the same stop."""
        o, d = self.index[origin], self.index[dest]
        ready = {o: depart}
        marked = {o: depart}
        best = INF
        for _round in range(max_legs):
            nxt: dict[int, int] = {}
            for u, r in marked.items():
                for dep, tid, k in self.departures[u]:
                    if dep < r:
                        continue
                    if dep >= best:
                        break
                    for v, arr, _, _, drop in self.trips[tid][2][k + 1 :]:
                        if arr >= best:
                            break
                        if not drop:
                            continue
                        if v == d:
                            best = arr
                            break
                        cands = [(v, arr + CHANGE_S)]
                        if walking:
                            cands += [(w, arr + ws) for w, ws in self.footpaths[v]]
                        for w, t in cands:
                            if t < nxt.get(w, INF) and t < ready.get(w, INF):
                                nxt[w] = t
            marked = nxt
            ready.update(nxt)
            if not marked:
                break
        return None if best >= INF else best

    def best_direct(self, origin: str, dest: str, depart: int) -> int | None:
        """Earliest arrival using a single trip (no change)."""
        o, d = self.index[origin], self.index[dest]
        best = None
        for dep, tid, k in self.departures[o]:
            if dep < depart:
                continue
            for s, arr, _, _, drop in self.trips[tid][2][k + 1 :]:
                if s == d and drop:
                    if best is None or arr < best:
                        best = arr
                    break
        return best

    def check(self, origin: str, dest: str, depart: int, legs: list[Leg]) -> Check:
        if not legs:
            return Check(False, None, "no legs")
        if len(legs) > 12:
            return Check(False, None, "too many legs")
        prev_stop: int | None = None
        prev_arr = depart
        for n, leg in enumerate(legs, 1):
            if leg.trip_id not in self.trips:
                return Check(False, None, f"leg {n}: unknown trip {leg.trip_id}")
            if leg.board_stop_id not in self.index or leg.alight_stop_id not in self.index:
                return Check(False, None, f"leg {n}: unknown stop")
            b, a = self.index[leg.board_stop_id], self.index[leg.alight_stop_id]
            if prev_stop is None:
                if b != self.index[origin]:
                    return Check(False, None, "first leg must board at the origin")
                earliest = depart
            elif b == prev_stop:
                earliest = prev_arr + CHANGE_S
            elif (prev_stop, b) in self._walk:
                earliest = prev_arr + self._walk[(prev_stop, b)]
            else:
                return Check(False, None, f"leg {n}: boarding stop not reachable from the previous stop")
            rows = self.trips[leg.trip_id][2]
            board_row = next(
                (k for k, r in enumerate(rows) if r[0] == b and r[3] and r[2] >= earliest and k + 1 < len(rows)),
                None,
            )
            if board_row is None:
                return Check(False, None, f"leg {n}: trip does not depart from the boarding stop in time")
            alight = next((r for r in rows[board_row + 1 :] if r[0] == a and r[4]), None)
            if alight is None:
                return Check(False, None, f"leg {n}: trip does not reach the alighting stop after boarding")
            prev_stop, prev_arr = a, alight[1]
        if prev_stop != self.index[dest]:
            return Check(False, None, "last leg must alight at the destination")
        return Check(True, prev_arr, "ok")

    def search(self, query: str, limit: int = 10) -> list[int]:
        q = query.strip().lower()
        if not q:
            return []
        exact = [i for i, nm in enumerate(self.stop_names) if nm.lower() == q]
        prefix = [i for i, nm in enumerate(self.stop_names) if nm.lower().startswith(q) and i not in exact]
        inner = [i for i, nm in enumerate(self.stop_names) if q in nm.lower() and not nm.lower().startswith(q)]
        return (exact + prefix + inner)[:limit]


@lru_cache(maxsize=4)
def load_network(path: str) -> Network:
    return Network.load(path)
