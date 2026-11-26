"""An independent, slow earliest-arrival search used only to cross-check the CSA router."""

import heapq

from gtfs_trip_planner.network import CHANGE_S, INF, Network


def reference_earliest_arrival(net: Network, origin: str, dest: str, depart: int) -> int | None:
    o, d = net.index[origin], net.index[dest]
    ready = {o: depart}
    arrived = INF
    heap = [(depart, o)]
    done = set()
    while heap:
        r, u = heapq.heappop(heap)
        if u in done or r >= arrived:
            continue
        done.add(u)
        for dep, tid, k in net.departures[u]:
            if dep < r:
                continue
            if dep >= arrived:
                break
            for v, arr, _dep, _pick, drop in net.trips[tid][2][k + 1 :]:
                if not drop:
                    continue
                if v == d:
                    arrived = min(arrived, arr)
                cands = [(v, arr + CHANGE_S)] + [(w, arr + ws) for w, ws in net.footpaths[v]]
                for w, t in cands:
                    if t < ready.get(w, INF):
                        ready[w] = t
                        heapq.heappush(heap, (t, w))
    return None if arrived >= INF else arrived
