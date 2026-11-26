"""Download the pinned Entur GTFS snapshot and build one network cache per region and day."""

from __future__ import annotations

import datetime as dt
import logging
import os
from pathlib import Path

from gtfs_trip_planner.network import Network

logger = logging.getLogger(__name__)

HF_REPO = "knuthp/GTFS_Entur"
HF_REVISION = "be37f77a4b9405535f00e72ebdb26617650bb555"
TABLES = ("agency", "routes", "trips", "calendar", "calendar_dates", "stops", "stop_times")
SERVICE_DATE = "2026-04-15"

# Region -> Entur authority ids. Train and eval regions share no stops or trips.
REGIONS: dict[str, dict] = {
    "bergen": {"name": "Skyss (Bergen and Vestland)", "agencies": ["SKY:Authority:SKY"], "split": "train"},
    "stavanger": {"name": "Kolumbus (Stavanger and Rogaland)", "agencies": ["KOL:Authority:8"], "split": "train"},
    "kristiansand": {"name": "AKT (Kristiansand and Agder)", "agencies": ["AKT:Authority:AKT_ID"], "split": "train"},
    "drammen": {"name": "Brakar (Drammen and Buskerud)", "agencies": ["BRA:Authority:4"], "split": "train"},
    "ostfold": {"name": "Ostfold kollektivtrafikk", "agencies": ["OST:Authority:1"], "split": "train"},
    "trondheim": {"name": "AtB (Trondheim and Trondelag)", "agencies": ["ATB:Authority:2"], "split": "eval"},
    "tromso": {"name": "Troms fylkestrafikk", "agencies": ["TRO:Authority:1"], "split": "eval"},
    "oslo": {"name": "Ruter (Oslo and Akershus)", "agencies": ["RUT:Authority:RUT"], "split": "eval"},
}


def cache_dir() -> Path:
    root = Path(os.environ.get("GTFS_TRIP_PLANNER_CACHE", str(Path.home() / ".cache" / "gtfs_trip_planner")))
    path = root / HF_REVISION[:7]
    path.mkdir(parents=True, exist_ok=True)
    return path


def network_path(region: str, date: str = SERVICE_DATE) -> Path:
    return cache_dir() / f"{region}_{date}.json.gz"


def _download_tables() -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    return {
        t: Path(hf_hub_download(HF_REPO, f"{t}.parquet", repo_type="dataset", revision=HF_REVISION)) for t in TABLES
    }


def _active_services(cal, cal_dates, date: str) -> set[str]:
    day = dt.date.fromisoformat(date)
    ds = int(day.strftime("%Y%m%d"))
    weekday = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"][day.weekday()]
    start, end = cal.start_date.astype(int), cal.end_date.astype(int)
    services = set(cal[(start <= ds) & (end >= ds) & (cal[weekday] == 1)].service_id)
    exc = cal_dates[cal_dates.date.astype(int) == ds]
    services |= set(exc[exc.exception_type == 1].service_id)
    services -= set(exc[exc.exception_type == 2].service_id)
    return services


def _seconds(series):
    parts = series.str.split(":", expand=True).astype(int)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def build_networks(regions: list[str], date: str = SERVICE_DATE) -> dict[str, Path]:
    """Build (or reuse) the cache file of each region; returns region -> path."""
    todo = [r for r in regions if not network_path(r, date).exists()]
    if todo:
        import pandas as pd

        logger.info("building transit networks for %s on %s", ", ".join(todo), date)
        paths = _download_tables()
        routes = pd.read_parquet(
            paths["routes"], columns=["route_id", "agency_id", "route_short_name", "route_long_name"]
        )
        trips = pd.read_parquet(paths["trips"], columns=["route_id", "trip_id", "service_id", "trip_headsign"])
        services = _active_services(pd.read_parquet(paths["calendar"]), pd.read_parquet(paths["calendar_dates"]), date)
        trips = trips[trips.service_id.isin(services)].merge(routes, on="route_id")
        stops = pd.read_parquet(
            paths["stops"], columns=["stop_id", "stop_name", "stop_lat", "stop_lon", "parent_station"]
        )
        place = stops.set_index("stop_id").parent_station.fillna(pd.Series(stops.stop_id.values, index=stops.stop_id))
        info = stops.set_index("stop_id")[["stop_name", "stop_lat", "stop_lon"]]
        cols = ["trip_id", "stop_id", "arrival_time", "departure_time", "stop_sequence", "pickup_type", "drop_off_type"]
        st_all = pd.read_parquet(paths["stop_times"], columns=cols)
        for region in todo:
            spec = REGIONS[region]
            rtrips = trips[trips.agency_id.isin(spec["agencies"])]
            st = st_all[st_all.trip_id.isin(set(rtrips.trip_id))].copy()
            st["place"] = st.stop_id.map(place)
            st["arr"] = _seconds(st.arrival_time)
            st["dep"] = _seconds(st.departure_time)
            st["pickok"] = st.pickup_type.fillna(0).astype(int) == 0
            st["dropok"] = st.drop_off_type.fillna(0).astype(int) == 0
            st = st.sort_values(["trip_id", "stop_sequence"])
            used = sorted(set(st.place))
            stop_rows = [
                [
                    sid,
                    str(info.at[sid, "stop_name"]),
                    round(float(info.at[sid, "stop_lat"]), 6),
                    round(float(info.at[sid, "stop_lon"]), 6),
                ]
                for sid in used
            ]
            idx = {sid: i for i, sid in enumerate(used)}
            meta_by_trip = rtrips.set_index("trip_id")
            trip_rows: dict[str, list] = {}
            for tid, g in st.groupby("trip_id", sort=False):
                rows, last = [], None
                for p, a, d, pk, dr in zip(g["place"], g["arr"], g["dep"], g["pickok"], g["dropok"]):
                    i = idx[p]
                    if last is not None and last[0] == i:
                        # Two consecutive platforms of the same stop place: merge them.
                        rows[-1] = [i, last[1], int(d), bool(last[3] or pk), bool(last[4] or dr)]
                    else:
                        rows.append([i, int(a), int(d), bool(pk), bool(dr)])
                    last = rows[-1]
                if len(rows) < 2:
                    continue
                m = meta_by_trip.loc[tid]
                name = (
                    m.route_short_name
                    if isinstance(m.route_short_name, str) and m.route_short_name
                    else m.route_long_name
                )
                trip_rows[tid] = [str(name), str(m.trip_headsign) if isinstance(m.trip_headsign, str) else "", rows]
            meta = {
                "region": region,
                "region_name": spec["name"],
                "date": date,
                "source": f"hf://datasets/{HF_REPO}@{HF_REVISION}",
                "attribution": "Data from Entur AS (entur.org), NLOD 2.0",
            }
            Network(meta, stop_rows, trip_rows).save(network_path(region, date))
            logger.info("built %s: %d stops, %d trips", region, len(stop_rows), len(trip_rows))
    return {r: network_path(r, date) for r in regions}
