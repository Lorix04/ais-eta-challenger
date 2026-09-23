"""M17C causal historical maritime corridor / knowledge-graph utilities.

The graph is built exclusively from AIS position transitions.  For every query,
callers must filter out blocked/validation MMSIs and pass only transition events
recorded at or before the query timestamp.  Edge direction is historical vessel
motion.  Edge speed is selected through a deterministic hierarchical fallback:

    edge + vessel type + 6h time bin  -> edge + vessel type
    -> edge + 6h time bin -> edge global.

No ETA/target value is used to construct graph topology, edge weights, snapping
or support.  The resulting travel time is therefore a target-free physical /
historical-corridor expert.  It is not claimed to be an ENC-authoritative route.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

M17C_VERSION = "m17c-sicily-historical-corridor-graph-v1-20260922"
M17C_GRID_DEG = 0.20
M17C_LAT_MIN = 30.0
M17C_LAT_MAX = 45.0
M17C_LON_MIN = -10.0
M17C_LON_MAX = 40.0
M17C_MAX_GAP_MIN = 15.0
M17C_MAX_TRANSITION_NM = 30.0
M17C_MIN_SPEED_KN = 1.0
M17C_MAX_SPEED_KN = 30.0
M17C_MIN_CONTEXT_SUPPORT = 2
M17C_MAX_SNAP_NM = 40.0
M17C_TIME_BIN_HOURS = 6
M17C_MIN_GRAPH_SUPPORTED_ROWS = 20


def _haversine_nm(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1 = np.asarray(lat1, dtype=float)
    lon1 = np.asarray(lon1, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dlat = p2 - p1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * 3440.065 * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def _cell_center(ilat: int, ilon: int, resolution_deg: float = M17C_GRID_DEG) -> tuple[float, float]:
    return ((int(ilat) + 0.5) * resolution_deg, (int(ilon) + 0.5) * resolution_deg)


def prepare_transition_events(
    states: pd.DataFrame,
    *,
    owner_ship_type: dict[int, str] | None = None,
    resolution_deg: float = M17C_GRID_DEG,
) -> pd.DataFrame:
    """Convert raw AIS states to target-free directed cell-transition events."""
    required = {"mmsi", "recorded_at", "lat_clean", "lon_clean", "sog_clean", "position_valid"}
    missing = required - set(states.columns)
    if missing:
        raise ValueError(f"M17C missing state columns: {sorted(missing)}")
    x = states.loc[
        states["position_valid"].fillna(False).astype(bool)
        & pd.to_numeric(states["lat_clean"], errors="coerce").between(M17C_LAT_MIN, M17C_LAT_MAX)
        & pd.to_numeric(states["lon_clean"], errors="coerce").between(M17C_LON_MIN, M17C_LON_MAX),
        ["mmsi", "recorded_at", "lat_clean", "lon_clean", "sog_clean"],
    ].copy()
    x["recorded_at"] = pd.to_datetime(x["recorded_at"], errors="coerce")
    x = x.dropna(subset=["recorded_at", "lat_clean", "lon_clean"]).sort_values(["mmsi", "recorded_at"], kind="mergesort")
    x["ilat"] = np.floor(pd.to_numeric(x["lat_clean"], errors="coerce") / resolution_deg).astype(int)
    x["ilon"] = np.floor(pd.to_numeric(x["lon_clean"], errors="coerce") / resolution_deg).astype(int)
    grp = x.groupby("mmsi", sort=False)
    for col in ["ilat", "ilon", "lat_clean", "lon_clean", "sog_clean", "recorded_at"]:
        x[f"prev_{col}"] = grp[col].shift()
    x["gap_min"] = (x["recorded_at"] - x["prev_recorded_at"]).dt.total_seconds() / 60.0
    changed = x["ilat"].ne(x["prev_ilat"]) | x["ilon"].ne(x["prev_ilon"])
    ev = x.loc[x["gap_min"].gt(0) & x["gap_min"].le(M17C_MAX_GAP_MIN) & changed].copy()
    if ev.empty:
        return pd.DataFrame(columns=[
            "mmsi", "recorded_at", "src_ilat", "src_ilon", "dst_ilat", "dst_ilon",
            "speed_kn", "observed_step_nm", "time_bin", "ship_type",
        ])
    ev["observed_step_nm"] = _haversine_nm(
        ev["prev_lat_clean"], ev["prev_lon_clean"], ev["lat_clean"], ev["lon_clean"]
    )
    a = pd.to_numeric(ev["prev_sog_clean"], errors="coerce").to_numpy(float)
    b = pd.to_numeric(ev["sog_clean"], errors="coerce").to_numpy(float)
    with np.errstate(invalid="ignore"):
        ev["speed_kn"] = np.nanmean(np.vstack([a, b]), axis=0)
    ev = ev.loc[
        ev["observed_step_nm"].between(0.01, M17C_MAX_TRANSITION_NM)
        & pd.to_numeric(ev["speed_kn"], errors="coerce").between(M17C_MIN_SPEED_KN, M17C_MAX_SPEED_KN)
    ].copy()
    ev["src_ilat"] = ev["prev_ilat"].astype(int)
    ev["src_ilon"] = ev["prev_ilon"].astype(int)
    ev["dst_ilat"] = ev["ilat"].astype(int)
    ev["dst_ilon"] = ev["ilon"].astype(int)
    ev["time_bin"] = (ev["recorded_at"].dt.hour // M17C_TIME_BIN_HOURS).astype(int)
    owner_ship_type = owner_ship_type or {}
    ev["ship_type"] = ev["mmsi"].astype(int).map(owner_ship_type).fillna("UNLABELED").astype(str)
    cols = [
        "mmsi", "recorded_at", "src_ilat", "src_ilon", "dst_ilat", "dst_ilon",
        "speed_kn", "observed_step_nm", "time_bin", "ship_type",
    ]
    return ev[cols].sort_values(["recorded_at", "mmsi", "src_ilat", "src_ilon", "dst_ilat", "dst_ilon"], kind="mergesort").reset_index(drop=True)


def _aggregate_edge_stats(events: pd.DataFrame, query_ship_type: str, query_time_bin: int) -> pd.DataFrame:
    keys = ["src_ilat", "src_ilon", "dst_ilat", "dst_ilon"]
    base = events.groupby(keys, sort=True)["speed_kn"].agg(edge_speed_global="median", edge_count_global="size").reset_index()
    timed = (
        events.loc[events["time_bin"].eq(int(query_time_bin))]
        .groupby(keys, sort=True)["speed_kn"].agg(edge_speed_time="median", edge_count_time="size").reset_index()
    )
    typed = (
        events.loc[events["ship_type"].eq(str(query_ship_type))]
        .groupby(keys, sort=True)["speed_kn"].agg(edge_speed_type="median", edge_count_type="size").reset_index()
    )
    both = (
        events.loc[events["time_bin"].eq(int(query_time_bin)) & events["ship_type"].eq(str(query_ship_type))]
        .groupby(keys, sort=True)["speed_kn"].agg(edge_speed_type_time="median", edge_count_type_time="size").reset_index()
    )
    out = base.merge(timed, on=keys, how="left").merge(typed, on=keys, how="left").merge(both, on=keys, how="left")
    out["selected_speed_kn"] = out["edge_speed_global"].astype(float)
    out["selected_support"] = out["edge_count_global"].astype(int)
    out["selected_level"] = "edge_global"
    for speed_col, count_col, level in [
        ("edge_speed_time", "edge_count_time", "edge_time"),
        ("edge_speed_type", "edge_count_type", "edge_type"),
        ("edge_speed_type_time", "edge_count_type_time", "edge_type_time"),
    ]:
        mask = pd.to_numeric(out[count_col], errors="coerce").fillna(0).ge(M17C_MIN_CONTEXT_SUPPORT)
        out.loc[mask, "selected_speed_kn"] = pd.to_numeric(out.loc[mask, speed_col], errors="coerce")
        out.loc[mask, "selected_support"] = pd.to_numeric(out.loc[mask, count_col], errors="coerce").fillna(0).astype(int)
        out.loc[mask, "selected_level"] = level
    out["selected_speed_kn"] = out["selected_speed_kn"].clip(M17C_MIN_SPEED_KN, M17C_MAX_SPEED_KN)
    return out


@dataclass(frozen=True)
class GraphPrediction:
    supported: bool
    prediction_h: float
    reason: str
    source_snap_nm: float
    destination_snap_nm: float
    path_edges: int
    path_distance_nm: float
    graph_path_h: float
    source_snap_h: float
    destination_snap_h: float
    graph_nodes: int
    graph_edges: int
    min_path_support: int
    median_path_support: float
    contextual_edge_share: float
    graph_median_speed_kn: float
    path_rows: tuple[dict, ...]


def predict_corridor_eta(
    events: pd.DataFrame,
    *,
    query_mmsi: int,
    query_at: pd.Timestamp,
    query_lat: float,
    query_lon: float,
    destination_lat: float,
    destination_lon: float,
    query_ship_type: str,
    recent_speed_kn: float,
    resolution_deg: float = M17C_GRID_DEG,
    max_snap_nm: float = M17C_MAX_SNAP_NM,
) -> GraphPrediction:
    """Query one causal directed historical corridor graph.

    `events` must already exclude forbidden MMSIs. This function additionally
    enforces the timestamp cutoff as a second line of defence.
    """
    query_at = pd.Timestamp(query_at)
    ev = events.loc[pd.to_datetime(events["recorded_at"]) <= query_at].copy()
    if len(ev) < 2:
        return _unsupported("insufficient_past_transitions")
    if int(query_mmsi) in set(ev["mmsi"].astype(int)):
        # Owner self-history is allowed only when the caller intentionally left it
        # in the graph. M17C outer-fold builder excludes every outer-valid owner.
        pass
    time_bin = int(query_at.hour // M17C_TIME_BIN_HOURS)
    stats = _aggregate_edge_stats(ev, str(query_ship_type), time_bin)
    if stats.empty:
        return _unsupported("empty_graph")

    src_cells = list(zip(stats["src_ilat"].astype(int), stats["src_ilon"].astype(int)))
    dst_cells = list(zip(stats["dst_ilat"].astype(int), stats["dst_ilon"].astype(int)))
    nodes = sorted(set(src_cells) | set(dst_cells))
    if not nodes:
        return _unsupported("empty_nodes")
    node_index = {node: i for i, node in enumerate(nodes)}
    coords = np.asarray([_cell_center(*node, resolution_deg) for node in nodes], dtype=float)
    coslat = float(np.cos(np.deg2rad(float(query_lat))))
    tree = cKDTree(np.column_stack([coords[:, 0], coords[:, 1] * coslat]))
    source_deg, source_idx = tree.query([float(query_lat), float(query_lon) * coslat])
    dest_deg, dest_idx = tree.query([float(destination_lat), float(destination_lon) * coslat])
    source_snap_nm = float(source_deg * 60.0)
    destination_snap_nm = float(dest_deg * 60.0)
    if source_snap_nm > max_snap_nm:
        return _unsupported("source_outside_graph", source_snap_nm=source_snap_nm, destination_snap_nm=destination_snap_nm, graph_nodes=len(nodes), graph_edges=len(stats))
    if destination_snap_nm > max_snap_nm:
        return _unsupported("destination_outside_graph", source_snap_nm=source_snap_nm, destination_snap_nm=destination_snap_nm, graph_nodes=len(nodes), graph_edges=len(stats))

    rows: list[int] = []
    cols: list[int] = []
    weights: list[float] = []
    edge_meta: dict[tuple[int, int], dict] = {}
    for row in stats.itertuples(index=False):
        src = (int(row.src_ilat), int(row.src_ilon))
        dst = (int(row.dst_ilat), int(row.dst_ilon))
        u, v = node_index[src], node_index[dst]
        lat1, lon1 = _cell_center(*src, resolution_deg)
        lat2, lon2 = _cell_center(*dst, resolution_deg)
        distance_nm = float(_haversine_nm(lat1, lon1, lat2, lon2))
        speed_kn = max(float(row.selected_speed_kn), 2.0)
        travel_h = distance_nm / speed_kn
        rows.append(u); cols.append(v); weights.append(travel_h)
        edge_meta[(u, v)] = {
            "src_ilat": src[0], "src_ilon": src[1], "dst_ilat": dst[0], "dst_ilon": dst[1],
            "distance_nm": distance_nm, "speed_kn": speed_kn, "travel_h": travel_h,
            "support": int(row.selected_support), "level": str(row.selected_level),
        }
    matrix = csr_matrix((weights, (rows, cols)), shape=(len(nodes), len(nodes)) )
    dist, predecessors = dijkstra(matrix, directed=True, indices=int(source_idx), return_predecessors=True)
    destination_idx = int(dest_idx)
    if not np.isfinite(dist[destination_idx]):
        return _unsupported("no_directed_path", source_snap_nm=source_snap_nm, destination_snap_nm=destination_snap_nm, graph_nodes=len(nodes), graph_edges=len(stats))

    path: list[dict] = []
    cur = destination_idx
    source_idx = int(source_idx)
    safety = 0
    while cur != source_idx:
        prev = int(predecessors[cur])
        if prev < 0 or (prev, cur) not in edge_meta:
            return _unsupported("path_reconstruction_failed", source_snap_nm=source_snap_nm, destination_snap_nm=destination_snap_nm, graph_nodes=len(nodes), graph_edges=len(stats))
        path.append(dict(edge_meta[(prev, cur)]))
        cur = prev
        safety += 1
        if safety > len(nodes) + 1:
            return _unsupported("path_cycle_guard", source_snap_nm=source_snap_nm, destination_snap_nm=destination_snap_nm, graph_nodes=len(nodes), graph_edges=len(stats))
    path.reverse()
    median_speed = float(np.median(pd.to_numeric(stats["selected_speed_kn"], errors="coerce").dropna()))
    median_speed = max(median_speed, 2.0)
    recent_speed = float(recent_speed_kn) if np.isfinite(recent_speed_kn) else median_speed
    recent_speed = min(max(recent_speed, 2.0), 25.0)
    source_snap_h = source_snap_nm / recent_speed
    destination_snap_h = destination_snap_nm / median_speed
    graph_path_h = float(dist[destination_idx])
    prediction_h = graph_path_h + source_snap_h + destination_snap_h
    supports = [int(x["support"]) for x in path]
    levels = [str(x["level"]) for x in path]
    return GraphPrediction(
        supported=True,
        prediction_h=float(prediction_h),
        reason="supported",
        source_snap_nm=source_snap_nm,
        destination_snap_nm=destination_snap_nm,
        path_edges=len(path),
        path_distance_nm=float(sum(float(x["distance_nm"]) for x in path)),
        graph_path_h=graph_path_h,
        source_snap_h=float(source_snap_h),
        destination_snap_h=float(destination_snap_h),
        graph_nodes=len(nodes),
        graph_edges=len(stats),
        min_path_support=int(min(supports)) if supports else 0,
        median_path_support=float(np.median(supports)) if supports else 0.0,
        contextual_edge_share=float(np.mean([lvl != "edge_global" for lvl in levels])) if levels else 0.0,
        graph_median_speed_kn=median_speed,
        path_rows=tuple(path),
    )


def _unsupported(
    reason: str,
    *,
    source_snap_nm: float = math.nan,
    destination_snap_nm: float = math.nan,
    graph_nodes: int = 0,
    graph_edges: int = 0,
) -> GraphPrediction:
    return GraphPrediction(
        supported=False, prediction_h=math.nan, reason=reason,
        source_snap_nm=float(source_snap_nm), destination_snap_nm=float(destination_snap_nm),
        path_edges=0, path_distance_nm=math.nan, graph_path_h=math.nan,
        source_snap_h=math.nan, destination_snap_h=math.nan,
        graph_nodes=int(graph_nodes), graph_edges=int(graph_edges),
        min_path_support=0, median_path_support=math.nan, contextual_edge_share=math.nan,
        graph_median_speed_kn=math.nan, path_rows=tuple(),
    )
