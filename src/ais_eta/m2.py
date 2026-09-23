"""M2: train-only empirical route representations for the AIS ETA challenger.

M2 asks one narrow question: does historical route geometry add stable out-of-fold
ETA signal beyond a geodesic-distance / robust-speed baseline?

The implementation is deliberately low-capacity and leakage-aware:
- Catania keeps its M1 chronological holdout untouched;
- Augusta receives its own locked chronological holdout;
- primary M2 estimates use expanding temporal folds;
- every route prototype and historical-path index is fitted only on calls whose
  arrival precedes the validation block;
- validation prefixes are projected onto those training-only representations;
- final chronological test calls are never scored in M2.

Targets remain research-gate entry events.  Nothing in this module upgrades them
to official PBP, ATA berth or all-fast timestamps.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .m0d import CataniaGeometryV1, _haversine_km
from .m1 import HORIZON_BINS_H, HORIZON_LABELS, KM_PER_NM
from .m1x import AugustaGeometryV1


@dataclass(frozen=True)
class M2Config:
    """Frozen M2 design choices.

    The values are engineering choices fixed before the formal M2 run, not
    hyperparameters tuned against ETA performance.
    """

    augusta_final_chronological_calls: int = 10
    temporal_folds: int = 3
    catania_warmup_calls: int = 8
    augusta_warmup_calls: int = 15
    max_route_radius_km: float = 120.0
    max_ring_crossing_gap_s: float = 10 * 60.0
    min_level_support_calls: int = 3
    min_family_calls: int = 4
    knn_k: int = 3
    speed_floor_kn: float = 1.0

    # Route prototype levels: intentionally coarser offshore and denser close
    # to the gate, where approach geometry changes fastest.
    prototype_levels_km: tuple[float, ...] = (
        120.0, 100.0, 80.0, 60.0, 40.0, 30.0, 20.0, 15.0,
        10.0, 7.5, 5.0, 3.0, 2.0, 1.0, 0.5,
    )
    # Historical individual paths retain more rings for route-kNN retrieval.
    knn_levels_km: tuple[float, ...] = (
        120.0, 110.0, 100.0, 90.0, 80.0, 70.0, 60.0, 50.0,
        40.0, 35.0, 30.0, 25.0, 20.0, 15.0, 12.0, 10.0,
        8.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5,
    )


@dataclass(frozen=True)
class M2GateCriteria:
    """Pre-specified engineering gate for retaining route complexity.

    Route geometry should help in the navigation-dominated <=24 h regime,
    improve a clear majority of port x temporal-fold slices, and generalize to
    vessels unseen by the corresponding fold.  Overall OOF performance must not
    materially degrade when very-long-horizon waiting dominates the error.
    """

    min_under24_mae_improvement_fraction: float = 0.10
    min_fold_win_fraction: float = 2.0 / 3.0
    min_cold_under24_improvement_fraction: float = 0.05
    max_overall_mae_degradation_fraction: float = 0.05


ROUTE_FAMILY_LABELS = ("NNE", "E", "SSE", "OTHER")


def target_gate(port: str, entrance: str | None = None) -> tuple[float, float]:
    """Return (lon, lat) for the versioned research-gate midpoint."""
    p = str(port).upper()
    if p == "CATANIA":
        g = CataniaGeometryV1()
        return (
            (g.gate_west[0] + g.gate_east[0]) / 2.0,
            (g.gate_west[1] + g.gate_east[1]) / 2.0,
        )
    if p == "AUGUSTA":
        name = str(entrance or "LEVANTE").upper()
        gates = {g.name: g for g in AugustaGeometryV1().gates}
        if name not in gates:
            raise ValueError(f"Unknown Augusta entrance: {name}")
        return gates[name].midpoint
    raise ValueError(f"Unknown port: {port}")


def bearing_from_gate_deg(gate_lon: float, gate_lat: float, lon: float, lat: float) -> float:
    """Initial great-circle bearing from gate toward the vessel point."""
    phi1 = np.radians(float(gate_lat))
    phi2 = np.radians(float(lat))
    dl = np.radians(float(lon) - float(gate_lon))
    y = np.sin(dl) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dl)
    return float((np.degrees(np.arctan2(y, x)) + 360.0) % 360.0)


def route_family_from_bearing(bearing_deg: float) -> str:
    """Fixed east-coast approach sectors; not learned from ETA outcomes."""
    b = float(bearing_deg) % 360.0
    if 0.0 <= b < 60.0:
        return "NNE"
    if 60.0 <= b < 120.0:
        return "E"
    if 120.0 <= b < 180.0:
        return "SSE"
    return "OTHER"


def assign_m2_splits(
    catania_cohort: pd.DataFrame,
    augusta_cohort: pd.DataFrame,
    catania_m1_splits: pd.DataFrame,
    config: M2Config | None = None,
) -> pd.DataFrame:
    """Freeze combined development/final sets and expanding temporal folds.

    Catania's M1 chronological holdout is preserved exactly.  Augusta reserves
    its last ``augusta_final_chronological_calls``.  Within each port's
    development set, the first warm-up calls are training-only; the remainder
    are partitioned into contiguous validation blocks.  Fold ``k`` may train on
    all development calls strictly earlier than that block, including earlier
    validation blocks, which simulates an expanding production history.
    """
    cfg = config or M2Config()

    cat = catania_cohort[["session_id", "mmsi", "name", "ground_truth_time"]].copy()
    cat["port"] = "CATANIA"
    cat["inbound_entrance"] = "CATANIA"
    cat = cat.merge(
        catania_m1_splits[["session_id", "m1_split"]],
        on="session_id",
        how="left",
        validate="one_to_one",
    )
    if cat["m1_split"].isna().any():
        raise ValueError("Catania M1 split coverage mismatch")
    cat["m2_split"] = cat["m1_split"]

    aug = augusta_cohort[[
        "session_id", "mmsi", "name", "ground_truth_time", "inbound_entrance"
    ]].copy()
    aug["port"] = "AUGUSTA"
    aug["ground_truth_time"] = pd.to_datetime(aug["ground_truth_time"], errors="raise")
    aug = aug.sort_values(["ground_truth_time", "session_id"], kind="mergesort").reset_index(drop=True)
    if len(aug) <= cfg.augusta_final_chronological_calls:
        raise ValueError("Augusta cohort too small for requested M2 final holdout")
    aug["m2_split"] = "development"
    aug.loc[aug.index[-cfg.augusta_final_chronological_calls :], "m2_split"] = "final_chronological_test"

    calls = pd.concat([cat, aug], ignore_index=True, sort=False)
    calls["ground_truth_time"] = pd.to_datetime(calls["ground_truth_time"], errors="raise")
    calls["m2_temporal_fold"] = pd.Series(pd.NA, index=calls.index, dtype="Int64")
    calls["m2_role"] = np.where(
        calls["m2_split"].eq("final_chronological_test"),
        "final_chronological_test",
        "warmup_train",
    )

    warmups = {
        "CATANIA": int(cfg.catania_warmup_calls),
        "AUGUSTA": int(cfg.augusta_warmup_calls),
    }
    for port, warmup_n in warmups.items():
        dev_idx = calls.index[(calls["port"].eq(port)) & calls["m2_split"].eq("development")]
        dev = calls.loc[dev_idx].sort_values(["ground_truth_time", "session_id"], kind="mergesort")
        if len(dev) <= warmup_n:
            raise ValueError(f"Not enough {port} development calls after warm-up")
        validation = dev.iloc[warmup_n:].copy()
        chunks = np.array_split(np.arange(len(validation)), cfg.temporal_folds)
        for fold, local_positions in enumerate(chunks, start=1):
            if len(local_positions) == 0:
                continue
            idx = validation.iloc[local_positions].index
            calls.loc[idx, "m2_temporal_fold"] = fold
            calls.loc[idx, "m2_role"] = "oof_validation"

        final = calls[(calls["port"].eq(port)) & calls["m2_split"].eq("final_chronological_test")]
        if len(final):
            if pd.Timestamp(dev["ground_truth_time"].max()) >= pd.Timestamp(final["ground_truth_time"].min()):
                raise AssertionError(f"{port} final holdout is not strictly chronological")

    calls["m2_call_index_port"] = (
        calls.sort_values(["port", "ground_truth_time", "session_id"], kind="mergesort")
        .groupby("port")
        .cumcount()
        .reindex(calls.index)
        .astype(int)
    )
    return calls.sort_values(["ground_truth_time", "port", "session_id"], kind="mergesort").reset_index(drop=True)


def _quality_track(
    states: pd.DataFrame,
    mmsi: int,
    start: pd.Timestamp,
    target: pd.Timestamp,
) -> pd.DataFrame:
    """Extract a quality-controlled pre-arrival trajectory for route fitting."""
    q = states[
        states["mmsi"].astype(int).eq(int(mmsi))
        & (states["recorded_at"] >= pd.Timestamp(start))
        & (states["recorded_at"] < pd.Timestamp(target))
        & states["position_valid"].astype(bool)
        & ~states["position_jump_candidate"].astype(bool)
        & ~states["stale_risk_level"].astype(str).eq("HIGH")
    ].copy()
    if q.empty:
        return q
    q = q.sort_values(["recorded_at", "id"], kind="mergesort")
    # Repeated provider states do not add route geometry.
    same = q["lon_clean"].eq(q["lon_clean"].shift()) & q["lat_clean"].eq(q["lat_clean"].shift())
    return q.loc[~same].reset_index(drop=True)


def _add_gate_distance(track: pd.DataFrame, gate_lon: float, gate_lat: float) -> pd.DataFrame:
    q = track.copy()
    if q.empty:
        q["route_gate_distance_km"] = pd.Series(dtype=float)
        return q
    q["route_gate_distance_km"] = _haversine_km(
        q["lat_clean"].to_numpy(dtype=float),
        q["lon_clean"].to_numpy(dtype=float),
        float(gate_lat),
        float(gate_lon),
    )
    return q


def _final_ring_crossing(
    track: pd.DataFrame,
    level_km: float,
    max_gap_s: float,
) -> tuple[float, float] | None:
    """Last well-observed inward crossing of a distance ring before arrival."""
    if len(track) < 2:
        return None
    d = track["route_gate_distance_km"].to_numpy(dtype=float)
    lon = track["lon_clean"].to_numpy(dtype=float)
    lat = track["lat_clean"].to_numpy(dtype=float)
    t = pd.to_datetime(track["recorded_at"]).to_numpy(dtype="datetime64[ns]")
    candidates = np.flatnonzero((d[:-1] >= float(level_km)) & (d[1:] < float(level_km)))
    if not len(candidates):
        return None
    for i in candidates[::-1]:
        dt_s = float((t[i + 1] - t[i]) / np.timedelta64(1, "s"))
        if not (0.0 < dt_s <= float(max_gap_s)):
            continue
        denom = d[i] - d[i + 1]
        w = 0.5 if denom == 0 else (d[i] - float(level_km)) / denom
        w = float(np.clip(w, 0.0, 1.0))
        return (
            float(lon[i] + w * (lon[i + 1] - lon[i])),
            float(lat[i] + w * (lat[i + 1] - lat[i])),
        )
    return None


def route_path_from_track(
    track: pd.DataFrame,
    gate_lon: float,
    gate_lat: float,
    levels_km: Iterable[float],
    max_gap_s: float,
) -> list[dict]:
    """Represent one historical voyage by its final inward ring crossings."""
    q = _add_gate_distance(track, gate_lon, gate_lat)
    points: list[dict] = []
    for level in levels_km:
        p = _final_ring_crossing(q, float(level), max_gap_s)
        if p is not None:
            points.append({"lon": p[0], "lat": p[1], "level_km": float(level)})
    points.append({"lon": float(gate_lon), "lat": float(gate_lat), "level_km": 0.0})
    points = sorted(points, key=lambda x: -float(x["level_km"]))
    # Remove effectively duplicate neighbouring vertices.
    deduped: list[dict] = []
    for p in points:
        if not deduped or abs(p["lon"] - deduped[-1]["lon"]) > 1e-8 or abs(p["lat"] - deduped[-1]["lat"]) > 1e-8:
            deduped.append(p)
    return deduped


def training_call_family(path: list[dict], gate_lon: float, gate_lat: float) -> str:
    if len(path) < 2:
        return "OTHER"
    far = path[0]
    return route_family_from_bearing(
        bearing_from_gate_deg(gate_lon, gate_lat, far["lon"], far["lat"])
    )


def _fit_prototype_from_paths(
    paths: dict[str, list[dict]],
    levels_km: Iterable[float],
    gate_lon: float,
    gate_lat: float,
    min_level_support_calls: int,
) -> dict | None:
    if not paths:
        return None
    points: list[dict] = []
    for level in levels_km:
        vals = []
        for path in paths.values():
            hit = next((p for p in path if np.isclose(float(p["level_km"]), float(level))), None)
            if hit is not None:
                vals.append((float(hit["lon"]), float(hit["lat"])))
        if len(vals) >= int(min_level_support_calls):
            a = np.asarray(vals, dtype=float)
            points.append(
                {
                    "lon": float(np.median(a[:, 0])),
                    "lat": float(np.median(a[:, 1])),
                    "level_km": float(level),
                    "support_calls": int(len(vals)),
                }
            )
    points.append(
        {
            "lon": float(gate_lon),
            "lat": float(gate_lat),
            "level_km": 0.0,
            "support_calls": int(len(paths)),
        }
    )
    points = sorted(points, key=lambda x: -float(x["level_km"]))
    if len(points) < 3:
        return None
    return {"points": points, "training_calls": int(len(paths))}


def fit_train_only_route_model(
    states: pd.DataFrame,
    train_calls: pd.DataFrame,
    voyage_start_lookup: dict[str, pd.Timestamp],
    config: M2Config | None = None,
) -> dict:
    """Fit port-specific prototype and individual historical paths.

    ``train_calls`` must contain one port/gate target and must already exclude
    validation/final calls.  The returned object is JSON-serializable.
    """
    cfg = config or M2Config()
    if train_calls.empty:
        raise ValueError("Cannot fit route model with zero training calls")
    ports = train_calls["port"].astype(str).str.upper().unique()
    if len(ports) != 1:
        raise ValueError("fit_train_only_route_model expects one port at a time")
    port = str(ports[0])

    individual: dict[str, dict] = {}
    prototype_source_paths: dict[str, list[dict]] = {}
    families: dict[str, str] = {}
    gate_keys = set()

    for call in train_calls.sort_values(["ground_truth_time", "session_id"]).itertuples(index=False):
        sid = str(call.session_id)
        gate_lon, gate_lat = target_gate(port, getattr(call, "inbound_entrance", None))
        gate_keys.add((round(gate_lon, 7), round(gate_lat, 7)))
        start = pd.Timestamp(voyage_start_lookup[sid])
        target = pd.Timestamp(call.ground_truth_time)
        track = _quality_track(states, int(call.mmsi), start, target)
        prototype_path = route_path_from_track(
            track,
            gate_lon,
            gate_lat,
            cfg.prototype_levels_km,
            cfg.max_ring_crossing_gap_s,
        )
        knn_path = route_path_from_track(
            track,
            gate_lon,
            gate_lat,
            cfg.knn_levels_km,
            cfg.max_ring_crossing_gap_s,
        )
        if len(prototype_path) >= 3:
            prototype_source_paths[sid] = prototype_path
            families[sid] = training_call_family(prototype_path, gate_lon, gate_lat)
        if len(knn_path) >= 3:
            individual[sid] = {
                "session_id": sid,
                "mmsi": int(call.mmsi),
                "family": training_call_family(knn_path, gate_lon, gate_lat),
                "points": knn_path,
            }

    if len(gate_keys) != 1:
        # Current primary Augusta ETA cohort is Levante-only.  If a future
        # cohort mixes entrances, fit each entrance independently instead of
        # silently merging different target gates.
        raise ValueError(f"Training calls mix target gates: {sorted(gate_keys)}")
    gate_lon, gate_lat = next(iter(gate_keys))

    port_prototype = _fit_prototype_from_paths(
        prototype_source_paths,
        cfg.prototype_levels_km,
        gate_lon,
        gate_lat,
        cfg.min_level_support_calls,
    )
    family_prototypes: dict[str, dict] = {}
    family_counts: dict[str, int] = {}
    for family in ROUTE_FAMILY_LABELS:
        ids = [sid for sid, fam in families.items() if fam == family]
        family_counts[family] = int(len(ids))
        if len(ids) < cfg.min_family_calls:
            continue
        subset = {sid: prototype_source_paths[sid] for sid in ids}
        proto = _fit_prototype_from_paths(
            subset,
            cfg.prototype_levels_km,
            gate_lon,
            gate_lat,
            cfg.min_level_support_calls,
        )
        if proto is not None:
            family_prototypes[family] = proto

    if port_prototype is None or not individual:
        raise ValueError(f"Insufficient route support for {port}")
    return {
        "port": port,
        "gate_lon": float(gate_lon),
        "gate_lat": float(gate_lat),
        "training_call_ids": [str(x) for x in train_calls["session_id"].tolist()],
        "training_call_count": int(len(train_calls)),
        "training_unique_vessels": int(train_calls["mmsi"].astype(int).nunique()),
        "family_counts": family_counts,
        "port_prototype": port_prototype,
        "family_prototypes": family_prototypes,
        "individual_paths": individual,
    }


def _local_xy_km(lon, lat, gate_lon: float, gate_lat: float) -> tuple[np.ndarray, np.ndarray]:
    radius_km = 6371.0088
    lon_arr = np.asarray(lon, dtype=float)
    lat_arr = np.asarray(lat, dtype=float)
    x = np.radians(lon_arr - float(gate_lon)) * radius_km * np.cos(np.radians(float(gate_lat)))
    y = np.radians(lat_arr - float(gate_lat)) * radius_km
    return x, y


def project_to_route(
    lon: float,
    lat: float,
    points: list[dict],
    gate_lon: float,
    gate_lat: float,
) -> dict:
    """Estimate remaining route distance via nearest polyline projection.

    The vessel joins the historical route at the nearest point, then follows it
    to the gate.  The estimate is floored by geodesic distance so the route
    representation cannot claim a physically shorter path than straight-line
    distance.
    """
    if len(points) < 2:
        raise ValueError("Route requires at least two points")
    lons = [float(p["lon"]) for p in points]
    lats = [float(p["lat"]) for p in points]
    xs, ys = _local_xy_km(lons, lats, gate_lon, gate_lat)
    px, py = _local_xy_km([lon], [lat], gate_lon, gate_lat)
    px = float(px[0]); py = float(py[0])
    seg = np.hypot(np.diff(xs), np.diff(ys))
    remaining_at_vertex = np.zeros(len(points), dtype=float)
    for i in range(len(points) - 2, -1, -1):
        remaining_at_vertex[i] = remaining_at_vertex[i + 1] + seg[i]

    best = None
    for i in range(len(points) - 1):
        ax, ay = float(xs[i]), float(ys[i])
        bx, by = float(xs[i + 1]), float(ys[i + 1])
        vx, vy = bx - ax, by - ay
        denom = vx * vx + vy * vy
        frac = 0.0 if denom == 0 else ((px - ax) * vx + (py - ay) * vy) / denom
        frac = float(np.clip(frac, 0.0, 1.0))
        qx, qy = ax + frac * vx, ay + frac * vy
        cross_track = float(np.hypot(px - qx, py - qy))
        along = float((1.0 - frac) * seg[i] + remaining_at_vertex[i + 1])
        candidate = (cross_track, along, i, frac)
        if best is None or candidate[0] < best[0]:
            best = candidate

    geodesic_km = float(_haversine_km(
        np.asarray([float(lat)]), np.asarray([float(lon)]), float(gate_lat), float(gate_lon)
    )[0])
    connector_plus_route = float(best[0] + best[1])
    return {
        "route_remaining_km": float(max(geodesic_km, connector_plus_route)),
        "cross_track_km": float(best[0]),
        "along_route_remaining_km": float(best[1]),
        "geodesic_km": geodesic_km,
        "segment_index": int(best[2]),
        "segment_fraction": float(best[3]),
    }


def predict_route_distances(model: dict, lon: float, lat: float, config: M2Config | None = None) -> dict:
    cfg = config or M2Config()
    gate_lon = float(model["gate_lon"]); gate_lat = float(model["gate_lat"])
    bearing = bearing_from_gate_deg(gate_lon, gate_lat, lon, lat)
    family = route_family_from_bearing(bearing)
    family_proto = model["family_prototypes"].get(family)
    proto_kind = "sector" if family_proto is not None else "port_fallback"
    proto = family_proto or model["port_prototype"]
    pp = project_to_route(lon, lat, proto["points"], gate_lon, gate_lat)

    candidates = []
    for sid, path in model["individual_paths"].items():
        pr = project_to_route(lon, lat, path["points"], gate_lon, gate_lat)
        candidates.append((pr["cross_track_km"], pr["route_remaining_km"], sid, path["mmsi"]))
    candidates.sort(key=lambda x: (x[0], x[2]))
    top = candidates[: min(cfg.knn_k, len(candidates))]
    if not top:
        raise ValueError("No historical paths available for route-kNN")
    knn_distance = float(np.median([x[1] for x in top]))
    return {
        "current_route_family": family,
        "prototype_kind": proto_kind,
        "prototype_route_km": float(pp["route_remaining_km"]),
        "prototype_cross_track_km": float(pp["cross_track_km"]),
        "knn_route_km": knn_distance,
        "knn_k_used": int(len(top)),
        "knn_mean_cross_track_km": float(np.mean([x[0] for x in top])),
        "knn_neighbor_sessions": "|".join(str(x[2]) for x in top),
        "knn_neighbor_mmsi": "|".join(str(int(x[3])) for x in top),
        "geodesic_km": float(pp["geodesic_km"]),
    }


def equal_call_weights(df: pd.DataFrame) -> pd.Series:
    counts = df.groupby("session_id")["session_id"].transform("size").astype(float)
    return 1.0 / counts


def weighted_quantile(values: Iterable[float], weights: Iterable[float], q: float) -> float:
    v = np.asarray(list(values), dtype=float)
    w = np.asarray(list(weights), dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not mask.any():
        return np.nan
    v = v[mask]; w = w[mask]
    order = np.argsort(v); v = v[order]; w = w[order]
    cdf = np.cumsum(w) / np.sum(w)
    return float(v[min(np.searchsorted(cdf, q, side="left"), len(v) - 1)])


def eta_metrics(df: pd.DataFrame, prediction_columns: Iterable[str]) -> pd.DataFrame:
    """Voyage-balanced OOF metrics; every call receives equal total weight."""
    rows = []
    for pred in prediction_columns:
        q = df[df[pred].notna()].copy()
        if q.empty:
            continue
        q["error_h"] = q[pred] - q["true_tta_h"]
        q["abs_error_h"] = q["error_h"].abs()
        q["weight"] = equal_call_weights(q)
        e = q["error_h"].to_numpy(float)
        ae = q["abs_error_h"].to_numpy(float)
        w = q["weight"].to_numpy(float)
        rows.append({
            "baseline": pred,
            "prediction_points": int(len(q)),
            "calls_covered": int(q["session_id"].nunique()),
            "unique_vessels": int(q["mmsi"].nunique()),
            "voyage_balanced_mae_h": float(np.average(ae, weights=w)),
            "voyage_balanced_rmse_h": float(np.sqrt(np.average(e ** 2, weights=w))),
            "voyage_balanced_median_ae_h": weighted_quantile(ae, w, 0.50),
            "voyage_balanced_p90_ae_h": weighted_quantile(ae, w, 0.90),
            "voyage_balanced_p95_ae_h": weighted_quantile(ae, w, 0.95),
            "within_30m": float(np.average(ae <= 0.5, weights=w)),
            "within_60m": float(np.average(ae <= 1.0, weights=w)),
            "within_120m": float(np.average(ae <= 2.0, weights=w)),
        })
    return pd.DataFrame(rows)


def add_horizon_band(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["horizon_band"] = pd.cut(
        out["true_tta_h"].astype(float),
        bins=HORIZON_BINS_H,
        labels=HORIZON_LABELS,
        include_lowest=True,
        right=True,
    )
    return out


def evaluate_m2_gate(
    overall_geo_mae_h: float,
    overall_knn_mae_h: float,
    under24_geo_mae_h: float,
    under24_knn_mae_h: float,
    fold_wins: int,
    fold_total: int,
    cold_under24_geo_mae_h: float,
    cold_under24_knn_mae_h: float,
    criteria: M2GateCriteria | None = None,
) -> dict:
    c = criteria or M2GateCriteria()
    overall_degradation = (overall_knn_mae_h - overall_geo_mae_h) / overall_geo_mae_h
    under24_improvement = (under24_geo_mae_h - under24_knn_mae_h) / under24_geo_mae_h
    cold_improvement = (cold_under24_geo_mae_h - cold_under24_knn_mae_h) / cold_under24_geo_mae_h
    fold_win_fraction = fold_wins / fold_total if fold_total else 0.0
    checks = {
        "under24_route_signal": under24_improvement >= c.min_under24_mae_improvement_fraction,
        "fold_consistency": fold_win_fraction >= c.min_fold_win_fraction,
        "cold_vessel_route_signal": cold_improvement >= c.min_cold_under24_improvement_fraction,
        "no_material_overall_degradation": overall_degradation <= c.max_overall_mae_degradation_fraction,
    }
    return {
        "criteria": {
            "min_under24_mae_improvement_fraction": c.min_under24_mae_improvement_fraction,
            "min_fold_win_fraction": c.min_fold_win_fraction,
            "min_cold_under24_improvement_fraction": c.min_cold_under24_improvement_fraction,
            "max_overall_mae_degradation_fraction": c.max_overall_mae_degradation_fraction,
        },
        "observed": {
            "overall_mae_improvement_fraction": -overall_degradation,
            "under24_mae_improvement_fraction": under24_improvement,
            "fold_wins": int(fold_wins),
            "fold_total": int(fold_total),
            "fold_win_fraction": fold_win_fraction,
            "cold_under24_mae_improvement_fraction": cold_improvement,
        },
        "checks": checks,
        "status": "GO_M3_ROUTE_SIGNAL_CONFIRMED" if all(checks.values()) else "M2_ROUTE_COMPLEXITY_NOT_JUSTIFIED",
    }
