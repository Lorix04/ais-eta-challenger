"""M3: shallow physics + residual ETA modelling with causal port-state ablation.

M3 deliberately remains low-capacity.  It asks whether standard tabular models can
learn a stable correction to the M2 route-kNN / robust-speed physical ETA while
preserving the strict temporal/grouped design established earlier.

Design rules:
- the locked chronological final holdout is never scored in M3;
- every validation block is later than its training calls;
- route representations are fitted on the fold training calls only;
- training rows may use historical route paths from the fold training set, but
  the current call is excluded from its route-kNN retrieval;
- every call receives equal total training weight, and occupied broad horizon
  bands receive equal weight within a call;
- no MMSI, vessel name, session id, true horizon, ground-truth confidence or
  future information enters a model feature;
- port-state features are same-snapshot AIS proxies, not true berth availability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .m0d import _haversine_km
from .m1 import KM_PER_NM
from .m2 import (
    M2Config,
    bearing_from_gate_deg,
    project_to_route,
    route_family_from_bearing,
)


@dataclass(frozen=True)
class M3Config:
    """Frozen M3 choices; no large hyperparameter search is permitted."""

    ridge_alpha: float = 10.0
    huber_alpha: float = 0.01
    huber_epsilon: float = 1.35
    lightgbm_estimators: int = 120
    lightgbm_max_depth: int = 3
    lightgbm_num_leaves: int = 7
    lightgbm_learning_rate: float = 0.03
    lightgbm_reg_lambda: float = 10.0
    random_seed: int = 42
    min_prediction_h: float = 0.0
    stacking_warmup_calls: int = 10
    stacking_temporal_folds: int = 3


@dataclass(frozen=True)
class M3GateCriteria:
    """Pre-specified gate for retaining residual ML complexity."""

    min_under24_mae_improvement_fraction: float = 0.05
    min_fold_win_fraction: float = 2.0 / 3.0
    min_cold_under24_improvement_fraction: float = 0.05
    max_overall_mae_degradation_fraction: float = 0.05
    max_top_positive_call_gain_share: float = 0.50


@dataclass(frozen=True)
class PortStateGateCriteria:
    """Conservative gate for keeping AIS-derived port-state proxies."""

    min_under24_improvement_fraction: float = 0.02
    min_fold_win_fraction: float = 2.0 / 3.0
    max_overall_degradation_fraction: float = 0.03


MODEL_NUMERIC_FEATURES = [
    "pred_route_physics_h",
    "geodesic_distance_km",
    "route_knn_distance_km",
    "route_to_geodesic_ratio",
    "knn_mean_cross_track_km",
    "sog_median_30m_kn",
    "sog_current_kn",
    "progress_to_gate_30m_kn",
    "cog_to_gate_error_deg",
    "provider_observation_age_s",
    "log_dynamic_state_age_min",
    "log_stop_duration_min",
    "draught_clean",
    "turn_candidate",
    "destination_support_port",
    "route_match_confidence",
    "decision_hour_sin",
    "decision_hour_cos",
]

MODEL_CATEGORICAL_FEATURES = [
    "port",
    "inbound_entrance",
    "route_family",
    "motion_state",
    "nav_status_name",
    "stale_risk_level",
]

PORT_STATE_FEATURES = [
    "port_traffic_5km",
    "port_traffic_15km",
    "port_traffic_30km",
    "port_stopped_5km",
    "port_stopped_15km",
    "port_slow_15km",
    "port_moving_15km",
    "port_stop_pressure_15km",
]

# Explicitly forbidden from model features.  Tests enforce this contract.
FORBIDDEN_MODEL_FEATURES = {
    "mmsi",
    "name",
    "session_id",
    "true_tta_h",
    "ground_truth_time",
    "ground_truth_confidence",
    "horizon_band",
    "knn_neighbor_sessions",
    "knn_neighbor_mmsi",
}


def broad_horizon_band(true_tta_h: pd.Series) -> pd.Series:
    """Broad bands used only for training weights and reporting, never features."""
    return pd.cut(
        true_tta_h.astype(float),
        bins=[-np.inf, 6.0, 12.0, 24.0, np.inf],
        labels=["<6h", "6-12h", "12-24h", ">24h"],
        right=True,
    )


def call_horizon_balanced_weights(df: pd.DataFrame) -> pd.Series:
    """Equal call weight; equal occupied-band weight inside each call.

    For a call that occupies B broad horizon bands, each band gets 1/B of that
    call's total weight; rows within a band share it equally.  We rescale to
    mean weight 1 because some estimators are sensitive to absolute weight scale.
    """
    if df.empty:
        return pd.Series(dtype=float, index=df.index)
    x = df[["session_id", "true_tta_h"]].copy()
    x["_band"] = broad_horizon_band(x["true_tta_h"]).astype(str)
    band_count = x.groupby("session_id")["_band"].transform("nunique").astype(float)
    rows_in_band = x.groupby(["session_id", "_band"])["_band"].transform("size").astype(float)
    w = 1.0 / (band_count * rows_in_band)
    return w / float(w.mean())


def circular_abs_diff_deg(a: np.ndarray | pd.Series, b: np.ndarray | pd.Series) -> np.ndarray:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    return np.abs((aa - bb + 180.0) % 360.0 - 180.0)


def bearing_to_target_deg(lon: float, lat: float, target_lon: float, target_lat: float) -> float:
    """Initial great-circle bearing from vessel toward target."""
    # M2 helper is from gate -> vessel; reverse endpoints here.
    return bearing_from_gate_deg(float(lon), float(lat), float(target_lon), float(target_lat))


def destination_supports_port(value: object, port: str) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    s = str(value).upper().strip()
    p = str(port).upper()
    if p == "CATANIA":
        return any(token in s for token in ["ITCTA", "CATANIA", "IT CTA", ">ITCTA"])
    if p == "AUGUSTA":
        return any(token in s for token in ["ITAUG", "AUGUSTA", "IT AUG", ">ITAUG"]) or s == "AUG"
    return False




def prepare_fast_knn_model(model: dict) -> dict:
    """Attach precomputed local-XY path arrays for fast repeated kNN projection.

    M2's reference implementation intentionally favors clarity over speed. M3
    projects many training rows against the same fold-local paths, so repeated
    coordinate conversion and Python segment loops are unnecessary overhead.
    The geometry is mathematically identical to M2's local equirectangular
    projection around the gate.
    """
    gate_lon = float(model["gate_lon"])
    gate_lat = float(model["gate_lat"])
    radius_km = 6371.0088
    cos_lat = float(np.cos(np.radians(gate_lat)))
    fast = {}
    for sid, path in model["individual_paths"].items():
        lons = np.asarray([float(p["lon"]) for p in path["points"]], dtype=float)
        lats = np.asarray([float(p["lat"]) for p in path["points"]], dtype=float)
        xs = np.radians(lons - gate_lon) * radius_km * cos_lat
        ys = np.radians(lats - gate_lat) * radius_km
        dx = np.diff(xs); dy = np.diff(ys)
        seg = np.hypot(dx, dy)
        remaining = np.zeros(len(xs), dtype=float)
        if len(seg):
            remaining[:-1] = np.cumsum(seg[::-1])[::-1]
        fast[str(sid)] = {
            "mmsi": int(path["mmsi"]),
            "xs": xs, "ys": ys, "dx": dx, "dy": dy, "seg": seg,
            "remaining": remaining,
        }
    out = dict(model)
    out["_fast_individual_paths"] = fast
    out["_fast_radius_km"] = radius_km
    out["_fast_cos_gate_lat"] = cos_lat
    return out


def _fast_project_prepared(model: dict, prepared: dict, lon: float, lat: float) -> tuple[float, float]:
    gate_lon = float(model["gate_lon"]); gate_lat = float(model["gate_lat"])
    radius_km = float(model["_fast_radius_km"]); cos_lat = float(model["_fast_cos_gate_lat"])
    px = float(np.radians(float(lon) - gate_lon) * radius_km * cos_lat)
    py = float(np.radians(float(lat) - gate_lat) * radius_km)
    xs = prepared["xs"]; ys = prepared["ys"]
    dx = prepared["dx"]; dy = prepared["dy"]; seg = prepared["seg"]
    denom = dx * dx + dy * dy
    numer = (px - xs[:-1]) * dx + (py - ys[:-1]) * dy
    frac = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)
    frac = np.clip(frac, 0.0, 1.0)
    qx = xs[:-1] + frac * dx
    qy = ys[:-1] + frac * dy
    cross = np.hypot(px - qx, py - qy)
    i = int(np.argmin(cross))
    along = float((1.0 - frac[i]) * seg[i] + prepared["remaining"][i + 1])
    cross_track = float(cross[i])
    geo = float(_haversine_km(
        np.asarray([float(lat)]), np.asarray([float(lon)]), gate_lat, gate_lon
    )[0])
    return cross_track, float(max(geo, cross_track + along))


def predict_route_knn_excluding(
    model: dict,
    lon: float,
    lat: float,
    config: M2Config | None = None,
    excluded_session_ids: Iterable[str] | None = None,
) -> dict:
    """Route-kNN prediction with optional self-call exclusion for training rows."""
    cfg = config or M2Config()
    excluded = {str(x) for x in (excluded_session_ids or [])}
    gate_lon = float(model["gate_lon"])
    gate_lat = float(model["gate_lat"])
    bearing = bearing_from_gate_deg(gate_lon, gate_lat, lon, lat)
    family = route_family_from_bearing(bearing)

    candidates = []
    fast_paths = model.get("_fast_individual_paths")
    for sid, path in model["individual_paths"].items():
        if str(sid) in excluded:
            continue
        if fast_paths is not None and str(sid) in fast_paths:
            cross_track, route_remaining = _fast_project_prepared(model, fast_paths[str(sid)], lon, lat)
        else:
            pr = project_to_route(lon, lat, path["points"], gate_lon, gate_lat)
            cross_track, route_remaining = pr["cross_track_km"], pr["route_remaining_km"]
        candidates.append((cross_track, route_remaining, str(sid), int(path["mmsi"])))
    candidates.sort(key=lambda x: (x[0], x[2]))
    top = candidates[: min(cfg.knn_k, len(candidates))]
    if not top:
        raise ValueError("No eligible historical paths after route-kNN exclusion")
    geo = float(_haversine_km(
        np.asarray([float(lat)]), np.asarray([float(lon)]), gate_lat, gate_lon
    )[0])
    route_km = float(max(geo, np.median([x[1] for x in top])))
    cross_track = float(np.mean([x[0] for x in top]))
    return {
        "route_knn_distance_km": route_km,
        "knn_mean_cross_track_km": cross_track,
        "route_family": family,
        "knn_neighbor_sessions": "|".join(x[2] for x in top),
        "knn_neighbor_mmsi": "|".join(str(x[3]) for x in top),
        "knn_k_used": len(top),
        "route_match_confidence": float(1.0 / (1.0 + cross_track)),
        "geodesic_km": geo,
    }


def _port_reference_distance_km(states: pd.DataFrame, port: str, gate_points: list[tuple[float, float]]) -> np.ndarray:
    lat = states["lat_clean"].to_numpy(dtype=float)
    lon = states["lon_clean"].to_numpy(dtype=float)
    dists = []
    for glon, glat in gate_points:
        dists.append(_haversine_km(lat, lon, float(glat), float(glon)))
    return np.nanmin(np.vstack(dists), axis=0)


def build_same_snapshot_port_state(
    states: pd.DataFrame,
    catania_gate: tuple[float, float],
    augusta_gates: list[tuple[float, float]],
) -> pd.DataFrame:
    """Aggregate causal same-provider-snapshot port-state proxies.

    These are traffic proxies only.  They are not berth availability, a queue,
    or a port-operations system.  Exact `recorded_at` is used so no observation
    after the target vessel's source snapshot can enter the features.
    """
    base = states[
        states["position_valid"].astype(bool)
        & ~states["position_jump_candidate"].astype(bool)
    ][["recorded_at", "mmsi", "lat_clean", "lon_clean", "sog_clean", "motion_state"]].copy()
    base["recorded_at"] = pd.to_datetime(base["recorded_at"], errors="raise")
    outputs = []
    for port, gates in [("CATANIA", [catania_gate]), ("AUGUSTA", augusta_gates)]:
        x = base.copy()
        x["_d"] = _port_reference_distance_km(x, port, gates)
        stopped = x["motion_state"].astype(str).eq("STOPPED") | x["sog_clean"].fillna(np.inf).le(0.5)
        slow = x["motion_state"].astype(str).eq("SLOW_MOTION") | x["sog_clean"].fillna(np.inf).le(3.0)
        moving = x["motion_state"].astype(str).eq("MOVING") & x["sog_clean"].fillna(0).gt(3.0)
        x["traffic_5"] = x["_d"].le(5.0).astype(int)
        x["traffic_15"] = x["_d"].le(15.0).astype(int)
        x["traffic_30"] = x["_d"].le(30.0).astype(int)
        x["stopped_5"] = (x["_d"].le(5.0) & stopped).astype(int)
        x["stopped_15"] = (x["_d"].le(15.0) & stopped).astype(int)
        x["slow_15"] = (x["_d"].le(15.0) & slow).astype(int)
        x["moving_15"] = (x["_d"].le(15.0) & moving).astype(int)
        agg = x.groupby("recorded_at", sort=False).agg(
            port_traffic_5km=("traffic_5", "sum"),
            port_traffic_15km=("traffic_15", "sum"),
            port_traffic_30km=("traffic_30", "sum"),
            port_stopped_5km=("stopped_5", "sum"),
            port_stopped_15km=("stopped_15", "sum"),
            port_slow_15km=("slow_15", "sum"),
            port_moving_15km=("moving_15", "sum"),
        ).reset_index()
        agg["port"] = port
        denom = agg["port_traffic_15km"].clip(lower=1)
        agg["port_stop_pressure_15km"] = (agg["port_slow_15km"] / denom).astype(float)
        outputs.append(agg)
    return pd.concat(outputs, ignore_index=True)


def subtract_target_from_port_state(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the target vessel itself from same-snapshot count features."""
    out = df.copy()
    d = out["port_reference_distance_km"].astype(float)
    stopped = out["motion_state"].astype(str).eq("STOPPED") | out["sog_current_kn"].fillna(np.inf).le(0.5)
    slow = out["motion_state"].astype(str).eq("SLOW_MOTION") | out["sog_current_kn"].fillna(np.inf).le(3.0)
    moving = out["motion_state"].astype(str).eq("MOVING") & out["sog_current_kn"].fillna(0).gt(3.0)
    adjustments = {
        "port_traffic_5km": d.le(5.0),
        "port_traffic_15km": d.le(15.0),
        "port_traffic_30km": d.le(30.0),
        "port_stopped_5km": d.le(5.0) & stopped,
        "port_stopped_15km": d.le(15.0) & stopped,
        "port_slow_15km": d.le(15.0) & slow,
        "port_moving_15km": d.le(15.0) & moving,
    }
    for c, mask in adjustments.items():
        out[c] = (out[c].astype(float) - mask.astype(int)).clip(lower=0)
    denom = out["port_traffic_15km"].clip(lower=1)
    out["port_stop_pressure_15km"] = out["port_slow_15km"] / denom
    return out


def finalize_m3_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive model-safe features from fold-local route and causal source state."""
    out = df.copy()
    speed = out["sog_median_30m_kn"].astype(float).clip(lower=1.0)
    out["pred_route_physics_h"] = (out["route_knn_distance_km"].astype(float) / KM_PER_NM) / speed
    out["route_to_geodesic_ratio"] = (
        out["route_knn_distance_km"].astype(float)
        / out["geodesic_distance_km"].astype(float).clip(lower=0.05)
    )
    out["log_dynamic_state_age_min"] = np.log1p(out["dynamic_state_age_s"].fillna(0).astype(float).clip(lower=0) / 60.0)
    out["log_stop_duration_min"] = np.log1p(out["stop_duration_s"].fillna(0).astype(float).clip(lower=0) / 60.0)
    out["turn_candidate"] = out["turn_candidate"].fillna(False).astype(int)
    out["destination_support_port"] = out["destination_support_port"].fillna(False).astype(int)
    hour = pd.to_datetime(out["decision_time"], errors="raise").dt.hour + pd.to_datetime(out["decision_time"], errors="raise").dt.minute / 60.0
    out["decision_hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["decision_hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["route_match_confidence"] = out["route_match_confidence"].astype(float).clip(0, 1)
    return out


def validate_feature_contract(feature_columns: Iterable[str]) -> None:
    cols = set(str(c) for c in feature_columns)
    bad = sorted(cols & FORBIDDEN_MODEL_FEATURES)
    if bad:
        raise ValueError(f"Forbidden M3 feature(s): {bad}")


def evaluate_m3_gate(
    baseline_overall_mae_h: float,
    model_overall_mae_h: float,
    baseline_under24_mae_h: float,
    model_under24_mae_h: float,
    fold_wins: int,
    fold_total: int,
    cold_baseline_under24_mae_h: float,
    cold_model_under24_mae_h: float,
    top_positive_call_gain_share: float,
    criteria: M3GateCriteria | None = None,
) -> dict:
    c = criteria or M3GateCriteria()
    under24_imp = (baseline_under24_mae_h - model_under24_mae_h) / baseline_under24_mae_h
    cold_imp = (cold_baseline_under24_mae_h - cold_model_under24_mae_h) / cold_baseline_under24_mae_h
    overall_deg = (model_overall_mae_h - baseline_overall_mae_h) / baseline_overall_mae_h
    win_fraction = fold_wins / fold_total if fold_total else 0.0
    checks = {
        "under24_gain": under24_imp >= c.min_under24_mae_improvement_fraction,
        "fold_consistency": win_fraction >= c.min_fold_win_fraction,
        "cold_vessel_gain": cold_imp >= c.min_cold_under24_improvement_fraction,
        "no_material_overall_degradation": overall_deg <= c.max_overall_mae_degradation_fraction,
        "not_single_call_dominated": top_positive_call_gain_share <= c.max_top_positive_call_gain_share,
    }
    return {
        "criteria": c.__dict__,
        "observed": {
            "under24_mae_improvement_fraction": float(under24_imp),
            "cold_under24_mae_improvement_fraction": float(cold_imp),
            "overall_mae_improvement_fraction": float(-overall_deg),
            "fold_wins": int(fold_wins),
            "fold_total": int(fold_total),
            "fold_win_fraction": float(win_fraction),
            "top_positive_call_gain_share": float(top_positive_call_gain_share),
        },
        "checks": checks,
        "status": "GO_M4_RESIDUAL_SIGNAL_CONFIRMED" if all(checks.values()) else "M3_COMPLEXITY_NOT_JUSTIFIED",
    }


def evaluate_port_state_gate(
    base_overall_mae_h: float,
    port_overall_mae_h: float,
    base_under24_mae_h: float,
    port_under24_mae_h: float,
    fold_wins: int,
    fold_total: int,
    criteria: PortStateGateCriteria | None = None,
) -> dict:
    c = criteria or PortStateGateCriteria()
    u24 = (base_under24_mae_h - port_under24_mae_h) / base_under24_mae_h
    overall_deg = (port_overall_mae_h - base_overall_mae_h) / base_overall_mae_h
    wf = fold_wins / fold_total if fold_total else 0.0
    checks = {
        "under24_gain": u24 >= c.min_under24_improvement_fraction,
        "fold_consistency": wf >= c.min_fold_win_fraction,
        "no_material_overall_degradation": overall_deg <= c.max_overall_degradation_fraction,
    }
    return {
        "criteria": c.__dict__,
        "observed": {
            "under24_mae_improvement_fraction": float(u24),
            "overall_mae_improvement_fraction": float(-overall_deg),
            "fold_wins": int(fold_wins),
            "fold_total": int(fold_total),
            "fold_win_fraction": float(wf),
        },
        "checks": checks,
        "status": "KEEP_PORT_STATE" if all(checks.values()) else "DROP_PORT_STATE_FOR_NOW",
    }
