"""M18G1 full-development operational scorer for blind external evaluation.

This module turns the frozen M16G/M16H development evidence into a deployable
*refit* scorer that can score previously unseen rows before labels are revealed.
It does not change the historical OOF benchmark and does not open the M18G
holdout.  Hyperparameter/model-family choices are derived deterministically from
already-frozen outer-fold selections; all learnable components are then refit on
all 386 development owners.

The scorer requires a prediction-time reference-row table plus causal AIS state
history for M16D.  It refuses target/reference-ETA columns in fresh inputs.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .m16b import DestinationResolver
from .m16c import add_m16c_features, predict_hierarchical_prior
from .m16d import M16D_DTW_BAND, dtw_distance, resample_causal_trajectory, representation_spec, weighted_median
from .m16e import add_physics_features, predict_physics_expert
from .m16f import M16F_FEATURES, prepare_features
from .m16g import M16G_EXPERT_NAMES, build_gating_features
from .m16h import build_error_features, predict_error
from .m18g import M18G_SELECTIVE_RISK_THRESHOLD_H, canonical_decision_time, sample_key, validate_prediction_ledger

M18G1_VERSION = "m18g1-full-development-scoring-bundle-v1-20260923"
M18G1_EXPECTED_DEV_ROWS = 386
M18G1_EXPECTED_OLD_FINAL_ROWS = 53

# Fresh rows must be label-free.  Reference ETA itself is also forbidden because
# M18G must seal predictions before labels/reference outcomes are revealed.
M18G1_FORBIDDEN_FRESH_COLUMNS = {
    "target_tte_h", "reference_eta_status", "eta_reference_raw", "eta_reference_dt",
    "abs_error_h", "signed_error_h", "covered_80", "error_h",
}

# Columns needed before M16B/M16E-derived features are constructed.
M18G1_REQUIRED_ROW_COLUMNS = (
    "mmsi", "last_update", "history_last_at",
    "track_lat", "track_lon", "track_sog", "track_cog", "track_heading", "track_draught",
    "track_msg_count", "destination_norm", "ship_type_cat", "nav_status_cat", "flag_cat",
    "last_hour_sin", "last_hour_cos", "last_dow", "position_age_min", "history_n", "history_span_h",
    "dynamic_state_age_min", "motion_state_cat", "stale_risk_cat", "hist_destination_unique",
    "hist_destination_matches_track", "track_vs_hist_position_gap_km",
    "sog_median_180m", "sog_std_180m", "moving_fraction_180m", "stopped_fraction_180m",
    "turn_count_180m", "distance_travelled_km_180m", "net_displacement_km_180m",
    "sog_median_360m", "sog_std_360m", "moving_fraction_360m", "stopped_fraction_360m",
    "turn_count_360m", "distance_travelled_km_360m", "net_displacement_km_360m",
)

M18G1_REQUIRED_STATE_COLUMNS = (
    "mmsi", "recorded_at", "position_valid", "lat_clean", "lon_clean", "sog_clean", "cog_clean",
)


def plurality_choice(values: Iterable[str]) -> tuple[str, dict[str, int]]:
    """Deterministic plurality rule with lexical tie-break.

    Choices are not reselected from pooled labels.  They are aggregated from the
    already-frozen nested outer-fold choices, which prevents a new post-hoc model
    search while defining one full-development refit recipe.
    """
    vals = [str(v) for v in values]
    if not vals:
        raise ValueError("empty plurality choice")
    counts = Counter(vals)
    top = max(counts.values())
    selected = sorted(k for k, v in counts.items() if v == top)[0]
    return selected, dict(sorted(counts.items()))


def validate_fresh_rows(rows: pd.DataFrame) -> None:
    missing = [c for c in M18G1_REQUIRED_ROW_COLUMNS if c not in rows.columns]
    if missing:
        raise ValueError(f"fresh-row input missing columns: {missing}")
    bad = sorted(M18G1_FORBIDDEN_FRESH_COLUMNS & set(rows.columns))
    if bad:
        raise ValueError(f"fresh-row input contains label/reference columns: {bad}")
    if len(rows) == 0:
        raise ValueError("fresh-row input is empty")
    if rows["mmsi"].isna().any() or rows["last_update"].isna().any():
        raise ValueError("mmsi/last_update must be present")
    keys = [sample_key(m, t) for m, t in zip(rows.mmsi, rows.last_update)]
    if len(set(keys)) != len(keys):
        raise ValueError("fresh-row sample keys must be unique")


def validate_states(states: pd.DataFrame) -> None:
    missing = [c for c in M18G1_REQUIRED_STATE_COLUMNS if c not in states.columns]
    if missing:
        raise ValueError(f"fresh-state input missing columns: {missing}")
    if len(states) == 0:
        raise ValueError("fresh-state input is empty")


def prepare_fresh_rows(rows: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    """Construct exactly the target-free semantic/physics features needed downstream."""
    validate_fresh_rows(rows)
    x = rows.copy().reset_index(drop=True)
    x["last_update"] = pd.to_datetime(x["last_update"])
    x["history_last_at"] = pd.to_datetime(x["history_last_at"])
    resolver = DestinationResolver(catalog)
    resolved = resolver.transform(x["destination_norm"])
    # Avoid collisions if callers provided stale derived columns: the scorer is
    # authoritative for M16B-derived values.
    for c in resolved.columns:
        if c in x.columns:
            x = x.drop(columns=[c])
    x = pd.concat([x.reset_index(drop=True), resolved.reset_index(drop=True)], axis=1)
    x = add_physics_features(x)
    x = add_m16c_features(x)
    return x


def _fresh_route_predictions(bundle: dict[str, Any], fresh: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    """Score fresh trajectories against frozen full-development route memory."""
    validate_states(states)
    route = bundle["route_memory"]
    config = bundle["configs"]["m16d"]
    rep = str(config["representation"])
    gate = str(config["gate"])
    k = int(config["k"])
    hours, kin = representation_spec(rep)
    train_seq = np.asarray(route["sequences"], dtype=float)
    train_y = np.asarray(route["targets_h"], dtype=float)
    train_dest = np.asarray(route["destinations"], dtype=object)
    train_mmsi = np.asarray(route["mmsis"], dtype=np.int64)
    if len(train_seq) != len(train_y) or len(train_y) != len(train_dest):
        raise AssertionError("route memory shape drift")

    s = states.copy()
    s["recorded_at"] = pd.to_datetime(s["recorded_at"])
    groups = {int(m): g for m, g in s.groupby("mmsi", sort=False)}
    rows: list[dict[str, Any]] = []
    for rec in fresh.itertuples(index=False):
        mmsi = int(rec.mmsi)
        g = groups.get(mmsi)
        if g is None:
            raise ValueError(f"fresh route scorer missing causal states for MMSI {mmsi}")
        try:
            seq, meta = resample_causal_trajectory(
                g,
                end_at=pd.Timestamp(rec.history_last_at),
                hours=hours,
                include_kinematics=kin,
            )
        except Exception as exc:  # make row-level failure auditable
            raise ValueError(f"fresh route scorer cannot build {rep} for MMSI {mmsi}: {exc}") from exc
        d = np.asarray([dtw_distance(seq, q, band=M16D_DTW_BAND) for q in train_seq], dtype=float)
        candidates = np.arange(len(train_y), dtype=int)
        gate_used = "global"
        dest = str(rec.canonical_destination)
        if gate == "destination":
            same = candidates[train_dest == dest]
            if len(same) >= 3:
                candidates = same
                gate_used = "destination"
            else:
                gate_used = "destination_fallback_global"
        order = np.argsort(d[candidates], kind="mergesort")[: min(k, len(candidates))]
        nn = candidates[order]
        nd = d[nn]
        if len(nn) == 0 or not np.isfinite(nd).all():
            raise ValueError(f"fresh route scorer produced invalid neighbours for MMSI {mmsi}")
        scale = max(float(np.median(nd)), 1e-9)
        weights = np.exp(-nd / scale)
        pred = weighted_median(train_y[nn], weights)
        second = float(nd[1]) if len(nd) > 1 else float(nd[0])
        gap = float((second - nd[0]) / max(abs(float(nd[0])), 1e-9)) if len(nd) > 1 else 0.0
        rows.append({
            "prediction_h": float(pred),
            "neighbour_count": int(len(nn)),
            "nearest_distance": float(nd[0]),
            "median_neighbour_distance": float(np.median(nd)),
            "similarity_gap": gap,
            "gate_used": gate_used,
            "nearest_mmsi": int(train_mmsi[nn[0]]),
            "history_raw_points": int(meta["raw_points"]),
            "history_observed_span_h": float(meta["observed_span_h"]),
        })
    return pd.DataFrame(rows, index=fresh.index)


def _confidence_tier(risk: np.ndarray, q1: float, q2: float) -> np.ndarray:
    return np.where(risk <= q1, "HIGH", np.where(risk <= q2, "MEDIUM", "LOW")).astype(object)


def score_fresh_rows(bundle: dict[str, Any], rows: pd.DataFrame, states: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score label-free fresh rows and return (M18G ledger, detailed audit frame)."""
    if bundle.get("version") != M18G1_VERSION:
        raise ValueError("unsupported M18G1 bundle version")
    fresh = prepare_fresh_rows(rows, bundle["catalog"])
    train = bundle["training_frame"]

    # Base expert 1: full-development hierarchical prior.
    prior = predict_hierarchical_prior(train, fresh, bundle["objects"]["m16c_config"])

    # Base expert 2: full-development route memory.
    route = _fresh_route_predictions(bundle, fresh, states)

    # Base expert 3: physics + train-only residual correction, then the frozen
    # target-free hard gate to route fallback.
    physics = predict_physics_expert(train, fresh, bundle["objects"]["m16e_config"])
    route_pred = route["prediction_h"].to_numpy(float)
    physics_pred = physics["prediction_h"].to_numpy(float)
    eligible = fresh["physics_eligible"].astype(bool).to_numpy()
    physics_gate = np.where(eligible & np.isfinite(physics_pred), physics_pred, route_pred)

    # Base expert 4: full-development tabular refit.
    tab_x = prepare_features(fresh)[M16F_FEATURES]
    tabular = np.asarray(bundle["objects"]["m16f_model"].predict(tab_x), dtype=float)

    base = np.column_stack([
        prior["prediction_h"].to_numpy(float),
        route_pred,
        physics_gate,
        tabular,
    ])
    if not np.isfinite(base).all():
        raise ValueError("fresh base expert matrix contains non-finite values")

    quality = pd.DataFrame({
        "m16c_deepest_support": prior["deepest_support"].to_numpy(float),
        "m16c_deepest_weight": prior["deepest_weight"].to_numpy(float),
        "m16c_fallback_count": prior["fallback_count"].to_numpy(float),
        "m16d_neighbour_count": route["neighbour_count"].to_numpy(float),
        "m16d_nearest_distance": route["nearest_distance"].to_numpy(float),
        "m16d_median_neighbour_distance": route["median_neighbour_distance"].to_numpy(float),
        "m16d_similarity_gap": route["similarity_gap"].to_numpy(float),
        "m16d_gate_used": route["gate_used"].astype(str).to_numpy(),
        "physics_eligible": fresh["physics_eligible"].astype(bool).to_numpy(),
        "resolution_confidence": pd.to_numeric(fresh["resolution_confidence"], errors="coerce").to_numpy(float),
        "physics_distance_gc_nm": pd.to_numeric(fresh["physics_distance_gc_nm"], errors="coerce").to_numpy(float),
        "physics_course_alignment_deg": pd.to_numeric(fresh["physics_course_alignment_deg"], errors="coerce").to_numpy(float),
        "physics_recent_speed_max_kn": pd.to_numeric(fresh["physics_recent_speed_max_kn"], errors="coerce").to_numpy(float),
        "m16e_effective_speed_kn": physics["effective_speed_kn"].to_numpy(float),
        "m16e_distance_factor": physics["distance_factor"].to_numpy(float),
        "m16e_route_distance_proxy_nm": physics["route_distance_proxy_nm"].to_numpy(float),
    })
    gating = build_gating_features(base, quality).reindex(columns=bundle["feature_columns"]["gating"], fill_value=0.0)
    meta_model = bundle["objects"]["m16g_meta_model"]
    label_idx = np.asarray(meta_model.predict(gating), dtype=int)
    if np.any((label_idx < 0) | (label_idx >= len(M16G_EXPERT_NAMES))):
        raise ValueError("M16G operational meta model returned invalid expert label")
    point = base[np.arange(len(base)), label_idx]
    expert_name = np.asarray([M16G_EXPERT_NAMES[int(i)] for i in label_idx], dtype=object)

    # Full-development M16H risk refit.  The risk model was fit to official
    # nested-OOF M16G errors; only prediction-time features are used here.
    hframe = pd.DataFrame({
        "pred_m16c_prior_h": base[:, 0],
        "pred_m16d_route_h": base[:, 1],
        "pred_m16e_physics_gate_h": base[:, 2],
        "pred_m16f_tabular_h": base[:, 3],
        "pred_m16g_selected_h": point,
        "m16g_selected_meta_id": bundle["configs"]["m16g_meta_id"],
        "m16c_deepest_support": quality["m16c_deepest_support"],
        "m16c_deepest_weight": quality["m16c_deepest_weight"],
        "m16c_fallback_count": quality["m16c_fallback_count"],
        "m16d_neighbour_count": quality["m16d_neighbour_count"],
        "m16d_nearest_distance": quality["m16d_nearest_distance"],
        "m16d_median_neighbour_distance": quality["m16d_median_neighbour_distance"],
        "m16d_similarity_gap": quality["m16d_similarity_gap"],
        "m16d_gate_used": quality["m16d_gate_used"],
        "physics_eligible": quality["physics_eligible"],
        "resolution_confidence": quality["resolution_confidence"],
        "physics_distance_gc_nm": quality["physics_distance_gc_nm"],
        "physics_course_alignment_deg": quality["physics_course_alignment_deg"],
        "physics_recent_speed_max_kn": quality["physics_recent_speed_max_kn"],
    })
    error_features = build_error_features(hframe).reindex(columns=bundle["feature_columns"]["risk"], fill_value=0.0)
    risk = predict_error(bundle["objects"]["m16h_risk_model"], error_features)
    q1 = float(bundle["uncertainty"]["confidence_q1_h"])
    q2 = float(bundle["uncertainty"]["confidence_q2_h"])
    tier = _confidence_tier(risk, q1, q2)
    floor_h = float(bundle["uncertainty"]["interval_floor_h"])
    scale_q = float(bundle["uncertainty"]["interval_scale_q"])
    half = scale_q * (floor_h + np.maximum(risk, 0.0))
    lo = point - half
    hi = point + half

    training_mmsis = set(int(x) for x in bundle["training_mmsis"])
    decision = [canonical_decision_time(v) for v in fresh["last_update"]]
    ledger = pd.DataFrame({
        "sample_key": [sample_key(m, t) for m, t in zip(fresh["mmsi"], decision)],
        "mmsi": fresh["mmsi"].astype(int).to_numpy(),
        "decision_time": decision,
        "baseline_pred_h": point,
        "m16h_risk_h": risk,
        "m16h_lower_h": lo,
        "m16h_upper_h": hi,
        "slice_vessel_seen": ["SEEN" if int(m) in training_mmsis else "UNSEEN" for m in fresh["mmsi"]],
        "slice_confidence_tier": tier,
        "slice_physics_eligible": np.where(eligible, "ELIGIBLE", "INELIGIBLE"),
        "slice_destination_resolved": np.where(fresh["is_resolved_port"].astype(bool), "RESOLVED", "UNRESOLVED"),
    })
    validate_prediction_ledger(ledger, require_candidate=False)

    detail = ledger.copy()
    detail["m16g_selected_expert"] = expert_name
    detail["canonical_destination"] = fresh["canonical_destination"].astype(str).to_numpy()
    detail["resolution_method"] = fresh["resolution_method"].astype(str).to_numpy()
    detail["m16d_gate_used"] = route["gate_used"].astype(str).to_numpy()
    detail["m16d_nearest_mmsi"] = route["nearest_mmsi"].astype(int).to_numpy()
    detail["m18f_action"] = np.where(risk <= M18G_SELECTIVE_RISK_THRESHOLD_H, "AUTO_ETA_WITH_UNCERTAINTY", "DEFER_LOW_CONFIDENCE")
    detail["pred_m16c_prior_h"] = base[:, 0]
    detail["pred_m16d_route_h"] = base[:, 1]
    detail["pred_m16e_physics_gate_h"] = base[:, 2]
    detail["pred_m16f_tabular_h"] = base[:, 3]
    return ledger, detail


def load_rows(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    s = p.name.lower()
    if s.endswith(".pkl") or s.endswith(".pkl.gz") or s.endswith(".pickle") or s.endswith(".pickle.gz"):
        return pd.read_pickle(p)
    if s.endswith(".parquet"):
        return pd.read_parquet(p)
    if s.endswith(".csv") or s.endswith(".csv.gz"):
        return pd.read_csv(p)
    raise ValueError(f"unsupported row file format: {p}")


def load_states(path: str | Path) -> pd.DataFrame:
    return load_rows(path)
