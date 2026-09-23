from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import pandas as pd

from .m3 import destination_supports_port


@dataclass(frozen=True)
class M4Config:
    calibration_calls: int = 24
    confidence_level: float = 0.90
    max_supported_pred_h: float = 24.0
    max_route_cross_track_km: float = 5.0
    max_provider_age_s: float = 90.0
    min_robust_speed_kn: float = 1.0
    smoothing_alpha_grid: tuple[float, ...] = (1.0, 0.8, 0.6, 0.4, 0.2)
    smoothing_reset_gap_h: float = 2.0
    reported_eta_alignment_tolerance_min: float = 5.0
    min_eval_calls: int = 12
    min_eval_trajectory_coverage: float = 0.90
    max_primary_half_width_h: float = 3.0


def conformal_order_statistic(scores: Iterable[float], confidence_level: float) -> tuple[float, int, int]:
    """Finite-sample split-conformal order statistic.

    q = kth sorted conformity score, k=ceil((n+1)*confidence), clipped to n.
    The caller decides the exchangeability unit. In M4 the unit is a voyage/call,
    not an AIS row.
    """
    arr = np.asarray(list(scores), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("No finite conformity scores")
    if not (0 < confidence_level < 1):
        raise ValueError("confidence_level must be in (0,1)")
    arr.sort()
    n = int(arr.size)
    k = int(min(n, max(1, math.ceil((n + 1) * confidence_level))))
    return float(arr[k - 1]), k, n


def enrich_with_source_state(m2_oof: pd.DataFrame, states: pd.DataFrame, cfg: M4Config | None = None) -> pd.DataFrame:
    """Join only the exact source state available at prediction time."""
    cfg = cfg or M4Config()
    x = m2_oof.copy()
    for c in ["ground_truth_time", "decision_time", "source_observation_time"]:
        x[c] = pd.to_datetime(x[c], errors="raise")
    s = states[[
        "mmsi", "recorded_at", "destination_clean", "stale_risk_level", "motion_state",
        "dynamic_state_age_s", "stop_duration_s",
    ]].drop_duplicates(["mmsi", "recorded_at"]).copy()
    s["recorded_at"] = pd.to_datetime(s["recorded_at"], errors="raise")
    x = x.merge(
        s,
        left_on=["mmsi", "source_observation_time"],
        right_on=["mmsi", "recorded_at"],
        how="left",
        validate="many_to_one",
    ).drop(columns=["recorded_at"])
    if x["stale_risk_level"].isna().any():
        raise AssertionError("M4 exact source-state join failed")
    x["provider_observation_age_s"] = (
        x["decision_time"] - x["source_observation_time"]
    ).dt.total_seconds().clip(lower=0)
    x["destination_support_port"] = [
        int(destination_supports_port(v, p))
        for v, p in zip(x["destination_clean"], x["port"])
    ]
    x["pred_route_physics_h"] = x["pred_m2_route_knn_h"].astype(float)
    return assign_reliability(x, cfg)


def assign_reliability(df: pd.DataFrame, cfg: M4Config | None = None) -> pd.DataFrame:
    """Transparent, causal reliability diagnostics; not a probability.

    HIGH requires all four observable checks plus a supported prediction horizon:
    target intent in current AIS destination, route proximity, source freshness,
    and non-trivial recent robust speed. MEDIUM is partial support; LOW is outside
    the supported horizon or has fewer than two quality checks.
    """
    cfg = cfg or M4Config()
    out = df.copy()
    out["rel_horizon_supported"] = out["pred_route_physics_h"].astype(float).le(cfg.max_supported_pred_h)
    out["rel_intent_supported"] = out["destination_support_port"].astype(int).eq(1)
    out["rel_route_supported"] = out["knn_mean_cross_track_km"].astype(float).le(cfg.max_route_cross_track_km)
    out["rel_source_fresh"] = (
        out["provider_observation_age_s"].astype(float).le(cfg.max_provider_age_s)
        & ~out["stale_risk_level"].astype(str).eq("HIGH")
    )
    out["rel_speed_supported"] = out["sog_median_30m_kn"].astype(float).ge(cfg.min_robust_speed_kn)
    checks = ["rel_intent_supported", "rel_route_supported", "rel_source_fresh", "rel_speed_supported"]
    out["reliability_checks_passed"] = out[checks].sum(axis=1).astype(int)
    high = out["rel_horizon_supported"] & out["reliability_checks_passed"].eq(len(checks))
    medium = out["rel_horizon_supported"] & out["reliability_checks_passed"].between(2, len(checks) - 1)
    out["reliability_tier"] = np.where(high, "HIGH", np.where(medium, "MEDIUM", "LOW"))
    out["reliability_definition"] = "diagnostic_not_probability"
    return out


def assign_temporal_roles(panel: pd.DataFrame, cfg: M4Config | None = None) -> pd.DataFrame:
    cfg = cfg or M4Config()
    calls = (
        panel[["session_id", "port", "mmsi", "ground_truth_time"]]
        .drop_duplicates("session_id")
        .sort_values(["ground_truth_time", "session_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    if len(calls) <= cfg.calibration_calls:
        raise ValueError("Insufficient M2 OOF calls for M4 calibration/evaluation split")
    calls["m4_role"] = "temporal_evaluation"
    calls.loc[: cfg.calibration_calls - 1, "m4_role"] = "calibration"
    return calls


def simultaneous_call_scores(df: pd.DataFrame, pred_col: str, reliability_tier: str = "HIGH") -> pd.DataFrame:
    x = df[df["reliability_tier"].eq(reliability_tier)].copy()
    if x.empty:
        raise ValueError(f"No {reliability_tier} rows available for conformity scores")
    x["abs_error_h"] = (x[pred_col].astype(float) - x["true_tta_h"].astype(float)).abs()
    scores = (
        x.groupby("session_id", as_index=False)
        .agg(
            conformity_score_h=("abs_error_h", "max"),
            points=("abs_error_h", "size"),
            port=("port", "first"),
            mmsi=("mmsi", "first"),
            ground_truth_time=("ground_truth_time", "first"),
        )
    )
    return scores


def add_symmetric_interval(df: pd.DataFrame, pred_col: str, half_width_h: float, prefix: str = "m4_pi") -> pd.DataFrame:
    out = df.copy()
    pred = out[pred_col].astype(float)
    out[f"{prefix}_lower_tta_h"] = np.maximum(0.0, pred - float(half_width_h))
    out[f"{prefix}_upper_tta_h"] = pred + float(half_width_h)
    out[f"{prefix}_effective_width_h"] = out[f"{prefix}_upper_tta_h"] - out[f"{prefix}_lower_tta_h"]
    out[f"{prefix}_covered"] = (
        out["true_tta_h"].astype(float).ge(out[f"{prefix}_lower_tta_h"])
        & out["true_tta_h"].astype(float).le(out[f"{prefix}_upper_tta_h"])
    )
    dt = pd.to_datetime(out["decision_time"], errors="raise")
    out[f"{prefix}_lower_arrival"] = dt + pd.to_timedelta(out[f"{prefix}_lower_tta_h"], unit="h")
    out[f"{prefix}_upper_arrival"] = dt + pd.to_timedelta(out[f"{prefix}_upper_tta_h"], unit="h")
    return out


def interval_metrics(df: pd.DataFrame, covered_col: str, width_col: str) -> dict:
    x = df.copy()
    if x.empty:
        return {
            "rows": 0, "calls": 0, "row_coverage": np.nan,
            "voyage_balanced_point_coverage": np.nan, "trajectory_coverage": np.nan,
            "mean_effective_width_h": np.nan, "median_effective_width_h": np.nan,
        }
    per_call_point = x.groupby("session_id")[covered_col].mean()
    per_call_all = x.groupby("session_id")[covered_col].all()
    return {
        "rows": int(len(x)),
        "calls": int(x["session_id"].nunique()),
        "row_coverage": float(x[covered_col].mean()),
        "voyage_balanced_point_coverage": float(per_call_point.mean()),
        "trajectory_coverage": float(per_call_all.mean()),
        "mean_effective_width_h": float(x[width_col].mean()),
        "median_effective_width_h": float(x[width_col].median()),
    }


def voyage_balanced_mae_h(df: pd.DataFrame, pred_col: str, max_true_h: float | None = 24.0) -> float:
    x = df.copy()
    if max_true_h is not None:
        x = x[x["true_tta_h"].astype(float).le(float(max_true_h))]
    if x.empty:
        return float("nan")
    x["ae"] = (x[pred_col].astype(float) - x["true_tta_h"].astype(float)).abs()
    return float(x.groupby("session_id")["ae"].mean().mean())


def apply_causal_arrival_ewma(
    df: pd.DataFrame,
    pred_col: str,
    alpha: float,
    reset_gap_h: float = 2.0,
    out_pred_col: str = "pred_stabilized_h",
    out_arrival_col: str = "pred_stabilized_arrival",
) -> pd.DataFrame:
    """Causal EWMA on absolute arrival timestamps, reset after long observation gaps."""
    if not (0 < alpha <= 1):
        raise ValueError("alpha must be in (0,1]")
    parts = []
    for _, g in df.sort_values(["session_id", "decision_time"], kind="mergesort").groupby("session_id", sort=False):
        q = g.copy()
        times = pd.to_datetime(q["decision_time"], errors="raise")
        raw_arr = times + pd.to_timedelta(q[pred_col].astype(float), unit="h")
        filt: list[pd.Timestamp] = []
        prev_arr: pd.Timestamp | None = None
        prev_t: pd.Timestamp | None = None
        for t, a in zip(times, raw_arr):
            if prev_arr is None or prev_t is None or (t - prev_t).total_seconds() / 3600.0 > reset_gap_h:
                f = a
            else:
                # Convex combination in ns. Integer conversion makes the operation deterministic.
                f = pd.Timestamp(int(alpha * a.value + (1.0 - alpha) * prev_arr.value))
            filt.append(f)
            prev_arr, prev_t = f, t
        q[out_arrival_col] = filt
        q[out_pred_col] = [max(0.0, (a - t).total_seconds() / 3600.0) for a, t in zip(q[out_arrival_col], times)]
        parts.append(q)
    return pd.concat(parts, ignore_index=True)


def revision_metrics(df: pd.DataFrame, arrival_col: str, reset_gap_h: float = 2.0) -> dict:
    x = df.sort_values(["session_id", "decision_time"], kind="mergesort").copy()
    x["decision_time"] = pd.to_datetime(x["decision_time"], errors="raise")
    x[arrival_col] = pd.to_datetime(x[arrival_col], errors="raise")
    x["step_gap_h"] = x.groupby("session_id")["decision_time"].diff().dt.total_seconds() / 3600.0
    x["revision_min"] = x.groupby("session_id")[arrival_col].diff().dt.total_seconds().abs() / 60.0
    x.loc[x["step_gap_h"].gt(reset_gap_h), "revision_min"] = np.nan
    q = x[x["revision_min"].notna()]
    if q.empty:
        return {"updates": 0, "calls": 0}
    return {
        "updates": int(len(q)),
        "calls": int(q["session_id"].nunique()),
        "median_revision_min": float(q["revision_min"].median()),
        "p90_revision_min": float(q["revision_min"].quantile(0.90)),
        "p95_revision_min": float(q["revision_min"].quantile(0.95)),
        "revision_gt15_fraction": float(q["revision_min"].gt(15).mean()),
        "revision_gt30_fraction": float(q["revision_min"].gt(30).mean()),
        "revision_gt60_fraction": float(q["revision_min"].gt(60).mean()),
    }


def parse_ais_eta_nearest_year(eta_value: object, observation_time: pd.Timestamp) -> pd.Timestamp | pd.NaT:
    """Parse AIS MM/DD HH:MM by choosing the candidate year nearest observation time."""
    if eta_value is None or (isinstance(eta_value, float) and np.isnan(eta_value)):
        return pd.NaT
    text = str(eta_value).strip()
    try:
        md, hm = text.split()
        month, day = [int(v) for v in md.split("/")]
        hour, minute = [int(v) for v in hm.split(":")]
    except Exception:
        return pd.NaT
    if month <= 0 or day <= 0 or hour >= 24 or minute >= 60:
        return pd.NaT
    obs = pd.Timestamp(observation_time)
    candidates = []
    for year in [obs.year - 1, obs.year, obs.year + 1]:
        try:
            candidates.append(pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute))
        except ValueError:
            continue
    if not candidates:
        return pd.NaT
    return min(candidates, key=lambda t: abs((t - obs).total_seconds()))
