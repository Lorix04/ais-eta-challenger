from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import pandas as pd

from .m4 import conformal_order_statistic, revision_metrics, voyage_balanced_mae_h


@dataclass(frozen=True)
class M5Config:
    max_eval_h: float = 24.0
    min_port_oof_calls: int = 8
    min_cold_calls_per_port: int = 5
    max_route_degradation_fraction: float = 0.10
    min_temporal_high_calls_per_port: int = 5
    min_port_trajectory_coverage: float = 0.80
    min_route_family_calls_for_warning: int = 8
    route_family_warning_degradation_fraction: float = 0.10
    min_seen_calls_for_warning: int = 5
    bootstrap_draws: int = 10000
    bootstrap_seed: int = 20260918
    confidence_level: float = 0.90


def paired_bootstrap_mean_gain(
    gains_h: Iterable[float],
    draws: int = 10000,
    seed: int = 20260918,
) -> tuple[float, float, float]:
    """Call-level paired bootstrap for geodesic AE - route AE.

    The caller must supply one gain per independent call. Positive values favor
    the route model. This is deliberately not a row bootstrap.
    """
    arr = np.asarray(list(gains_h), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = np.empty(int(draws), dtype=float)
    for i in range(int(draws)):
        boot[i] = rng.choice(arr, size=arr.size, replace=True).mean()
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(arr.mean()), float(lo), float(hi)


def _call_mae(df: pd.DataFrame, pred_col: str, max_true_h: float | None) -> pd.Series:
    x = df.copy()
    if max_true_h is not None:
        x = x[x["true_tta_h"].astype(float).le(float(max_true_h))]
    if x.empty:
        return pd.Series(dtype=float)
    x["ae_h"] = (x[pred_col].astype(float) - x["true_tta_h"].astype(float)).abs()
    return x.groupby("session_id")["ae_h"].mean()


def build_call_stress_table(
    panel: pd.DataFrame,
    ground_truth_confidence: pd.DataFrame,
    max_true_h: float = 24.0,
) -> pd.DataFrame:
    """Build one stress-test row per independent OOF call.

    Regime labels are diagnostic only. ``initial_*`` values come from the first
    OOF decision available for each call, while fractions summarize how often a
    causal state was present over that call's OOF prediction points.
    """
    x = panel.copy()
    x["decision_time"] = pd.to_datetime(x["decision_time"], errors="raise")
    x["ground_truth_time"] = pd.to_datetime(x["ground_truth_time"], errors="raise")
    x["ae_geo_h"] = (x["pred_m2_geodesic_h"].astype(float) - x["true_tta_h"].astype(float)).abs()
    x["ae_route_h"] = (x["pred_m2_route_knn_h"].astype(float) - x["true_tta_h"].astype(float)).abs()

    first = (
        x.sort_values(["session_id", "decision_time"], kind="mergesort")
        .groupby("session_id", as_index=False)
        .first()[[
            "session_id", "route_family", "destination_support_port",
            "reliability_tier", "knn_mean_cross_track_km",
        ]]
        .rename(columns={
            "route_family": "initial_route_family",
            "destination_support_port": "initial_intent_supported",
            "reliability_tier": "initial_reliability_tier",
            "knn_mean_cross_track_km": "initial_cross_track_km",
        })
    )

    all_metrics = (
        x.groupby(["port", "session_id", "mmsi", "name", "cold_vessel_in_fold"], as_index=False)
        .agg(
            all_geo_mae_h=("ae_geo_h", "mean"),
            all_route_mae_h=("ae_route_h", "mean"),
            prediction_points=("session_id", "size"),
            intent_support_fraction=("destination_support_port", "mean"),
            route_support_fraction=("rel_route_supported", "mean"),
            high_reliability_fraction=("reliability_tier", lambda s: float((s == "HIGH").mean())),
            median_cross_track_km=("knn_mean_cross_track_km", "median"),
            max_true_tta_h=("true_tta_h", "max"),
        )
    )

    under = x[x["true_tta_h"].astype(float).le(float(max_true_h))].copy()
    under_metrics = (
        under.groupby("session_id", as_index=False)
        .agg(
            under24_geo_mae_h=("ae_geo_h", "mean"),
            under24_route_mae_h=("ae_route_h", "mean"),
            under24_points=("session_id", "size"),
        )
    )

    out = all_metrics.merge(under_metrics, on="session_id", how="left", validate="one_to_one")
    out = out.merge(first, on="session_id", how="left", validate="one_to_one")
    conf = ground_truth_confidence[["session_id", "ground_truth_confidence"]].drop_duplicates("session_id")
    out = out.merge(conf, on="session_id", how="left", validate="one_to_one")
    out["under24_gain_h"] = out["under24_geo_mae_h"] - out["under24_route_mae_h"]
    out["all_gain_h"] = out["all_geo_mae_h"] - out["all_route_mae_h"]
    out["route_better_under24"] = out["under24_gain_h"].gt(0)
    return out


def grouped_route_metrics(
    call_table: pd.DataFrame,
    group_cols: list[str],
    cfg: M5Config | None = None,
) -> pd.DataFrame:
    cfg = cfg or M5Config()
    rows: list[dict] = []
    for keys, g in call_table.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        q = g.dropna(subset=["under24_geo_mae_h", "under24_route_mae_h"]).copy()
        if q.empty:
            continue
        geo = float(q["under24_geo_mae_h"].mean())
        route = float(q["under24_route_mae_h"].mean())
        mean_gain, ci_lo, ci_hi = paired_bootstrap_mean_gain(
            q["under24_gain_h"], cfg.bootstrap_draws, cfg.bootstrap_seed + len(rows)
        )
        row = {c: k for c, k in zip(group_cols, keys)}
        row.update({
            "calls": int(q["session_id"].nunique()),
            "unique_vessels": int(q["mmsi"].nunique()),
            "geodesic_voyage_mae_h": geo,
            "route_voyage_mae_h": route,
            "route_improvement_fraction": float(1.0 - route / geo) if geo > 0 else np.nan,
            "mean_paired_gain_h": mean_gain,
            "bootstrap_gain_ci95_low_h": ci_lo,
            "bootstrap_gain_ci95_high_h": ci_hi,
            "route_better_call_fraction": float(q["route_better_under24"].mean()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def temporal_reliability_by_port(eval_df: pd.DataFrame) -> pd.DataFrame:
    x = eval_df.copy()
    x["decision_time"] = pd.to_datetime(x["decision_time"], errors="raise")
    rows: list[dict] = []
    for (port, tier), g in x.groupby(["port", "reliability_tier"], dropna=False):
        rows.append({
            "port": port,
            "reliability_tier": tier,
            "rows": int(len(g)),
            "calls": int(g["session_id"].nunique()),
            "raw_voyage_balanced_mae_under24_h": voyage_balanced_mae_h(g, "pred_raw_h", 24.0),
            "stabilized_voyage_balanced_mae_under24_h": voyage_balanced_mae_h(g, "pred_stabilized_h", 24.0),
            "intent_supported_fraction": float(g["destination_support_port"].mean()),
            "route_supported_fraction": float(g["rel_route_supported"].mean()),
            "source_fresh_fraction": float(g["rel_source_fresh"].mean()),
        })
    return pd.DataFrame(rows)


def temporal_stability_by_port(eval_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for port, g in eval_df.groupby("port"):
        raw = revision_metrics(g, "pred_raw_arrival")
        smooth = revision_metrics(g, "pred_stabilized_arrival")
        rows.append({
            "port": port,
            "calls": int(g["session_id"].nunique()),
            "raw_under24_mae_h": voyage_balanced_mae_h(g, "pred_raw_h", 24.0),
            "stabilized_under24_mae_h": voyage_balanced_mae_h(g, "pred_stabilized_h", 24.0),
            "raw_p90_revision_min": raw["p90_revision_min"],
            "stabilized_p90_revision_min": smooth["p90_revision_min"],
            "raw_revision_gt30_fraction": raw["revision_gt30_fraction"],
            "stabilized_revision_gt30_fraction": smooth["revision_gt30_fraction"],
        })
    return pd.DataFrame(rows)


def port_interval_diagnostics(
    calibration_scores: pd.DataFrame,
    eval_df: pd.DataFrame,
    pooled_half_width_h: float,
    confidence_level: float = 0.90,
) -> pd.DataFrame:
    """Diagnostic per-port uncertainty; does not replace pooled M4 interval."""
    rows: list[dict] = []
    for port in sorted(set(calibration_scores["port"]) | set(eval_df["port"])):
        cal = calibration_scores[calibration_scores["port"].eq(port)].copy()
        ev = eval_df[(eval_df["port"].eq(port)) & eval_df["reliability_tier"].eq("HIGH")].copy()
        if cal.empty:
            q = np.nan; k = 0; n = 0
        else:
            q, k, n = conformal_order_statistic(cal["conformity_score_h"], confidence_level)
        pooled_cov = np.nan
        pooled_traj = np.nan
        port_cov = np.nan
        port_traj = np.nan
        if not ev.empty:
            pooled_covered = (
                ev["true_tta_h"].astype(float).ge(np.maximum(0.0, ev["pred_route_physics_h"].astype(float) - pooled_half_width_h))
                & ev["true_tta_h"].astype(float).le(ev["pred_route_physics_h"].astype(float) + pooled_half_width_h)
            )
            pooled_cov = float(pooled_covered.mean())
            pooled_traj = float(pooled_covered.groupby(ev["session_id"]).all().mean())
            if np.isfinite(q):
                port_covered = (
                    ev["true_tta_h"].astype(float).ge(np.maximum(0.0, ev["pred_route_physics_h"].astype(float) - q))
                    & ev["true_tta_h"].astype(float).le(ev["pred_route_physics_h"].astype(float) + q)
                )
                port_cov = float(port_covered.mean())
                port_traj = float(port_covered.groupby(ev["session_id"]).all().mean())
        rows.append({
            "port": port,
            "calibration_calls": int(n),
            "port_specific_conformal_order_k": int(k),
            "port_specific_q_h": float(q) if np.isfinite(q) else np.nan,
            "uses_max_calibration_score": bool(n > 0 and k == n),
            "temporal_high_rows": int(len(ev)),
            "temporal_high_calls": int(ev["session_id"].nunique()),
            "pooled_m4_half_width_h": float(pooled_half_width_h),
            "pooled_row_coverage": pooled_cov,
            "pooled_trajectory_coverage": pooled_traj,
            "port_specific_row_coverage_diagnostic": port_cov,
            "port_specific_trajectory_coverage_diagnostic": port_traj,
            "port_specific_interval_status": "DIAGNOSTIC_ONLY_SMALL_N",
        })
    return pd.DataFrame(rows)


def evaluate_m5_gate(
    port_metrics: pd.DataFrame,
    cold_metrics: pd.DataFrame,
    interval_metrics: pd.DataFrame,
    route_family_metrics: pd.DataFrame,
    cfg: M5Config | None = None,
) -> dict:
    cfg = cfg or M5Config()
    port_checks: list[dict] = []
    for r in port_metrics.itertuples(index=False):
        if int(r.calls) < cfg.min_port_oof_calls:
            continue
        degradation = -float(r.route_improvement_fraction)
        port_checks.append({
            "port": str(r.port),
            "calls": int(r.calls),
            "passed": bool(degradation <= cfg.max_route_degradation_fraction),
            "route_improvement_fraction": float(r.route_improvement_fraction),
        })

    cold_checks: list[dict] = []
    cold = cold_metrics[cold_metrics["cold_vessel_in_fold"].astype(bool)] if not cold_metrics.empty else cold_metrics
    for r in cold.itertuples(index=False):
        if int(r.calls) < cfg.min_cold_calls_per_port:
            continue
        degradation = -float(r.route_improvement_fraction)
        cold_checks.append({
            "port": str(r.port),
            "calls": int(r.calls),
            "passed": bool(degradation <= cfg.max_route_degradation_fraction),
            "route_improvement_fraction": float(r.route_improvement_fraction),
        })

    interval_checks: list[dict] = []
    for r in interval_metrics.itertuples(index=False):
        if int(r.temporal_high_calls) < cfg.min_temporal_high_calls_per_port:
            continue
        interval_checks.append({
            "port": str(r.port),
            "calls": int(r.temporal_high_calls),
            "passed": bool(float(r.pooled_trajectory_coverage) >= cfg.min_port_trajectory_coverage),
            "pooled_trajectory_coverage": float(r.pooled_trajectory_coverage),
        })

    warnings: list[dict] = []
    if not route_family_metrics.empty:
        for r in route_family_metrics.itertuples(index=False):
            if int(r.calls) >= cfg.min_route_family_calls_for_warning and float(r.route_improvement_fraction) < -cfg.route_family_warning_degradation_fraction:
                warnings.append({
                    "type": "ROUTE_FAMILY_DEGRADATION",
                    "port": str(r.port),
                    "route_family": str(r.initial_route_family),
                    "calls": int(r.calls),
                    "route_improvement_fraction": float(r.route_improvement_fraction),
                })

    passed = (
        bool(port_checks) and all(v["passed"] for v in port_checks)
        and bool(cold_checks) and all(v["passed"] for v in cold_checks)
        and bool(interval_checks) and all(v["passed"] for v in interval_checks)
    )
    status = "M5_GO_M6_SCOPE_LIMITED" if passed else "M5_GENERALIZATION_RISK_HIGH"
    return {
        "status": status,
        "passed": bool(passed),
        "port_checks": port_checks,
        "cold_vessel_checks": cold_checks,
        "interval_checks": interval_checks,
        "warnings": warnings,
        "unseen_port_generalization": "NOT_ESTABLISHED",
    }
