#!/usr/bin/env python3
"""M4 voyage-level uncertainty, reliability diagnostics and ETA stability.

Design constraints:
- retained point predictor is M2 route-kNN distance / robust trailing speed;
- M3 residual model is not revived;
- only M2 cross-fitted OOF calls are used here;
- 16 locked M2 chronological final calls remain untouched;
- calibration unit is an independent port call, not an AIS row;
- primary interval is issued only in a causal HIGH-reliability regime.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m4 import (  # noqa: E402
    M4Config,
    add_symmetric_interval,
    apply_causal_arrival_ewma,
    assign_temporal_roles,
    conformal_order_statistic,
    enrich_with_source_state,
    interval_metrics,
    parse_ais_eta_nearest_year,
    revision_metrics,
    simultaneous_call_scores,
    voyage_balanced_mae_h,
)

REPORTS = ROOT / "reports"
STATES_PATH = ROOT / "data" / "derived" / "m0c_ship_states.pkl.gz"
M2_OOF = REPORTS / "m2_oof_route_predictions.csv"
M2_SPLITS = REPORTS / "m2_call_splits.csv"
POINT_MODEL = "pred_route_physics_h"
PRIMARY_TIER = "HIGH"


def _jsonify(value):
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def reliability_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tier, g in df.groupby("reliability_tier", sort=True):
        z = g.copy()
        z["abs_error_h"] = (z[POINT_MODEL] - z["true_tta_h"]).abs()
        under24 = z[z["true_tta_h"].le(24)].copy()
        rows.append({
            "reliability_tier": tier,
            "rows": int(len(z)),
            "calls": int(z.session_id.nunique()),
            "row_mae_h": float(z.abs_error_h.mean()),
            "row_median_ae_h": float(z.abs_error_h.median()),
            "voyage_balanced_mae_h": float(z.groupby("session_id").abs_error_h.mean().mean()),
            "under24_rows": int(len(under24)),
            "under24_calls": int(under24.session_id.nunique()),
            "under24_voyage_balanced_mae_h": (
                float(under24.groupby("session_id").apply(
                    lambda q: (q[POINT_MODEL] - q["true_tta_h"]).abs().mean(), include_groups=False
                ).mean()) if not under24.empty else np.nan
            ),
        })
    return pd.DataFrame(rows)


def select_stabilizer(calibration: pd.DataFrame, cfg: M4Config) -> tuple[pd.DataFrame, float]:
    rows = []
    for alpha in cfg.smoothing_alpha_grid:
        q = apply_causal_arrival_ewma(
            calibration, POINT_MODEL, float(alpha), cfg.smoothing_reset_gap_h,
            out_pred_col="pred_stabilized_h", out_arrival_col="pred_stabilized_arrival",
        )
        rev = revision_metrics(q, "pred_stabilized_arrival", cfg.smoothing_reset_gap_h)
        rows.append({
            "alpha": float(alpha),
            "calibration_voyage_balanced_mae_under24_h": voyage_balanced_mae_h(q, "pred_stabilized_h", 24.0),
            **rev,
        })
    grid = pd.DataFrame(rows).sort_values(
        ["calibration_voyage_balanced_mae_under24_h", "p90_revision_min", "alpha"],
        ascending=[True, True, False], kind="mergesort"
    ).reset_index(drop=True)
    selected = float(grid.iloc[0].alpha)
    grid["selected_on_calibration"] = grid.alpha.eq(selected)
    return grid, selected


def stability_comparison(eval_panel: pd.DataFrame, selected_alpha: float, cfg: M4Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = apply_causal_arrival_ewma(
        eval_panel, POINT_MODEL, 1.0, cfg.smoothing_reset_gap_h,
        out_pred_col="pred_raw_h", out_arrival_col="pred_raw_arrival",
    )
    stable = apply_causal_arrival_ewma(
        eval_panel, POINT_MODEL, selected_alpha, cfg.smoothing_reset_gap_h,
        out_pred_col="pred_stabilized_h", out_arrival_col="pred_stabilized_arrival",
    )
    out = raw.drop(columns=[c for c in ["pred_stabilized_h", "pred_stabilized_arrival"] if c in raw.columns], errors="ignore")
    out = out.merge(
        stable[["session_id", "decision_time", "pred_stabilized_h", "pred_stabilized_arrival"]],
        on=["session_id", "decision_time"], how="left", validate="one_to_one"
    )

    rows = []
    for label, pred_col, arrival_col in [
        ("raw_route_physics", "pred_raw_h", "pred_raw_arrival"),
        (f"causal_ewma_alpha_{selected_alpha:.1f}", "pred_stabilized_h", "pred_stabilized_arrival"),
    ]:
        for scope, sub in [
            ("all", out),
            ("high_reliability", out[out.reliability_tier.eq("HIGH")]),
        ]:
            rev = revision_metrics(sub, arrival_col, cfg.smoothing_reset_gap_h)
            rows.append({
                "model": label,
                "scope": scope,
                "voyage_balanced_mae_under24_h": voyage_balanced_mae_h(sub, pred_col, 24.0),
                **rev,
            })
    return out, pd.DataFrame(rows)


def _load_tracks(data_zip: Path) -> pd.DataFrame:
    with zipfile.ZipFile(data_zip) as zf:
        raw = zf.read("data/vessel_tracks.csv")
    tracks = pd.read_csv(io.BytesIO(raw))
    tracks["last_update"] = pd.to_datetime(tracks["last_update"], errors="coerce")
    return tracks


def reported_eta_alignment(eval_panel: pd.DataFrame, data_zip: Path | None, cfg: M4Config) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if data_zip is None or not data_zip.exists():
        return pd.DataFrame(), pd.DataFrame(), {
            "status": "DATA_ZIP_UNAVAILABLE",
            "aligned_rows_5min": 0,
            "aligned_calls_5min": 0,
        }
    tracks = _load_tracks(data_zip)
    t = tracks[["mmsi", "eta", "destination", "last_update"]].copy()
    x = eval_panel.merge(t, on="mmsi", how="left", validate="many_to_one", suffixes=("", "_snapshot"))
    x["snapshot_offset_min"] = (
        x["source_observation_time"] - x["last_update"]
    ).abs().dt.total_seconds() / 60.0
    x["reported_eta_parsed"] = [
        parse_ais_eta_nearest_year(v, u) if pd.notna(u) else pd.NaT
        for v, u in zip(x["eta"], x["last_update"])
    ]
    x["model_arrival_raw"] = pd.to_datetime(x["decision_time"]) + pd.to_timedelta(x[POINT_MODEL], unit="h")
    tol = float(cfg.reported_eta_alignment_tolerance_min)
    aligned = x[
        x["reported_eta_parsed"].notna()
        & x["snapshot_offset_min"].le(tol)
    ].copy()
    if not aligned.empty:
        aligned["reported_minus_model_h"] = (
            aligned["reported_eta_parsed"] - aligned["model_arrival_raw"]
        ).dt.total_seconds() / 3600.0
    nearest = x[x["eta"].notna() & x["last_update"].notna()].sort_values("snapshot_offset_min").head(50).copy()
    summary = {
        "status": "ALIGNED_ROWS_AVAILABLE" if not aligned.empty else "NO_TEMPORALLY_ALIGNED_REPORTED_ETA",
        "alignment_tolerance_min": tol,
        "aligned_rows_5min": int(len(aligned)),
        "aligned_calls_5min": int(aligned.session_id.nunique()) if not aligned.empty else 0,
        "rows_within_60min": int((x["eta"].notna() & x["snapshot_offset_min"].le(60)).sum()),
        "calls_within_60min": int(x.loc[x["eta"].notna() & x["snapshot_offset_min"].le(60), "session_id"].nunique()),
        "closest_snapshot_offset_min": float(nearest.snapshot_offset_min.min()) if not nearest.empty else np.nan,
    }
    return aligned, nearest, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-zip", type=Path, default=None, help="Optional original data ZIP used only for reported-ETA timestamp alignment")
    args = ap.parse_args()
    cfg = M4Config()

    m2 = pd.read_csv(M2_OOF)
    states = pd.read_pickle(STATES_PATH)
    panel = enrich_with_source_state(m2, states, cfg)
    roles = assign_temporal_roles(panel, cfg)
    panel = panel.merge(roles[["session_id", "m4_role"]], on="session_id", how="left", validate="many_to_one")

    locked = pd.read_csv(M2_SPLITS)
    locked_ids = set(locked.loc[locked.m2_split.eq("final_chronological_test"), "session_id"].astype(str))
    if set(panel.session_id.astype(str)) & locked_ids:
        raise AssertionError("M4 panel unexpectedly contains locked final calls")

    calibration = panel[panel.m4_role.eq("calibration")].copy()
    evaluation = panel[panel.m4_role.eq("temporal_evaluation")].copy()
    if calibration.session_id.nunique() != cfg.calibration_calls:
        raise AssertionError("Unexpected M4 calibration call count")

    # Primary uncertainty: simultaneous path-level band over causal HIGH-reliability rows.
    scores = simultaneous_call_scores(calibration, POINT_MODEL, PRIMARY_TIER)
    sensitivity_rows = []
    primary_q = None
    primary_metrics = None
    primary_eval = None
    for level in [0.80, 0.85, cfg.confidence_level]:
        q, order_k, n_scores = conformal_order_statistic(scores.conformity_score_h, level)
        e = evaluation[evaluation.reliability_tier.eq(PRIMARY_TIER)].copy()
        e = add_symmetric_interval(e, POINT_MODEL, q, prefix="m4_pi")
        met = interval_metrics(e, "m4_pi_covered", "m4_pi_effective_width_h")
        sensitivity_rows.append({
            "confidence_level": float(level),
            "half_width_h": float(q),
            "conformal_order_k": int(order_k),
            "calibration_calls": int(n_scores),
            **met,
        })
        if abs(level - cfg.confidence_level) < 1e-12:
            primary_q, primary_metrics, primary_eval = q, met, e
    sensitivity = pd.DataFrame(sensitivity_rows)
    assert primary_q is not None and primary_metrics is not None and primary_eval is not None

    # Transparent reliability diagnostics on the temporal evaluation calls.
    rel_metrics = reliability_metrics(evaluation)

    # Stability: choose a tiny fixed EWMA grid on calibration only; evaluate once later in time.
    smoothing_grid, selected_alpha = select_stabilizer(calibration, cfg)
    eval_with_stability, stability = stability_comparison(evaluation, selected_alpha, cfg)
    raw_eval = stability[(stability.model.eq("raw_route_physics")) & stability.scope.eq("all")].iloc[0]
    smooth_eval = stability[(stability.model.str.startswith("causal_ewma")) & stability.scope.eq("all")].iloc[0]
    jitter_p90_improvement = 1.0 - float(smooth_eval.p90_revision_min) / float(raw_eval.p90_revision_min)
    mae_change = 1.0 - float(smooth_eval.voyage_balanced_mae_under24_h) / float(raw_eval.voyage_balanced_mae_under24_h)
    stability_status = (
        "KEEP_CAUSAL_STABILIZER"
        if float(smooth_eval.voyage_balanced_mae_under24_h) <= float(raw_eval.voyage_balanced_mae_under24_h) * 1.02
        and jitter_p90_improvement >= 0.20
        else "REPORT_ONLY_NO_STABILIZER"
    )

    # Merge primary intervals into the full temporal-evaluation output; withhold for non-HIGH rows.
    eval_output = eval_with_stability.copy()
    interval_cols = [
        "session_id", "decision_time", "m4_pi_lower_tta_h", "m4_pi_upper_tta_h",
        "m4_pi_effective_width_h", "m4_pi_covered", "m4_pi_lower_arrival", "m4_pi_upper_arrival",
    ]
    eval_output = eval_output.merge(primary_eval[interval_cols], on=["session_id", "decision_time"], how="left", validate="one_to_one")
    eval_output["m4_interval_status"] = np.where(
        eval_output.reliability_tier.eq("HIGH"),
        f"VOYAGE_LEVEL_{int(cfg.confidence_level*100)}PCT_SIMULTANEOUS",
        "WITHHELD_NON_HIGH_RELIABILITY",
    )

    aligned, nearest, reported_summary = reported_eta_alignment(evaluation, args.data_zip, cfg)

    # Pre-specified M4 gate: high-reliability simultaneous interval only.
    interval_gate_pass = (
        primary_metrics["calls"] >= cfg.min_eval_calls
        and primary_metrics["trajectory_coverage"] >= cfg.min_eval_trajectory_coverage
        and float(primary_q) <= cfg.max_primary_half_width_h
    )
    gate_status = "M4_CALIBRATED_HIGH_RELIABILITY_GO_M5" if interval_gate_pass else "M4_CALIBRATION_INSUFFICIENT"

    # Persist artifacts.
    REPORTS.mkdir(exist_ok=True)
    roles.to_csv(REPORTS / "m4_call_roles.csv", index=False)
    scores.to_csv(REPORTS / "m4_calibration_call_scores.csv", index=False)
    sensitivity.to_csv(REPORTS / "m4_interval_sensitivity.csv", index=False)
    rel_metrics.to_csv(REPORTS / "m4_reliability_metrics.csv", index=False)
    smoothing_grid.to_csv(REPORTS / "m4_stabilizer_calibration_grid.csv", index=False)
    stability.to_csv(REPORTS / "m4_stability_metrics.csv", index=False)
    eval_output.to_csv(REPORTS / "m4_temporal_eval_predictions.csv", index=False)
    aligned.to_csv(REPORTS / "m4_reported_eta_aligned.csv", index=False)
    nearest.to_csv(REPORTS / "m4_reported_eta_nearest_snapshots.csv", index=False)

    high_rel = rel_metrics[rel_metrics.reliability_tier.eq("HIGH")].iloc[0]
    summary = {
        "status": gate_status,
        "retained_point_model": "M2 route-kNN distance / robust trailing speed",
        "m3_residual_model_used": False,
        "m2_oof_calls_available": int(panel.session_id.nunique()),
        "calibration_calls": int(calibration.session_id.nunique()),
        "temporal_evaluation_calls": int(evaluation.session_id.nunique()),
        "locked_final_calls_scored": 0,
        "primary_interval": {
            "scope": "HIGH reliability rows only; reliability defined using prediction-time information",
            "confidence_level": cfg.confidence_level,
            "conformity_unit": "maximum absolute error within each calibration voyage/call",
            "half_width_h": float(primary_q),
            **primary_metrics,
        },
        "reliability": {
            "definition": "diagnostic tier, not probability of correctness",
            "high_eval_rows": int(high_rel.rows),
            "high_eval_calls": int(high_rel.calls),
            "high_under24_voyage_balanced_mae_h": float(high_rel.under24_voyage_balanced_mae_h),
            "high_requirements": {
                "predicted_horizon_max_h": cfg.max_supported_pred_h,
                "AIS_destination_supports_target_port": True,
                "route_cross_track_max_km": cfg.max_route_cross_track_km,
                "provider_source_age_max_s": cfg.max_provider_age_s,
                "robust_speed_min_kn": cfg.min_robust_speed_kn,
            },
        },
        "stability": {
            "selected_alpha_from_calibration": selected_alpha,
            "selection_rule": "minimum calibration voyage-balanced MAE <=24h over fixed alpha grid; p90 revision as tie-breaker",
            "status": stability_status,
            "raw_eval_under24_mae_h": float(raw_eval.voyage_balanced_mae_under24_h),
            "stabilized_eval_under24_mae_h": float(smooth_eval.voyage_balanced_mae_under24_h),
            "eval_mae_improvement_fraction": float(mae_change),
            "raw_eval_p90_revision_min": float(raw_eval.p90_revision_min),
            "stabilized_eval_p90_revision_min": float(smooth_eval.p90_revision_min),
            "eval_p90_revision_improvement_fraction": float(jitter_p90_improvement),
        },
        "reported_eta": reported_summary,
        "gate": {
            "min_eval_calls": cfg.min_eval_calls,
            "min_trajectory_coverage": cfg.min_eval_trajectory_coverage,
            "max_primary_half_width_h": cfg.max_primary_half_width_h,
            "passed": bool(interval_gate_pass),
        },
    }
    (REPORTS / "m4_summary.json").write_text(json.dumps(_jsonify(summary), indent=2))
    (REPORTS / "m4_gate_result.json").write_text(json.dumps(_jsonify({
        "status": gate_status,
        "passed": bool(interval_gate_pass),
        "primary_half_width_h": float(primary_q),
        "trajectory_coverage": float(primary_metrics["trajectory_coverage"]),
        "evaluation_calls": int(primary_metrics["calls"]),
        "stability_status": stability_status,
        "reported_eta_status": reported_summary["status"],
    }), indent=2))

    print(json.dumps(_jsonify(summary), indent=2))


if __name__ == "__main__":
    main()
