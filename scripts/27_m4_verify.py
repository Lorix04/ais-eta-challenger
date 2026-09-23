#!/usr/bin/env python3
"""Integration verification for M4 uncertainty/reliability/stability."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m4 import M4Config, conformal_order_statistic  # noqa: E402

REPORTS = ROOT / "reports"


def ok(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS {message}")


def main() -> None:
    cfg = M4Config()
    summary = json.loads((REPORTS / "m4_summary.json").read_text())
    gate = json.loads((REPORTS / "m4_gate_result.json").read_text())
    roles = pd.read_csv(REPORTS / "m4_call_roles.csv")
    scores = pd.read_csv(REPORTS / "m4_calibration_call_scores.csv")
    intervals = pd.read_csv(REPORTS / "m4_interval_sensitivity.csv")
    rel = pd.read_csv(REPORTS / "m4_reliability_metrics.csv")
    grid = pd.read_csv(REPORTS / "m4_stabilizer_calibration_grid.csv")
    stability = pd.read_csv(REPORTS / "m4_stability_metrics.csv")
    pred = pd.read_csv(REPORTS / "m4_temporal_eval_predictions.csv")
    m2splits = pd.read_csv(REPORTS / "m2_call_splits.csv")
    locked_ids = set(m2splits.loc[m2splits.m2_split.eq("final_chronological_test"), "session_id"].astype(str))

    ok(roles.session_id.nunique() == 40, "M4 uses exactly 40 M2 OOF calls")
    ok((roles.m4_role == "calibration").sum() == 24, "24 earliest calls assigned to M4 calibration")
    ok((roles.m4_role == "temporal_evaluation").sum() == 16, "16 later calls assigned to M4 temporal evaluation")
    cal = roles[roles.m4_role.eq("calibration")]
    ev = roles[roles.m4_role.eq("temporal_evaluation")]
    ok(pd.to_datetime(cal.ground_truth_time).max() < pd.to_datetime(ev.ground_truth_time).min(),
       "M4 calibration arrivals precede temporal evaluation arrivals")
    ok(not (set(roles.session_id.astype(str)) & locked_ids), "locked final chronological calls remain untouched")
    ok(summary["locked_final_calls_scored"] == 0, "summary records zero locked-final scoring")
    ok(summary["m3_residual_model_used"] is False, "M4 does not revive rejected M3 residual model")

    ok(scores.session_id.nunique() == 24, "one primary conformity score per calibration call")
    q, k, n = conformal_order_statistic(scores.conformity_score_h, cfg.confidence_level)
    primary = intervals[np.isclose(intervals.confidence_level, cfg.confidence_level)].iloc[0]
    ok(abs(float(primary.half_width_h) - q) < 1e-12, "saved primary half-width matches call-level conformal order statistic")
    ok(int(primary.conformal_order_k) == k and int(primary.calibration_calls) == n,
       "saved conformal order and calibration count are correct")
    ok(int(primary.calls) >= cfg.min_eval_calls, "primary interval evaluated on enough independent calls")
    ok(float(primary.trajectory_coverage) >= cfg.min_eval_trajectory_coverage,
       "primary high-reliability trajectory coverage reaches gate")
    ok(float(primary.half_width_h) <= cfg.max_primary_half_width_h,
       "primary high-reliability interval half-width remains within gate")

    high = pred[pred.reliability_tier.eq("HIGH")]
    nonhigh = pred[~pred.reliability_tier.eq("HIGH")]
    ok(high.session_id.nunique() == 16, "all 16 temporal-evaluation calls contribute HIGH-reliability points")
    ok(high.rel_horizon_supported.astype(bool).all(), "HIGH rows remain within supported predicted horizon")
    ok(high.rel_intent_supported.astype(bool).all(), "HIGH rows have contemporaneous target-intent support")
    ok(high.rel_route_supported.astype(bool).all(), "HIGH rows satisfy route-match diagnostic")
    ok(high.rel_source_fresh.astype(bool).all(), "HIGH rows satisfy source-freshness diagnostic")
    ok(high.rel_speed_supported.astype(bool).all(), "HIGH rows satisfy robust-speed diagnostic")
    ok(high.m4_pi_lower_tta_h.notna().all() and high.m4_pi_upper_tta_h.notna().all(),
       "primary interval is present on all HIGH rows")
    ok(nonhigh.m4_pi_lower_tta_h.isna().all() and nonhigh.m4_pi_upper_tta_h.isna().all(),
       "interval is withheld on non-HIGH reliability rows")
    ok(high.m4_pi_covered.astype(bool).groupby(high.session_id).all().mean() >= cfg.min_eval_trajectory_coverage,
       "saved primary intervals achieve required whole-trajectory coverage")

    selected = grid[grid.selected_on_calibration.astype(bool)]
    ok(len(selected) == 1, "exactly one causal EWMA alpha selected on calibration")
    selected_row = selected.iloc[0]
    min_mae = grid.calibration_voyage_balanced_mae_under24_h.min()
    ok(abs(float(selected_row.calibration_voyage_balanced_mae_under24_h) - float(min_mae)) < 1e-12,
       "EWMA alpha selected only by minimum calibration MAE")
    raw = stability[(stability.model == "raw_route_physics") & (stability.scope == "all")].iloc[0]
    smooth = stability[(stability.model.str.startswith("causal_ewma")) & (stability.scope == "all")].iloc[0]
    ok(float(smooth.voyage_balanced_mae_under24_h) <= float(raw.voyage_balanced_mae_under24_h) * 1.02,
       "selected stabilizer does not materially degrade temporal-evaluation <=24h MAE")
    ok(float(smooth.p90_revision_min) < float(raw.p90_revision_min),
       "selected stabilizer reduces temporal-evaluation P90 ETA revision")

    ok(summary["reported_eta"]["aligned_rows_5min"] == 0, "no reported AIS ETA is falsely treated as time-aligned")
    ok(summary["reported_eta"]["rows_within_60min"] == 0, "even 60-minute reported-ETA alignment has zero rows")
    ok(gate["status"] == "M4_CALIBRATED_HIGH_RELIABILITY_GO_M5", "M4 gate result is reproducible")
    ok(gate["passed"] is True, "M4 gate passed")

    print("M4 verification complete")


if __name__ == "__main__":
    main()
