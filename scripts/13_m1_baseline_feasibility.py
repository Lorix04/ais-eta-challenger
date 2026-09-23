#!/usr/bin/env python3
"""Run M1 ETA baseline-feasibility analysis on the audited Catania cohort."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m1 import (  # noqa: E402
    M1Config,
    assign_call_splits,
    baseline_metrics,
    build_decision_panel,
    call_level_metrics,
    horizon_metrics,
    scope_coverage,
    split_diagnostics,
)

REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "derived" / "m0c_ship_states.pkl.gz"
COHORT = REPORTS / "catania_m0e_eta_primary_cohort.csv"
EVENTS = REPORTS / "catania_m0d_gate_events.csv"


def _jsonify(value):
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    cfg = M1Config(decision_interval_min=15)
    states = pd.read_pickle(DATA)
    cohort = pd.read_csv(COHORT)
    events = pd.read_csv(EVENTS)

    splits = assign_call_splits(cohort, cfg)
    panel = build_decision_panel(states, cohort, events, splits, cfg)
    # A dedicated near-term scope makes the strong final-approach baseline
    # visible without hiding the long-horizon failure mode.
    panel["scope_inbound_approach_under6h"] = (
        panel["scope_inbound_approach"].astype(bool) & panel["true_tta_h"].le(6.0)
    )

    scopes = [
        "scope_conditional_target_known",
        "scope_declared_destination",
        "scope_inbound_approach",
        "scope_inbound_approach_under6h",
    ]
    metrics = pd.concat([baseline_metrics(panel, s) for s in scopes], ignore_index=True)
    horizon = horizon_metrics(
        panel,
        "scope_inbound_approach",
        "pred_b2_geodesic_median30_sog_floor_h",
    )
    calls = call_level_metrics(
        panel,
        "scope_inbound_approach",
        "pred_b2_geodesic_median30_sog_floor_h",
    )
    coverage = scope_coverage(panel)
    split_diag = split_diagnostics(splits)

    # Cadence sensitivity is frozen as a methodology check, not as model tuning.
    # Both 15 and 30 minutes were pre-authorized in the workflow; 15 minutes is
    # retained because it preserves more independent calls near the final approach.
    cadence_rows = []
    for interval in [15, 30]:
        alt_cfg = M1Config(decision_interval_min=interval)
        alt_splits = assign_call_splits(cohort, alt_cfg)
        alt_panel = panel if interval == 15 else build_decision_panel(states, cohort, events, alt_splits, alt_cfg)
        alt_panel = alt_panel.copy()
        alt_panel["scope_inbound_approach_under6h"] = (
            alt_panel["scope_inbound_approach"].astype(bool) & alt_panel["true_tta_h"].le(6.0)
        )
        alt_cov = scope_coverage(alt_panel)
        alt_cov = alt_cov[alt_cov["scope"].eq("scope_inbound_approach")].iloc[0]
        alt_m = baseline_metrics(alt_panel, "scope_inbound_approach_under6h")
        alt_m = alt_m[alt_m["baseline"].eq("pred_b2_geodesic_median30_sog_floor_h")].iloc[0]
        cadence_rows.append({
            "decision_interval_min": interval,
            "development_approach_calls": int(alt_cov["calls_covered"]),
            "development_approach_points": int(alt_cov["decision_points"]),
            "under6h_calls": int(alt_m["calls_covered"]),
            "under6h_mae_min": float(alt_m["voyage_balanced_mae_h"] * 60.0),
            "under6h_p90_min": float(alt_m["voyage_balanced_p90_ae_h"] * 60.0),
        })
    cadence_sensitivity = pd.DataFrame(cadence_rows)

    approach_dev = panel[
        panel["m1_split"].eq("development") & panel["scope_inbound_approach"].astype(bool)
    ]
    long_horizon_calls = int(approach_dev.loc[approach_dev["true_tta_h"].gt(6.0), "session_id"].nunique())
    short_horizon_calls = int(approach_dev.loc[approach_dev["true_tta_h"].le(6.0), "session_id"].nunique())
    declared_dev_calls = int(
        panel.loc[
            panel["m1_split"].eq("development") & panel["scope_declared_destination"].astype(bool),
            "session_id",
        ].nunique()
    )

    robust_short = metrics[
        metrics["scope"].eq("scope_inbound_approach_under6h")
        & metrics["baseline"].eq("pred_b2_geodesic_median30_sog_floor_h")
    ].iloc[0]
    robust_approach = metrics[
        metrics["scope"].eq("scope_inbound_approach")
        & metrics["baseline"].eq("pred_b2_geodesic_median30_sog_floor_h")
    ].iloc[0]

    # M1 stop rule: we do not fit residual/high-capacity ML when the clean
    # approach benchmark has fewer than 20 development calls or when fewer
    # than 8 development calls support horizons >6h.  These thresholds are
    # intentionally conservative and are documented as project gates, not as
    # universal statistical constants.
    enough_approach_calls = int(robust_approach["calls_covered"]) >= 20
    enough_long_horizon_calls = long_horizon_calls >= 8
    if enough_approach_calls and enough_long_horizon_calls:
        m1_gate = "PROCEED_TO_M2_ROUTE_DIAGNOSTICS"
        next_action = "M2 train-only route model on Catania"
    else:
        m1_gate = "EXPAND_GROUND_TRUTH_BEFORE_COMPLEXITY"
        next_action = "Augusta ground-truth expansion before route/residual ML"

    summary = {
        "config": {
            "decision_interval_min": cfg.decision_interval_min,
            "robust_speed_window_min": cfg.robust_speed_window_min,
            "max_provider_observation_age_s": cfg.max_provider_observation_age_s,
            "approach_max_distance_km": cfg.approach_max_distance_km,
            "approach_min_progress_kn": cfg.approach_min_progress_kn,
            "final_chronological_test_calls": cfg.final_chronological_test_calls,
        },
        "split_diagnostics": split_diag,
        "panel_rows": int(len(panel)),
        "development_scope_coverage": coverage.to_dict("records"),
        "declared_destination_development_calls": declared_dev_calls,
        "approach_development_calls": int(robust_approach["calls_covered"]),
        "approach_under6h_development_calls": short_horizon_calls,
        "approach_over6h_development_calls": long_horizon_calls,
        "under6h_robust_baseline": {
            "mae_h": float(robust_short["voyage_balanced_mae_h"]),
            "mae_min": float(robust_short["voyage_balanced_mae_h"] * 60.0),
            "median_ae_min": float(robust_short["voyage_balanced_median_ae_h"] * 60.0),
            "p90_ae_min": float(robust_short["voyage_balanced_p90_ae_h"] * 60.0),
            "within_30m": float(robust_short["within_30m"]),
            "within_60m": float(robust_short["within_60m"]),
            "within_120m": float(robust_short["within_120m"]),
        },
        "all_approach_robust_baseline": {
            "mae_h": float(robust_approach["voyage_balanced_mae_h"]),
            "median_ae_h": float(robust_approach["voyage_balanced_median_ae_h"]),
            "p90_ae_h": float(robust_approach["voyage_balanced_p90_ae_h"]),
        },
        "m1_gate": m1_gate,
        "next_action": next_action,
        "cadence_sensitivity": cadence_sensitivity.to_dict("records"),
        "interpretation": (
            "Final-approach physics is already strong, while long-horizon evidence is too sparse and "
            "dominated by a handful of loitering/multi-leg calls. Expand independent ground truth before "
            "adding route/residual model capacity."
        ),
    }

    splits.to_csv(REPORTS / "catania_m1_call_splits.csv", index=False)
    panel.to_csv(REPORTS / "catania_m1_decision_panel.csv", index=False)
    coverage.to_csv(REPORTS / "catania_m1_scope_coverage.csv", index=False)
    metrics.to_csv(REPORTS / "catania_m1_baseline_metrics.csv", index=False)
    horizon.to_csv(REPORTS / "catania_m1_horizon_metrics.csv", index=False)
    calls.to_csv(REPORTS / "catania_m1_call_metrics.csv", index=False)
    cadence_sensitivity.to_csv(REPORTS / "catania_m1_cadence_sensitivity.csv", index=False)
    (REPORTS / "m1_summary.json").write_text(json.dumps(_jsonify(summary), indent=2), encoding="utf-8")

    print(json.dumps(_jsonify(summary), indent=2))


if __name__ == "__main__":
    main()
