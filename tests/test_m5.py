from __future__ import annotations

import numpy as np
import pandas as pd

from ais_eta.m5 import (
    M5Config,
    build_call_stress_table,
    evaluate_m5_gate,
    grouped_route_metrics,
    paired_bootstrap_mean_gain,
)


def test_paired_bootstrap_uses_one_gain_per_call_and_is_deterministic():
    gains = [0.1, 0.2, -0.05, 0.3]
    a = paired_bootstrap_mean_gain(gains, draws=1000, seed=7)
    b = paired_bootstrap_mean_gain(gains, draws=1000, seed=7)
    assert a == b
    assert abs(a[0] - np.mean(gains)) < 1e-12


def test_call_stress_table_uses_initial_causal_regime_and_call_balanced_errors():
    x = pd.DataFrame({
        "port": ["AUGUSTA"] * 4,
        "session_id": ["A", "A", "B", "B"],
        "mmsi": [1, 1, 2, 2],
        "name": ["X", "X", "Y", "Y"],
        "cold_vessel_in_fold": [True, True, False, False],
        "decision_time": pd.to_datetime(["2026-04-01 00:00", "2026-04-01 00:15", "2026-04-02 00:00", "2026-04-02 00:15"]),
        "ground_truth_time": pd.to_datetime(["2026-04-01 03:00"] * 2 + ["2026-04-02 03:00"] * 2),
        "true_tta_h": [3.0, 2.75, 3.0, 2.75],
        "pred_m2_geodesic_h": [4.0, 3.0, 3.5, 3.0],
        "pred_m2_route_knn_h": [3.5, 2.9, 3.2, 2.8],
        "route_family": ["E", "NNE", "SSE", "SSE"],
        "destination_support_port": [0, 1, 1, 1],
        "reliability_tier": ["MEDIUM", "HIGH", "HIGH", "HIGH"],
        "knn_mean_cross_track_km": [7.0, 1.0, 1.0, 1.0],
        "rel_route_supported": [False, True, True, True],
    })
    conf = pd.DataFrame({"session_id": ["A", "B"], "ground_truth_confidence": ["B", "A"]})
    out = build_call_stress_table(x, conf)
    a = out[out.session_id.eq("A")].iloc[0]
    assert a.initial_route_family == "E"
    assert a.initial_intent_supported == 0
    assert a.initial_reliability_tier == "MEDIUM"
    assert abs(a.intent_support_fraction - 0.5) < 1e-12
    assert a.ground_truth_confidence == "B"


def test_grouped_metrics_are_voyage_balanced_not_row_weighted():
    calls = pd.DataFrame({
        "port": ["AUGUSTA", "AUGUSTA"],
        "session_id": ["A", "B"],
        "mmsi": [1, 2],
        "under24_geo_mae_h": [1.0, 3.0],
        "under24_route_mae_h": [0.5, 2.5],
        "under24_gain_h": [0.5, 0.5],
        "route_better_under24": [True, True],
    })
    m = grouped_route_metrics(calls, ["port"], M5Config(bootstrap_draws=100))
    assert len(m) == 1
    assert abs(m.iloc[0].geodesic_voyage_mae_h - 2.0) < 1e-12
    assert abs(m.iloc[0].route_voyage_mae_h - 1.5) < 1e-12


def test_m5_gate_can_pass_with_route_family_warning():
    cfg = M5Config()
    ports = pd.DataFrame({
        "port": ["CATANIA", "AUGUSTA"], "calls": [10, 30],
        "route_improvement_fraction": [0.2, 0.1],
    })
    cold = pd.DataFrame({
        "port": ["CATANIA", "AUGUSTA"], "cold_vessel_in_fold": [True, True],
        "calls": [6, 22], "route_improvement_fraction": [0.2, 0.2],
    })
    intervals = pd.DataFrame({
        "port": ["CATANIA", "AUGUSTA"], "temporal_high_calls": [2, 14],
        "pooled_trajectory_coverage": [1.0, 1.0],
    })
    fam = pd.DataFrame({
        "port": ["AUGUSTA"], "initial_route_family": ["E"], "calls": [10],
        "route_improvement_fraction": [-0.12],
    })
    gate = evaluate_m5_gate(ports, cold, intervals, fam, cfg)
    assert gate["passed"] is True
    assert gate["status"] == "M5_GO_M6_SCOPE_LIMITED"
    assert gate["warnings"][0]["type"] == "ROUTE_FAMILY_DEGRADATION"
    assert gate["unseen_port_generalization"] == "NOT_ESTABLISHED"
