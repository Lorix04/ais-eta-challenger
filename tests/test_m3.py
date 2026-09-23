from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ais_eta.m3 import (
    FORBIDDEN_MODEL_FEATURES,
    MODEL_CATEGORICAL_FEATURES,
    MODEL_NUMERIC_FEATURES,
    M3GateCriteria,
    PortStateGateCriteria,
    call_horizon_balanced_weights,
    circular_abs_diff_deg,
    destination_supports_port,
    evaluate_m3_gate,
    evaluate_port_state_gate,
    finalize_m3_features,
    predict_route_knn_excluding,
    validate_feature_contract,
)


def test_call_horizon_weights_equalize_calls_and_bands():
    df = pd.DataFrame({
        "session_id": ["A"] * 4 + ["B"] * 2,
        "true_tta_h": [1, 2, 8, 30, 1, 2],
    })
    w = call_horizon_balanced_weights(df)
    sums = w.groupby(df["session_id"]).sum()
    assert np.isclose(sums["A"], sums["B"])
    # For call A, each occupied band receives equal total weight.
    bands = pd.cut(df.true_tta_h, [-np.inf, 6, 12, 24, np.inf], labels=["<6h","6-12h","12-24h",">24h"])
    a = pd.DataFrame({"w": w[df.session_id.eq("A")], "band": bands[df.session_id.eq("A")]}).groupby("band", observed=True).w.sum()
    assert np.allclose(a.to_numpy(), a.iloc[0])


def test_circular_difference_handles_wraparound():
    out = circular_abs_diff_deg(np.array([359, 1, 180]), np.array([1, 359, 0]))
    assert np.allclose(out, [2, 2, 180])


def test_destination_support_is_conservative():
    assert destination_supports_port("ITGOA>ITCTA", "CATANIA")
    assert destination_supports_port("ITAUG", "AUGUSTA")
    assert not destination_supports_port("FOR ORDERS", "CATANIA")
    assert not destination_supports_port("FOR ORDERS", "AUGUSTA")


def test_route_knn_excludes_self_call():
    model = {
        "gate_lon": 15.0,
        "gate_lat": 37.5,
        "individual_paths": {
            "SELF": {"mmsi": 1, "points": [{"lon": 15.2, "lat": 37.6}, {"lon": 15.0, "lat": 37.5}]},
            "OTHER": {"mmsi": 2, "points": [{"lon": 15.25, "lat": 37.65}, {"lon": 15.0, "lat": 37.5}]},
        },
    }
    out = predict_route_knn_excluding(model, 15.19, 37.59, excluded_session_ids=["SELF"])
    assert out["knn_neighbor_sessions"] == "OTHER"
    assert "SELF" not in out["knn_neighbor_sessions"]
    assert out["route_knn_distance_km"] >= out["geodesic_km"]


def test_feature_contract_rejects_identity_or_truth():
    validate_feature_contract(MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES)
    for bad in sorted(FORBIDDEN_MODEL_FEATURES):
        with pytest.raises(ValueError):
            validate_feature_contract(MODEL_NUMERIC_FEATURES + [bad])


def test_finalize_features_keeps_target_out_and_physics_nonnegative():
    df = pd.DataFrame({
        "route_knn_distance_km": [18.52],
        "geodesic_distance_km": [15.0],
        "sog_median_30m_kn": [10.0],
        "dynamic_state_age_s": [60.0],
        "stop_duration_s": [0.0],
        "turn_candidate": [False],
        "destination_support_port": [True],
        "decision_time": [pd.Timestamp("2026-04-01 06:00")],
        "route_match_confidence": [0.5],
    })
    out = finalize_m3_features(df)
    assert np.isclose(out.loc[0, "pred_route_physics_h"], 1.0)
    assert out.loc[0, "route_to_geodesic_ratio"] >= 1.0
    assert out.loc[0, "pred_route_physics_h"] >= 0


def test_m3_gate_requires_stable_not_single_call_gain():
    good = evaluate_m3_gate(
        baseline_overall_mae_h=5.0,
        model_overall_mae_h=4.9,
        baseline_under24_mae_h=1.0,
        model_under24_mae_h=0.90,
        fold_wins=5,
        fold_total=6,
        cold_baseline_under24_mae_h=1.0,
        cold_model_under24_mae_h=0.90,
        top_positive_call_gain_share=0.2,
        criteria=M3GateCriteria(),
    )
    assert good["status"] == "GO_M4_RESIDUAL_SIGNAL_CONFIRMED"
    bad = evaluate_m3_gate(
        baseline_overall_mae_h=5.0,
        model_overall_mae_h=5.0,
        baseline_under24_mae_h=1.0,
        model_under24_mae_h=0.99,
        fold_wins=2,
        fold_total=6,
        cold_baseline_under24_mae_h=1.0,
        cold_model_under24_mae_h=1.1,
        top_positive_call_gain_share=0.8,
    )
    assert bad["status"] == "M3_COMPLEXITY_NOT_JUSTIFIED"


def test_port_state_gate_is_conservative():
    keep = evaluate_port_state_gate(5.0, 4.95, 1.0, 0.96, 5, 6, PortStateGateCriteria())
    assert keep["status"] == "KEEP_PORT_STATE"
    drop = evaluate_port_state_gate(5.0, 5.2, 1.0, 0.99, 2, 6)
    assert drop["status"] == "DROP_PORT_STATE_FOR_NOW"


def test_fast_prepared_knn_matches_reference_projection():
    from ais_eta.m3 import prepare_fast_knn_model
    model = {
        "gate_lon": 15.0,
        "gate_lat": 37.5,
        "individual_paths": {
            "A": {"mmsi": 1, "points": [
                {"lon": 15.25, "lat": 37.70}, {"lon": 15.12, "lat": 37.58}, {"lon": 15.0, "lat": 37.5}
            ]},
            "B": {"mmsi": 2, "points": [
                {"lon": 15.30, "lat": 37.60}, {"lon": 15.15, "lat": 37.54}, {"lon": 15.0, "lat": 37.5}
            ]},
        },
    }
    ref = predict_route_knn_excluding(model, 15.18, 37.59)
    fast = predict_route_knn_excluding(prepare_fast_knn_model(model), 15.18, 37.59)
    assert np.isclose(ref["route_knn_distance_km"], fast["route_knn_distance_km"], atol=1e-10)
    assert np.isclose(ref["knn_mean_cross_track_km"], fast["knn_mean_cross_track_km"], atol=1e-10)
