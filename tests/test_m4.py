import numpy as np
import pandas as pd

from ais_eta.m4 import (
    M4Config,
    add_symmetric_interval,
    apply_causal_arrival_ewma,
    assign_reliability,
    conformal_order_statistic,
    parse_ais_eta_nearest_year,
)


def test_conformal_quantile_uses_call_level_finite_sample_order_statistic():
    q, k, n = conformal_order_statistic(range(1, 25), 0.90)
    assert n == 24
    assert k == 23
    assert q == 23


def test_high_reliability_requires_all_causal_quality_checks():
    cfg = M4Config()
    x = pd.DataFrame({
        "pred_route_physics_h": [2.0, 2.0, 30.0],
        "destination_support_port": [1, 0, 1],
        "knn_mean_cross_track_km": [1.0, 1.0, 1.0],
        "provider_observation_age_s": [20.0, 20.0, 20.0],
        "stale_risk_level": ["LOW", "LOW", "LOW"],
        "sog_median_30m_kn": [5.0, 5.0, 5.0],
    })
    y = assign_reliability(x, cfg)
    assert y.reliability_tier.tolist() == ["HIGH", "MEDIUM", "LOW"]
    assert y.reliability_definition.eq("diagnostic_not_probability").all()


def test_interval_clips_remaining_time_at_zero():
    x = pd.DataFrame({
        "decision_time": pd.to_datetime(["2026-04-10 10:00:00"]),
        "true_tta_h": [0.2],
        "pred": [0.1],
    })
    y = add_symmetric_interval(x, "pred", 1.0, prefix="pi")
    assert y.loc[0, "pi_lower_tta_h"] == 0.0
    assert y.loc[0, "pi_upper_tta_h"] == 1.1


def test_causal_ewma_is_prefix_invariant():
    x = pd.DataFrame({
        "session_id": ["A"] * 4,
        "decision_time": pd.to_datetime([
            "2026-04-10 10:00", "2026-04-10 10:15", "2026-04-10 10:30", "2026-04-10 10:45"
        ]),
        "pred": [2.0, 2.5, 1.8, 2.1],
    })
    full = apply_causal_arrival_ewma(x, "pred", 0.4)
    prefix = apply_causal_arrival_ewma(x.iloc[:3], "pred", 0.4)
    assert np.allclose(full.iloc[:3].pred_stabilized_h, prefix.pred_stabilized_h)


def test_causal_ewma_resets_after_long_gap():
    x = pd.DataFrame({
        "session_id": ["A", "A"],
        "decision_time": pd.to_datetime(["2026-04-10 10:00", "2026-04-10 14:30"]),
        "pred": [10.0, 1.0],
    })
    y = apply_causal_arrival_ewma(x, "pred", 0.2, reset_gap_h=2.0)
    assert abs(y.iloc[1].pred_stabilized_h - 1.0) < 1e-12


def test_eta_parser_handles_year_rollover():
    obs = pd.Timestamp("2026-12-31 23:00:00")
    parsed = parse_ais_eta_nearest_year("01/01 02:00", obs)
    assert parsed == pd.Timestamp("2027-01-01 02:00:00")
