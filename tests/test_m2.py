from __future__ import annotations

import numpy as np
import pandas as pd

from ais_eta.m2 import (
    M2Config,
    M2GateCriteria,
    assign_m2_splits,
    evaluate_m2_gate,
    project_to_route,
    route_family_from_bearing,
)


def _cohort(prefix: str, n: int, start: str, port: str) -> pd.DataFrame:
    times = pd.date_range(start, periods=n, freq="6h")
    out = pd.DataFrame(
        {
            "session_id": [f"{prefix}-{i:03d}" for i in range(n)],
            "mmsi": [1000 + i for i in range(n)],
            "name": [f"V{i}" for i in range(n)],
            "ground_truth_time": times,
        }
    )
    if port == "AUGUSTA":
        out["inbound_entrance"] = "LEVANTE"
    return out


def test_m2_splits_preserve_catania_holdout_and_lock_augusta_tail():
    cat = _cohort("CT", 12, "2026-04-01", "CATANIA")
    aug = _cohort("AU", 20, "2026-04-01", "AUGUSTA")
    m1 = pd.DataFrame(
        {
            "session_id": cat.session_id,
            "m1_split": ["development"] * 10 + ["final_chronological_test"] * 2,
        }
    )
    cfg = M2Config(
        augusta_final_chronological_calls=4,
        temporal_folds=2,
        catania_warmup_calls=4,
        augusta_warmup_calls=6,
    )
    out = assign_m2_splits(cat, aug, m1, cfg)
    cat_final = out[(out.port.eq("CATANIA")) & out.m2_split.eq("final_chronological_test")]
    aug_final = out[(out.port.eq("AUGUSTA")) & out.m2_split.eq("final_chronological_test")]
    assert cat_final.session_id.tolist() == ["CT-010", "CT-011"]
    assert aug_final.session_id.tolist() == ["AU-016", "AU-017", "AU-018", "AU-019"]
    assert out[out.m2_role.eq("oof_validation")].m2_temporal_fold.notna().all()
    assert out[out.m2_role.eq("warmup_train")].m2_temporal_fold.isna().all()


def test_route_family_sectors_are_fixed_and_exhaustive():
    assert route_family_from_bearing(0) == "NNE"
    assert route_family_from_bearing(59.999) == "NNE"
    assert route_family_from_bearing(60) == "E"
    assert route_family_from_bearing(119.999) == "E"
    assert route_family_from_bearing(120) == "SSE"
    assert route_family_from_bearing(179.999) == "SSE"
    assert route_family_from_bearing(180) == "OTHER"
    assert route_family_from_bearing(359) == "OTHER"


def test_route_projection_never_claims_shorter_than_geodesic():
    # A deliberately bent route ending at the gate (0, 0).
    points = [
        {"lon": 0.10, "lat": 0.10, "level_km": 15},
        {"lon": 0.05, "lat": 0.02, "level_km": 7},
        {"lon": 0.00, "lat": 0.00, "level_km": 0},
    ]
    result = project_to_route(0.09, 0.09, points, 0.0, 0.0)
    assert result["route_remaining_km"] >= result["geodesic_km"]
    assert result["cross_track_km"] >= 0
    assert result["along_route_remaining_km"] >= 0


def test_m2_gate_requires_navigation_signal_consistency_and_cold_vessel_value():
    c = M2GateCriteria()
    good = evaluate_m2_gate(
        overall_geo_mae_h=5.0,
        overall_knn_mae_h=4.9,
        under24_geo_mae_h=1.0,
        under24_knn_mae_h=0.85,
        fold_wins=5,
        fold_total=6,
        cold_under24_geo_mae_h=1.0,
        cold_under24_knn_mae_h=0.8,
        criteria=c,
    )
    assert good["status"] == "GO_M3_ROUTE_SIGNAL_CONFIRMED"

    bad = evaluate_m2_gate(
        overall_geo_mae_h=5.0,
        overall_knn_mae_h=5.5,
        under24_geo_mae_h=1.0,
        under24_knn_mae_h=0.99,
        fold_wins=2,
        fold_total=6,
        cold_under24_geo_mae_h=1.0,
        cold_under24_knn_mae_h=1.1,
        criteria=c,
    )
    assert bad["status"] == "M2_ROUTE_COMPLEXITY_NOT_JUSTIFIED"


def test_projected_route_is_finite_for_simple_on_route_point():
    points = [
        {"lon": 15.2, "lat": 37.6, "level_km": 20},
        {"lon": 15.15, "lat": 37.55, "level_km": 10},
        {"lon": 15.10, "lat": 37.50, "level_km": 0},
    ]
    result = project_to_route(15.16, 37.56, points, 15.10, 37.50)
    assert np.isfinite(result["route_remaining_km"])
    assert np.isfinite(result["cross_track_km"])
