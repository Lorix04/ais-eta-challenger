from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m16e import (
    PhysicsConfig, add_physics_features, candidate_configs, haversine_nm,
    predict_physics_expert,
)


def _rows():
    return pd.DataFrame({
        "mmsi": [1, 2, 3, 4],
        "track_lat": [37.0, 37.0, 37.0, 37.0],
        "track_lon": [15.0, 15.0, 15.0, 15.0],
        "track_cog": [0.0, 0.0, 0.0, 0.0],
        "canonical_lat": [38.0, 38.0, np.nan, 38.0],
        "canonical_lon": [15.0, 15.0, np.nan, 15.0],
        "canonical_destination": ["ITAUG", "ITAUG", "FOR_ORDERS", "ITAUG"],
        "is_resolved_port": [True, True, False, True],
        "resolution_confidence": [1.0, 1.0, 1.0, 1.0],
        "sog_median_180m": [10.0, 8.0, 12.0, 0.0],
        "sog_median_360m": [9.0, 7.0, 11.0, 0.0],
        "distance_travelled_km_180m": [50.0, 40.0, 60.0, 0.0],
        "net_displacement_km_180m": [45.0, 35.0, 55.0, 0.0],
        "distance_travelled_km_360m": [100.0, 80.0, 120.0, 0.0],
        "net_displacement_km_360m": [90.0, 70.0, 110.0, 0.0],
        "target_tte_h": [6.0, 8.0, 10.0, 12.0],
    })


def test_haversine_one_degree_latitude_is_about_60_nm():
    d = float(haversine_nm([37.0], [15.0], [38.0], [15.0])[0])
    assert 59.5 < d < 60.5


def test_physics_eligibility_requires_resolved_destination_and_motion():
    x = add_physics_features(_rows())
    assert x["physics_eligible"].tolist() == [True, True, False, False]


def test_candidate_grid_is_small_and_predeclared():
    cfgs = candidate_configs()
    assert len(cfgs) == 12
    assert {c.speed_window_min for c in cfgs} == {180, 360}
    assert {c.distance_mode for c in cfgs} == {"geodesic", "local_sinuosity"}
    assert {c.correction for c in cfgs} == {"none", "global_residual", "destination_residual"}


def test_valid_targets_do_not_affect_physics_prediction():
    x = add_physics_features(_rows())
    tr = x.iloc[:2].copy(); va = x.iloc[[2]].copy()
    # Make the validation row eligible without touching the train.
    va.loc[:, "canonical_lat"] = 38.0; va.loc[:, "canonical_lon"] = 15.0
    va.loc[:, "canonical_destination"] = "ITAUG"; va.loc[:, "is_resolved_port"] = True
    va = add_physics_features(va)
    cfg = PhysicsConfig(180, "geodesic", "destination_residual")
    p1 = predict_physics_expert(tr, va, cfg)
    va2 = va.copy(); va2.loc[:, "target_tte_h"] = -123456.0
    p2 = predict_physics_expert(tr, va2, cfg)
    np.testing.assert_allclose(p1["prediction_h"], p2["prediction_h"])


def test_m16e_artifacts_are_dev_only_and_physics_signal_is_retained():
    sp = ROOT / "reports/M16E_SUMMARY.json"
    lp = ROOT / "reports/m16e_oof_predictions.csv"
    if not sp.exists() or not lp.exists():
        return
    s = json.loads(sp.read_text())
    ledger = pd.read_csv(lp)
    manifest = json.loads((ROOT / "reports/M16A_DEVELOPMENT_MANIFEST.json").read_text())
    assert len(ledger) == 386 and ledger["mmsi"].nunique() == 386
    assert set(ledger["mmsi"].astype(int)).isdisjoint(manifest["blocked_old_final_mmsi"])
    assert s["final_test_used_for_selection"] is False
    assert s["eligible_rows"] >= 100
    assert s["physics_wins_vs_route_share_on_eligible"] > 0.5
    assert s["hard_gate_mae_gain_h_vs_route"] > 0
    assert s["gate"].startswith("PASS")


def test_project_state_can_advance_beyond_m16e_without_invalidating_freeze():
    state = json.loads((ROOT / "PROJECT_STATE.json").read_text())
    if "m16e_summary" not in state:
        return
    assert state["m16e_summary"]["final_test_used_for_selection"] is False
    assert state["current_phase"].startswith(("M16E_", "M16F_", "M16G_", "M16H_", "M16I_", "M16J_", "M17A_", "M17B_", "M17C_", "M17D_", "M17E_", "M17F_", "M17G_", "M17H_", "M18A_", "M18B_", "M18C_", "M18D_", "M18E_", "M18F_", "M18G_", "M19_"))
