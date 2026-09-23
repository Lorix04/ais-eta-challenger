from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m16d import RouteConfig, candidate_configs, dtw_distance, predict_route_analogue, resample_causal_trajectory


def test_dtw_is_zero_on_identity_and_symmetric():
    a = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 1.0]])
    b = np.array([[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]])
    assert dtw_distance(a, a) == 0.0
    assert np.isclose(dtw_distance(a, b), dtw_distance(b, a))


def test_resampling_is_causal_and_ignores_future_points():
    t0 = pd.Timestamp("2026-04-10T12:00:00")
    base = pd.DataFrame({
        "recorded_at": [t0 - pd.Timedelta(minutes=60), t0 - pd.Timedelta(minutes=30), t0, t0 + pd.Timedelta(minutes=1)],
        "lat_clean": [37.0, 37.1, 37.2, 99.0],
        "lon_clean": [15.0, 15.1, 15.2, 99.0],
        "sog_clean": [5, 6, 7, 99],
        "cog_clean": [10, 20, 30, 99],
        "position_valid": [True, True, True, True],
    })
    a, meta = resample_causal_trajectory(base, end_at=t0, hours=3, n_points=6, include_kinematics=True)
    b = base.iloc[:3].copy()
    c, _ = resample_causal_trajectory(b, end_at=t0, hours=3, n_points=6, include_kinematics=True)
    np.testing.assert_allclose(a, c)
    assert meta["last_at"].startswith("2026-04-10T12:00:00")


def test_route_prediction_uses_only_training_targets():
    D = np.array([[0,1,2,3],[1,0,1,2],[2,1,0,1],[3,2,1,0]], dtype=float)
    targets = np.array([10.0, 20.0, 9999.0, -9999.0])
    dest = np.array(["A","A","A","A"])
    cfg = RouteConfig("geo_3h", "global", 2)
    p1 = predict_route_analogue(train_indices=np.array([0,1]), valid_indices=np.array([2]), distance_matrix=D, targets=targets, destinations=dest, config=cfg)
    targets2 = targets.copy(); targets2[2] = -123456.0
    p2 = predict_route_analogue(train_indices=np.array([0,1]), valid_indices=np.array([2]), distance_matrix=D, targets=targets2, destinations=dest, config=cfg)
    assert float(p1.iloc[0].prediction_h) == float(p2.iloc[0].prediction_h)


def test_candidate_grid_is_small_predeclared_and_contains_required_ablations():
    cfgs = candidate_configs()
    assert len(cfgs) == 12
    assert {c.gate for c in cfgs} == {"global", "destination"}
    assert {c.k for c in cfgs} == {5, 9}
    assert {c.representation for c in cfgs} == {"geo_3h", "geo_kin_3h", "geo_kin_6h"}


def test_m16d_artifacts_are_dev_only_and_route_signal_passes_history_gate():
    sp = ROOT / "reports/M16D_SUMMARY.json"
    lp = ROOT / "reports/m16d_oof_predictions.csv"
    if not sp.exists() or not lp.exists():
        return
    s = json.loads(sp.read_text())
    ledger = pd.read_csv(lp)
    manifest = json.loads((ROOT / "reports/M16A_DEVELOPMENT_MANIFEST.json").read_text())
    assert len(ledger) == 386 and ledger["mmsi"].nunique() == 386
    assert set(ledger["mmsi"].astype(int)).isdisjoint(manifest["blocked_old_final_mmsi"])
    assert s["final_test_used_for_selection"] is False
    assert s["route_mae_gain_h_vs_history_aggregate_catboost"] > 0
    assert s["route_fold_wins_vs_history_aggregate_catboost"] >= 3
    assert s["gate"].startswith("PASS")


def test_project_state_can_advance_beyond_m16d_without_invalidating_freeze():
    state = json.loads((ROOT / "PROJECT_STATE.json").read_text())
    if "m16d_summary" not in state:
        return
    assert state["m16d_summary"]["final_test_used_for_selection"] is False
    assert state["current_phase"].startswith(("M16D_", "M16E_", "M16F_", "M16G_", "M16H_", "M16I_", "M16J_", "M17A_", "M17B_", "M17C_", "M17D_", "M17E_", "M17F_", "M17G_", "M17H_", "M18A_", "M18B_", "M18C_", "M18D_", "M18E_", "M18F_", "M18G_", "M19_"))
