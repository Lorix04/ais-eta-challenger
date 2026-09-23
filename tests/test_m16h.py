from __future__ import annotations
from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m16h import (
    M16H_CANDIDATES, assign_confidence_tier, build_error_features,
    calibrate_interval, conformal_upper_quantile, promotion_gate,
)


def _toy_frame(n=12):
    x = np.arange(n, dtype=float)
    return pd.DataFrame({
        "pred_m16c_prior_h": x,
        "pred_m16d_route_h": x + 1,
        "pred_m16e_physics_gate_h": x + 2,
        "pred_m16f_tabular_h": x + 3,
        "pred_m16g_selected_h": x + 1.5,
        "m16c_deepest_support": 3,
        "m16c_deepest_weight": .5,
        "m16c_fallback_count": 1,
        "m16d_neighbour_count": 5,
        "m16d_nearest_distance": .2,
        "m16d_median_neighbour_distance": .3,
        "m16d_similarity_gap": .1,
        "m16d_gate_used": "destination",
        "physics_eligible": [i % 2 == 0 for i in range(n)],
        "resolution_confidence": 1.0,
        "physics_distance_gc_nm": 50.0,
        "physics_course_alignment_deg": 20.0,
        "physics_recent_speed_max_kn": 10.0,
        "m16g_selected_meta_id": "median_blend",
    })


def test_conformal_quantile_is_finite_sample_upper_order_statistic():
    assert conformal_upper_quantile([1, 2, 3, 4], .8) == 4.0


def test_error_features_are_prediction_time_only():
    f = _toy_frame()
    out = build_error_features(f)
    assert len(out) == len(f) and np.isfinite(out.to_numpy()).all()
    f["reference_eta_status"] = "FUTURE_0_7D"
    try:
        build_error_features(f)
    except ValueError as exc:
        assert "forbidden" in str(exc)
    else:
        raise AssertionError("target-derived confidence feature should be rejected")


def test_interval_candidates_preserve_p50_and_ordering():
    p50 = np.array([10.0, 20.0])
    err = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    risk_tr = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    risk_va = np.array([2, 5], dtype=float)
    for candidate in M16H_CANDIDATES:
        r = calibrate_interval(err, risk_tr, p50, risk_va, candidate)
        assert np.allclose(r.p50_h, p50)
        assert np.all(r.lower_h <= r.p50_h)
        assert np.all(r.p50_h <= r.upper_h)


def test_confidence_tier_is_target_free_thresholding():
    tier, q1, q2 = assign_confidence_tier([1, 2, 3, 4, 5, 6], [1, 3.5, 8])
    assert list(tier) == ["HIGH", "MEDIUM", "LOW"]
    assert q1 < q2


def test_m16h_gate_requires_calibration_sharpness_and_ordered_confidence():
    ok = promotion_gate(pooled_coverage=.80, adaptive_mean_width_h=160, global_mean_width_h=220,
                        stable_folds=4, high_medae_h=4, medium_medae_h=20, low_medae_h=80)
    assert ok == "PASS_PROBABILISTIC_CALIBRATION_CONFIDENCE"
    bad = promotion_gate(pooled_coverage=.70, adaptive_mean_width_h=160, global_mean_width_h=220,
                         stable_folds=4, high_medae_h=4, medium_medae_h=20, low_medae_h=80)
    assert bad == "FAIL_PROBABILISTIC_CALIBRATION_GATE"


def test_if_m16h_artifacts_exist_they_are_dev_only_and_p50_is_m16g():
    p = ROOT / "reports/M16H_SUMMARY.json"
    if not p.exists():
        return
    s = json.loads(p.read_text())
    f = json.loads((ROOT / "reports/M16H_PROBABILISTIC_FREEZE.json").read_text())
    h = pd.read_csv(ROOT / "reports/m16h_oof_intervals.csv")
    g = pd.read_csv(ROOT / "reports/m16g_oof_predictions.csv")[["mmsi", "pred_m16g_selected_h"]]
    manifest = json.loads((ROOT / "reports/M16A_DEVELOPMENT_MANIFEST.json").read_text())
    assert len(h) == 386 and h.mmsi.nunique() == 386
    assert set(h.mmsi.astype(int)).isdisjoint(set(manifest["blocked_old_final_mmsi"]))
    z = h.merge(g, on="mmsi", suffixes=("", "_g"))
    assert np.allclose(z["p50_h"], z["pred_m16g_selected_h_g"], atol=1e-10)
    assert (h.p10_h <= h.p50_h).all() and (h.p50_h <= h.p90_h).all()
    assert s["final_test_used_for_calibration"] is False
    assert s["gate"] == "PASS_PROBABILISTIC_CALIBRATION_CONFIDENCE"
    for name, digest in f["artifact_sha256"].items():
        assert hashlib.sha256((ROOT / "reports" / name).read_bytes()).hexdigest() == digest, name
    state = json.loads((ROOT / "PROJECT_STATE.json").read_text())
    assert state["current_phase"].startswith(("M16H_", "M16I_", "M16J_", "M17A_", "M17B_", "M17C_", "M17D_", "M17E_", "M17F_", "M17G_", "M17H_", "M18A_", "M18B_", "M18C_", "M18D_", "M18E_", "M18F_", "M18G_", "M19_"))
