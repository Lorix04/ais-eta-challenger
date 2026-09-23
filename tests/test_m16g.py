from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m16g import (
    M16G_EXPERT_NAMES, M16G_META_CANDIDATES, build_gating_features,
    fit_predict_meta, promotion_gate, score_candidates_inner_cv,
)


def _toy(n=24):
    rng = np.random.default_rng(42)
    y = np.linspace(-20, 120, n)
    base = np.column_stack([
        y + rng.normal(0, 15, n),
        y + rng.normal(0, 10, n),
        y + rng.normal(0, 8, n),
        y + rng.normal(0, 12, n),
    ])
    q = pd.DataFrame({
        "m16c_deepest_support": np.arange(n) % 5,
        "m16c_deepest_weight": 0.5,
        "m16c_fallback_count": 1,
        "m16d_neighbour_count": 5,
        "m16d_nearest_distance": 0.2,
        "m16d_median_neighbour_distance": 0.3,
        "m16d_similarity_gap": 0.1,
        "m16d_gate_used": ["destination" if i % 2 else "global" for i in range(n)],
        "physics_eligible": [i % 3 == 0 for i in range(n)],
        "resolution_confidence": 1.0,
        "physics_distance_gc_nm": 50.0,
        "physics_course_alignment_deg": 20.0,
        "physics_recent_speed_max_kn": 10.0,
        "m16e_effective_speed_kn": 9.0,
        "m16e_distance_factor": 1.0,
        "m16e_route_distance_proxy_nm": 50.0,
    })
    return y, base, q


def test_meta_panel_is_predeclared_and_contains_required_baselines():
    assert len(M16G_EXPERT_NAMES) == 4
    assert "median_blend" in M16G_META_CANDIDATES
    assert "nnls_convex" in M16G_META_CANDIDATES
    assert any("gate" in x for x in M16G_META_CANDIDATES)


def test_gating_features_reject_target_columns():
    y, base, q = _toy()
    q["reference_eta_status"] = "FUTURE_0_7D"
    try:
        build_gating_features(base, q)
    except ValueError as exc:
        assert "forbidden" in str(exc)
    else:
        raise AssertionError("target-derived gating feature should be rejected")


def test_all_meta_candidates_predict_finite_values():
    y, base, q = _toy()
    g = build_gating_features(base, q)
    for candidate in M16G_META_CANDIDATES:
        out = fit_predict_meta(candidate, base[:18], y[:18], g.iloc[:18], base[18:], g.iloc[18:])
        assert len(out.prediction) == 6
        assert np.isfinite(out.prediction).all()


def test_inner_cv_does_not_score_on_training_rows_in_place():
    y, base, q = _toy()
    g = build_gating_features(base, q)
    folds = np.arange(len(y)) % 4
    scored = score_candidates_inner_cv(base, y, g, folds)
    assert set(scored.candidate_id) == set(M16G_META_CANDIDATES)
    assert np.isfinite(scored.mae_h).all()


def test_promotion_gate_requires_pooled_and_fold_stability():
    best = {"mae_h": 193.0}
    assert promotion_gate(best, {"mae_h": 190.0}, fold_wins=3, changed_row_win_share=0.55) == "PASS_MOE_STABLE_GAIN"
    assert promotion_gate(best, {"mae_h": 192.5}, fold_wins=5, changed_row_win_share=0.9) == "FAIL_MOE_PROMOTION_GATE"
    assert promotion_gate(best, {"mae_h": 188.0}, fold_wins=2, changed_row_win_share=0.9) == "FAIL_MOE_PROMOTION_GATE"


def test_if_m16g_artifacts_exist_they_are_dev_only_and_state_advanced():
    import json, hashlib
    p = ROOT / "reports/M16G_SUMMARY.json"
    if not p.exists():
        return
    s = json.loads(p.read_text())
    f = json.loads((ROOT / "reports/M16G_MIXTURE_FREEZE.json").read_text())
    ledger = pd.read_csv(ROOT / "reports/m16g_oof_predictions.csv")
    manifest = json.loads((ROOT / "reports/M16A_DEVELOPMENT_MANIFEST.json").read_text())
    assert len(ledger) == 386 and ledger.mmsi.nunique() == 386
    assert set(ledger.mmsi.astype(int)).isdisjoint(set(manifest["blocked_old_final_mmsi"]))
    assert s["gate"] == "PASS_MOE_STABLE_GAIN"
    assert s["final_test_used_for_selection"] is False
    assert s["changed_row_win_share_vs_best_single"] >= 0.50
    for name, digest in f["artifact_sha256"].items():
        got = hashlib.sha256((ROOT / "reports" / name).read_bytes()).hexdigest()
        assert got == digest, name
    state = json.loads((ROOT / "PROJECT_STATE.json").read_text())
    assert state["current_phase"].startswith(("M16G_", "M16H_", "M16I_", "M16J_", "M17A_", "M17B_", "M17C_", "M17D_", "M17E_", "M17F_", "M17G_", "M17H_", "M18A_", "M18B_", "M18C_", "M18D_", "M18E_", "M18F_", "M18G_", "M19_"))
