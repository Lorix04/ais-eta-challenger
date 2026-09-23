from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m18g import (
    evaluate_point_candidate,
    evaluate_selective_layer,
    paired_bootstrap_mae_gain,
    sample_key,
    validate_gate_contract,
    validate_prediction_ledger,
)


def _slices(n: int) -> dict[str, list]:
    return {
        "slice_vessel_seen": ["seen"] * (n // 2) + ["unseen"] * (n - n // 2),
        "slice_confidence_tier": (["HIGH", "LOW"] * ((n + 1) // 2))[:n],
        "slice_physics_eligible": ([True, False] * ((n + 1) // 2))[:n],
        "slice_destination_resolved": ([True, False] * ((n + 1) // 2))[:n],
    }


def test_m18g_paired_bootstrap_is_deterministic_and_positive_for_better_candidate():
    y = np.arange(60.0)
    b = y + np.linspace(2.0, 8.0, 60)
    c = y + np.linspace(0.2, 1.0, 60)
    a = paired_bootstrap_mae_gain(y, b, c, reps=1000, seed=18023)
    z = paired_bootstrap_mae_gain(y, b, c, reps=1000, seed=18023)
    assert a == z
    assert a["mae_gain_h"] > 0
    assert a["ci95_lower_h"] > 0


def test_m18g_point_gate_requires_paired_and_slice_stability():
    n = 80
    y = np.zeros(n)
    d = pd.DataFrame({
        "target_tte_h": y,
        "baseline_pred_h": np.full(n, 10.0),
        "candidate_pred_h": np.full(n, 5.0),
        **_slices(n),
    })
    r = evaluate_point_candidate(d)
    assert r["decision"] == "PROMOTE_POINT_CANDIDATE"
    assert r["critical_slices_all_pass"] is True


def test_m18g_selective_gate_uses_frozen_threshold_without_target_selection():
    n = 100
    risk = np.arange(1, n + 1, dtype=float)
    # Error rises with risk; threshold 61.116 retains 61/100.
    y = np.zeros(n)
    p = risk.copy()
    d = pd.DataFrame({
        "target_tte_h": y,
        "baseline_pred_h": p,
        "m16h_risk_h": risk,
        "m16h_lower_h": p - 200.0,
        "m16h_upper_h": p + 200.0,
    })
    # This fixture falls below the frozen 70% coverage floor and must fail,
    # proving the external gate does not retune the threshold to the test set.
    r = evaluate_selective_layer(d)
    assert r["realized_coverage"] == pytest.approx(0.61)
    assert r["decision"] == "DO_NOT_EXTERNALLY_VALIDATE_M18F_LAYER"


def test_m18g_prediction_ledger_rejects_target_columns_and_bad_sample_keys():
    d = pd.DataFrame({
        "sample_key": [sample_key(1, "2026-05-01T00:00:00Z")],
        "mmsi": [1],
        "decision_time": ["2026-05-01T00:00:00Z"],
        "baseline_pred_h": [2.0],
        "m16h_risk_h": [1.0],
        "m16h_lower_h": [0.0],
        "m16h_upper_h": [4.0],
        "slice_vessel_seen": ["unseen"],
        "slice_confidence_tier": ["HIGH"],
        "slice_physics_eligible": [True],
        "slice_destination_resolved": [True],
    })
    validate_prediction_ledger(d, require_candidate=False)
    d["target_tte_h"] = 2.0
    with pytest.raises(AssertionError, match="target/post-label"):
        validate_prediction_ledger(d, require_candidate=False)


def test_m18g_frozen_artifacts_if_present():
    p = ROOT / "reports/M18G_EXTERNAL_PROMOTION_GATE.json"
    if not p.exists():
        return
    gate = json.loads(p.read_text())
    validate_gate_contract(gate)
    forbidden = pd.read_csv(ROOT / "reports/m18g_forbidden_sample_keys.csv")
    registry = pd.read_csv(ROOT / "reports/m18g_candidate_registry.csv")
    assert len(forbidden) == 439
    assert forbidden.sample_key.nunique() == 439
    assert int(forbidden["split"].eq("final_test").sum()) == 53
    assert gate["scope"]["fresh_holdout_opened"] is False
    assert gate["opening_readiness"]["ready_now"] is False
    assert registry.loc[registry.track.eq("POINT_MODEL"), "candidate_id"].iloc[0] == "UNREGISTERED"
