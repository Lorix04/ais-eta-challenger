from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m18f import risk_threshold, selective_gate, selective_metrics, validate_m18f_audit


def test_m18f_risk_threshold_is_target_free_quantile():
    r = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert risk_threshold(r, 1.0) == float("inf")
    assert risk_threshold(r, 0.8) == 5.0
    assert risk_threshold(r, 0.5) == 3.0


def test_m18f_selective_metrics_retains_lower_risk_subset():
    y = [0.0, 0.0, 0.0, 0.0]
    p = [1.0, 2.0, 20.0, 30.0]
    lo = [-2.0, -3.0, -30.0, -40.0]
    hi = [2.0, 3.0, 30.0, 40.0]
    m = selective_metrics(y, p, lo, hi, [True, True, False, False])
    assert m["retained_rows"] == 2
    assert m["realized_coverage"] == 0.5
    assert m["retained_mae_h"] == 1.5
    assert m["deferred_mae_h"] == 25.0


def test_m18f_gate_requires_all_fold_improvements_and_no_coverage_collapse():
    ok = selective_gate(
        balanced_realized_coverage=0.8,
        balanced_retained_mae_h=10.0,
        full_mae_h=20.0,
        fold_wins=5,
        fold_count=5,
        near_term_retention=0.9,
        retained_empirical_interval_coverage=0.82,
    )
    assert ok == "PASS_SELECTIVE_ETA_ADVISORY_LAYER"
    bad = selective_gate(
        balanced_realized_coverage=0.8,
        balanced_retained_mae_h=10.0,
        full_mae_h=20.0,
        fold_wins=4,
        fold_count=5,
        near_term_retention=0.9,
        retained_empirical_interval_coverage=0.82,
    )
    assert bad == "NO_SELECTIVE_ETA_PROMOTION"


def test_m18f_frozen_artifacts_if_present():
    p = ROOT / "reports/M18F_SELECTIVE_ETA.json"
    if not p.exists():
        return
    audit = json.loads(p.read_text())
    validate_m18f_audit(audit)
    ledger = pd.read_csv(ROOT / "reports/m18f_selective_eta_ledger.csv")
    curve = pd.read_csv(ROOT / "reports/m18f_risk_coverage_curve.csv")
    assert len(ledger) == 386
    assert audit["scope"]["fresh_holdout_opened"] is False
    assert audit["scope"]["selection_uses_validation_target"] is False
    assert audit["scope"]["selection_conditional_conformal_guarantee_claimed"] is False
    b = curve[(curve.scope == "STRICT_386") & np.isclose(curve.target_coverage, 0.8)].iloc[0]
    assert b.retained_rows > 0
    assert b.deferred_rows > 0
