from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m18d import (
    categorical_slice_metrics,
    expert_diagnostics,
    rank_correlation,
    validate_attribution_dict,
)


def test_m18d_rank_correlation_is_monotonic_diagnostic():
    x = pd.Series([1, 2, 3, 4, 5], dtype=float)
    y = pd.Series([10, 20, 30, 40, 50], dtype=float)
    assert np.isclose(rank_correlation(x, y), 1.0)
    assert np.isclose(rank_correlation(x, y.iloc[::-1].reset_index(drop=True)), -1.0)


def test_m18d_slice_metrics_preserve_scope_and_support_grades():
    d = pd.DataFrame({
        "g": ["A"] * 20 + ["B"] * 8 + ["C"] * 3,
        "m16g_abs_error_h": np.arange(31, dtype=float),
        "m16g_signed_error_h": np.arange(31, dtype=float) - 10,
    })
    out = categorical_slice_metrics(d, ["g"], "TEST")
    grades = dict(zip(out.slice_value, out.support_grade))
    assert grades["A"] == "ROBUST_N_GE20"
    assert grades["B"] == "SMALL_N_8_19"
    assert grades["C"] == "TINY_N_LT8"
    assert out.rows.sum() == len(d)


def test_m18d_expert_oracle_is_diagnostic_lower_bound():
    d = pd.DataFrame({
        "m16g_abs_error_h": [10.0, 20.0],
        "oracle_expert_abs_error_h": [5.0, 15.0],
        "m16g_selection_regret_h": [5.0, 5.0],
        "selected_expert_is_oracle": [False, False],
        "oracle_expert": ["route", "tabular"],
    })
    s = expert_diagnostics(d, "TEST")
    assert s["oracle_existing_expert_mae_h"] <= s["m16g_selected_mae_h"]
    assert s["diagnostic_only"] is True


def test_m18d_frozen_artifacts_if_present():
    p = ROOT / "reports/M18D_RESIDUAL_ATTRIBUTION.json"
    if not p.exists():
        return
    audit = json.loads(p.read_text())
    validate_attribution_dict(audit)
    freeze = json.loads((ROOT / "reports/M18D_ATTRIBUTION_FREEZE.json").read_text())
    ledger = pd.read_csv(ROOT / "reports/m18d_residual_ledger.csv")
    matrix = pd.read_csv(ROOT / "reports/m18d_missing_information_matrix.csv")
    assert freeze["model_change"] is False
    assert freeze["fresh_holdout_opened"] is False
    assert len(ledger) == 386 and ledger.mmsi.nunique() == 386
    assert ledger.m18d_scope_near_term.sum() == audit["findings"]["near_term_rows"]
    assert set(matrix["hypothesis"]).issuperset({
        "authoritative_target_ground_truth",
        "destination_and_route_intent",
        "raw_ais_freshness_and_state",
        "port_operations",
        "weather_and_ocean",
    })
    assert audit["policy"]["attribution_is_causal_proof"] is False
