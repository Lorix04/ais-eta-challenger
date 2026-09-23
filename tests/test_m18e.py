from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m18e import paired_gain_summary, validate_ablation_audit, validate_source_registry


def test_m18e_paired_gain_positive_means_candidate_better():
    y = [10.0, 20.0, 30.0, 40.0]
    b = [20.0, 30.0, 40.0, 50.0]
    c = [11.0, 19.0, 31.0, 39.0]
    s = paired_gain_summary(y, b, c, seed=7, draws=500)
    assert s["mean_abs_error_gain_h"] > 0
    assert s["win_fraction"] == 1.0
    assert s["paired_bootstrap_ci95_h"][0] > 0


def test_m18e_source_registry_rejects_unpinned_ready_source():
    d = pd.DataFrame([{
        "source_id":"x", "information_family":"x", "source_name":"x", "source_url":"x",
        "source_authority":"x", "data_version":"x", "decision_time_safe":True,
        "artifact_pinned":False, "status":"READY_NEW_DATA_PINNED", "m18e_action":"x", "notes":"x",
    }])
    try:
        validate_source_registry(d)
    except AssertionError:
        pass
    else:
        raise AssertionError("unpinned READY source should be rejected")


def test_m18e_frozen_artifacts_if_present():
    p = ROOT / "reports/M18E_EXTERNAL_ABLATION.json"
    if not p.exists():
        return
    audit = json.loads(p.read_text())
    validate_ablation_audit(audit)
    reg = pd.read_csv(ROOT / "reports/m18e_external_source_registry.csv")
    validate_source_registry(reg)
    abl = pd.read_csv(ROOT / "reports/m18e_ablation_results.csv")
    assert audit["scope"]["fresh_holdout_opened"] is False
    assert audit["scope"]["new_external_model_promoted"] is False
    assert len(abl) == 4
    assert int(reg.status.eq("READY_NEW_DATA_PINNED").sum()) == 0
    assert (abl.rows > 0).all()


def test_m18e_blocked_sources_are_unmeasured_not_zero_gain():
    p = ROOT / "reports/M18E_EXTERNAL_ABLATION.json"
    if not p.exists():
        return
    audit = json.loads(p.read_text())
    assert audit["policy"]["blocked_sources_are_treated_as_zero_gain"] is False
    reg = pd.read_csv(ROOT / "reports/m18e_external_source_registry.csv")
    blocked = reg[reg.status.str.startswith("BLOCKED_")]
    assert len(blocked) >= 4
