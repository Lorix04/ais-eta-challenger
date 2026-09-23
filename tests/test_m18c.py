from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m18c import (
    build_ais_proxy_alignment,
    diagnostic_reliability_tier,
    validate_audit_dict,
)


def test_m18c_reliability_tiers_are_reference_only_diagnostics():
    base = {
        "reference_eta_status": "FUTURE_0_7D",
        "eta_placeholder_like_pattern": False,
        "year_assignment_margin_lt90d": False,
    }
    assert diagnostic_reliability_tier(pd.Series(base)) == "R3_REFERENCE_NEAR_TERM"
    assert diagnostic_reliability_tier(pd.Series({**base, "reference_eta_status": "FUTURE_7_14D"})) == "R2_REFERENCE_EXTENDED_HORIZON"
    assert diagnostic_reliability_tier(pd.Series({**base, "reference_eta_status": "PAST_1_24H"})) == "R1_REFERENCE_QUESTIONABLE"
    assert diagnostic_reliability_tier(pd.Series({**base, "reference_eta_status": "PAST_GT24H"})) == "R0_REFERENCE_AMBIGUOUS"
    assert diagnostic_reliability_tier(pd.Series({**base, "eta_placeholder_like_pattern": True})) == "R0_REFERENCE_AMBIGUOUS"


def test_m18c_proxy_alignment_cannot_be_authoritative_ground_truth():
    f = pd.DataFrame({
        "mmsi": [1], "name": ["X"], "last_update": [pd.Timestamp("2026-04-03 12:00")],
        "eta_reference_dt": [pd.Timestamp("2026-04-03 10:00")],
        "reference_eta_status": ["PAST_1_24H"],
        "m18c_reliability_tier": ["R1_REFERENCE_QUESTIONABLE"],
    })
    calls = pd.DataFrame({"mmsi": [1], "ground_truth_time": ["2026-04-03 09:55:00"], "port": ["TEST"]})
    out = build_ais_proxy_alignment(f, calls)
    assert len(out) == 1
    assert bool(out.loc[0, "proxy_alignment_is_posthoc_only"]) is True
    assert bool(out.loc[0, "proxy_is_authoritative_ata"]) is False
    assert out.loc[0, "proxy_calls_on_or_after_decision"] == 0


def test_m18c_frozen_artifacts_if_present():
    p = ROOT / "reports/M18C_TARGET_LABEL_AUDIT.json"
    if not p.exists():
        return
    audit = json.loads(p.read_text())
    validate_audit_dict(audit)
    freeze = json.loads((ROOT / "reports/M18C_AUDIT_FREEZE.json").read_text())
    f = pd.read_csv(ROOT / "reports/m18c_reference_forensics.csv")
    assert freeze["model_change"] is False
    assert freeze["relabeling"] is False
    assert freeze["strict_rows_deleted"] == 0
    assert freeze["fresh_holdout_opened"] is False
    assert len(f) == 386 and f.mmsi.nunique() == 386
    assert audit["policy"]["ais_proxy_replaces_company_target"] is False
    assert audit["policy"]["authoritative_port_call_ground_truth_still_missing"] is True
