#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m18g import (  # noqa: E402
    M18G_SELECTIVE_RISK_THRESHOLD_H,
    sha256_file,
    validate_gate_contract,
)


def main() -> int:
    gate = json.loads((R / "M18G_EXTERNAL_PROMOTION_GATE.json").read_text())
    freeze = json.loads((R / "M18G_PROMOTION_FREEZE.json").read_text())
    state = json.loads((ROOT / "PROJECT_STATE.json").read_text())
    validate_gate_contract(gate)

    assert freeze["development_population"] == 386
    assert freeze["blocked_old_final_population"] == 53
    assert freeze["fresh_holdout_opened"] is False
    assert freeze["holdout_labels_observed"] is False
    assert freeze["one_time_evaluation_executed"] is False
    assert freeze["opening_ready_now"] is False
    assert sha256_file(R / "M18A_BASELINE_FREEZE.json") == freeze["m18a_freeze_sha256"]
    assert sha256_file(R / "M18B_VALIDATION_FREEZE.json") == freeze["m18b_freeze_sha256"]
    assert sha256_file(R / "M18C_AUDIT_FREEZE.json") == freeze["m18c_freeze_sha256"]
    assert sha256_file(R / "M18D_ATTRIBUTION_FREEZE.json") == freeze["m18d_freeze_sha256"]
    assert sha256_file(R / "M18E_ABLATION_FREEZE.json") == freeze["m18e_freeze_sha256"]
    assert sha256_file(R / "M18F_SELECTIVE_FREEZE.json") == freeze["m18f_freeze_sha256"]
    assert sha256_file(R / "m16g_oof_predictions.csv") == freeze["m16g_oof_sha256"]
    assert sha256_file(R / "m16h_oof_intervals.csv") == freeze["m16h_oof_intervals_sha256"]
    for rel, digest in freeze["artifact_sha256"].items():
        assert sha256_file(ROOT / rel) == digest, rel

    forbidden = pd.read_csv(R / "m18g_forbidden_sample_keys.csv")
    registry = pd.read_csv(R / "m18g_candidate_registry.csv")
    checklist = pd.read_csv(R / "m18g_opening_readiness_checklist.csv")
    assert len(forbidden) == 439 and forbidden.sample_key.nunique() == 439
    assert int(forbidden["split"].eq("final_test").sum()) == 53
    assert registry.loc[registry.track.eq("POINT_MODEL"), "candidate_id"].iloc[0] == "UNREGISTERED"
    assert not bool(registry.external_opening_allowed_now.astype(bool).any())
    assert checklist.loc[checklist.check.eq("gate_freeze"), "passed_now"].astype(str).str.lower().eq("true").all()
    assert not checklist.loc[checklist.check.eq("labels_observed"), "passed_now"].astype(str).str.lower().eq("true").any()
    assert np.isclose(float(gate["selective_eta_gate"]["frozen_risk_threshold_h"]), M18G_SELECTIVE_RISK_THRESHOLD_H)
    assert state["current_phase"].startswith(("M18G_", "M19_"))
    assert state["m18g_summary"]["fresh_holdout_opened"] is False
    assert state["m18g_summary"]["opening_ready_now"] is False

    print("PASS M18G frozen external promotion gate and artifact hashes")
    print("PASS historical M18G pre-opening freeze remains immutable (opened/labels/evaluation flags are frozen false by construction)")
    m19_status_path = R / "M19_STATUS.json"
    if m19_status_path.exists():
        m19_status = json.loads(m19_status_path.read_text())
        if m19_status.get("external_holdout_opened") is True:
            print("PASS M19 records the later one-shot external opening separately; M18G gate freeze was not mutated")
    print("PASS original 439 supervised sample keys frozen and excluded (53 old-final included)")
    print("PASS point gate: paired bootstrap / P90 / four pre-label critical-slice guards frozen")
    print(f"PASS M18F external selective threshold frozen at {M18G_SELECTIVE_RISK_THRESHOLD_H:.9f} h")
    print("PASS M18G pre-opening readiness rules remain frozen and auditable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
