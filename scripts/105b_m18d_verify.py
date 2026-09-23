#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m18a import sha256_file  # noqa: E402
from ais_eta.m18d import validate_attribution_dict  # noqa: E402


def main() -> int:
    audit = json.loads((R / "M18D_RESIDUAL_ATTRIBUTION.json").read_text())
    freeze = json.loads((R / "M18D_ATTRIBUTION_FREEZE.json").read_text())
    state = json.loads((ROOT / "PROJECT_STATE.json").read_text())
    validate_attribution_dict(audit)

    assert freeze["model_change"] is False
    assert freeze["relabeling"] is False
    assert freeze["strict_rows_deleted"] == 0
    assert freeze["fresh_holdout_opened"] is False
    assert freeze["development_population"] == 386
    assert freeze["near_term_diagnostic_population"] == 279
    assert freeze["blocked_old_final_population"] == 53
    assert sha256_file(R / "M18A_BASELINE_FREEZE.json") == freeze["m18a_baseline_freeze_sha256"]
    assert sha256_file(R / "M18B_VALIDATION_FREEZE.json") == freeze["m18b_validation_freeze_sha256"]
    assert sha256_file(R / "M18C_AUDIT_FREEZE.json") == freeze["m18c_audit_freeze_sha256"]
    assert sha256_file(R / "m16g_oof_predictions.csv") == freeze["m16g_oof_sha256"]
    for rel, digest in freeze["attribution_artifact_sha256"].items():
        assert sha256_file(ROOT / rel) == digest, rel

    ledger = pd.read_csv(R / "m18d_residual_ledger.csv")
    slices = pd.read_csv(R / "m18d_categorical_slice_metrics.csv")
    matrix = pd.read_csv(R / "m18d_missing_information_matrix.csv")
    assert len(ledger) == 386 and ledger.mmsi.nunique() == 386
    assert int(ledger.m18d_scope_near_term.sum()) == 279
    assert set(slices.scope.unique()) == {"STRICT_386", "NEAR_TERM_FUTURE_0_7D"}
    assert matrix.iloc[0].hypothesis == "authoritative_target_ground_truth"
    assert "weather_and_ocean" in set(matrix.hypothesis)
    assert state["current_phase"].startswith(("M18D_", "M18E_", "M18F_", "M18G_", "M19_"))
    assert state["m18d_summary"]["fresh_holdout_opened"] is False
    assert state["m18d_summary"]["attribution_is_causal_proof"] is False

    print("PASS M18D frozen residual attribution artifacts and hashes")
    print("PASS 386 strict rows / 279 near-term diagnostic rows / 53 old-final blocked")
    print("PASS M18A/M18B/M18C freezes and M16G OOF unchanged")
    print("PASS Missing Information Matrix present; causal claims prohibited")
    print("PASS fresh holdout remains sealed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
