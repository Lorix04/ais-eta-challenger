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

from ais_eta.m18f import (  # noqa: E402
    M18F_BALANCED_TARGET_COVERAGE,
    sha256_file,
    validate_m18f_audit,
)


def main() -> int:
    audit = json.loads((R / "M18F_SELECTIVE_ETA.json").read_text())
    freeze = json.loads((R / "M18F_SELECTIVE_FREEZE.json").read_text())
    state = json.loads((ROOT / "PROJECT_STATE.json").read_text())
    validate_m18f_audit(audit)

    assert freeze["development_population"] == 386
    assert freeze["blocked_old_final_population"] == 53
    assert freeze["fresh_holdout_opened"] is False
    assert freeze["point_model_changed"] is False
    assert freeze["uncertainty_model_changed"] is False
    assert freeze["selection_uses_validation_target"] is False
    assert sha256_file(R / "M18A_BASELINE_FREEZE.json") == freeze["m18a_baseline_freeze_sha256"]
    assert sha256_file(R / "M18B_VALIDATION_FREEZE.json") == freeze["m18b_validation_freeze_sha256"]
    assert sha256_file(R / "M18C_AUDIT_FREEZE.json") == freeze["m18c_audit_freeze_sha256"]
    assert sha256_file(R / "M18D_ATTRIBUTION_FREEZE.json") == freeze["m18d_attribution_freeze_sha256"]
    assert sha256_file(R / "M18E_ABLATION_FREEZE.json") == freeze["m18e_ablation_freeze_sha256"]
    assert sha256_file(R / "m16g_oof_predictions.csv") == freeze["m16g_oof_sha256"]
    assert sha256_file(R / "m16h_oof_intervals.csv") == freeze["m16h_oof_intervals_sha256"]
    for rel, digest in freeze["artifact_sha256"].items():
        assert sha256_file(ROOT / rel) == digest, rel

    ledger = pd.read_csv(R / "m18f_selective_eta_ledger.csv")
    curve = pd.read_csv(R / "m18f_risk_coverage_curve.csv")
    folds = pd.read_csv(R / "m18f_fold_risk_coverage.csv")
    assert len(ledger) == 386 and ledger.mmsi.nunique() == 386
    assert set(ledger.m18f_balanced_action) == {"AUTO_ETA_WITH_UNCERTAINTY", "DEFER_LOW_CONFIDENCE"}
    balanced = curve[(curve.scope == "STRICT_386") & np.isclose(curve.target_coverage, M18F_BALANCED_TARGET_COVERAGE)].iloc[0]
    assert 0.70 <= float(balanced.realized_coverage) <= 0.90
    assert float(balanced.retained_mae_h) < float(curve[(curve.scope == "STRICT_386") & np.isclose(curve.target_coverage, 1.0)].retained_mae_h.iloc[0])
    bfold = folds[np.isclose(folds.target_coverage, M18F_BALANCED_TARGET_COVERAGE)]
    assert len(bfold) == 5
    assert (bfold.retained_mae_gain_vs_full_fold_h > 0).all()
    assert not bfold.threshold_uses_validation_target.astype(bool).any()
    assert state["current_phase"].startswith(("M18F_", "M18G_", "M19_"))
    assert state["m18f_summary"]["fresh_holdout_opened"] is False
    assert state["m18f_summary"]["selection_conditional_conformal_guarantee_claimed"] is False

    print("PASS M18F uncertainty/selective ETA frozen hashes and policy")
    print("PASS M16G point + M16H uncertainty unchanged; 386 development / 53 old-final blocked")
    print(f"PASS BALANCED_80 realized coverage: {float(balanced.realized_coverage):.6f}")
    print(f"PASS BALANCED_80 retained MAE: {float(balanced.retained_mae_h):.6f} h")
    print("PASS BALANCED_80 retained MAE improves in 5/5 outer folds")
    print("PASS retained interval coverage is empirical only; no selection-conditional guarantee claimed")
    print("PASS fresh holdout remains sealed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
