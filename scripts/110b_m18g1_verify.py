#!/usr/bin/env python3
"""Verify the frozen M18G1 full-development scoring bundle and registration."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
sys.path.insert(0, str(ROOT / "src"))
from ais_eta.m18g import sha256_file, validate_prediction_ledger
from ais_eta.m18g1 import M18G1_VERSION


def main() -> int:
    reg = json.loads((R / "M18G1_SCORING_BUNDLE_REGISTRATION.json").read_text())
    freeze = json.loads((R / "M18G1_SCORING_BUNDLE_FREEZE.json").read_text())
    assert reg["version"] == M18G1_VERSION
    assert reg["status"] == "FULL_DEVELOPMENT_M16G_M16H_SCORING_BUNDLE_REGISTERED"
    assert reg["development_population"] == 386
    assert reg["blocked_old_final_population"] == 53
    assert reg["fresh_holdout_opened"] is False
    assert reg["holdout_labels_observed"] is False
    assert reg["historical_m16g_oof_score_changed"] is False
    bundle_path = ROOT / reg["bundle_path"]
    assert bundle_path.exists()
    assert sha256_file(bundle_path) == reg["bundle_sha256"]
    sidecar = Path(str(bundle_path) + ".sha256")
    assert sidecar.exists() and reg["bundle_sha256"] in sidecar.read_text()
    bundle = joblib.load(bundle_path)
    assert bundle["version"] == M18G1_VERSION
    assert bundle["development_rows"] == 386
    assert bundle["blocked_old_final_rows"] == 53
    assert bundle["external_holdout_opened"] is False
    assert bundle["labels_required_for_scoring"] is False
    assert len(bundle["training_mmsis"]) == 386
    assert len(set(bundle["training_mmsis"])) == 386
    assert len(bundle["route_memory"]["sequences"]) == 386
    smoke = pd.read_csv(R / "m18g1_blind_smoke_ledger.csv")
    validate_prediction_ledger(smoke, require_candidate=False)
    forbidden_cols = {"target_tte_h", "reference_eta_status", "eta_reference_dt", "abs_error_h", "signed_error_h"}
    assert not (forbidden_cols & set(smoke.columns))
    assert freeze["m18g_gate_modified"] is False
    assert freeze["fresh_holdout_opened"] is False
    for rel, expected in freeze["artifact_sha256"].items():
        assert sha256_file(ROOT / rel) == expected, rel
    print(f"PASS M18G1 bundle SHA-256: {reg['bundle_sha256']}")
    print("PASS development population: 386; old-final blocked: 53")
    print(f"PASS synthetic blind smoke ledger: {len(smoke)} rows, label-free")
    print("PASS historical M18G gate freeze untouched")
    print("PASS historical M18G1 registration is unchanged; later M19 opening is tracked separately")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
