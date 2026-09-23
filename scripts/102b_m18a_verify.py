#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m18a import sha256_file, validate_contract_dict  # noqa: E402


def main() -> int:
    contract = json.loads((R / "M18A_PREDICTION_CONTRACT.json").read_text())
    freeze = json.loads((R / "M18A_BASELINE_FREEZE.json").read_text())
    state = json.loads((ROOT / "PROJECT_STATE.json").read_text())
    validate_contract_dict(contract)

    assert freeze["model_change"] is False
    assert freeze["prediction_artifact_rewrite"] is False
    assert freeze["fresh_holdout_opened"] is False
    assert freeze["point_champion"] == "M16G"
    assert freeze["uncertainty_sidecar"] == "M16H"
    assert freeze["development_population"] == 386
    assert freeze["blocked_old_final_population"] == 53

    for rel, digest in freeze["immutable_baseline_sha256"].items():
        assert sha256_file(ROOT / rel) == digest, rel
    for rel, digest in freeze["contract_artifact_sha256"].items():
        assert sha256_file(ROOT / rel) == digest, rel

    g = json.loads((R / "M16G_SUMMARY.json").read_text())
    led = pd.read_csv(R / "m16g_oof_predictions.csv")
    dev = json.loads((R / "M16A_DEVELOPMENT_MANIFEST.json").read_text())
    assert len(led) == 386 and led.mmsi.nunique() == 386
    assert set(led.mmsi.astype(int)).isdisjoint(set(dev["blocked_old_final_mmsi"]))
    mae = float(np.mean(np.abs(led.target_tte_h - led.pred_m16g_selected_h)))
    assert np.isclose(mae, g["mixture_mae_h"], rtol=0, atol=1e-9)

    assert state["current_phase"].startswith(("M18A_", "M18B_", "M18C_", "M18D_", "M18E_", "M18F_", "M18G_", "M19_"))
    assert state["m18a_summary"]["point_champion"] == "M16G"
    assert state["m18a_summary"]["fresh_holdout_opened"] is False

    print("PASS M18A machine-readable prediction contract")
    print("PASS frozen target: Tracks.eta reference at decision_time=Tracks.last_update")
    print("PASS causal boundary: Positions.recorded_at <= Tracks.last_update")
    print(f"PASS frozen M16G baseline n=386 MAE={mae:.6f} h; old-final 53 hard-blocked")
    print(f"PASS immutable baseline hashes: {len(freeze['immutable_baseline_sha256'])}")
    print("PASS fresh holdout unopened and forbidden for selection/tuning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
