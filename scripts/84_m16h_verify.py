#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    s = json.loads((R / "M16H_SUMMARY.json").read_text())
    f = json.loads((R / "M16H_PROBABILISTIC_FREEZE.json").read_text())
    h = pd.read_csv(R / "m16h_oof_intervals.csv")
    g = pd.read_csv(R / "m16g_oof_predictions.csv")[["mmsi", "pred_m16g_selected_h"]]
    manifest = json.loads((R / "M16A_DEVELOPMENT_MANIFEST.json").read_text())
    assert len(h) == 386 and h.mmsi.nunique() == 386
    assert set(h.mmsi.astype(int)).isdisjoint(set(manifest["blocked_old_final_mmsi"]))
    z = h.merge(g, on="mmsi", how="inner", suffixes=("", "_g"))
    assert len(z) == 386
    assert np.allclose(z["p50_h"], z["pred_m16g_selected_h_g"], atol=1e-10)
    assert (h["p10_h"] <= h["p50_h"]).all() and (h["p50_h"] <= h["p90_h"]).all()
    assert s["final_test_used_for_calibration"] is False
    assert s["target_derived_confidence_features_used"] is False
    assert s["gate"] == "PASS_PROBABILISTIC_CALIBRATION_CONFIDENCE"
    assert 0.78 <= s["pooled_coverage"] <= 0.86
    assert s["mean_width_reduction_fraction"] >= 0.10
    assert s["stable_outer_folds_at_or_above_75pct_coverage"] >= 4
    med = s["confidence_medae_h"]
    assert med["HIGH"] <= med["MEDIUM"] <= med["LOW"]
    assert f["p50_is_frozen_m16g"] is True
    assert f["error_risk_cross_fitted_inside_outer_train"] is True
    assert f["interval_candidate_selected_on_inner_only"] is True
    for name, digest in f["artifact_sha256"].items():
        assert sha(R / name) == digest, name
    for name, digest in f["prior_freeze_sha256"].items():
        assert sha(R / name) == digest, name
    state = json.loads((ROOT / "PROJECT_STATE.json").read_text())
    assert state["current_phase"].startswith(("M16H_", "M16I_", "M16J_", "M17", "M18", "M19"))
    print("PASS M16H development-only probabilistic calibration")
    print(f"PASS empirical coverage: {s['pooled_coverage']:.6f}")
    print(f"PASS mean width: {s['pooled_mean_width_h']:.6f} h")
    print(f"PASS width reduction vs global: {s['mean_width_reduction_fraction']:.6f}")
    print(f"PASS stable folds: {s['stable_outer_folds_at_or_above_75pct_coverage']}/5")
    print("PASS P50 exactly equals frozen M16G")
    print("PASS old final blocked: 53/53")
    print(f"PASS frozen M16H artifacts: {len(f['artifact_sha256'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
