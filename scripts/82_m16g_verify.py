#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main() -> int:
    s = json.loads((R / "M16G_SUMMARY.json").read_text())
    f = json.loads((R / "M16G_MIXTURE_FREEZE.json").read_text())
    ledger = pd.read_csv(R / "m16g_oof_predictions.csv")
    manifest = json.loads((R / "M16A_DEVELOPMENT_MANIFEST.json").read_text())
    assert len(ledger) == 386 and ledger.mmsi.nunique() == 386
    assert set(ledger.mmsi.astype(int)).isdisjoint(set(manifest["blocked_old_final_mmsi"]))
    assert s["blocked_old_final_rows"] == 53 and s["final_test_used_for_selection"] is False
    assert s["base_predictions_for_meta_training"].startswith("regenerated inner-OOF")
    assert s["target_derived_gating_features_used"] is False
    assert s["gain_h_vs_best_single"] >= 1.0
    assert s["fold_wins_vs_best_single"] >= 3
    assert s["changed_row_win_share_vs_best_single"] >= 0.50
    assert s["gate"] == "PASS_MOE_STABLE_GAIN"
    assert f["base_predictions_regenerated_inner_oof"] is True
    assert f["target_derived_gating_features_used"] is False
    for name, digest in f["artifact_sha256"].items():
        assert sha(R / name) == digest, name
    for name in [
        "M16A_BENCHMARK_FREEZE.json", "M16B_RESOLVER_FREEZE.json", "M16C_HIERARCHICAL_PRIOR_FREEZE.json",
        "M16D_ROUTE_ANALOGUE_FREEZE.json", "M16E_MARITIME_PHYSICS_FREEZE.json", "M16F_TABULAR_PANEL_FREEZE.json",
        "M14_FINAL_FREEZE.json", "M10_FREEZE.json", "M6_FREEZE.json",
    ]:
        assert (R / name).exists(), name
    state = json.loads((ROOT / "PROJECT_STATE.json").read_text())
    assert state["current_phase"].startswith(("M16G_", "M16H_", "M16I_", "M16J_", "M17", "M18", "M19"))
    print("PASS M16G development-only nested OOF mixture")
    print(f"PASS mixture MAE: {s['mixture_mae_h']:.6f} h")
    print(f"PASS gain vs best single: {s['gain_h_vs_best_single']:.6f} h")
    print(f"PASS fold wins: {s['fold_wins_vs_best_single']}/5")
    print(f"PASS changed-row win share: {s['changed_row_win_share_vs_best_single']:.6f}")
    print("PASS old final blocked: 53/53")
    print(f"PASS frozen M16G artifacts: {len(f['artifact_sha256'])}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
