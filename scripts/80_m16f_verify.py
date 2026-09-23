#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    R = ROOT / "reports"
    s = json.loads((R / "M16F_SUMMARY.json").read_text())
    f = json.loads((R / "M16F_TABULAR_PANEL_FREEZE.json").read_text())
    ledger = pd.read_csv(R / "m16f_oof_predictions.csv")
    manifest = json.loads((R / "M16A_DEVELOPMENT_MANIFEST.json").read_text())
    assert len(ledger) == 386 and ledger.mmsi.nunique() == 386
    assert set(ledger.mmsi.astype(int)).isdisjoint(manifest["blocked_old_final_mmsi"])
    assert s["final_test_used_for_selection"] is False and f["final_test_used_for_selection"] is False
    assert s["expert_predictions_used_as_features"] is False and f["expert_predictions_used_as_features"] is False
    assert f["candidate_config_count"] == 10
    assert set(f["families"]) == {"catboost", "extra_trees", "hist_gb", "random_forest", "svr"}
    assert s["fold_wins_vs_m16a_current_catboost"] >= 0
    for name, digest in f["artifact_sha256"].items():
        assert sha(R / name) == digest, name
    for name in [
        "M16A_BENCHMARK_FREEZE.json", "M16B_RESOLVER_FREEZE.json", "M16C_HIERARCHICAL_PRIOR_FREEZE.json",
        "M16D_ROUTE_ANALOGUE_FREEZE.json", "M16E_MARITIME_PHYSICS_FREEZE.json", "M14_FINAL_FREEZE.json",
        "M10_FREEZE.json", "M6_FREEZE.json",
    ]:
        assert (R / name).exists(), name
    print("PASS M16F dev-only tabular challenger panel")
    print(f"PASS best family: {s['best_family']}")
    print(f"PASS best MAE: {s['best_mae_h']:.6f} h")
    print(f"PASS gain vs M16A current CatBoost: {s['gain_h_vs_m16a_current_catboost']:.6f} h")
    print(f"PASS fold wins vs M16A current CatBoost: {s['fold_wins_vs_m16a_current_catboost']}/5")
    print(f"PASS old final blocked: {len(manifest['blocked_old_final_mmsi'])}/53")
    print(f"PASS frozen artifacts: {len(f['artifact_sha256'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
