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
    s = json.loads((R / "M16E_SUMMARY.json").read_text())
    f = json.loads((R / "M16E_MARITIME_PHYSICS_FREEZE.json").read_text())
    ledger = pd.read_csv(R / "m16e_oof_predictions.csv")
    manifest = json.loads((R / "M16A_DEVELOPMENT_MANIFEST.json").read_text())
    assert len(ledger) == 386 and ledger.mmsi.nunique() == 386
    assert set(ledger.mmsi.astype(int)).isdisjoint(manifest["blocked_old_final_mmsi"])
    assert s["final_test_used_for_selection"] is False and f["final_test_used_for_selection"] is False
    assert s["reference_eta_status_used_for_selection_or_gate"] is False
    assert s["eligible_rows"] >= 100
    assert s["physics_wins_vs_route_share_on_eligible"] > 0.5
    assert s["hard_gate_mae_gain_h_vs_route"] > 0
    assert s["hard_gate_fold_wins_vs_route"] >= 3
    assert s["gate"].startswith("PASS")
    for name, digest in f["artifact_sha256"].items():
        assert sha(R / name) == digest, name
    for name in [
        "M16A_BENCHMARK_FREEZE.json", "M16B_RESOLVER_FREEZE.json", "M16C_HIERARCHICAL_PRIOR_FREEZE.json",
        "M16D_ROUTE_ANALOGUE_FREEZE.json", "M14_FINAL_FREEZE.json", "M10_FREEZE.json", "M6_FREEZE.json",
    ]:
        assert (R / name).exists(), name
    print("PASS M16E dev-only maritime physics expert")
    print(f"PASS eligible rows: {s['eligible_rows']}/386")
    print(f"PASS paired physics win share vs route: {s['physics_wins_vs_route_share_on_eligible']:.6f}")
    print(f"PASS hard-gate MAE gain vs route: {s['hard_gate_mae_gain_h_vs_route']:.6f} h")
    print(f"PASS hard-gate fold wins: {s['hard_gate_fold_wins_vs_route']}/5")
    print(f"PASS old final blocked: {len(manifest['blocked_old_final_mmsi'])}/53")
    print(f"PASS frozen artifacts: {len(f['artifact_sha256'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
