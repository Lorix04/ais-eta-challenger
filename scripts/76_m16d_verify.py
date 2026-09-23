#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()

def main() -> int:
    R = ROOT / "reports"
    s = json.loads((R / "M16D_SUMMARY.json").read_text())
    f = json.loads((R / "M16D_ROUTE_ANALOGUE_FREEZE.json").read_text())
    ledger = pd.read_csv(R / "m16d_oof_predictions.csv")
    manifest = json.loads((R / "M16A_DEVELOPMENT_MANIFEST.json").read_text())
    assert len(ledger) == 386 and ledger.mmsi.nunique() == 386
    assert set(ledger.mmsi.astype(int)).isdisjoint(manifest["blocked_old_final_mmsi"])
    assert s["final_test_used_for_selection"] is False and f["final_test_used_for_selection"] is False
    assert s["history_gate_pass"] is True
    assert s["route_mae_gain_h_vs_history_aggregate_catboost"] > 0
    assert s["route_fold_wins_vs_history_aggregate_catboost"] >= 3
    assert s["manual_geometry_audit_rows"] >= 10
    for name, digest in f["artifact_sha256"].items():
        assert sha(R / name) == digest, name
    # All prior immutable branch freezes still exist; M16D does not rewrite them.
    for name in ["M16A_BENCHMARK_FREEZE.json", "M16B_RESOLVER_FREEZE.json", "M16C_HIERARCHICAL_PRIOR_FREEZE.json", "M14_FINAL_FREEZE.json", "M10_FREEZE.json", "M6_FREEZE.json"]:
        assert (R / name).exists(), name
    print("PASS M16D dev-only route analogue")
    print(f"PASS route gain vs history aggregate: {s['route_mae_gain_h_vs_history_aggregate_catboost']:.6f} h")
    print(f"PASS fold wins vs history aggregate: {s['route_fold_wins_vs_history_aggregate_catboost']}/5")
    print(f"PASS old final blocked: {len(manifest['blocked_old_final_mmsi'])}/53")
    print(f"PASS frozen artifacts: {len(f['artifact_sha256'])}")
    return 0

if __name__ == "__main__": raise SystemExit(main())
