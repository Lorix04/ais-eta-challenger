#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ais_eta.m16c import candidate_configs

EXPECTED = {
    "dist/ais_eta_takehome_submission_final.zip": "65684d016775deea6b111db1a3ad76fe442cf84f1ceb1ac17f4327ea145c553c",
    "models/m10_strict_reference_catboost.cbm": "efbb86375751e901c48c2f5724514828a10f9eb8ff08ffaabf2d875e369ef86a",
    "reports/m10_final_predictions.csv": "fc6334d82f0b3c382e40805dc1a325163001a363ff6a6329da33160734011ee1",
    "reports/M6_FREEZE.json": "b551c12da6d05fbea22c365f0b2167879669d948c4d790d46ccd28a9dc8e17de",
    "reports/M16A_BENCHMARK_FREEZE.json": "2d3b1cdbe0e9efe170180aa4f55bdb00862f38a84f80406f7e8b7bf9f5fdbe1d",
    "reports/M16B_RESOLVER_FREEZE.json": "eff1bb3fc559d5d32a10004aaf8dddc082934103c9876ef6c397d003108e578c",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    summary = json.loads((ROOT / "reports/M16C_SUMMARY.json").read_text())
    freeze = json.loads((ROOT / "reports/M16C_HIERARCHICAL_PRIOR_FREEZE.json").read_text())
    ledger = pd.read_csv(ROOT / "reports/m16c_oof_predictions.csv")
    selections = pd.read_csv(ROOT / "reports/m16c_outer_selected_configs.csv")
    manifest = json.loads((ROOT / "reports/M16A_DEVELOPMENT_MANIFEST.json").read_text())

    assert len(ledger) == 386 and ledger["mmsi"].nunique() == 386
    assert set(ledger["mmsi"].astype(int)).isdisjoint(manifest["blocked_old_final_mmsi"])
    assert len(selections) == 5 and selections["outer_fold"].nunique() == 5
    assert selections["selected_on_inner_only"].astype(bool).all()
    assert summary["final_test_used_for_selection"] is False
    assert freeze["final_test_used_for_selection"] is False
    assert freeze["candidate_config_count"] == len(candidate_configs())
    assert summary["m16c_mae_gain_h_vs_m16a_raw_destination"] > 0
    assert summary["gate"] in {"PASS_STANDALONE", "PASS_COMPONENT_NOT_STANDALONE"}
    assert summary["standalone_promoted"] is False
    assert summary["equal_blend_50_50_diagnostic"]["weight_selected_or_tuned"] is False
    assert ledger["pred_m16c_hierarchical_prior_h"].notna().all()

    for rel, expected in EXPECTED.items():
        got = sha(ROOT / rel)
        assert got == expected, f"immutable hash drift: {rel}: {got} != {expected}"
        print(f"PASS immutable {rel} {got}")

    for name, expected in freeze["artifact_sha256"].items():
        got = sha(ROOT / "reports" / name)
        assert got == expected, f"M16C artifact drift {name}: {got} != {expected}"

    m = summary["m16c_hierarchical_prior_oof"]
    print(f"PASS M16C dev-only rows={len(ledger)} blocked_old_final={len(manifest['blocked_old_final_mmsi'])}")
    print(f"PASS nested selection outer=5 inner={summary['inner_folds']} candidate_configs={freeze['candidate_config_count']}")
    print(f"PASS M16C OOF MAE={m['mae_h']:.3f} h gain_vs_raw_dest={summary['m16c_mae_gain_h_vs_m16a_raw_destination']:.3f} h")
    print(f"PASS gate={summary['gate']} final_test_used_for_selection={summary['final_test_used_for_selection']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
