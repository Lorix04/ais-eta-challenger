#!/usr/bin/env python3
"""Verify M16A hard-blocks, determinism and frozen-branch integrity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
EXPECTED_IMMUTABLE = {
    "dist/ais_eta_takehome_submission_final.zip": "65684d016775deea6b111db1a3ad76fe442cf84f1ceb1ac17f4327ea145c553c",
    "models/m10_strict_reference_catboost.cbm": "efbb86375751e901c48c2f5724514828a10f9eb8ff08ffaabf2d875e369ef86a",
    "reports/m10_final_predictions.csv": "fc6334d82f0b3c382e40805dc1a325163001a363ff6a6329da33160734011ee1",
    "reports/M6_FREEZE.json": "b551c12da6d05fbea22c365f0b2167879669d948c4d790d46ccd28a9dc8e17de",
    "reports/M10_FREEZE.json": "0b9970550ab4bc08f77fccfbac0f299cc8e87b8be57b8fa09868590e539d89ba",
}
COMPARE_FILES = [
    "M16A_DEVELOPMENT_MANIFEST.json",
    "m16a_oof_predictions.csv",
    "m16a_oof_model_metrics.csv",
    "m16a_oof_fold_metrics.csv",
    "m16a_oof_status_metrics.csv",
    "m16a_fold_gains.csv",
    "m16a_catboost_tree_counts.csv",
    "m16a_summary.json",
    "M16A_BENCHMARK_FREEZE.json",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_clean(out: Path) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/69_m16a_build_oof_benchmark.py"), "--output-dir", str(out)],
        cwd=ROOT, check=True, stdout=subprocess.DEVNULL,
    )


def main() -> int:
    for rel, expected in EXPECTED_IMMUTABLE.items():
        got = sha(ROOT / rel)
        assert got == expected, f"immutable hash changed: {rel} {got} != {expected}"
        print(f"PASS immutable {rel} {got}")

    manifest = json.loads((R / "M16A_DEVELOPMENT_MANIFEST.json").read_text())
    assert manifest["development_rows"] == 386
    assert len(manifest["allowed_mmsi"]) == 386
    assert manifest["blocked_old_final_rows"] == 53
    assert len(manifest["blocked_old_final_mmsi"]) == 53
    assert not (set(manifest["allowed_mmsi"]) & set(manifest["blocked_old_final_mmsi"]))
    assert manifest["old_final_may_be_used_for_selection"] is False
    print("PASS development manifest 386 allowed / 53 hard-blocked / zero overlap")

    ledger = pd.read_csv(R / "m16a_oof_predictions.csv")
    assert len(ledger) == 386 and ledger["mmsi"].nunique() == 386
    assert not set(ledger["mmsi"].astype(int)) & set(manifest["blocked_old_final_mmsi"])
    assert sorted(ledger["m16a_outer_fold"].value_counts().tolist()) == [77, 77, 77, 77, 78]
    for col in ["pred_global_train_median_h", "pred_train_destination_median_h", "pred_catboost_current_snapshot_h"]:
        assert ledger[col].notna().all(), f"missing OOF predictions: {col}"
    print("PASS OOF ledger exactly 386 development MMSIs with complete predictions")

    # Rebuild twice in clean temp directories and require byte-identical outputs.
    with tempfile.TemporaryDirectory(prefix="m16a_repeat1_") as a, tempfile.TemporaryDirectory(prefix="m16a_repeat2_") as b:
        pa, pb = Path(a), Path(b)
        run_clean(pa); run_clean(pb)
        for name in COMPARE_FILES:
            h_main, h_a, h_b = sha(R / name), sha(pa / name), sha(pb / name)
            assert h_main == h_a == h_b, f"non-deterministic M16A output: {name}"
        print(f"PASS deterministic rebuild x2 ({len(COMPARE_FILES)} artifacts byte-identical)")

    freeze = json.loads((R / "M16A_BENCHMARK_FREEZE.json").read_text())
    assert freeze["primary_metric"] == "mae_h"
    assert freeze["development_population"] == 386
    assert freeze["blocked_old_final_population"] == 53
    for name, expected in freeze["artifact_sha256"].items():
        assert sha(R / name) == expected, f"M16A freeze artifact mismatch: {name}"
    print(f"PASS M16A benchmark freeze {len(freeze['artifact_sha256'])} artifact hashes")

    state = json.loads((ROOT / "PROJECT_STATE.json").read_text())
    assert state["current_phase"] == "M16A_DONE_DEVELOPMENT_ONLY_OOF_BENCHMARK"
    assert state["m16_plan"]["status"] == "IN_PROGRESS_M16A_DONE"
    print("PASS project state marks M16A done and M16B next")

    summary = json.loads((R / "m16a_summary.json").read_text())
    assert summary["final_test_used_for_selection"] is False
    assert summary["development_rows"] == 386 and summary["blocked_old_final_rows"] == 53
    print("PASS summary claim boundary: development-only OOF, old final unused for selection")
    print("PASS M16A verification")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
