#!/usr/bin/env python3
"""Freeze M10 target, split and candidate-model policy before final-test scoring."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "derived" / "m10_reference_rows.pkl.gz"
R = ROOT / "reports"
OUT = R / "M10_FREEZE.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if OUT.exists():
        raise RuntimeError("M10_FREEZE.json already exists; refusing to overwrite the frozen protocol")
    df = pd.read_pickle(D)
    final_ids = sorted(int(x) for x in df.loc[df["split"].eq("final_test"), "mmsi"])
    policy = {
        "created_at": "2026-09-21T15:56:00+02:00",
        "company_target_policy": "Tracks.eta is the exercise reference/ground-truth value per company clarification",
        "semantic_caveat": "Tracks.eta is an Estimated Time of Arrival reference, not observed Actual Time of Arrival",
        "unit_of_supervision": "one Tracks row / one MMSI",
        "target": "target_tte_h = parsed Tracks.eta - Tracks.last_update",
        "eta_year_resolution": "nearest valid previous/current/next calendar year because AIS Message 5 ETA omits year",
        "strict_benchmark": "all parseable ship ETA references",
        "operational_diagnostic": "future-reference subset target_tte_h >= 0, reported separately by 0-7d/7-14d/14-30d/>30d bands",
        "split": "deterministic target-free MMSI hash: 70% train / 15% calibration / 15% final_test",
        "final_test_mmsi": final_ids,
        "candidate_models": [
            "global_train_median",
            "train_destination_median_with_global_fallback",
            "ridge_current_plus_history",
            "catboost_current_snapshot",
            "catboost_current_plus_causal_history",
        ],
        "selection_metric": "calibration MAE hours; tie-break P90 absolute error",
        "selection_scope_strict": "all parseable reference labels",
        "selection_scope_future": "future-reference labels only",
        "no_final_test_model_selection": True,
        "no_eta_as_feature": True,
        "no_future_positions": True,
        "reference_dataset_sha256": sha256(D),
        "m6_predictor_unchanged": True,
        "m6_holdout_not_reused": True,
    }
    OUT.write_text(json.dumps(policy, indent=2) + "\n")
    print(f"PASS M10 freeze written with {len(final_ids)} final-test MMSIs")
    print(f"PASS dataset sha256 {policy['reference_dataset_sha256']}")
    print("PASS model selection is calibration-only; final test remains unopened by this script")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
