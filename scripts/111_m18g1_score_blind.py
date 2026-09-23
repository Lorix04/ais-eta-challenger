#!/usr/bin/env python3
"""Produce and SHA-256 seal an M18G blind prediction ledger before labels.

This command must be run on an *unlabelled* fresh cohort.  It refuses target or
reference-ETA columns, verifies the registered full-development scorer hash,
checks that sample keys do not reuse the original 439 supervised rows, and
writes a pre-label opening manifest consumable by script 109.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
MODELS = ROOT / "models"
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m18g import sha256_file, validate_prediction_ledger
from ais_eta.m18g1 import load_rows, load_states, score_fresh_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, help="Fresh target-free reference-row table (CSV/Parquet/Pickle)")
    ap.add_argument("--states", required=True, help="Matching causal AIS state history (CSV/Parquet/Pickle)")
    ap.add_argument("--output", required=True, help="Blind prediction ledger CSV to create")
    ap.add_argument("--holdout-id", required=True)
    ap.add_argument("--provenance-sha256", required=True, help="Already-sealed hash of the fresh cohort provenance manifest")
    ap.add_argument("--opening-manifest", required=True, help="Pre-label opening manifest JSON to create")
    ap.add_argument("--detail-output", help="Optional target-free diagnostic detail CSV")
    args = ap.parse_args()

    output = Path(args.output)
    opening_path = Path(args.opening_manifest)
    detail_path = Path(args.detail_output) if args.detail_output else None
    for p in [output, opening_path, detail_path]:
        if p is not None and p.exists():
            raise SystemExit(f"refusing to overwrite existing pre-label artifact: {p}")

    registration = json.loads((R / "M18G1_SCORING_BUNDLE_REGISTRATION.json").read_text())
    bundle_path = ROOT / registration["bundle_path"]
    actual_bundle_sha = sha256_file(bundle_path)
    if actual_bundle_sha != registration["bundle_sha256"]:
        raise SystemExit("registered scoring bundle SHA-256 mismatch")
    bundle = joblib.load(bundle_path)

    rows = load_rows(args.rows)
    states = load_states(args.states)
    ledger, detail = score_fresh_rows(bundle, rows, states)
    validate_prediction_ledger(ledger, require_candidate=False)

    forbidden = set(pd.read_csv(R / "m18g_forbidden_sample_keys.csv").sample_key.astype(str))
    overlap = set(ledger.sample_key.astype(str)) & forbidden
    if overlap:
        raise SystemExit(f"refusing blind ledger: {len(overlap)} sample keys reuse original supervised rows")

    output.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(output, index=False, float_format="%.12g")
    pred_sha = sha256_file(output)
    Path(str(output) + ".sha256").write_text(f"{pred_sha}  {output.name}\n")

    if detail_path is not None:
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        detail.to_csv(detail_path, index=False, float_format="%.12g")
        Path(str(detail_path) + ".sha256").write_text(f"{sha256_file(detail_path)}  {detail_path.name}\n")

    manifest = {
        "status": "SEALED_BEFORE_LABEL_REVEAL",
        "gate_freeze_sha256": sha256_file(R / "M18G_PROMOTION_FREEZE.json"),
        "holdout_id": str(args.holdout_id),
        "holdout_provenance_sha256": str(args.provenance_sha256),
        "prediction_ledger_sha256": pred_sha,
        "registered_scoring_bundle_sha256": actual_bundle_sha,
        "scoring_bundle_registration_sha256": sha256_file(R / "M18G1_SCORING_BUNDLE_REGISTRATION.json"),
        "point_candidate_registered": False,
        "point_candidate_bundle_sha256": None,
        "labels_observed_when_prediction_ledger_sealed": False,
        "opening_count_before_this_evaluation": 0,
        "rows": int(len(ledger)),
    }
    opening_path.parent.mkdir(parents=True, exist_ok=True)
    opening_path.write_text(json.dumps(manifest, indent=2) + "\n")
    opening_sha = sha256_file(opening_path)
    Path(str(opening_path) + ".sha256").write_text(f"{opening_sha}  {opening_path.name}\n")

    print(f"PASS blind prediction ledger rows: {len(ledger)}")
    print(f"PASS prediction ledger SHA-256: {pred_sha}")
    print(f"PASS scoring bundle SHA-256: {actual_bundle_sha}")
    print(f"PASS opening manifest SHA-256: {opening_sha}")
    print("PASS labels_observed_when_prediction_ledger_sealed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
