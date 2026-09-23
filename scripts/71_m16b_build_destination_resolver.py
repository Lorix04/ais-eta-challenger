#!/usr/bin/env python3
"""Build M16B canonical destination artifacts and OOF diagnostic comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ais_eta.m16a import assert_development_only, extended_metrics
from ais_eta.m16b import DestinationResolver, M16B_RESOLVER_VERSION

D = ROOT / "data/derived/m10_reference_rows.pkl.gz"
CATALOG = ROOT / "data/static/m16b_destination_catalog.csv"
M16A_LEDGER = ROOT / "reports/m16a_oof_predictions.csv"
M16A_MANIFEST = ROOT / "reports/M16A_DEVELOPMENT_MANIFEST.json"
R = ROOT / "reports"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_pickle(D)
    final = raw.loc[raw["split"].eq("final_test")].copy()
    dev = raw.loc[raw["split"].isin(["train", "calibration"])].copy()
    assert_development_only(dev, final["mmsi"])
    if len(dev) != 386 or len(final) != 53:
        raise AssertionError("M16B population drift")

    m16a = pd.read_csv(M16A_LEDGER)
    manifest = json.loads(M16A_MANIFEST.read_text())
    if set(m16a["mmsi"].astype(int)) != set(manifest["allowed_mmsi"]):
        raise AssertionError("M16B M16A ledger/manifest population mismatch")
    if set(m16a["mmsi"].astype(int)) & set(manifest["blocked_old_final_mmsi"]):
        raise AssertionError("M16B old final contamination")

    resolver = DestinationResolver.from_csv(CATALOG)
    dev = dev.sort_values("mmsi", kind="mergesort").reset_index(drop=True)
    resolved = resolver.transform(dev["destination_norm"])
    ledger = pd.concat([
        dev[["mmsi", "split", "destination_norm", "target_tte_h"]].reset_index(drop=True),
        resolved.reset_index(drop=True),
    ], axis=1)
    fold_map = dict(zip(m16a["mmsi"].astype(int), m16a["m16a_outer_fold"].astype(int)))
    ledger["m16a_outer_fold"] = ledger["mmsi"].astype(int).map(fold_map)
    if ledger["m16a_outer_fold"].isna().any():
        raise AssertionError("M16B fold assignment missing")

    # M16B diagnostic: same train-only destination median as M16A, but grouping
    # by target-free canonical destination.  This is not the M16C hierarchy.
    preds = np.full(len(ledger), np.nan)
    fold_rows = []
    for fold in sorted(ledger["m16a_outer_fold"].unique()):
        tr = ledger.loc[ledger["m16a_outer_fold"].ne(fold)]
        va = ledger.loc[ledger["m16a_outer_fold"].eq(fold)]
        global_med = float(tr["target_tte_h"].median())
        med = tr.groupby("canonical_destination")["target_tte_h"].median()
        pred = va["canonical_destination"].map(med).fillna(global_med).to_numpy(float)
        preds[va.index.to_numpy()] = pred
        fold_rows.append({"outer_fold": int(fold), **extended_metrics(va["target_tte_h"], pred)})
    if not np.isfinite(preds).all():
        raise AssertionError("non-finite M16B OOF prediction")
    ledger["pred_train_canonical_destination_median_h"] = preds

    raw_pred_map = dict(zip(m16a["mmsi"].astype(int), m16a["pred_train_destination_median_h"].astype(float)))
    ledger["pred_m16a_raw_destination_median_h"] = ledger["mmsi"].astype(int).map(raw_pred_map)
    raw_metrics = extended_metrics(ledger["target_tte_h"], ledger["pred_m16a_raw_destination_median_h"])
    canonical_metrics = extended_metrics(ledger["target_tte_h"], ledger["pred_train_canonical_destination_median_h"])

    unique_raw = int(ledger["destination_norm"].nunique(dropna=False))
    unique_canon = int(ledger["canonical_destination"].nunique(dropna=False))
    resolved_n = int(ledger["is_resolved_port"].sum())
    nonspecific_n = int(ledger["is_non_specific"].sum())
    route_n = int(ledger["is_route_expression"].sum())
    method_counts = ledger["resolution_method"].value_counts().sort_index().to_dict()

    variant = (ledger.groupby(["destination_norm", "canonical_destination", "resolution_method"], dropna=False)
               .size().reset_index(name="n").sort_values(["n", "destination_norm"], ascending=[False, True]))
    canon = (ledger.groupby(["canonical_destination", "canonical_unlocode", "canonical_port_name", "canonical_country", "resolution_method"], dropna=False)
             .agg(n=("mmsi", "size"), raw_variants=("destination_norm", "nunique"))
             .reset_index().sort_values(["n", "canonical_destination"], ascending=[False, True]))
    fold_metrics = pd.DataFrame(fold_rows).sort_values("outer_fold")

    out_cols = [
        "mmsi", "split", "destination_norm", "normalized_destination", "terminal_segment",
        "canonical_destination", "canonical_unlocode", "canonical_port_name", "canonical_country",
        "canonical_lat", "canonical_lon", "resolution_method", "resolution_confidence",
        "is_resolved_port", "is_non_specific", "is_route_expression", "catalog_version",
        "m16a_outer_fold", "target_tte_h", "pred_m16a_raw_destination_median_h",
        "pred_train_canonical_destination_median_h",
    ]
    ledger[out_cols].to_csv(output_dir / "m16b_destination_resolutions.csv", index=False, float_format="%.12g")
    variant.to_csv(output_dir / "m16b_destination_variant_audit.csv", index=False)
    canon.to_csv(output_dir / "m16b_canonical_destination_summary.csv", index=False)
    fold_metrics.to_csv(output_dir / "m16b_canonical_destination_oof_fold_metrics.csv", index=False, float_format="%.12g")

    coverage = {
        "milestone": "M16B",
        "resolver_version": M16B_RESOLVER_VERSION,
        "resolver_inputs": ["destination_norm", "versioned destination catalog"],
        "target_or_reference_eta_used_in_resolution": False,
        "development_rows": 386,
        "blocked_old_final_rows": 53,
        "raw_destination_unique": unique_raw,
        "canonical_destination_unique": unique_canon,
        "cardinality_reduction": unique_raw - unique_canon,
        "cardinality_reduction_pct": (unique_raw - unique_canon) / unique_raw,
        "resolved_port_rows": resolved_n,
        "resolved_port_coverage": resolved_n / len(ledger),
        "non_specific_rows": nonspecific_n,
        "route_expression_rows": route_n,
        "method_counts": {str(k): int(v) for k, v in method_counts.items()},
        "raw_destination_median_oof": raw_metrics,
        "canonical_destination_median_oof": canonical_metrics,
        "canonical_mae_gain_h_vs_m16a_raw_destination": raw_metrics["mae_h"] - canonical_metrics["mae_h"],
        "final_test_used_for_selection": False,
        "catalog_file": "data/static/m16b_destination_catalog.csv",
        "catalog_sha256": sha(CATALOG),
    }
    (output_dir / "M16B_CARDINALITY_COVERAGE.json").write_text(json.dumps(coverage, indent=2) + "\n")

    freeze_files = [
        "m16b_destination_resolutions.csv", "m16b_destination_variant_audit.csv",
        "m16b_canonical_destination_summary.csv", "m16b_canonical_destination_oof_fold_metrics.csv",
        "M16B_CARDINALITY_COVERAGE.json",
    ]
    freeze = {
        "milestone": "M16B",
        "status": "CANONICAL_DESTINATION_RESOLVER_FROZEN",
        "resolver_version": M16B_RESOLVER_VERSION,
        "frozen_catalog_sha256": sha(CATALOG),
        "development_population": 386,
        "old_final_population_blocked": 53,
        "resolver_target_free": True,
        "artifact_sha256": {name: sha(output_dir / name) for name in freeze_files},
    }
    (output_dir / "M16B_RESOLVER_FREEZE.json").write_text(json.dumps(freeze, indent=2) + "\n")
    return coverage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=R)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
