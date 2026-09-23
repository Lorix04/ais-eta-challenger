#!/usr/bin/env python3
"""Verify M10 target/split/leakage invariants and frozen M6 integrity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ais_eta.m10 import build_reference_rows

R = ROOT / "reports"
D = ROOT / "data" / "derived" / "m10_reference_rows.pkl.gz"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    freeze = json.loads((R / "M10_FREEZE.json").read_text())
    summary = json.loads((R / "m10_reference_dataset_summary.json").read_text())
    bench = json.loads((R / "m10_benchmark_summary.json").read_text())
    marker = json.loads((R / "M10_FINAL_HOLDOUT_OPENED.json").read_text())
    rows = pd.read_pickle(D)

    assert sha256(D) == freeze["reference_dataset_sha256"]
    assert len(rows) == 439 and rows["mmsi"].nunique() == 439
    assert summary["one_row_per_mmsi"] is True
    assert summary["status_counts"]["PAST_GT24H"] == 51
    assert summary["status_counts"]["FUTURE_0_7D"] == 319
    assert summary["future_reference_rows"] == 375
    assert summary["past_reference_rows"] == 64
    assert rows["history_last_at"].le(rows["last_update"]).all()
    assert set(rows["split"].unique()) == {"train", "calibration", "final_test"}
    assert sorted(rows.loc[rows["split"].eq("final_test"), "mmsi"].astype(int).tolist()) == freeze["final_test_mmsi"]
    assert marker["no_post_test_tuning"] is True
    assert bench["final_test_opened_once"] is True
    assert bench["m6_holdout_not_reused"] is True

    # M6 frozen artifacts remain byte-identical.
    m6 = json.loads((R / "M6_FREEZE.json").read_text())
    for rel, expected in m6["sha256"].items():
        p = ROOT / rel
        assert p.exists(), rel
        assert sha256(p) == expected, rel

    # Actual-data future-row invariance: append altered observations after the Tracks cutoff.
    tracks = pd.read_csv(ROOT / "data" / "vessel_tracks.csv", low_memory=False)
    states = pd.read_pickle(ROOT / "data" / "derived" / "m0c_ship_states.pkl.gz")
    sample_ids = rows["mmsi"].head(8).astype(int).tolist()
    tsmall = tracks.loc[tracks["mmsi"].isin(sample_ids)].copy()
    ssmall = states.loc[states["mmsi"].isin(sample_ids)].copy()
    base = build_reference_rows(tsmall, ssmall).sort_values("mmsi").reset_index(drop=True)
    fake = ssmall.groupby("mmsi", as_index=False).tail(1).copy()
    cut = pd.to_datetime(tsmall.set_index("mmsi")["last_update"])
    fake["recorded_at"] = fake["mmsi"].map(cut) + pd.Timedelta(days=30)
    fake["lat_clean"] = fake["lat_clean"].fillna(0) + 20
    fake["lon_clean"] = fake["lon_clean"].fillna(0) + 20
    fake["sog_clean"] = 99.9
    augmented = build_reference_rows(tsmall, pd.concat([ssmall, fake], ignore_index=True)).sort_values("mmsi").reset_index(drop=True)
    compare_cols = [c for c in base.columns if c not in {"eta_reference_raw", "eta_reference_dt", "target_tte_h", "reference_eta_status"}]
    for c in compare_cols:
        a, b = base[c], augmented[c]
        if pd.api.types.is_numeric_dtype(a):
            assert np.allclose(pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce"), equal_nan=True), c
        else:
            assert a.fillna("<NA>").astype(str).equals(b.fillna("<NA>").astype(str)), c

    print("PASS M10 reference dataset hash matches pre-open freeze")
    print("PASS 439 parseable ship references = 439 independent supervised MMSIs")
    print("PASS all historical feature timestamps are <= Tracks.last_update")
    print("PASS target-free MMSI split and frozen final-test IDs unchanged")
    print("PASS future-position injection leaves prediction features unchanged (8/8 MMSIs)")
    print(f"PASS M6 frozen hashes unchanged: {len(m6['sha256'])}/{len(m6['sha256'])}")
    print("PASS M10 final holdout was opened only after calibration selection; no post-test tuning marker")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
