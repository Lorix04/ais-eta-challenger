#!/usr/bin/env python3
"""Verify M11 is diagnostic-only and preserves the M10 freeze."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    freeze = json.loads((R / "M10_FREEZE.json").read_text())
    marker = json.loads((R / "M10_FINAL_HOLDOUT_OPENED.json").read_text())
    summary = json.loads((R / "m11_summary.json").read_text())
    f = pd.read_csv(R / "m11_reference_eta_forensics.csv")
    strict = pd.read_csv(R / "m11_strict_final_error_forensics.csv")
    concentration = pd.read_csv(R / "m11_final_error_concentration.csv")

    d = ROOT / "data" / "derived" / "m10_reference_rows.pkl.gz"
    assert sha256(d) == freeze["reference_dataset_sha256"]
    assert marker["no_post_test_tuning"] is True
    assert len(f) == 439 and f["mmsi"].nunique() == 439
    assert len(strict) == 53 and strict["mmsi"].nunique() == 53
    assert summary["interpretation"]["rows_deleted_from_strict_benchmark"] == 0
    assert summary["interpretation"]["m10_model_changed"] is False
    assert summary["interpretation"]["m10_split_changed"] is False
    assert summary["interpretation"]["final_test_reselected"] is False
    assert summary["past_reference_rows"] == 64
    assert summary["future_0_7d_rows"] == 319
    assert summary["future_gt30d_rows"] == 8
    assert summary["placeholder_like_rows"] == 7
    assert summary["year_assignment_margin_lt90d_rows"] == 5
    assert summary["year_assignment_margin_lt30d_rows"] == 2
    assert abs(summary["eta_exact_hour_fraction"] - (378 / 439)) < 1e-12
    assert abs(summary["eta_00_or_30_fraction"] - (412 / 439)) < 1e-12
    assert summary["unique_reference_timestamps"] == 300
    assert summary["max_reference_timestamp_multiplicity"] == 7
    top5 = concentration.loc[concentration["top_k"].eq(5), "share_of_total_absolute_error"].iloc[0]
    assert 0.78 < top5 < 0.80
    assert abs(summary["strict_final_mae_h"] - 312.01882325980154) < 1e-9
    assert 28.5 < summary["strict_final_future_0_7d_mae_h"] < 28.7

    # M6 frozen artifacts still byte-identical as inherited through M10.
    m6 = json.loads((R / "M6_FREEZE.json").read_text())
    for rel, expected in m6["sha256"].items():
        p = ROOT / rel
        assert p.exists(), rel
        assert sha256(p) == expected, rel

    print("PASS M10 reference dataset still matches pre-open freeze")
    print("PASS M11 is diagnostic-only: 0 strict rows removed; no M10 model/split reselection")
    print("PASS 439-reference and 53-final forensic ledgers are one-row-per-MMSI")
    print("PASS reference-quality counts and ETA granularity invariants")
    print("PASS top-error concentration and post-hoc sensitivity invariants")
    print(f"PASS M6 frozen hashes unchanged: {len(m6['sha256'])}/{len(m6['sha256'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
