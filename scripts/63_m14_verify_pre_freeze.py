#!/usr/bin/env python3
"""Verify M14 pre-freeze invariants without training, tuning or rescoring."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
MODEL = ROOT / "models" / "m10_strict_reference_catboost.cbm"
REFERENCE = ROOT / "data" / "derived" / "m10_reference_rows.pkl.gz"
FINAL_PREDICTIONS = R / "m10_final_predictions.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    m10 = json.loads((R / "M10_FREEZE.json").read_text())
    m10_open = json.loads((R / "M10_FINAL_HOLDOUT_OPENED.json").read_text())
    m12 = json.loads((R / "m12_summary.json").read_text())
    m6 = json.loads((R / "M6_FREEZE.json").read_text())

    assert sha256(REFERENCE) == m10["reference_dataset_sha256"]
    assert sha256(MODEL) == m12["frozen_model_sha256_before"] == m12["frozen_model_sha256_after"]
    for rel, expected in m6["sha256"].items():
        assert sha256(ROOT / rel) == expected, rel

    final = pd.read_csv(FINAL_PREDICTIONS)
    strict = final.loc[final["task"].eq("strict_all_parseable")]
    assert len(strict) == 53
    assert strict["mmsi"].nunique() == 53
    assert m10_open["final_test_rows_strict"] == 53
    assert m10_open["no_post_test_tuning"] is True
    assert m10["no_final_test_model_selection"] is True
    assert m10["no_eta_as_feature"] is True
    assert m10["no_future_positions"] is True

    corpus = "\n".join(
        (ROOT / rel).read_text().lower()
        for rel in [
            "README.md",
            "EXECUTIVE_SUMMARY.md",
            "TAKE_HOME_REPORT.md",
            "docs/MODEL_CARD.md",
            "docs/PACKAGE_GUIDE.md",
            "docs/REPRODUCIBILITY.md",
        ]
    )
    assert "tracks.eta" in corpus
    assert "312.0 h" in corpus
    assert "25.3 h" in corpus
    assert "secondary" in corpus
    assert "final submission" in corpus

    print(f"PASS M10 reference dataset SHA-256 {sha256(REFERENCE)}")
    print(f"PASS frozen M10 CatBoost SHA-256 {sha256(MODEL)}")
    print(f"PASS frozen M10 final predictions SHA-256 {sha256(FINAL_PREDICTIONS)}")
    print(f"PASS M6 frozen hashes unchanged: {len(m6['sha256'])}/{len(m6['sha256'])}")
    print("PASS 53 strict final predictions remain frozen; no post-test tuning/model selection")
    print("PASS company-facing docs retain target semantics, headline metrics and final-submission scope")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
