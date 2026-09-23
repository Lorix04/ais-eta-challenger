#!/usr/bin/env python3
"""Verify M13 company-facing reconciliation without changing frozen models/results."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
MODEL = ROOT / "models" / "m10_strict_reference_catboost.cbm"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    m10 = json.loads((R / "M10_FREEZE.json").read_text())
    m12 = json.loads((R / "m12_summary.json").read_text())
    m6 = json.loads((R / "M6_FREEZE.json").read_text())
    assert sha256(MODEL) == m12["frozen_model_sha256_before"] == m12["frozen_model_sha256_after"]
    d = ROOT / "data" / "derived" / "m10_reference_rows.pkl.gz"
    assert sha256(d) == m10["reference_dataset_sha256"]
    for rel, expected in m6["sha256"].items():
        assert sha256(ROOT / rel) == expected, rel

    readme = (ROOT / "README.md").read_text()
    executive = (ROOT / "EXECUTIVE_SUMMARY.md").read_text()
    report = (ROOT / "TAKE_HOME_REPORT.md").read_text()
    card = (ROOT / "docs" / "MODEL_CARD.md").read_text()
    guide = (ROOT / "docs" / "COMPANY_SUBMISSION_GUIDE.md").read_text()
    reconciliation = (R / "M13_RECONCILIATION_REPORT.md").read_text()

    docs = "\n".join([readme, executive, report, card, guide, reconciliation]).lower()
    assert "tracks.eta" in docs
    assert "primary" in docs
    assert "secondary" in docs
    assert "312.0 h" in docs
    assert "25.3 h" in docs
    assert "79.1%" in docs
    assert "destination_norm" in docs

    # The physical branch may appear, but it must be explicitly secondary.
    assert re.search(r"secondary[^\n]{0,120}(domain|physical|research)", docs)

    # Prevent accidental unsupported claims in recruiter-facing docs.
    banned_patterns = [
        r"we (have )?proven .*beat.*external",
        r"we beat the .*model",
        r"tracks\.eta is .*actual time of arrival",
        r"unseen-port generalization .*established",
    ]
    for pat in banned_patterns:
        assert not re.search(pat, docs), pat

    claims = pd.read_csv(R / "m13_claim_map.csv")
    assert len(claims) >= 12
    assert {"PRIMARY", "DIAGNOSTIC_ONLY", "SECONDARY", "BLOCKED"}.issubset(set(claims["delivery_role"]))
    blocked = claims.loc[claims["delivery_role"].eq("BLOCKED")]
    assert len(blocked) >= 4
    assert (blocked["status"] == "NOT_ESTABLISHED").all()

    final = pd.read_csv(R / "m10_final_predictions.csv")
    strict = final.loc[final["task"].eq("strict_all_parseable")]
    assert len(strict) == 53

    print("PASS M10 reference dataset/model and M6 freeze hashes unchanged")
    print("PASS company-facing docs lead with Tracks.eta primary exercise and label physical branch secondary")
    print("PASS frozen headline metrics and M11/M12 diagnostics are present")
    print("PASS unsupported superiority/ATA/unseen-port claims blocked")
    print(f"PASS claim map rows: {len(claims)}; blocked claims: {len(blocked)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
