#!/usr/bin/env python3
"""Verify the deterministic M10 company-facing archive."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
ZIP = DIST / "ais_eta_takehome_submission_m10.zip"
MANIFEST = DIST / "ais_eta_takehome_submission_m10_manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    m = json.loads(MANIFEST.read_text())
    assert sha256(ZIP) == m["sha256"]
    assert m["status"] == "M10_COMPANY_REFERENCE_ETA_BENCHMARK_READY"
    assert m["contains_company_raw_ais"] is False
    assert m["contains_row_level_derived_company_data"] is False
    assert m["post_m10_final_test_tuning"] is False

    with zipfile.ZipFile(ZIP) as zf:
        names = zf.namelist()
        assert any(x.endswith("reports/M10_REPORT.md") for x in names)
        assert any(x.endswith("reports/M10_FREEZE.json") for x in names)
        assert any(x.endswith("models/m10_strict_reference_catboost.cbm") for x in names)
        banned = [
            "vessel_positions_part1.csv", "vessel_positions_part2.csv", "vessel_tracks.csv",
            "m10_reference_rows.pkl.gz", "CALL_PREP_IT.md", "EMAIL_DRAFT_IT.md",
        ]
        assert not any(any(b in n for b in banned) for n in names)
        assert not any("/evidence/" in n for n in names)

        with tempfile.TemporaryDirectory() as td:
            zf.extractall(td)
            root = Path(td) / "ais_eta_challenger"
            report = (root / "reports" / "M10_REPORT.md").read_text().lower()
            card = (root / "docs" / "MODEL_CARD.md").read_text().lower()
            assert "company reference eta" in report
            assert ("not an observed actual time of arrival" in card or "this is **not** an observed actual time of arrival" in card or "not observed ata" in card)

    print(f"PASS M10 submission hash {m['sha256']}")
    print("PASS M10 report/model artifact present")
    print("PASS raw AIS, row-level derived data, internal evidence/correspondence excluded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
