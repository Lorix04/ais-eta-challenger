#!/usr/bin/env python3
"""Verify deterministic M13 draft submission archive."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "dist" / "ais_eta_takehome_submission_m13_draft.zip"
MANIFEST = ROOT / "dist" / "ais_eta_takehome_submission_m13_draft_manifest.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    m = json.loads(MANIFEST.read_text())
    assert m["status"] == "M13_COMPANY_SUBMISSION_RECONCILED_DRAFT"
    assert sha256(ZIP) == m["sha256"]
    assert m["post_m10_final_test_tuning"] is False
    assert m["m14_final_freeze_pending"] is True

    with zipfile.ZipFile(ZIP) as zf:
        names = zf.namelist()
        for suffix in [
            "README.md", "EXECUTIVE_SUMMARY.md", "TAKE_HOME_REPORT.md",
            "reports/M10_REPORT.md", "reports/M11_REPORT.md", "reports/M12_REPORT.md",
            "reports/M13_RECONCILIATION_REPORT.md", "docs/MODEL_CARD.md",
            "docs/COMPANY_SUBMISSION_GUIDE.md", "models/m10_strict_reference_catboost.cbm",
        ]:
            assert any(n.endswith(suffix) for n in names), suffix
        banned = ["vessel_positions_part1.csv", "vessel_positions_part2.csv", "vessel_tracks.csv",
                  "m10_reference_rows.pkl.gz", "CALL_PREP_IT.md", "EMAIL_DRAFT_IT.md",
                  "m12_shap_contributions.csv", "m11_final_top15_errors.csv"]
        assert not any(any(b in n for b in banned) for n in names)
        assert not any("/evidence/" in n for n in names)

        with tempfile.TemporaryDirectory() as td:
            zf.extractall(td)
            root = Path(td) / "ais_eta_challenger"
            readme = (root / "README.md").read_text().lower()
            report = (root / "TAKE_HOME_REPORT.md").read_text().lower()
            card = (root / "docs" / "MODEL_CARD.md").read_text().lower()
            assert "tracks.eta" in readme and "primary" in readme
            assert "312.0 h" in report and "25.3 h" in report
            assert "secondary" in report
            assert "not" in card and "actual time of arrival" in card

    print(f"PASS M13 draft package hash {m['sha256']}")
    print("PASS M10-M12 primary evidence and secondary M6 branch present")
    print("PASS no raw AIS, row-level diagnostics, evidence bundles or internal correspondence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
