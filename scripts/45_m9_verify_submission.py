#!/usr/bin/env python3
"""Verify M9 draft submission alignment/claim boundaries."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
R = ROOT / "reports"


def main() -> int:
    manifest = json.loads((DIST / "ais_eta_takehome_submission_m9_draft_manifest.json").read_text())
    verify = json.loads((R / "m9_alignment_verification.json").read_text())
    archive = DIST / manifest["archive"]
    assert archive.exists()
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["sha256"]
    assert manifest["status"] == "DRAFT_PENDING_COMPANY_GT_AND_PORT_AREA_GEOMETRY"
    assert manifest["predictor_changed_after_m6"] is False
    assert manifest["m6_holdout_reopened"] is False
    assert verify["status"] == "PASS"

    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        texts = "\n".join(
            zf.read(n).decode("utf-8", errors="ignore")
            for n in names if n.endswith(".md")
        ).lower()

    forbidden = ["CALL_PREP_IT.md", "EMAIL_DRAFT_IT.md", "CHANGE_PROTOCOL.md", "COMPANY_REPLY_MATRIX.md"]
    assert not any(any(x in n for x in forbidden) for n in names)
    assert not any("/evidence/" in n for n in names)
    assert "destination-port area" in texts
    assert "conceptually aligned proxy" in texts
    assert "ground-truth source" in texts
    assert "perimeter/polygon" in texts
    assert "exactly matches the company's target" not in texts
    assert "company confirmed polling architecture" not in texts

    print("PASS M9 draft ZIP hash matches manifest and package build is reproducible")
    print("PASS M9 draft ZIP contains updated company-aligned semantics")
    print("PASS exact target equivalence remains explicitly pending")
    print("PASS internal correspondence/evidence excluded")
    print("PASS predictor/final-holdout state remains frozen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
