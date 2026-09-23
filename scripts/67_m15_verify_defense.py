#!/usr/bin/env python3
"""Verify M15 defense material without touching the frozen M14 submission."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def main() -> int:
    manifest = json.loads((ROOT / "reports/M15_DEFENSE_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "M15_INTERVIEW_TECHNICAL_DEFENSE_READY"

    expected = manifest["immutable_hashes"]
    checks = {
        "m14_final_zip_sha256": "dist/ais_eta_takehome_submission_final.zip",
        "m14_final_manifest_sha256": "dist/ais_eta_takehome_submission_final_manifest.json",
        "m14_final_sha_file_sha256": "dist/ais_eta_takehome_submission_final.zip.sha256",
        "m10_model_sha256": "models/m10_strict_reference_catboost.cbm",
        "m10_final_predictions_sha256": "reports/m10_final_predictions.csv",
        "m6_freeze_sha256": "reports/M6_FREEZE.json",
    }
    for key, rel in checks.items():
        got = sha256(rel)
        assert got == expected[key], f"hash mismatch: {rel} {got} != {expected[key]}"
        print(f"PASS immutable {rel} {got}")

    docs = [
        "docs/M15_TECHNICAL_DEFENSE_IT.md",
        "docs/M15_QA_BANK_IT.md",
        "docs/M15_LIVE_DEMO_CHECKLIST_IT.md",
        "reports/M15_TECHNICAL_DEFENSE_REPORT.md",
    ]
    combined = "\n".join((ROOT / p).read_text(encoding="utf-8") for p in docs).lower()
    required = [
        "312,0 h",
        "25,3 h",
        "321 / 65 / 53",
        "79,1%",
        "diagnostic",
        "no model",
        "superior",
        "observed ata",
        "shap",
        "leakage",
    ]
    for token in required:
        assert token in combined, f"missing defense token: {token}"
    print(f"PASS defense docs {len(docs)} files and claim boundaries")

    state = json.loads((ROOT / "PROJECT_STATE.json").read_text(encoding="utf-8"))
    assert state["current_phase"] == "M15_DONE_INTERVIEW_TECHNICAL_DEFENSE_PREPARATION"
    assert state["m15_summary"]["model_changed"] is False
    assert state["m15_summary"]["m14_submission_changed"] is False
    print("PASS project state marks M15 as post-freeze non-modelling")

    print("PASS M15 interview/technical-defense verification")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
