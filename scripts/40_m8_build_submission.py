#!/usr/bin/env python3
"""Build the M8 company-facing submission package after red-team hardening."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
OUT = DIST / "ais_eta_takehome_submission_m8.zip"
MANIFEST = DIST / "ais_eta_takehome_submission_m8_manifest.json"

TOP_FILES = [
    "README.md", "EXECUTIVE_SUMMARY.md", "TAKE_HOME_REPORT.md", "REFERENCES.md",
    "requirements.txt", "PYTHON_VERSION.txt", "WORKFLOW.md",
]
DOCS = [
    "MODEL_CARD.md", "REPRODUCIBILITY.md", "DATA_REQUESTS.md",
    "M0C_DATA_CONTRACT.md", "M0D_CATANIA_GEOMETRY.md", "M0E_ROLE_PROVENANCE.md",
    "M1X_AUGUSTA_GEOMETRY.md", "PACKAGE_GUIDE.md",
]
REPORT_FILES = [
    "M2_REPORT.md", "M3_REPORT.md", "M4_REPORT.md", "M5_REPORT.md", "M6_FINAL_REPORT.md",
    "M6_FREEZE.json", "M6_FINAL_HOLDOUT_OPENED.json",
    "m2_route_ablation.png", "m3_model_ablation.png", "m4_interval_calibration.png",
    "m4_stability_tradeoff.png", "m6_result_panel.png", "m6_final_timeline_example.png",
    "m6_final_port_comparison.png", "m6_final_reliability.png",
    "m6_final_eta_metrics.csv", "m6_final_call_coverage.csv", "m6_final_claim_scope.csv",
    "m6_final_summary.json", "m8_submission_summary.json",
    "catania_m0d_candidate_calls.csv",
]
INTERNAL_DOC_BASENAMES = {
    "CALL_PREP_IT.md", "EMAIL_DRAFT_IT.md", "CHANGE_PROTOCOL.md", "COMPANY_REPLY_MATRIX.md"
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def include_files() -> list[Path]:
    files: list[Path] = []
    for rel in TOP_FILES:
        p = ROOT / rel
        if p.exists(): files.append(p)
    data_readme = ROOT / "data" / "README.md"
    if data_readme.exists(): files.append(data_readme)
    for d in ["src", "config"]:
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                files.append(p)
    # Ship the modelling/reproduction scripts only; packaging/red-team scripts are internal.
    for p in sorted((ROOT / "scripts").glob("*.py")):
        try:
            num = int(p.name.split("_", 1)[0])
        except ValueError:
            continue
        if num <= 36:
            files.append(p)
    # Model/contract tests only; M7/M8 packaging tests are internal.
    for p in sorted((ROOT / "tests").glob("test_*.py")):
        if p.name not in {"test_m7.py", "test_m8.py", "test_project_contract.py"}:
            files.append(p)
    for rel in DOCS:
        p = ROOT / "docs" / rel
        if p.exists(): files.append(p)
    for rel in REPORT_FILES:
        p = ROOT / "reports" / rel
        if p.exists(): files.append(p)
    return sorted(set(files), key=lambda p: str(p.relative_to(ROOT)))


def main() -> int:
    DIST.mkdir(exist_ok=True)
    files = include_files()
    rels = [str(p.relative_to(ROOT)) for p in files]
    if any("evidence" in p.parts for p in files):
        raise AssertionError("Company submission must not contain internal evidence bundles")
    if any(p.name in INTERNAL_DOC_BASENAMES for p in files):
        raise AssertionError("Company submission contains internal personalized/process docs")
    if any("data/derived" in rel for rel in rels):
        raise AssertionError("Company submission contains high-volume derived trajectory data")
    if any(rel.endswith(("vessel_positions_part1.csv", "vessel_positions_part2.csv", "vessel_tracks.csv")) for rel in rels):
        raise AssertionError("Company submission contains raw AIS files")

    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, arcname=str(Path("ais_eta_challenger") / p.relative_to(ROOT)))

    manifest = {
        "archive": OUT.name,
        "sha256": sha256(OUT),
        "files": rels,
        "contains_company_raw_ais": False,
        "contains_high_volume_derived_trajectories": False,
        "contains_compact_company_data_derived_annotations": True,
        "annotation_reason": "Versioned manual-audit/config annotations are included because they are required to reproduce event/cohort decisions; do not treat this ZIP as public-safe without a separate privacy review.",
        "contains_internal_evidence_bundles": False,
        "contains_internal_personalized_docs": False,
        "model_changed_after_final_holdout": False,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"PASS built {OUT.relative_to(ROOT)} with {len(files)} files")
    print(f"PASS sha256 {manifest['sha256']}")
    print("PASS excludes raw AIS, high-volume derived trajectories, evidence bundles and internal personalized docs")
    print("PASS compact derived annotations are explicitly disclosed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
