#!/usr/bin/env python3
"""Build deterministic company-facing M10 submission package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
OUT = DIST / "ais_eta_takehome_submission_m10.zip"
MANIFEST = DIST / "ais_eta_takehome_submission_m10_manifest.json"

TOP_FILES = [
    "README.md", "EXECUTIVE_SUMMARY.md", "TAKE_HOME_REPORT.md", "REFERENCES.md",
    "requirements.txt", "PYTHON_VERSION.txt", "WORKFLOW.md",
]
DOCS = [
    "MODEL_CARD.md", "REPRODUCIBILITY.md", "DATA_REQUESTS.md", "COMPANY_ALIGNMENT.md",
    "M0C_DATA_CONTRACT.md", "M0D_CATANIA_GEOMETRY.md", "M0E_ROLE_PROVENANCE.md",
    "M1X_AUGUSTA_GEOMETRY.md", "PACKAGE_GUIDE.md",
]
REPORT_FILES = [
    # Primary M10 benchmark.
    "M10_REPORT.md", "M10_FREEZE.json", "M10_FINAL_HOLDOUT_OPENED.json",
    "m10_reference_dataset_summary.json", "m10_reference_eta_status_counts.csv",
    "m10_split_counts.csv", "m10_calibration_model_comparison.csv",
    "m10_future_test_by_reference_status.csv", "m10_benchmark_summary.json",
    "m10_reference_eta_quality.png", "m10_calibration_ablation.png", "m10_result_panel.png",
    # Secondary frozen physical-arrival evidence.
    "M6_FINAL_REPORT.md", "M6_FREEZE.json", "M6_FINAL_HOLDOUT_OPENED.json",
    "m6_final_summary.json", "m6_final_claim_scope.csv", "m6_result_panel.png",
    # Compact M0E audit input required by included reproducibility tests.
    "catania_m0d_candidate_calls.csv",
]
MODEL_FILES = ["m10_strict_reference_catboost.cbm", "m10_strict_destination_medians.json"]
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
    dr = ROOT / "data" / "README.md"
    if dr.exists(): files.append(dr)
    for d in ["src", "config"]:
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                files.append(p)
    # Analytical scripts through M10 finalize; packaging/evidence scripts remain internal.
    for p in sorted((ROOT / "scripts").glob("*.py")):
        try:
            num = int(p.name.split("_", 1)[0])
        except ValueError:
            continue
        if num <= 51:
            files.append(p)
    for p in sorted((ROOT / "tests").glob("test_*.py")):
        if p.name not in {"test_m7.py", "test_m8.py", "test_m9.py", "test_project_contract.py"}:
            files.append(p)
    for rel in DOCS:
        p = ROOT / "docs" / rel
        if p.exists(): files.append(p)
    for rel in REPORT_FILES:
        p = ROOT / "reports" / rel
        if p.exists(): files.append(p)
    for rel in MODEL_FILES:
        p = ROOT / "models" / rel
        if p.exists(): files.append(p)
    return sorted(set(files), key=lambda p: str(p.relative_to(ROOT)))


def main() -> int:
    DIST.mkdir(exist_ok=True)
    files = include_files()
    rels = [str(p.relative_to(ROOT)) for p in files]
    if any("evidence" in p.parts for p in files):
        raise AssertionError("submission must not contain internal evidence bundles")
    if any(p.name in INTERNAL_DOC_BASENAMES for p in files):
        raise AssertionError("submission contains internal correspondence/process docs")
    if any("data/derived" in rel for rel in rels):
        raise AssertionError("submission contains derived row-level trajectory/reference data")
    if any(rel.endswith(("vessel_positions_part1.csv", "vessel_positions_part2.csv", "vessel_tracks.csv")) for rel in rels):
        raise AssertionError("submission contains raw AIS files")
    if "reports/M10_REPORT.md" not in rels or "reports/M10_FREEZE.json" not in rels:
        raise AssertionError("M10 primary deliverables missing")

    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in files:
            arcname = str(Path("ais_eta_challenger") / p.relative_to(ROOT))
            info = zipfile.ZipInfo(arcname, date_time=(2026, 9, 21, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100644 & 0xFFFF) << 16
            zf.writestr(info, p.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)

    manifest = {
        "archive": OUT.name,
        "sha256": sha256(OUT),
        "files": rels,
        "status": "M10_COMPANY_REFERENCE_ETA_BENCHMARK_READY",
        "primary_target": "company-provided Tracks.eta reference",
        "contains_company_raw_ais": False,
        "contains_row_level_derived_company_data": False,
        "contains_internal_correspondence_or_evidence": False,
        "m10_final_test_opened_once": True,
        "post_m10_final_test_tuning": False,
        "m6_frozen_branch_preserved": True,
        "public_safe": False,
        "public_safe_reason": "Company-facing analytical package; publication still requires a separate confidentiality/privacy review.",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"PASS built {OUT.relative_to(ROOT)} with {len(files)} files")
    print(f"PASS sha256 {manifest['sha256']}")
    print("PASS excludes raw AIS, row-level derived data, evidence bundles and internal correspondence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
