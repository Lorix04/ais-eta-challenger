#!/usr/bin/env python3
"""Build deterministic M13 draft company-facing submission."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
OUT = DIST / "ais_eta_takehome_submission_m13_draft.zip"
MANIFEST = DIST / "ais_eta_takehome_submission_m13_draft_manifest.json"

TOP = [
    "README.md", "EXECUTIVE_SUMMARY.md", "TAKE_HOME_REPORT.md", "REFERENCES.md",
    "requirements.txt", "PYTHON_VERSION.txt",
]
DOCS = [
    "MODEL_CARD.md", "REPRODUCIBILITY.md", "COMPANY_ALIGNMENT.md",
    "COMPANY_SUBMISSION_GUIDE.md", "PACKAGE_GUIDE.md",
    "M0C_DATA_CONTRACT.md", "M0D_CATANIA_GEOMETRY.md", "M0E_ROLE_PROVENANCE.md", "M1X_AUGUSTA_GEOMETRY.md",
]
REPORTS = [
    # Primary company task.
    "M10_REPORT.md", "M10_FREEZE.json", "M10_FINAL_HOLDOUT_OPENED.json",
    "m10_reference_dataset_summary.json", "m10_reference_eta_status_counts.csv", "m10_split_counts.csv",
    "m10_calibration_model_comparison.csv", "m10_benchmark_summary.json",
    "m10_reference_eta_quality.png", "m10_calibration_ablation.png", "m10_result_panel.png",
    # M11 aggregate forensics only.
    "M11_REPORT.md", "m11_summary.json", "m11_eta_granularity.csv",
    "m11_frozen_model_posthoc_sensitivity.csv", "m11_final_error_concentration.csv",
    "m11_reference_horizon_distribution.png", "m11_final_error_concentration.png", "m11_eta_minute_granularity.png",
    # M12 aggregate/frozen explainability only.
    "M12_REPORT.md", "m12_summary.json", "m12_prediction_values_change.csv",
    "m12_mean_abs_shap.csv", "m12_group_shap_importance.csv", "m12_frozen_calibration_ablation.csv",
    "m12_feature_importance.png", "m12_group_shap_stability.png", "m12_group_masking_stress.png", "m12_result_panel.png",
    # M13 scope/claim reconciliation.
    "M13_RECONCILIATION_REPORT.md", "m13_claim_map.csv", "m13_result_panel.png",
    # Secondary frozen branch — concise evidence only.
    "M6_FINAL_REPORT.md", "M6_FREEZE.json", "M6_FINAL_HOLDOUT_OPENED.json",
    "m6_final_summary.json", "m6_final_claim_scope.csv", "m6_result_panel.png",
    # Compact audit input required by the included M0E reproducibility tests.
    "catania_m0d_candidate_calls.csv",
]
MODELS = ["m10_strict_reference_catboost.cbm", "m10_strict_destination_medians.json"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def include_files() -> list[Path]:
    files: list[Path] = []
    for rel in TOP:
        p = ROOT / rel
        if p.exists(): files.append(p)
    dr = ROOT / "data" / "README.md"
    if dr.exists(): files.append(dr)
    for folder in ["src", "config"]:
        for p in sorted((ROOT / folder).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                files.append(p)
    # Analytical pipelines only: physical branch 00-36 and reference branch 46-59.
    for p in sorted((ROOT / "scripts").glob("*.py")):
        try:
            n = int(p.name.split("_", 1)[0])
        except ValueError:
            continue
        if n <= 36 or 46 <= n <= 59:
            files.append(p)
    # Include analytical tests, not internal package-process tests.
    exclude_tests = {"test_m7.py", "test_m8.py", "test_m9.py", "test_project_contract.py", "test_m13.py"}
    for p in sorted((ROOT / "tests").glob("test_*.py")):
        if p.name not in exclude_tests:
            files.append(p)
    for rel in DOCS:
        p = ROOT / "docs" / rel
        if p.exists(): files.append(p)
    for rel in REPORTS:
        p = ROOT / "reports" / rel
        if p.exists(): files.append(p)
    for rel in MODELS:
        p = ROOT / "models" / rel
        if p.exists(): files.append(p)
    return sorted(set(files), key=lambda p: str(p.relative_to(ROOT)))


def main() -> int:
    DIST.mkdir(exist_ok=True)
    files = include_files()
    rels = [str(p.relative_to(ROOT)) for p in files]
    required = {
        "reports/M10_REPORT.md", "reports/M11_REPORT.md", "reports/M12_REPORT.md",
        "reports/M13_RECONCILIATION_REPORT.md", "docs/COMPANY_SUBMISSION_GUIDE.md",
        "models/m10_strict_reference_catboost.cbm",
    }
    assert required.issubset(set(rels))
    banned = ["vessel_positions_part1.csv", "vessel_positions_part2.csv", "vessel_tracks.csv",
              "m10_reference_rows.pkl.gz", "CALL_PREP_IT.md", "EMAIL_DRAFT_IT.md", "COMPANY_REPLY_MATRIX.md"]
    assert not any(any(b in rel for b in banned) for rel in rels)
    assert not any(rel.startswith("evidence/") for rel in rels)
    assert not any("m12_shap_contributions.csv" in rel or "m11_final_top15_errors.csv" in rel for rel in rels)

    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in files:
            arc = str(Path("ais_eta_challenger") / p.relative_to(ROOT))
            info = zipfile.ZipInfo(arc, date_time=(2026, 9, 21, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100644 & 0xFFFF) << 16
            zf.writestr(info, p.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)

    manifest = {
        "archive": OUT.name,
        "sha256": sha256(OUT),
        "files": rels,
        "status": "M13_COMPANY_SUBMISSION_RECONCILED_DRAFT",
        "primary_target": "company-provided Tracks.eta reference",
        "primary_milestones": ["M10", "M11", "M12"],
        "secondary_branch": "M0-M9 physical port-entry research",
        "contains_company_raw_ais": False,
        "contains_row_level_derived_company_data": False,
        "contains_internal_correspondence_or_evidence": False,
        "post_m10_final_test_tuning": False,
        "m14_final_freeze_pending": True,
        "public_safe": False,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"PASS built {OUT.relative_to(ROOT)} with {len(files)} files")
    print(f"PASS sha256 {manifest['sha256']}")
    print("PASS primary story M10-M12; physical M0-M9 branch secondary")
    print("PASS raw AIS, row-level derived data, internal correspondence/evidence excluded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
