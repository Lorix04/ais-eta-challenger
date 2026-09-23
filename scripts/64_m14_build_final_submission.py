#!/usr/bin/env python3
"""Build deterministic M14 final company-facing submission with content manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
OUT = DIST / "ais_eta_takehome_submission_final.zip"
MANIFEST = DIST / "ais_eta_takehome_submission_final_manifest.json"
CONTENT_MANIFEST_NAME = "SUBMISSION_CONTENT_MANIFEST.json"
VERIFY_NAME = "VERIFY_SUBMISSION.py"
FIXED_ZIP_TIME = (2026, 9, 21, 0, 0, 0)

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
    # Submission reconciliation/final freeze.
    "M13_RECONCILIATION_REPORT.md", "m13_claim_map.csv", "m13_result_panel.png",
    "M14_REPRODUCIBILITY_REPORT.md", "M14_FINAL_FREEZE.json", "m14_result_panel.png",
    # Secondary frozen branch — concise evidence only.
    "M6_FINAL_REPORT.md", "M6_FREEZE.json", "M6_FINAL_HOLDOUT_OPENED.json",
    "m6_final_summary.json", "m6_final_claim_scope.csv", "m6_result_panel.png",
    # Compact audit input required by the included M0E reproducibility tests.
    "catania_m0d_candidate_calls.csv",
]
MODELS = ["m10_strict_reference_catboost.cbm", "m10_strict_destination_medians.json"]

VERIFY_SCRIPT = '''#!/usr/bin/env python3
"""Stdlib-only integrity verification for the frozen submission payload."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "SUBMISSION_CONTENT_MANIFEST.json"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main() -> int:
    m = json.loads(MANIFEST.read_text())
    assert m["status"] == "M14_FINAL_SUBMISSION_FROZEN"
    expected = {row["path"]: row for row in m["files"]}
    actual = sorted(
        str(p.relative_to(ROOT)).replace("\\\\", "/")
        for p in ROOT.rglob("*")
        if p.is_file() and p.name != MANIFEST.name and "__pycache__" not in p.parts and not p.name.endswith((".pyc", ".pyo"))
    )
    assert actual == sorted(expected), "payload file list differs from frozen manifest"
    for rel, row in expected.items():
        p = ROOT / rel
        assert p.stat().st_size == row["size"], rel
        assert sha256(p) == row["sha256"], rel
    model = ROOT / "models" / "m10_strict_reference_catboost.cbm"
    assert sha256(model) == m["frozen_artifacts"]["m10_model_sha256"]
    print(f"PASS final submission content manifest: {len(expected)} files")
    print(f"PASS frozen M10 model SHA-256: {sha256(model)}")
    print("PASS no file is missing, added or modified")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
'''


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
        if p.exists():
            files.append(p)
    dr = ROOT / "data" / "README.md"
    if dr.exists():
        files.append(dr)
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
    # Analytical tests only; internal packaging/project-control tests stay outside.
    exclude_tests = {
        "test_m7.py", "test_m8.py", "test_m9.py", "test_project_contract.py",
        "test_m13.py", "test_m14.py",
    }
    for p in sorted((ROOT / "tests").glob("test_*.py")):
        if p.name not in exclude_tests:
            files.append(p)
    for rel in DOCS:
        p = ROOT / "docs" / rel
        if p.exists():
            files.append(p)
    for rel in REPORTS:
        p = ROOT / "reports" / rel
        if p.exists():
            files.append(p)
    for rel in MODELS:
        p = ROOT / "models" / rel
        if p.exists():
            files.append(p)
    return sorted(set(files), key=lambda p: str(p.relative_to(ROOT)))


def zip_write_bytes(zf: zipfile.ZipFile, arcname: str, data: bytes) -> None:
    info = zipfile.ZipInfo(arcname, date_time=FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (0o100644 & 0xFFFF) << 16
    zf.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def build() -> dict:
    DIST.mkdir(exist_ok=True)
    files = include_files()
    rels = [str(p.relative_to(ROOT)).replace("\\", "/") for p in files]
    required = {
        "reports/M10_REPORT.md", "reports/M11_REPORT.md", "reports/M12_REPORT.md",
        "reports/M13_RECONCILIATION_REPORT.md", "reports/M14_REPRODUCIBILITY_REPORT.md",
        "reports/M14_FINAL_FREEZE.json", "docs/COMPANY_SUBMISSION_GUIDE.md",
        "models/m10_strict_reference_catboost.cbm",
    }
    assert required.issubset(set(rels))
    banned = [
        "vessel_positions_part1.csv", "vessel_positions_part2.csv", "vessel_tracks.csv",
        "m10_reference_rows.pkl.gz", "m10_final_predictions.csv", "CALL_PREP_IT.md",
        "EMAIL_DRAFT_IT.md", "COMPANY_REPLY_MATRIX.md", "m12_shap_contributions.csv",
        "m11_final_top15_errors.csv",
    ]
    assert not any(any(b in rel for b in banned) for rel in rels)
    assert not any(rel.startswith("evidence/") for rel in rels)
    assert not any("/mnt/data" in p.read_text(errors="ignore") for p in files if p.suffix == ".py")

    verify_bytes = VERIFY_SCRIPT.encode("utf-8")
    entries = [
        {"path": rel, "size": p.stat().st_size, "sha256": sha256(p)}
        for rel, p in zip(rels, files)
    ]
    entries.append({"path": VERIFY_NAME, "size": len(verify_bytes), "sha256": sha256_bytes(verify_bytes)})
    entries.sort(key=lambda row: row["path"])

    freeze = json.loads((ROOT / "reports" / "M14_FINAL_FREEZE.json").read_text())
    content_manifest = {
        "schema_version": 1,
        "status": "M14_FINAL_SUBMISSION_FROZEN",
        "freeze_date": "2026-09-21",
        "primary_target": "company-provided Tracks.eta reference",
        "primary_milestones": ["M10", "M11", "M12"],
        "secondary_branch": "M0-M9 physical port-entry research",
        "files": entries,
        "frozen_artifacts": {
            "m10_reference_dataset_sha256": freeze["frozen_artifacts"]["m10_reference_dataset_sha256"],
            "m10_model_sha256": freeze["frozen_artifacts"]["m10_model_sha256"],
            "m10_final_predictions_sha256": freeze["frozen_artifacts"]["m10_final_predictions_sha256"],
        },
        "contains_company_raw_ais": False,
        "contains_high_volume_row_level_derived_data": False,
        "contains_internal_evidence": False,
        "contains_internal_correspondence": False,
        "post_m10_final_test_tuning": False,
        "public_safe": False,
    }
    content_bytes = (json.dumps(content_manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")

    archive_entries = {f"ais_eta_challenger/{rel}": p.read_bytes() for rel, p in zip(rels, files)}
    archive_entries[f"ais_eta_challenger/{VERIFY_NAME}"] = verify_bytes
    archive_entries[f"ais_eta_challenger/{CONTENT_MANIFEST_NAME}"] = content_bytes
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for arcname in sorted(archive_entries):
            zip_write_bytes(zf, arcname, archive_entries[arcname])

    manifest = {
        "archive": OUT.name,
        "status": "M14_FINAL_SUBMISSION_FROZEN",
        "sha256": sha256(OUT),
        "archive_size": OUT.stat().st_size,
        "payload_files_hashed": len(entries),
        "archive_members": len(entries) + 1,
        "content_manifest_sha256": sha256_bytes(content_bytes),
        "deterministic_zip": {
            "sorted_input_order": True,
            "fixed_member_timestamp": "2026-09-21T00:00:00",
            "fixed_unix_mode": "0644",
            "compression": "ZIP_DEFLATED level 9",
        },
        "frozen_artifacts": content_manifest["frozen_artifacts"],
        "contains_company_raw_ais": False,
        "contains_high_volume_row_level_derived_data": False,
        "contains_internal_evidence": False,
        "contains_internal_correspondence": False,
        "post_m10_final_test_tuning": False,
        "final_freeze": True,
        "public_safe": False,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (DIST / "ais_eta_takehome_submission_final.zip.sha256").write_text(f"{manifest['sha256']}  {OUT.name}\n")
    return manifest


def main() -> int:
    m = build()
    print(f"PASS built {OUT.relative_to(ROOT)}")
    print(f"PASS archive SHA-256 {m['sha256']}")
    print(f"PASS content manifest hashes {m['payload_files_hashed']} payload files")
    print("PASS deterministic order/timestamp/mode policy applied")
    print("PASS raw AIS, high-volume row-level data, internal evidence/correspondence excluded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
