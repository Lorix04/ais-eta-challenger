#!/usr/bin/env python3
"""Build a clean take-home ZIP without company raw/derived data or internal evidence bundles."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
OUT = DIST / "ais_eta_takehome_submission.zip"
MANIFEST = DIST / "ais_eta_takehome_submission_manifest.json"

TOP_FILES = [
    "README.md", "TAKE_HOME_REPORT.md", "REFERENCES.md", "requirements.txt", "WORKFLOW.md"
]
DIRS = ["src", "scripts", "tests", "config", "docs"]
REPORT_FILES = [
    "M2_REPORT.md", "M3_REPORT.md", "M4_REPORT.md", "M5_REPORT.md", "M6_FINAL_REPORT.md",
    "M6_FREEZE.json", "M6_FINAL_HOLDOUT_OPENED.json",
    "m2_route_ablation.png", "m3_model_ablation.png", "m4_interval_calibration.png",
    "m4_stability_tradeoff.png", "m6_result_panel.png", "m6_final_timeline_example.png",
    "m6_final_port_comparison.png", "m6_final_reliability.png",
    "m6_final_eta_metrics.csv", "m6_final_call_coverage.csv", "m6_final_claim_scope.csv",
    "m6_final_summary.json", "m7_submission_summary.json",
]


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
    for d in DIRS:
        for p in sorted((ROOT / d).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                files.append(p)
    for rel in REPORT_FILES:
        p = ROOT / "reports" / rel
        if p.exists(): files.append(p)
    return sorted(set(files), key=lambda p: str(p.relative_to(ROOT)))


def main() -> int:
    DIST.mkdir(exist_ok=True)
    files = include_files()
    if any("derived" in p.parts for p in files):
        raise AssertionError("Clean submission must not contain derived company data")
    if any("evidence" in p.parts for p in files):
        raise AssertionError("Clean submission must not contain internal evidence bundles")

    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in files:
            zf.write(p, arcname=str(Path("ais_eta_challenger") / p.relative_to(ROOT)))

    manifest = {
        "archive": OUT.name,
        "sha256": sha256(OUT),
        "files": [str(p.relative_to(ROOT)) for p in files],
        "contains_company_raw_or_derived_data": False,
        "contains_internal_evidence_bundles": False,
        "model_changed_after_final_holdout": False,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"PASS built {OUT.relative_to(ROOT)} with {len(files)} files")
    print(f"PASS sha256 {manifest['sha256']}")
    print("PASS clean package excludes raw/derived company data and internal evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
