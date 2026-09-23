#!/usr/bin/env python3
"""Verify M14 final archive integrity and run data-free clean-room checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
ZIP = DIST / "ais_eta_takehome_submission_final.zip"
MANIFEST = DIST / "ais_eta_takehome_submission_final_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> str:
    p = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)
    out = (p.stdout or "") + (p.stderr or "")
    if p.returncode != 0:
        raise AssertionError(f"command failed ({p.returncode}): {' '.join(cmd)}\n{out}")
    return out.strip()


def main() -> int:
    m = json.loads(MANIFEST.read_text())
    assert m["status"] == "M14_FINAL_SUBMISSION_FROZEN"
    assert m["final_freeze"] is True
    assert sha256(ZIP) == m["sha256"]
    assert m["post_m10_final_test_tuning"] is False

    with zipfile.ZipFile(ZIP) as zf:
        names = zf.namelist()
        assert len(names) == len(set(names))
        assert names == sorted(names)
        required = [
            "README.md", "TAKE_HOME_REPORT.md", "VERIFY_SUBMISSION.py",
            "SUBMISSION_CONTENT_MANIFEST.json", "reports/M10_REPORT.md",
            "reports/M11_REPORT.md", "reports/M12_REPORT.md",
            "reports/M13_RECONCILIATION_REPORT.md", "reports/M14_REPRODUCIBILITY_REPORT.md",
            "reports/M14_FINAL_FREEZE.json", "models/m10_strict_reference_catboost.cbm",
        ]
        for suffix in required:
            assert any(n.endswith(suffix) for n in names), suffix
        banned = [
            "vessel_positions_part1.csv", "vessel_positions_part2.csv", "vessel_tracks.csv",
            "m10_reference_rows.pkl.gz", "m10_final_predictions.csv", "CALL_PREP_IT.md",
            "EMAIL_DRAFT_IT.md", "m12_shap_contributions.csv", "m11_final_top15_errors.csv",
        ]
        assert not any(any(b in n for b in banned) for n in names)
        assert not any("/evidence/" in n for n in names)
        for info in zf.infolist():
            assert info.date_time == (2026, 9, 21, 0, 0, 0), info.filename

        with tempfile.TemporaryDirectory() as td:
            zf.extractall(td)
            root = Path(td) / "ais_eta_challenger"
            integrity = run([sys.executable, "VERIFY_SUBMISSION.py"], root)
            env = dict(__import__("os").environ)
            env["PYTHONPATH"] = str(root / "src")
            tests = run([sys.executable, "-m", "pytest", "-q"], root, env)
            compile_out = run([sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests"], root)
            assert "passed" in tests.lower()
            assert "failed" not in tests.lower()

    print(f"PASS final archive SHA-256 {m['sha256']}")
    print("PASS ZIP member order and fixed timestamps are deterministic")
    print("PASS package content/confidentiality boundary")
    print(integrity)
    print(f"PASS extracted-package pytest: {tests.splitlines()[-1]}")
    print("PASS extracted-package compileall")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
