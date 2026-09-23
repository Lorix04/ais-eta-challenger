#!/usr/bin/env python3
"""M8 red-team audit: claim discipline, frozen-artifact integrity, portability, and packaging risks.

This script is reporting-only. It MUST NOT refit or rescore the frozen M6 predictor.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_integrity() -> list[dict[str, str]]:
    freeze = json.loads((REPORTS / "M6_FREEZE.json").read_text())
    out = []
    for rel, expected in freeze["sha256"].items():
        path = ROOT / rel
        actual = sha256(path) if path.exists() else "MISSING"
        out.append({
            "artifact": rel,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "status": "PASS" if actual == expected else "FAIL",
        })
    return out


def user_facing_corpus() -> str:
    paths = [
        ROOT / "README.md",
        ROOT / "EXECUTIVE_SUMMARY.md",
        ROOT / "TAKE_HOME_REPORT.md",
        ROOT / "docs" / "MODEL_CARD.md",
        ROOT / "docs" / "REPRODUCIBILITY.md",
    ]
    return "\n".join(p.read_text(errors="replace") for p in paths if p.exists()).lower()


def main() -> int:
    REPORTS.mkdir(exist_ok=True)
    frozen = frozen_integrity()
    pd_frozen = [r for r in frozen if r["status"] != "PASS"]

    corpus = user_facing_corpus()
    banned = {
        "we beat the company's model": "unsupported superiority claim",
        "beats the company's model": "unsupported superiority claim",
        "proven superior to reported ais eta": "reported ETA is not temporally aligned",
        "generalizes to unseen ports": "unseen-port transfer is not established",
    }
    overclaims = [phrase for phrase in banned if phrase in corpus]

    absolute_hits: list[str] = []
    scan_exclude = {"scripts/39_m8_red_team.py", "scripts/41_m8_verify_submission.py"}
    for base in [ROOT / "scripts", ROOT / "src", ROOT / "tests"]:
        for p in base.rglob("*.py"):
            rel = str(p.relative_to(ROOT))
            if rel in scan_exclude:
                continue
            text = p.read_text(errors="replace")
            if "/mnt/data" in text:
                absolute_hits.append(rel)

    findings = [
        {
            "id": "M8-01",
            "severity": "HIGH",
            "status": "FIXED",
            "finding": "Personalized internal call/email preparation documents were included in the M7 company ZIP.",
            "action": "M8 company package explicitly excludes CALL_PREP_IT.md, EMAIL_DRAFT_IT.md, CHANGE_PROTOCOL.md and the internal company-response matrix.",
        },
        {
            "id": "M8-02",
            "severity": "MEDIUM",
            "status": "FIXED",
            "finding": "M7 manifest said no company-derived data although compact human audit annotations derived from company AIS were included for reproducibility.",
            "action": "M8 manifest states the exact boundary: raw AIS and high-volume derived trajectories are excluded; compact annotation/config files are included and should not be published externally without review.",
        },
        {
            "id": "M8-03",
            "severity": "MEDIUM",
            "status": "FIXED" if not absolute_hits else "OPEN",
            "finding": "Environment-specific /mnt/data paths reduce portability.",
            "action": "Optional data-ZIP alignment is now CLI/environment configured; company package is scanned for absolute container paths.",
        },
        {
            "id": "M8-04",
            "severity": "HIGH",
            "status": "PASS" if not pd_frozen else "FAIL",
            "finding": "Post-holdout changes must not alter frozen M2/M4/M6 model artifacts or final-evaluation inputs.",
            "action": f"Verified {len(frozen) - len(pd_frozen)}/{len(frozen)} SHA-256 hashes against M6_FREEZE.json.",
        },
        {
            "id": "M8-05",
            "severity": "HIGH",
            "status": "PASS" if not overclaims else "FAIL",
            "finding": "Recruiter-facing text must not overclaim superiority, official ATA semantics or unseen-port transfer.",
            "action": "Automated phrase/claim checks plus frozen claim-scope verification.",
        },
        {
            "id": "M8-06",
            "severity": "MEDIUM",
            "status": "DOCUMENTED",
            "finding": "Full end-to-end replay requires the company AIS files and the versioned manual-audit annotations.",
            "action": "Reproducibility guide now separates data-free verification from data-backed replay and pins the tested Python version.",
        },
        {
            "id": "M8-07",
            "severity": "MEDIUM",
            "status": "DOCUMENTED",
            "finding": "The final 3.4% route gain is not statistically resolved; Catania final result is underpowered and Augusta is neutral.",
            "action": "Executive summary foregrounds claim boundaries rather than the best-looking subgroup metric.",
        },
        {
            "id": "M8-08",
            "severity": "MEDIUM",
            "status": "DOCUMENTED",
            "finding": "The 90% call-level interval has 91.7% empirical whole-call coverage on only 12 HIGH-reliability final calls.",
            "action": "Documented as small-sample empirical evidence, not a universal conformal guarantee under temporal dependence.",
        },
        {
            "id": "M8-09",
            "severity": "MEDIUM",
            "status": "FIXED",
            "finding": "The first clean-room ZIP test exposed package/test coupling: omitted M0D candidate data and internal project-control tests caused 8 failures.",
            "action": "The final company ZIP includes the compact candidate table needed by model tests and excludes packaging/project-control tests; clean-room result is 56/56 PASS.",
        },
        {
            "id": "M8-10",
            "severity": "LOW",
            "status": "DOCUMENTED",
            "finding": "A complete 969k-row clean-room M0C rebuild exceeded the 180 s interactive execution budget in this environment.",
            "action": "Do not claim a full clean-room replay was completed in M8; data-free package tests and a raw-data semantic-state smoke test passed. The full replay runbook remains documented.",
        },
    ]

    with (REPORTS / "m8_red_team_findings.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "severity", "status", "finding", "action"])
        w.writeheader(); w.writerows(findings)

    (REPORTS / "m8_frozen_integrity.json").write_text(json.dumps(frozen, indent=2) + "\n")

    summary = {
        "status": "PASS" if not pd_frozen and not overclaims and not absolute_hits else "FAIL",
        "frozen_hashes_checked": len(frozen),
        "frozen_hash_mismatches": len(pd_frozen),
        "unsupported_overclaims_found": overclaims,
        "absolute_container_path_hits": absolute_hits,
        "red_team_findings": len(findings),
        "predictor_changed_after_m6": False,
    }
    (REPORTS / "m8_red_team_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    report = "# M8 — Red-team report\n\n"
    report += "M8 is a **non-modelling** hardening pass. It does not refit, rescore, retune or change the frozen M6 predictor.\n\n"
    report += "## Outcome\n\n"
    report += f"- frozen artifact hashes: **{len(frozen) - len(pd_frozen)}/{len(frozen)} PASS**;\n"
    report += f"- unsupported recruiter-facing overclaims: **{len(overclaims)}**;\n"
    report += f"- absolute `/mnt/data` code references after portability fixes: **{len(absolute_hits)}**;\n"
    report += "- M7 packaging issue found: personalized internal call/email documents were being shipped; M8 removes them;\n"
    report += "- confidentiality wording corrected: compact audit annotations are derived from company AIS and remain in the company package only because they are required for exact reproducibility.\n\n"
    report += "## Findings\n\n"
    report += "| ID | Severity | Status | Finding / disposition |\n|---|---|---|---|\n"
    for r in findings:
        report += f"| {r['id']} | {r['severity']} | {r['status']} | {r['finding']} {r['action']} |\n"
    report += "\n## Claim posture after red-team\n\n"
    report += "The strongest defensible statement remains: the project demonstrates a leakage-resistant AIS port-entry ETA pipeline with auditable ground truth, train-only route retrieval, explicit reliability/uncertainty and a one-time chronological holdout. It does **not** establish superiority to the company model, superiority to reported AIS ETA, unseen-port transfer, berth/all-fast ETA or final long-horizon performance.\n"
    (REPORTS / "M8_RED_TEAM_REPORT.md").write_text(report)

    if summary["status"] != "PASS":
        raise SystemExit(f"FAIL M8 red-team: {summary}")
    print(f"PASS frozen artifacts unchanged: {len(frozen)}/{len(frozen)}")
    print("PASS recruiter-facing overclaim scan")
    print("PASS no absolute /mnt/data references in Python code")
    print(f"PASS wrote {len(findings)} red-team findings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
