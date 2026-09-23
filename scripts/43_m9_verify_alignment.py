#!/usr/bin/env python3
"""Verify M9 company-alignment wording against frozen M6 artifacts.

M9 is documentation-only. This script must not refit, rescore, or reopen the M6 holdout.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    answers = json.loads((R / "m9_company_answers.json").read_text())
    state = json.loads((ROOT / "PROJECT_STATE.json").read_text())
    freeze = json.loads((R / "M6_FREEZE.json").read_text())
    final = json.loads((R / "m6_final_summary.json").read_text())

    assert answers["target"]["company_definition"] == "vessel reaches/enters the area of the destination port"
    assert answers["target"]["alignment_with_current_research_gate"] == "conceptually_aligned_proxy"
    assert answers["target"]["exact_equivalence_confirmed"] is False
    assert answers["voyage_fields"]["historical_eta_in_positions"] is False
    assert answers["voyage_fields"]["historical_destination_in_positions"] is True
    assert answers["additional_history_available_for_exercise"] is False
    assert answers["existing_model"]["input_information_set_known"] is False
    assert answers["m9_policy"]["predictor_changed"] is False
    assert answers["m9_policy"]["final_holdout_reopened"] is False

    # Frozen M6 integrity must remain exact.
    for rel, expected in freeze["sha256"].items():
        path = ROOT / rel
        assert path.exists(), rel
        assert sha256(path) == expected, rel

    # Final metrics must remain untouched.
    assert int(final["locked_final_calls_scored"]) == 13
    route_min = float(final["primary_under24"]["route_mae_h"]) * 60
    geo_min = float(final["primary_under24"]["geodesic_mae_h"]) * 60
    assert round(route_min, 1) == 36.5
    assert round(geo_min, 1) == 37.7

    public_paths = [
        ROOT / "README.md",
        ROOT / "EXECUTIVE_SUMMARY.md",
        ROOT / "TAKE_HOME_REPORT.md",
        ROOT / "docs" / "MODEL_CARD.md",
        ROOT / "docs" / "COMPANY_ALIGNMENT.md",
    ]
    corpus = "\n".join(p.read_text(errors="replace") for p in public_paths).lower()

    required = [
        "destination-port area",
        "conceptually aligned proxy",
        "ground-truth source",
        "perimeter/polygon",
        "historical eta",
    ]
    for needle in required:
        assert needle in corpus, f"missing M9 alignment wording: {needle}"

    banned = [
        "exactly matches the company's target",
        "official company ground truth",
        "company confirmed polling architecture",
        "we beat the company's model",
    ]
    assert not any(x in corpus for x in banned)

    assert state["current_phase"] == "M9_DONE_WAITING_FINAL_GT_GEOMETRY_CLARIFICATION"
    assert state["m9_company_alignment"]["predictor_changed"] is False
    assert state["m9_company_alignment"]["final_holdout_reopened"] is False
    assert len(state["blocking_questions_for_company"]) == 2

    out = {
        "status": "PASS",
        "frozen_hashes_checked": len(freeze["sha256"]),
        "predictor_changed": False,
        "final_holdout_reopened": False,
        "target_alignment": "conceptually_aligned_proxy_pending_ground_truth_and_boundary",
        "remaining_target_questions": 2,
        "final_route_mae_min_preserved": route_min,
        "final_geodesic_mae_min_preserved": geo_min,
    }
    (R / "m9_alignment_verification.json").write_text(json.dumps(out, indent=2) + "\n")

    print(f"PASS M6 frozen hashes unchanged: {len(freeze['sha256'])}/{len(freeze['sha256'])}")
    print("PASS M9 company answers encoded without inventing missing semantics")
    print("PASS target wording = conceptually aligned proxy, not exact equivalence")
    print("PASS historical ETA comparison remains blocked")
    print("PASS final holdout not reopened and M6 metrics preserved")
    print("PASS exactly 2 target-label questions remain")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
