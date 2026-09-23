#!/usr/bin/env python3
"""Verify the recruiter-facing M7 delivery against frozen M6 artifacts."""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

REQUIRED = [
    ROOT / "README.md",
    ROOT / "TAKE_HOME_REPORT.md",
    ROOT / "requirements.txt",
    ROOT / "REFERENCES.md",
    ROOT / "docs" / "MODEL_CARD.md",
    ROOT / "docs" / "REPRODUCIBILITY.md",
    ROOT / "docs" / "DATA_REQUESTS.md",
    ROOT / "docs" / "CALL_PREP_IT.md",
    ROOT / "docs" / "EMAIL_DRAFT_IT.md",
    REPORTS / "M6_FINAL_REPORT.md",
    REPORTS / "m6_final_summary.json",
    REPORTS / "m6_final_claim_scope.csv",
    REPORTS / "m7_submission_summary.json",
]


def main() -> int:
    for path in REQUIRED:
        if not path.exists():
            raise SystemExit(f"FAIL missing delivery file: {path.relative_to(ROOT)}")

    summary = json.loads((REPORTS / "m6_final_summary.json").read_text())
    m7 = json.loads((REPORTS / "m7_submission_summary.json").read_text())
    report = (ROOT / "TAKE_HOME_REPORT.md").read_text()
    readme = (ROOT / "README.md").read_text()
    model_card = (ROOT / "docs" / "MODEL_CARD.md").read_text()
    claims = pd.read_csv(REPORTS / "m6_final_claim_scope.csv")

    route_min = float(summary["primary_under24"]["route_mae_h"]) * 60.0
    geo_min = float(summary["primary_under24"]["geodesic_mae_h"]) * 60.0
    gain = float(summary["primary_under24"]["route_improvement_fraction"])
    final_calls = int(summary["locked_final_calls_scored"])
    high_calls = int(summary["interval_high_reliability"]["calls"])
    coverage = float(summary["interval_high_reliability"]["trajectory_coverage"])

    assert abs(m7["final_route_mae_min"] - route_min) < 1e-9
    assert abs(m7["final_geodesic_mae_min"] - geo_min) < 1e-9
    assert abs(m7["final_route_gain_fraction"] - gain) < 1e-12
    assert m7["final_scorable_calls"] == final_calls == 13
    assert m7["high_reliability_final_calls"] == high_calls == 12
    assert abs(m7["high_whole_call_coverage"] - coverage) < 1e-12
    assert m7["predictor_changed_after_m6"] is False

    for text in [report, readme]:
        assert "36.5 min" in text
        assert "37.7 min" in text
        assert "3.4%" in text
        assert "91.7%" in text

    assert "not claimed to be official PBP" in model_card
    assert "superiority to reported AIS ETA" in model_card

    reported = claims.loc[claims["claim"].eq("Model beats reported AIS ETA"), "status"]
    assert len(reported) == 1 and reported.iloc[0] == "NOT_ESTABLISHED"
    unseen = claims.loc[claims["claim"].str.contains("unseen ports", case=False, na=False), "status"]
    assert len(unseen) == 1 and unseen.iloc[0] == "NOT_ESTABLISHED"

    banned_overclaims = [
        "we beat the company's model",
        "beats the company's model",
        "proven superior to reported ais eta",
        "generalizes to unseen ports",
    ]
    corpus = (report + "\n" + readme).lower()
    for phrase in banned_overclaims:
        if phrase in corpus:
            raise AssertionError(f"Unsupported overclaim found: {phrase}")

    print("PASS recruiter-facing delivery files present")
    print(f"PASS final metrics match frozen M6: route={route_min:.1f} min geo={geo_min:.1f} min gain={gain:.1%}")
    print(f"PASS final scope preserved: {final_calls} scorable calls; HIGH interval calls={high_calls}; whole-call coverage={coverage:.1%}")
    print("PASS unsupported reported-ETA and unseen-port claims remain blocked")
    print("PASS no M7 predictor modification declared")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
