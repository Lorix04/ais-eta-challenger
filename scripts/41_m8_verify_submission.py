#!/usr/bin/env python3
"""Verify M8 executive/package claims against frozen M6 artifacts and package contents."""
from __future__ import annotations

import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
DIST = ROOT / "dist"


def main() -> int:
    summary = json.loads((R / "m6_final_summary.json").read_text())
    m8 = json.loads((R / "m8_submission_summary.json").read_text())
    red = json.loads((R / "m8_red_team_summary.json").read_text())
    texts = [
        (ROOT / "README.md").read_text().lower(),
        (ROOT / "EXECUTIVE_SUMMARY.md").read_text().lower(),
        (ROOT / "TAKE_HOME_REPORT.md").read_text().lower(),
        (ROOT / "docs" / "MODEL_CARD.md").read_text().lower(),
    ]
    corpus = "\n".join(texts)

    route_min = float(summary["primary_under24"]["route_mae_h"]) * 60
    geo_min = float(summary["primary_under24"]["geodesic_mae_h"]) * 60
    gain = float(summary["primary_under24"]["route_improvement_fraction"])
    assert abs(m8["final_route_mae_min"] - route_min) < 1e-9
    assert abs(m8["final_geodesic_mae_min"] - geo_min) < 1e-9
    assert abs(m8["final_route_gain_fraction"] - gain) < 1e-12
    assert m8["final_scorable_calls"] == int(summary["locked_final_calls_scored"]) == 13
    assert m8["predictor_changed_after_m6"] is False
    assert red["status"] == "PASS"

    for needle in ["36.5", "37.7", "3.4%", "91.7%", "not established"]:
        assert needle in corpus, f"Missing claim-boundary token: {needle}"

    banned = [
        "we beat the company's model", "beats the company's model",
        "proven superior to reported ais eta", "generalizes to unseen ports",
    ]
    assert not any(p in corpus for p in banned)

    archive = DIST / "ais_eta_takehome_submission_m8.zip"
    manifest = json.loads((DIST / "ais_eta_takehome_submission_m8_manifest.json").read_text())
    assert archive.exists()
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        py_text = "\n".join(
            zf.read(n).decode("utf-8", errors="ignore")
            for n in names if n.endswith(".py")
        )
    forbidden = ["CALL_PREP_IT.md", "EMAIL_DRAFT_IT.md", "CHANGE_PROTOCOL.md", "COMPANY_REPLY_MATRIX.md"]
    assert not any(any(x in n for x in forbidden) for n in names)
    assert not any("/evidence/" in n for n in names)
    assert not any("data/derived" in n for n in names)
    assert not any(n.endswith(("vessel_positions_part1.csv", "vessel_positions_part2.csv", "vessel_tracks.csv")) for n in names)
    assert "/mnt/data" not in py_text
    assert manifest["contains_company_raw_ais"] is False
    assert manifest["contains_compact_company_data_derived_annotations"] is True
    assert manifest["contains_internal_personalized_docs"] is False

    print("PASS M8 executive metrics match frozen M6")
    print("PASS M8 red-team status PASS and predictor unchanged")
    print("PASS company ZIP excludes internal/personalized docs, raw/high-volume data and evidence bundles")
    print("PASS company ZIP Python code contains no /mnt/data absolute path")
    print("PASS compact derived annotations are explicitly disclosed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
