import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def test_m7_delivery_summary_matches_m6():
    m6 = json.loads((ROOT / "reports/m6_final_summary.json").read_text())
    m7 = json.loads((ROOT / "reports/m7_submission_summary.json").read_text())
    assert m7["final_scorable_calls"] == m6["locked_final_calls_scored"] == 13
    assert abs(m7["final_route_mae_min"] - 60 * m6["primary_under24"]["route_mae_h"]) < 1e-9
    assert m7["predictor_changed_after_m6"] is False


def test_recruiter_docs_preserve_scope():
    text = (ROOT / "TAKE_HOME_REPORT.md").read_text().lower()
    assert "not established" in text
    assert "reported ais eta" in text
    assert "unseen-port" in text
    assert "official pbp" in text


def test_clean_submission_has_no_company_data_or_evidence():
    archive = ROOT / "dist/ais_eta_takehome_submission.zip"
    if not archive.exists():
        return
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
    assert not any("data/derived" in n for n in names)
    assert not any("vessel_positions_part" in n for n in names)
    assert not any("vessel_tracks.csv" in n for n in names)
    assert not any("/evidence/" in n for n in names)
    assert any(n.endswith("TAKE_HOME_REPORT.md") for n in names)
