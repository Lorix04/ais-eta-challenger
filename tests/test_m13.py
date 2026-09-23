import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_m13_claim_map_has_blocked_and_primary_claims():
    rows = list(csv.DictReader((ROOT / "reports/m13_claim_map.csv").open()))
    roles = {r["delivery_role"] for r in rows}
    assert {"PRIMARY", "DIAGNOSTIC_ONLY", "SECONDARY", "BLOCKED"}.issubset(roles)
    assert all(r["status"] == "NOT_ESTABLISHED" for r in rows if r["delivery_role"] == "BLOCKED")


def test_company_report_leads_with_reference_eta_and_secondary_branch_is_explicit():
    report = (ROOT / "TAKE_HOME_REPORT.md").read_text().lower()
    assert report.index("tracks.eta") < report.index("secondary branch")
    assert "312.0 h" in report
    assert "25.3 h" in report
    assert "physical port-entry" in report


def test_model_card_keeps_reference_eta_distinct_from_observed_ata():
    card = (ROOT / "docs/MODEL_CARD.md").read_text().lower()
    assert "tracks.eta" in card
    assert "not" in card and "actual time of arrival" in card
