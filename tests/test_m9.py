import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_m9_company_target_is_only_conceptually_aligned_until_boundary_known():
    a = json.loads((ROOT / "reports/m9_company_answers.json").read_text())
    assert a["target"]["company_definition"] == "vessel reaches/enters the area of the destination port"
    assert a["target"]["alignment_with_current_research_gate"] == "conceptually_aligned_proxy"
    assert a["target"]["exact_equivalence_confirmed"] is False
    assert set(a["target"]["remaining_questions"]) == {"ground_truth_source", "port_area_perimeter_or_polygon"}


def test_m9_file_semantics_match_company_clarification_without_inventing_timestamp_provenance():
    a = json.loads((ROOT / "reports/m9_company_answers.json").read_text())
    assert a["data_semantics"]["positions"] == "historical AIS observations over time for each vessel"
    assert a["data_semantics"]["tracks"] == "most recent available vessel state"
    assert a["data_semantics"]["recorded_at_upstream_semantics"] == "unknown"


def test_m9_eta_history_stays_blocked():
    a = json.loads((ROOT / "reports/m9_company_answers.json").read_text())
    assert a["voyage_fields"]["historical_destination_in_positions"] is True
    assert a["voyage_fields"]["historical_eta_in_positions"] is False
    assert a["voyage_fields"]["eta_in_tracks_latest_state"] is True


def test_m9_frozen_m6_hashes_unchanged():
    freeze = json.loads((ROOT / "reports/M6_FREEZE.json").read_text())
    for rel, expected in freeze["sha256"].items():
        p = ROOT / rel
        assert p.exists(), rel
        assert hashlib.sha256(p.read_bytes()).hexdigest() == expected, rel


def test_m9_public_docs_remain_non_overclaiming_after_m10_supersedes_target():
    corpus = "\n".join([
        (ROOT / "README.md").read_text(),
        (ROOT / "EXECUTIVE_SUMMARY.md").read_text(),
        (ROOT / "TAKE_HOME_REPORT.md").read_text(),
        (ROOT / "docs/MODEL_CARD.md").read_text(),
        (ROOT / "docs/COMPANY_ALIGNMENT.md").read_text(),
    ]).lower()
    # M9's geometry question was superseded by the company's final M10 instruction:
    # use Tracks.eta as the exercise reference and do not reconstruct arrival.
    assert "company reference eta" in corpus or "reference eta" in corpus
    assert "not observed ata" in corpus or "not an observed actual time of arrival" in corpus
    assert "exactly matches the company's target" not in corpus
    assert "official company ground truth" not in corpus


def test_m9_draft_manifest_hash_matches_archive():
    manifest_path = ROOT / "dist/ais_eta_takehome_submission_m9_draft_manifest.json"
    archive = ROOT / "dist/ais_eta_takehome_submission_m9_draft.zip"
    if not manifest_path.exists() or not archive.exists():
        return
    m = json.loads(manifest_path.read_text())
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == m["sha256"]
    assert m["status"] == "DRAFT_PENDING_COMPANY_GT_AND_PORT_AREA_GEOMETRY"
