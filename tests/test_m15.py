import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_m15_preserves_m14_and_scientific_freezes():
    m = json.loads((ROOT / "reports/M15_DEFENSE_MANIFEST.json").read_text())
    h = m["immutable_hashes"]
    assert sha256(ROOT / "dist/ais_eta_takehome_submission_final.zip") == h["m14_final_zip_sha256"]
    assert sha256(ROOT / "dist/ais_eta_takehome_submission_final_manifest.json") == h["m14_final_manifest_sha256"]
    assert sha256(ROOT / "dist/ais_eta_takehome_submission_final.zip.sha256") == h["m14_final_sha_file_sha256"]
    assert sha256(ROOT / "models/m10_strict_reference_catboost.cbm") == h["m10_model_sha256"]
    assert sha256(ROOT / "reports/m10_final_predictions.csv") == h["m10_final_predictions_sha256"]
    assert sha256(ROOT / "reports/M6_FREEZE.json") == h["m6_freeze_sha256"]


def test_m15_playbook_keeps_strict_and_diagnostic_claims_separate():
    text = (ROOT / "docs/M15_TECHNICAL_DEFENSE_IT.md").read_text().lower()
    assert "312,0 h" in text
    assert "25,3 h" in text
    assert "28,6 h" in text
    assert "post-hoc" in text
    assert "non sostituisce" in text or "non sostituire" in text
    assert "non posso dirlo" in text
    assert "shap" in text and "non causal" in text


def test_m15_qa_bank_contains_core_protocol_and_reproducibility_answers():
    text = (ROOT / "docs/M15_QA_BANK_IT.md").read_text().lower()
    for token in [
        "439",
        "321 train, 65 calibration, 53 final",
        "target-only",
        "hash split",
        "final aperto una volta",
        "raw-data replay",
        "production-ready",
    ]:
        assert token in text


def test_m15_manifest_and_project_state_are_non_modelling():
    m = json.loads((ROOT / "reports/M15_DEFENSE_MANIFEST.json").read_text())
    s = json.loads((ROOT / "PROJECT_STATE.json").read_text())
    assert m["status"] == "M15_INTERVIEW_TECHNICAL_DEFENSE_READY"
    assert "no modelling" in m["scope"]
    assert s["current_phase"] == "M15_DONE_INTERVIEW_TECHNICAL_DEFENSE_PREPARATION" or s["current_phase"].startswith(("M16", "M17", "M18", "M19"))
    assert s["m15_summary"]["model_changed"] is False
    assert s["m15_summary"]["m14_submission_changed"] is False
