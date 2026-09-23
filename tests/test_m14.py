import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_m14_freeze_matches_frozen_m10_artifacts():
    f = json.loads((ROOT / "reports/M14_FINAL_FREEZE.json").read_text())
    assert f["status"] == "M14_FINAL_SUBMISSION_FROZEN"
    assert sha256(ROOT / "data/derived/m10_reference_rows.pkl.gz") == f["frozen_artifacts"]["m10_reference_dataset_sha256"]
    assert sha256(ROOT / "models/m10_strict_reference_catboost.cbm") == f["frozen_artifacts"]["m10_model_sha256"]
    assert sha256(ROOT / "reports/m10_final_predictions.csv") == f["frozen_artifacts"]["m10_final_predictions_sha256"]
    assert f["invariants"]["post_final_tuning"] is False


def test_m14_docs_mark_final_submission_without_changing_headline():
    text = "\n".join([
        (ROOT / "README.md").read_text(),
        (ROOT / "TAKE_HOME_REPORT.md").read_text(),
        (ROOT / "reports/M14_REPRODUCIBILITY_REPORT.md").read_text(),
    ]).lower()
    assert "final submission freeze" in text
    assert "312.0 h" in text
    assert "25.3 h" in text
    assert "no model" in text or "model" in text and "unchanged" in text


def test_m14_final_package_policy_is_confidentiality_bounded():
    f = json.loads((ROOT / "reports/M14_FINAL_FREEZE.json").read_text())
    p = f["submission_policy"]
    assert p["deterministic_archive"] is True
    assert p["cryptographic_content_manifest"] is True
    assert p["company_raw_ais_included"] is False
    assert p["high_volume_row_level_derived_data_included"] is False
    assert p["internal_evidence_included"] is False
