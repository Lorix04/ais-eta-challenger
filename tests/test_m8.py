import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def test_m8_frozen_artifacts_still_match_m6_freeze():
    freeze = json.loads((ROOT / "reports/M6_FREEZE.json").read_text())
    for rel, expected in freeze["sha256"].items():
        path = ROOT / rel
        assert path.exists(), rel
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == expected, rel


def test_m8_submission_does_not_ship_internal_personalized_docs():
    zpath = ROOT / "dist/ais_eta_takehome_submission_m8.zip"
    if not zpath.exists():
        return
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
    forbidden = ["CALL_PREP_IT.md", "EMAIL_DRAFT_IT.md", "CHANGE_PROTOCOL.md", "COMPANY_REPLY_MATRIX.md"]
    assert not any(any(x in n for x in forbidden) for n in names)


def test_m8_confidentiality_manifest_is_precise():
    path = ROOT / "dist/ais_eta_takehome_submission_m8_manifest.json"
    if not path.exists():
        return
    m = json.loads(path.read_text())
    assert m["contains_company_raw_ais"] is False
    assert m["contains_high_volume_derived_trajectories"] is False
    assert m["contains_compact_company_data_derived_annotations"] is True
    assert m["contains_internal_personalized_docs"] is False


def test_m8_public_claims_remain_scoped():
    corpus = "\n".join([
        (ROOT / "README.md").read_text(),
        (ROOT / "EXECUTIVE_SUMMARY.md").read_text(),
        (ROOT / "TAKE_HOME_REPORT.md").read_text(),
        (ROOT / "docs/MODEL_CARD.md").read_text(),
    ]).lower()
    assert "not established" in corpus
    assert "reported ais eta" in corpus
    assert "unseen-port" in corpus or "unseen ports" in corpus
    assert "we beat the company's model" not in corpus
    assert "proven superior to reported ais eta" not in corpus
