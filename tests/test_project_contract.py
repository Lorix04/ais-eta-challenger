import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_required_project_control_files_exist():
    required = [
        ROOT / "WORKFLOW.md",
        ROOT / "PROJECT_STATE.json",
        ROOT / "docs" / "CHANGE_PROTOCOL.md",
        ROOT / "README.md",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    assert not missing, f"Missing project control files: {missing}"


def test_project_state_is_valid_and_respects_go_no_go_gate():
    state = json.loads((ROOT / "PROJECT_STATE.json").read_text())
    phase = state["current_phase"]
    status = state["go_no_go_A"]["status"]
    if phase.startswith("M0"):
        assert status == "NOT_REACHED"
    else:
        assert status in {"GO_LIMITED", "GO"}
        assert "M0E manual audit" in " ".join(state["completed"])
    assert "berth/all-fast prediction" in state["do_not_start_yet"]
    assert state["evidence_policy"]["required_for_every_material_change"] is True
    assert len(state["evidence_policy"]["minimum_screenshots"]) >= 3


def test_workflow_contains_gates_and_evidence_protocol():
    text = (ROOT / "WORKFLOW.md").read_text()
    required_phrases = [
        "GO/NO-GO A",
        "M0C — AIS normalization + semantic motion states",
        "M0D — Catania geometry + port-call reconstruction",
        "M1 — ETA baselines",
        "Evidence rule for every project modification",
        "01_changes.png",
        "02_tests.png",
        "03_result.png",
    ]
    missing = [p for p in required_phrases if p not in text]
    assert not missing, f"Workflow missing required phrases: {missing}"


def test_raw_data_is_gitignored():
    text = (ROOT / ".gitignore").read_text()
    assert "/data/" in text
