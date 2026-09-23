from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def _state():
    return json.loads((ROOT / "PROJECT_STATE.json").read_text())


def test_m16_workflow_exists_and_tracks_execution_state():
    workflow = (ROOT / "WORKFLOW.md").read_text()
    state = _state()
    assert "### M16 — Smart Hybrid Challenger" in workflow
    assert "### M16 — Smart Hybrid Challenger" in workflow
    assert state["m16_plan"]["status"] in {"PLANNED_NOT_STARTED", "IN_PROGRESS_M16A_DONE", "IN_PROGRESS_M16B_DONE", "IN_PROGRESS_M16C_DONE", "IN_PROGRESS_M16D_DONE", "IN_PROGRESS_M16E_DONE", "IN_PROGRESS_M16F_DONE", "IN_PROGRESS_M16G_DONE", "IN_PROGRESS_M16H_DONE", "IN_PROGRESS_M16I_DONE", "M16J_DONE"}


def test_m16_selection_population_excludes_old_final():
    state = _state()
    plan = state["m16_plan"]
    assert plan["development_rows"] == 386
    assert plan["prohibited_selection_rows"] == 53
    assert plan["development_rows"] + plan["prohibited_selection_rows"] == 439
    assert plan["m10_final_may_be_used_for_selection"] is False


def test_m16_keeps_frozen_submission_contract():
    state = _state()
    assert state["m16_plan"]["m14_submission_must_remain_immutable"] is True
    assert state["m14_summary"]["final_submission_sha256"] == (
        "65684d016775deea6b111db1a3ad76fe442cf84f1ceb1ac17f4327ea145c553c"
    )
    assert state["m14_summary"]["m10_model_sha256"] == (
        "efbb86375751e901c48c2f5724514828a10f9eb8ff08ffaabf2d875e369ef86a"
    )
    assert state["m14_summary"]["m10_final_predictions_sha256"] == (
        "fc6334d82f0b3c382e40805dc1a325163001a363ff6a6329da33160734011ee1"
    )


def test_workflow_hard_blocks_old_final_from_model_selection():
    workflow = (ROOT / "WORKFLOW.md").read_text()
    required = [
        "Never use the 53-row M10 final set for selection",
        "321 train + 65 calibration = 386 development rows",
        "OOF Gating / Blend" if "OOF Gating / Blend" in workflow else "Mixture of Experts / OOF Stacking",
        "Canonical Destination Resolver",
        "Historical Route-Analogue Expert",
        "Maritime / Physics Expert",
        "Probabilistic ETA + Confidence",
    ]
    for marker in required:
        assert marker in workflow
