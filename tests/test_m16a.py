from pathlib import Path
import json
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m16a import assert_development_only, balanced_hash_folds, extended_metrics


def test_balanced_hash_folds_deterministic_target_free_shape():
    ids = list(range(1000, 1386))
    a = balanced_hash_folds(ids, n_splits=5, salt="fixture")
    b = balanced_hash_folds(reversed(ids), n_splits=5, salt="fixture")
    assert a == b
    counts = pd.Series(list(a.values())).value_counts().sort_index().tolist()
    assert sorted(counts) == [77, 77, 77, 77, 78]


def test_development_hard_block_rejects_old_final():
    df = pd.DataFrame({"mmsi": [1, 2], "split": ["train", "calibration"]})
    assert_development_only(df, [99])
    with pytest.raises(AssertionError, match="hard block"):
        assert_development_only(df, [2, 99])


def test_m16a_manifest_has_exact_population_and_zero_overlap():
    p = ROOT / "reports/M16A_DEVELOPMENT_MANIFEST.json"
    if not p.exists():
        pytest.skip("M16A artifacts not built yet")
    m = json.loads(p.read_text())
    assert m["development_rows"] == 386
    assert len(m["allowed_mmsi"]) == 386
    assert len(m["blocked_old_final_mmsi"]) == 53
    assert set(m["allowed_mmsi"]).isdisjoint(m["blocked_old_final_mmsi"])
    assert m["old_final_may_be_used_for_selection"] is False


def test_m16a_oof_ledger_excludes_old_final_and_is_one_row_per_mmsi():
    p = ROOT / "reports/m16a_oof_predictions.csv"
    mpath = ROOT / "reports/M16A_DEVELOPMENT_MANIFEST.json"
    if not p.exists() or not mpath.exists():
        pytest.skip("M16A artifacts not built yet")
    df = pd.read_csv(p)
    manifest = json.loads(mpath.read_text())
    assert len(df) == 386
    assert df["mmsi"].nunique() == 386
    assert set(df["mmsi"].astype(int)).isdisjoint(manifest["blocked_old_final_mmsi"])


def test_extended_metrics_schema_contains_m16_required_metrics():
    m = extended_metrics([0, 24, 48], [1, 20, 80])
    for key in ["mae_h", "medae_h", "rmse_h", "p90_ae_h", "p95_ae_h", "within_24h", "within_48h", "within_72h", "prediction_min_h", "prediction_max_h", "nonfinite_prediction_count", "abs_gt_1y_prediction_count"]:
        assert key in m


def test_project_state_preserves_m16a_freeze_after_later_m16_progress():
    state = json.loads((ROOT / "PROJECT_STATE.json").read_text())
    assert state["m16a_summary"]["status"] == "DONE_DEVELOPMENT_ONLY_OOF_BENCHMARK_FROZEN"
    assert state["m16a_summary"]["final_test_used_for_selection"] is False
    assert "M16B" in state["m16a_summary"]["next_submilestone"]
    assert state["current_phase"].startswith(("M16A_", "M16B_", "M16C_", "M16D_", "M16E_", "M16F_", "M16G_", "M16H_", "M16I_", "M16J_", "M17A_", "M17B_", "M17C_", "M17D_", "M17E_", "M17F_", "M17G_", "M17H_", "M18A_", "M18B_", "M18C_", "M18D_", "M18E_", "M18F_", "M18G_", "M19_"))
