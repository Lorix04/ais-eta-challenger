from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m18a import (
    M18A_DECISION_TIME_FIELD,
    M18A_PRIMARY_TASK_ID,
    M18A_TARGET_FIELD,
    assert_feature_contract,
    assert_history_not_after_decision_time,
    validate_contract_dict,
)


def test_m18a_contract_artifacts_and_identity():
    contract = json.loads((ROOT / "reports/M18A_PREDICTION_CONTRACT.json").read_text())
    validate_contract_dict(contract)
    task = contract["primary_prediction_contract"]
    assert task["task_id"] == M18A_PRIMARY_TASK_ID
    assert task["decision_time"] == M18A_DECISION_TIME_FIELD
    assert task["label_source"] == M18A_TARGET_FIELD
    assert contract["frozen_baseline"]["point_champion"].startswith("M16G")
    assert contract["fresh_holdout_policy"]["selection_use_allowed"] is False


def test_m18a_feature_contract_rejects_target_and_future_fields():
    assert_feature_contract(["track_lat", "destination_norm", "sog_median_60m"])
    with pytest.raises(AssertionError):
        assert_feature_contract(["track_lat", "target_tte_h"])
    with pytest.raises(AssertionError):
        assert_feature_contract(["future_weather_wave_h"])
    with pytest.raises(AssertionError):
        assert_feature_contract(["reference_eta_status"])


def test_m18a_causal_history_guard():
    rows = pd.DataFrame({
        "mmsi": [1, 1, 2],
        "recorded_at": ["2026-04-01T09:00:00", "2026-04-01T10:00:00", "2026-04-02T10:00:00"],
    })
    cut = {1: pd.Timestamp("2026-04-01T10:00:00"), 2: pd.Timestamp("2026-04-02T11:00:00")}
    assert_history_not_after_decision_time(rows, cut)
    bad = rows.copy()
    bad.loc[1, "recorded_at"] = "2026-04-01T10:00:01"
    with pytest.raises(AssertionError):
        assert_history_not_after_decision_time(bad, cut)


def test_m18a_freeze_does_not_promote_m17h_or_reopen_old_final():
    freeze = json.loads((ROOT / "reports/M18A_BASELINE_FREEZE.json").read_text())
    contract = json.loads((ROOT / "reports/M18A_PREDICTION_CONTRACT.json").read_text())
    assert freeze["point_champion"] == "M16G"
    assert freeze["model_change"] is False
    assert freeze["prediction_artifact_rewrite"] is False
    assert freeze["fresh_holdout_opened"] is False
    assert contract["frozen_baseline"]["blocked_old_final_rows"] == 53
    assert contract["frozen_baseline"]["development_rows"] == 386
