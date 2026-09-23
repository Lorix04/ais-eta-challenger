from __future__ import annotations

import numpy as np
import pandas as pd

from ais_eta.m1 import (
    M1Config,
    _decision_grid,
    assign_call_splits,
    baseline_metrics,
    destination_supports_catania,
)


def _cohort(n=10):
    times = pd.date_range("2026-04-01", periods=n, freq="12h")
    return pd.DataFrame(
        {
            "session_id": [f"S{i:02d}" for i in range(n)],
            "mmsi": [100 + (i % 4) for i in range(n)],
            "name": [f"V{i % 4}" for i in range(n)],
            "ground_truth_time": times,
            "ground_truth_confidence": ["A"] * n,
        }
    )


def test_destination_support_is_conservative():
    assert destination_supports_catania("IT CTA")
    assert destination_supports_catania("ITGOA>ITCTA")
    assert destination_supports_catania("CATANIA")
    assert not destination_supports_catania("FOR ORDERS")
    assert not destination_supports_catania(None)


def test_decision_grid_is_wall_clock_anchored_not_ata_anchored():
    start = pd.Timestamp("2026-04-01 10:07:00")
    target = pd.Timestamp("2026-04-01 11:02:13")
    grid = _decision_grid(start, target, 15)
    assert list(grid) == [
        pd.Timestamp("2026-04-01 10:15:00"),
        pd.Timestamp("2026-04-01 10:30:00"),
        pd.Timestamp("2026-04-01 10:45:00"),
        pd.Timestamp("2026-04-01 11:00:00"),
    ]


def test_call_split_locks_last_calls_and_keeps_final_out_of_cv():
    cfg = M1Config(final_chronological_test_calls=2, session_cv_folds=2, vessel_cv_folds=2)
    out = assign_call_splits(_cohort(10), cfg)
    final = out[out.m1_split.eq("final_chronological_test")]
    dev = out[out.m1_split.eq("development")]
    assert final.session_id.tolist() == ["S08", "S09"]
    assert final.session_cv_fold.isna().all()
    assert final.vessel_cv_fold.isna().all()
    assert dev.session_cv_fold.notna().all()
    assert dev.vessel_cv_fold.notna().all()
    assert final.ground_truth_time.min() > dev.ground_truth_time.max()


def test_vessel_cv_keeps_mmsi_whole():
    cfg = M1Config(final_chronological_test_calls=2, session_cv_folds=2, vessel_cv_folds=2)
    out = assign_call_splits(_cohort(12), cfg)
    dev = out[out.m1_split.eq("development")]
    for _, group in dev.groupby("mmsi"):
        assert group.vessel_cv_fold.nunique() == 1


def test_baseline_metrics_equalize_calls():
    panel = pd.DataFrame(
        {
            "session_id": ["A", "A", "A", "B"],
            "m1_split": ["development"] * 4,
            "scope": [True] * 4,
            "true_tta_h": [1.0, 1.0, 1.0, 1.0],
            "pred_b0_geodesic_current_sog_h": [1.0, 1.0, 1.0, 3.0],
            "pred_b1_geodesic_current_sog_floor_h": [1.0, 1.0, 1.0, 3.0],
            "pred_b2_geodesic_median30_sog_floor_h": [1.0, 1.0, 1.0, 3.0],
        }
    )
    m = baseline_metrics(panel, "scope")
    # Call A MAE=0, call B MAE=2, so equal-call MAE must be 1h rather
    # than pooled-row MAE of 0.5h.
    assert np.isclose(m.iloc[0].voyage_balanced_mae_h, 1.0)
