from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m16c import (
    PriorConfig,
    add_m16c_features,
    candidate_configs,
    predict_hierarchical_prior,
    select_config_inner_cv,
)


def _fixture():
    return pd.DataFrame({
        "mmsi": [1, 2, 3, 4, 5, 6, 7, 8],
        "canonical_destination": ["ITAUG"] * 4 + ["ITCTA"] * 4,
        "canonical_lat": [37.19] * 4 + [37.50] * 4,
        "canonical_lon": [15.20] * 4 + [15.10] * 4,
        "track_lat": [37.0, 37.1, 37.2, 37.3, 37.2, 37.3, 37.4, 37.5],
        "track_lon": [15.0, 15.1, 15.2, 15.3, 15.0, 15.1, 15.2, 15.3],
        "track_sog": [0, 2, 8, 12, 0, 3, 9, 11],
        "ship_type_cat": ["Cargo", "Cargo", "Tanker", "Tanker"] * 2,
        "target_tte_h": [10, 12, 20, 22, 30, 32, 40, 42],
    })


def test_m16c_feature_engineering_is_target_free():
    a = _fixture()
    b = a.copy()
    b["target_tte_h"] = b["target_tte_h"] * 999.0
    fa = add_m16c_features(a)
    fb = add_m16c_features(b)
    cols = [
        "dest_key", "ship_type_key", "distance_to_destination_km", "distance_band",
        "movement_bucket", "key_dest", "key_dest_ship", "key_dest_distance",
        "key_dest_distance_ship", "key_dest_distance_ship_movement",
    ]
    pd.testing.assert_frame_equal(fa[cols], fb[cols])


def test_hierarchical_prior_shrinks_child_toward_parent():
    x = add_m16c_features(_fixture())
    tr = x.iloc[:6].copy()
    va = x.iloc[[6]].copy()
    cfg = PriorConfig("dest_ship", min_support=2, shrinkage_alpha=10.0, clip_quantile=None)
    out = predict_hierarchical_prior(tr, va, cfg)
    p = float(out.iloc[0]["prediction_h"])
    assert np.isfinite(p)
    # Shrinkage should prevent raw child/group memorization and remain within
    # the broad training target range for this fixture.
    assert tr["target_tte_h"].min() <= p <= tr["target_tte_h"].max()
    assert float(out.iloc[0]["deepest_weight"]) < 1.0


def test_low_support_child_falls_back_deterministically():
    x = add_m16c_features(_fixture())
    tr = x.iloc[:6].copy()
    va = x.iloc[[6]].copy()
    cfg = PriorConfig("full", min_support=5, shrinkage_alpha=50.0, clip_quantile=None)
    a = predict_hierarchical_prior(tr, va, cfg)
    b = predict_hierarchical_prior(tr, va, cfg)
    pd.testing.assert_frame_equal(a, b)
    assert int(a.iloc[0]["deepest_level"]) <= len(("key_dest", "key_dest_distance", "key_dest_distance_ship", "key_dest_distance_ship_movement"))


def test_inner_selection_is_deterministic_and_has_multiple_safe_configs():
    x = add_m16c_features(_fixture())
    # Replicate MMSI-safe fixture to ensure all inner folds are non-empty.
    frames = []
    for j in range(8):
        z = x.copy()
        z["mmsi"] = z["mmsi"] + 100 * j
        z["target_tte_h"] = z["target_tte_h"] + j
        frames.append(z)
    work = pd.concat(frames, ignore_index=True)
    configs = candidate_configs()
    assert len(configs) >= 12
    c1, s1 = select_config_inner_cv(work, outer_fold=2, configs=configs[:12])
    c2, s2 = select_config_inner_cv(work, outer_fold=2, configs=configs[:12])
    assert c1 == c2
    pd.testing.assert_frame_equal(s1, s2)


def test_m16c_artifacts_are_dev_only_nested_and_not_standalone_promoted():
    sp = ROOT / "reports/M16C_SUMMARY.json"
    lp = ROOT / "reports/m16c_oof_predictions.csv"
    if not sp.exists() or not lp.exists():
        return
    summary = json.loads(sp.read_text())
    ledger = pd.read_csv(lp)
    manifest = json.loads((ROOT / "reports/M16A_DEVELOPMENT_MANIFEST.json").read_text())
    assert len(ledger) == 386 and ledger["mmsi"].nunique() == 386
    assert set(ledger["mmsi"].astype(int)).isdisjoint(manifest["blocked_old_final_mmsi"])
    assert summary["final_test_used_for_selection"] is False
    assert summary["m16c_mae_gain_h_vs_m16a_raw_destination"] > 0
    assert summary["standalone_promoted"] is False
    assert summary["gate"] == "PASS_COMPONENT_NOT_STANDALONE"
    assert summary["equal_blend_50_50_diagnostic"]["weight_selected_or_tuned"] is False


def test_project_state_preserves_m16c_and_advances_only_after_finalize():
    state = json.loads((ROOT / "PROJECT_STATE.json").read_text())
    if "m16c_summary" not in state:
        return
    assert state["m16c_summary"]["final_test_used_for_selection"] is False
    assert state["m16c_summary"]["standalone_promoted"] is False
    assert state["current_phase"].startswith(("M16C_", "M16D_", "M16E_", "M16F_", "M16G_", "M16H_", "M16I_", "M16J_", "M17A_", "M17B_", "M17C_", "M17D_", "M17E_", "M17F_", "M17G_", "M17H_", "M18A_", "M18B_", "M18C_", "M18D_", "M18E_", "M18F_", "M18G_", "M19_"))
