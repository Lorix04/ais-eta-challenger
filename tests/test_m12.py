import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from ais_eta.m12 import (
    CURRENT_FEATURES,
    FEATURE_GROUPS,
    group_shap,
    masking_stress,
    mean_abs_shap,
    prediction_values_change,
    prepare_current_features,
    shap_additivity_error,
    shap_values,
)


def _tiny_df():
    rows = []
    for i in range(8):
        rows.append({
            "track_lat": 37.0 + i * 0.01,
            "track_lon": 15.0 + i * 0.01,
            "track_sog": float(i + 1),
            "track_cog": float(i * 20),
            "track_heading": float(i * 20),
            "track_draught": 5.0 + i / 10,
            "track_msg_count": 10 + i,
            "destination_norm": "AUGUSTA" if i % 2 == 0 else "CATANIA",
            "ship_type_cat": "Cargo",
            "nav_status_cat": "Under way using engine",
            "flag_cat": "IT",
            "last_hour_sin": float(np.sin(i)),
            "last_hour_cos": float(np.cos(i)),
            "last_dow": i % 7,
            "target_tte_h": float(20 + i * 2),
        })
    return pd.DataFrame(rows)


def _tiny_model(df):
    x = prepare_current_features(df)
    m = CatBoostRegressor(iterations=8, depth=2, learning_rate=0.1, loss_function="MAE", verbose=False, allow_writing_files=False)
    m.fit(x[CURRENT_FEATURES], x["target_tte_h"], cat_features=["destination_norm", "ship_type_cat", "nav_status_cat", "flag_cat"], verbose=False)
    return m


def test_feature_group_contract_covers_every_current_feature_once():
    flattened = [f for feats in FEATURE_GROUPS.values() for f in feats]
    assert sorted(flattened) == sorted(CURRENT_FEATURES)
    assert len(flattened) == len(set(flattened))


def test_native_shap_additivity_on_frozen_style_catboost():
    df = _tiny_df()
    m = _tiny_model(df)
    shap_df, pred = shap_values(m, df)
    assert shap_additivity_error(shap_df, pred) < 1e-8


def test_importance_tables_have_all_features():
    df = _tiny_df()
    m = _tiny_model(df)
    shap_df, _ = shap_values(m, df)
    pvc = prediction_values_change(m)
    mas = mean_abs_shap(shap_df, "tiny")
    assert set(pvc["feature"]) == set(CURRENT_FEATURES)
    assert set(mas["feature"]) == set(CURRENT_FEATURES)


def test_group_shap_has_expected_groups():
    df = _tiny_df()
    m = _tiny_model(df)
    shap_df, _ = shap_values(m, df)
    out = group_shap(shap_df, "tiny")
    assert set(out["feature_group"]) == set(FEATURE_GROUPS)


def test_masking_stress_does_not_change_input_and_is_finite():
    df = _tiny_df()
    m = _tiny_model(df)
    before = df.copy(deep=True)
    out = masking_stress(m, df.iloc[:6], df.iloc[6:], level="group")
    pd.testing.assert_frame_equal(df, before)
    assert len(out) == len(FEATURE_GROUPS)
    assert np.isfinite(out["mean_abs_prediction_shift_h"]).all()
