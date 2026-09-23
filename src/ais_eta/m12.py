"""M12 frozen-model explainability helpers.

M12 is strictly diagnostic.  It explains the already-selected M10 strict
CatBoost model without fitting, tuning, selecting, pruning or replacing it.

Interpretation guardrails:
- CatBoost PredictionValuesChange is model-internal importance, not causality.
- SHAP values decompose the frozen model prediction, not the data-generating process.
- Masking stress tests use train-only replacement values and are sensitivity
  diagnostics, not feature-selection evidence.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

CURRENT_FEATURES = [
    "track_lat", "track_lon", "track_sog", "track_cog", "track_heading", "track_draught",
    "track_msg_count", "destination_norm", "ship_type_cat", "nav_status_cat", "flag_cat",
    "last_hour_sin", "last_hour_cos", "last_dow",
]
CURRENT_CAT_FEATURES = ["destination_norm", "ship_type_cat", "nav_status_cat", "flag_cat"]

FEATURE_GROUPS: dict[str, list[str]] = {
    "location": ["track_lat", "track_lon"],
    "motion": ["track_sog", "track_cog", "track_heading"],
    "voyage_state": ["destination_norm", "nav_status_cat"],
    "vessel_context": ["track_draught", "ship_type_cat", "flag_cat"],
    "feed_context": ["track_msg_count"],
    "time_context": ["last_hour_sin", "last_hour_cos", "last_dow"],
}


def prepare_current_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with CatBoost categorical columns in deterministic form."""
    out = df.copy()
    for col in CURRENT_CAT_FEATURES:
        if col not in out.columns:
            out[col] = "UNKNOWN"
        out[col] = out[col].fillna("UNKNOWN").astype(str)
    return out


def current_pool(df: pd.DataFrame) -> Pool:
    x = prepare_current_features(df)
    return Pool(x[CURRENT_FEATURES], cat_features=CURRENT_CAT_FEATURES)


def prediction_values_change(model: CatBoostRegressor) -> pd.DataFrame:
    vals = np.asarray(model.get_feature_importance(type="PredictionValuesChange"), dtype=float)
    if len(vals) != len(CURRENT_FEATURES):
        raise AssertionError("Frozen model feature count does not match M12 feature contract")
    return (
        pd.DataFrame({"feature": CURRENT_FEATURES, "prediction_values_change": vals})
        .sort_values("prediction_values_change", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )


def shap_values(model: CatBoostRegressor, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Native CatBoost SHAP decomposition for the frozen model.

    Returns one SHAP column per model input feature plus the expected value.
    """
    x = prepare_current_features(df)
    arr = np.asarray(model.get_feature_importance(current_pool(x), type="ShapValues"), dtype=float)
    if arr.shape != (len(x), len(CURRENT_FEATURES) + 1):
        raise AssertionError(f"Unexpected SHAP shape: {arr.shape}")
    cols = [f"shap__{c}" for c in CURRENT_FEATURES] + ["shap__expected_value"]
    return pd.DataFrame(arr, index=x.index, columns=cols), model.predict(x[CURRENT_FEATURES])


def shap_additivity_error(shap_df: pd.DataFrame, predictions: Iterable[float]) -> float:
    p = np.asarray(list(predictions), dtype=float)
    contrib = shap_df[[f"shap__{c}" for c in CURRENT_FEATURES]].sum(axis=1).to_numpy(float)
    expected = shap_df["shap__expected_value"].to_numpy(float)
    return float(np.max(np.abs(contrib + expected - p))) if len(p) else 0.0


def mean_abs_shap(shap_df: pd.DataFrame, split_label: str) -> pd.DataFrame:
    rows = []
    for feat in CURRENT_FEATURES:
        vals = shap_df[f"shap__{feat}"].to_numpy(float)
        rows.append({
            "split": split_label,
            "feature": feat,
            "mean_abs_shap_h": float(np.mean(np.abs(vals))),
            "median_abs_shap_h": float(np.median(np.abs(vals))),
            "mean_signed_shap_h": float(np.mean(vals)),
        })
    return pd.DataFrame(rows).sort_values("mean_abs_shap_h", ascending=False, kind="mergesort")


def group_shap(shap_df: pd.DataFrame, split_label: str) -> pd.DataFrame:
    rows = []
    for group, feats in FEATURE_GROUPS.items():
        vals = shap_df[[f"shap__{f}" for f in feats]].to_numpy(float)
        # Group effect is the algebraic sum of contributions in the group per object.
        signed_group = vals.sum(axis=1)
        rows.append({
            "split": split_label,
            "feature_group": group,
            "features": ", ".join(feats),
            "mean_abs_group_shap_h": float(np.mean(np.abs(signed_group))),
            "median_abs_group_shap_h": float(np.median(np.abs(signed_group))),
            "mean_signed_group_shap_h": float(np.mean(signed_group)),
        })
    return pd.DataFrame(rows).sort_values("mean_abs_group_shap_h", ascending=False, kind="mergesort")


def _replacement_values(train: pd.DataFrame) -> dict[str, object]:
    x = prepare_current_features(train)
    rep: dict[str, object] = {}
    for feat in CURRENT_FEATURES:
        if feat in CURRENT_CAT_FEATURES:
            mode = x[feat].mode(dropna=False)
            rep[feat] = str(mode.iloc[0]) if len(mode) else "UNKNOWN"
        else:
            v = pd.to_numeric(x[feat], errors="coerce").median()
            rep[feat] = float(v) if pd.notna(v) else np.nan
    return rep


def _mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p)))


def masking_stress(
    model: CatBoostRegressor,
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    level: str = "group",
) -> pd.DataFrame:
    """Frozen-model perturbation diagnostic using train-only replacement values.

    This intentionally does *not* retrain the model.  It is not a causal or
    model-selection ablation; it asks how sensitive the frozen prediction is
    when an input/group is collapsed to a train-derived typical value.
    """
    tr = prepare_current_features(train)
    ev = prepare_current_features(evaluation)
    y = pd.to_numeric(ev["target_tte_h"], errors="coerce").to_numpy(float)
    base = np.asarray(model.predict(ev[CURRENT_FEATURES]), dtype=float)
    base_mae = _mae(y, base)
    rep = _replacement_values(tr)

    if level == "group":
        items = list(FEATURE_GROUPS.items())
    elif level == "feature":
        items = [(f, [f]) for f in CURRENT_FEATURES]
    else:
        raise ValueError(level)

    rows: list[dict[str, object]] = []
    for label, feats in items:
        masked = ev.copy()
        for feat in feats:
            masked[feat] = rep[feat]
        pred = np.asarray(model.predict(masked[CURRENT_FEATURES]), dtype=float)
        shift = np.abs(pred - base)
        masked_mae = _mae(y, pred)
        rows.append({
            "level": level,
            "item": label,
            "features": ", ".join(feats),
            "baseline_mae_h": base_mae,
            "masked_mae_h": masked_mae,
            "delta_mae_h": masked_mae - base_mae,
            "mean_abs_prediction_shift_h": float(np.mean(shift)),
            "median_abs_prediction_shift_h": float(np.median(shift)),
            "p90_abs_prediction_shift_h": float(np.quantile(shift, 0.9)),
        })
    return pd.DataFrame(rows).sort_values(
        ["mean_abs_prediction_shift_h", "delta_mae_h"], ascending=[False, False], kind="mergesort"
    )
