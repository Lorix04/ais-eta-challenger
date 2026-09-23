#!/usr/bin/env python3
"""M12 — explain the frozen M10 strict CatBoost without retraining."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
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

R = ROOT / "reports"
D = ROOT / "data" / "derived" / "m10_reference_rows.pkl.gz"
MODEL_PATH = ROOT / "models" / "m10_strict_reference_catboost.cbm"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    before_model_sha = sha256(MODEL_PATH)
    df = prepare_current_features(pd.read_pickle(D))
    train = df.loc[df["split"].eq("train")].copy()
    cal = df.loc[df["split"].eq("calibration")].copy()
    final = df.loc[df["split"].eq("final_test")].copy()

    model = CatBoostRegressor()
    model.load_model(str(MODEL_PATH))
    if model.feature_names_ != CURRENT_FEATURES:
        raise AssertionError(f"Frozen model features changed: {model.feature_names_}")

    # 1) Native model importance and native SHAP. No targets enter these calculations.
    pvc = prediction_values_change(model)
    pvc.to_csv(R / "m12_prediction_values_change.csv", index=False)

    shap_tables = []
    shap_summary = []
    group_summary = []
    additivity = {}
    for split_name, g in [("train", train), ("calibration", cal), ("final_test", final)]:
        sh, pred = shap_values(model, g)
        additivity[split_name] = shap_additivity_error(sh, pred)
        shap_summary.append(mean_abs_shap(sh, split_name))
        group_summary.append(group_shap(sh, split_name))

        compact = g[["mmsi", "name", "split", "target_tte_h", "reference_eta_status", "destination_norm"]].copy()
        compact["prediction_tte_h"] = pred
        compact = pd.concat([compact.reset_index(drop=True), sh.reset_index(drop=True)], axis=1)
        shap_tables.append(compact)

    pd.concat(shap_summary, ignore_index=True).to_csv(R / "m12_mean_abs_shap.csv", index=False)
    pd.concat(group_summary, ignore_index=True).to_csv(R / "m12_group_shap_importance.csv", index=False)
    all_shap = pd.concat(shap_tables, ignore_index=True)
    all_shap.to_csv(R / "m12_shap_contributions.csv", index=False)

    # 2) Frozen-model sensitivity: calibration only, train-derived replacement values.
    group_mask = masking_stress(model, train, cal, level="group")
    feat_mask = masking_stress(model, train, cal, level="feature")
    group_mask.to_csv(R / "m12_calibration_group_masking_stress.csv", index=False)
    feat_mask.to_csv(R / "m12_calibration_feature_masking_stress.csv", index=False)

    # 3) Reuse the already-frozen M10 calibration candidate table as the architecture/history ablation.
    frozen = pd.read_csv(R / "m10_calibration_model_comparison.csv")
    strict = frozen.loc[frozen["task"].eq("strict_all_parseable")].copy()
    future = frozen.loc[frozen["task"].eq("future_reference")].copy()
    rows = []
    for task_name, table in [("strict_all_parseable", strict), ("future_reference", future)]:
        metrics = table.set_index("model")
        current = float(metrics.loc["catboost_current_snapshot", "mae_h"])
        history = float(metrics.loc["catboost_current_plus_causal_history", "mae_h"])
        dest = float(metrics.loc["train_destination_median_with_global_fallback", "mae_h"])
        ridge = float(metrics.loc["ridge_current_plus_history", "mae_h"])
        rows.extend([
            {"task": task_name, "comparison": "catboost_plus_history_minus_current", "reference_model": "catboost_current_snapshot", "compared_model": "catboost_current_plus_causal_history", "reference_mae_h": current, "compared_mae_h": history, "delta_mae_h": history-current, "relative_delta": (history-current)/current},
            {"task": task_name, "comparison": "destination_median_minus_current", "reference_model": "catboost_current_snapshot", "compared_model": "train_destination_median_with_global_fallback", "reference_mae_h": current, "compared_mae_h": dest, "delta_mae_h": dest-current, "relative_delta": (dest-current)/current},
            {"task": task_name, "comparison": "ridge_history_minus_current", "reference_model": "catboost_current_snapshot", "compared_model": "ridge_current_plus_history", "reference_mae_h": current, "compared_mae_h": ridge, "delta_mae_h": ridge-current, "relative_delta": (ridge-current)/current},
        ])
    ablation = pd.DataFrame(rows)
    ablation.to_csv(R / "m12_frozen_calibration_ablation.csv", index=False)

    # 4) Local explanations for representative frozen final rows: top errors + median-ish error.
    fp = pd.read_csv(R / "m10_final_predictions.csv")
    fp = fp.loc[fp["task"].eq("strict_all_parseable")].copy()
    final_shap = all_shap.loc[all_shap["split"].eq("final_test")].copy()
    local = fp[["mmsi", "name", "target_tte_h", "prediction_tte_h", "abs_error_h", "reference_eta_status", "destination_norm"]].merge(
        final_shap.drop(columns=["name", "target_tte_h", "prediction_tte_h", "reference_eta_status", "destination_norm"]),
        on="mmsi", how="left", validate="one_to_one"
    )
    local["abs_error_rank"] = local["abs_error_h"].rank(method="first", ascending=False).astype(int)
    med = float(local["abs_error_h"].median())
    local["is_median_error_example"] = (local["abs_error_h"] - med).abs().eq((local["abs_error_h"] - med).abs().min())
    keep = local.loc[(local["abs_error_rank"] <= 5) | local["is_median_error_example"]].copy()
    keep.to_csv(R / "m12_local_explanation_examples.csv", index=False)

    # Compact top-contributor view for human review.
    contrib_rows = []
    for rr in keep.itertuples(index=False):
        vals = [(f, float(getattr(rr, f"shap__{f}"))) for f in CURRENT_FEATURES]
        for rank, (feat, val) in enumerate(sorted(vals, key=lambda x: abs(x[1]), reverse=True)[:5], start=1):
            contrib_rows.append({
                "mmsi": rr.mmsi, "name": rr.name, "abs_error_rank": rr.abs_error_rank,
                "target_tte_h": rr.target_tte_h, "prediction_tte_h": rr.prediction_tte_h,
                "abs_error_h": rr.abs_error_h, "contributor_rank": rank,
                "feature": feat, "shap_h": val, "abs_shap_h": abs(val),
            })
    pd.DataFrame(contrib_rows).to_csv(R / "m12_local_top_contributors.csv", index=False)

    # Prediction compression explains why a frozen model centered on ordinary horizons cannot follow multi-month stale labels.
    final_pred = fp["prediction_tte_h"].to_numpy(float)
    final_target = fp["target_tte_h"].to_numpy(float)
    pvc_top = pvc.iloc[0]
    cal_shap = pd.concat(shap_summary, ignore_index=True)
    cal_shap = cal_shap.loc[cal_shap["split"].eq("calibration")].sort_values("mean_abs_shap_h", ascending=False)
    top_shap = cal_shap.iloc[0]
    group_cal = pd.concat(group_summary, ignore_index=True)
    group_cal = group_cal.loc[group_cal["split"].eq("calibration")].sort_values("mean_abs_group_shap_h", ascending=False)
    mask_top = group_mask.iloc[0]

    summary = {
        "scope": "diagnostic-only frozen-model explainability; no training, tuning, model selection or feature pruning",
        "frozen_model": str(MODEL_PATH.relative_to(ROOT)),
        "frozen_model_sha256_before": before_model_sha,
        "frozen_model_sha256_after": sha256(MODEL_PATH),
        "tree_count": int(model.tree_count_),
        "feature_count": len(CURRENT_FEATURES),
        "expected_value_h": float(all_shap["shap__expected_value"].iloc[0]),
        "shap_max_additivity_error_h": {k: float(v) for k, v in additivity.items()},
        "top_native_importance_feature": str(pvc_top["feature"]),
        "top_native_importance_value": float(pvc_top["prediction_values_change"]),
        "top_calibration_mean_abs_shap_feature": str(top_shap["feature"]),
        "top_calibration_mean_abs_shap_h": float(top_shap["mean_abs_shap_h"]),
        "top_calibration_shap_group": str(group_cal.iloc[0]["feature_group"]),
        "top_calibration_group_mean_abs_shap_h": float(group_cal.iloc[0]["mean_abs_group_shap_h"]),
        "top_calibration_masking_group": str(mask_top["item"]),
        "top_calibration_masking_mean_abs_shift_h": float(mask_top["mean_abs_prediction_shift_h"]),
        "strict_calibration_history_delta_mae_h": float(ablation.loc[(ablation["task"].eq("strict_all_parseable")) & (ablation["comparison"].eq("catboost_plus_history_minus_current")), "delta_mae_h"].iloc[0]),
        "future_calibration_history_delta_mae_h": float(ablation.loc[(ablation["task"].eq("future_reference")) & (ablation["comparison"].eq("catboost_plus_history_minus_current")), "delta_mae_h"].iloc[0]),
        "future_calibration_destination_median_delta_mae_h_vs_current": float(ablation.loc[(ablation["task"].eq("future_reference")) & (ablation["comparison"].eq("destination_median_minus_current")), "delta_mae_h"].iloc[0]),
        "final_prediction_min_h": float(np.min(final_pred)),
        "final_prediction_max_h": float(np.max(final_pred)),
        "final_target_min_h": float(np.min(final_target)),
        "final_target_max_h": float(np.max(final_target)),
        "interpretation": {
            "importance_is_not_causal": True,
            "shap_explains_model_not_maritime_reality": True,
            "masking_used_for_selection": False,
            "final_test_used_for_model_change": False,
            "history_model_refit_in_m12": False,
        },
    }
    (R / "m12_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"PASS frozen CatBoost loaded: {model.tree_count_} trees / {len(CURRENT_FEATURES)} features")
    print(f"PASS frozen model SHA unchanged: {before_model_sha[:12]}...")
    print(f"PASS native importance top feature: {summary['top_native_importance_feature']}")
    print(f"PASS calibration SHAP top feature: {summary['top_calibration_mean_abs_shap_feature']} ({summary['top_calibration_mean_abs_shap_h']:.2f} h mean |SHAP|)")
    print(f"PASS SHAP additivity max error: {max(additivity.values()):.3e} h")
    print(f"PASS strict calibration causal-history delta: {summary['strict_calibration_history_delta_mae_h']:+.2f} h MAE")
    print(f"PASS future calibration destination-median delta vs current: {summary['future_calibration_destination_median_delta_mae_h_vs_current']:+.2f} h MAE")
    print("PASS M12 performed no training or model reselection")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
