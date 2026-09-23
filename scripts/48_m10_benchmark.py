#!/usr/bin/env python3
"""Run calibration-only selection then one-time M10 final-test evaluation."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ais_eta.m10 import error_metrics

D = ROOT / "data" / "derived" / "m10_reference_rows.pkl.gz"
R = ROOT / "reports"
M = ROOT / "models"
FREEZE = R / "M10_FREEZE.json"
OPEN_MARKER = R / "M10_FINAL_HOLDOUT_OPENED.json"

ID_COLS = {
    "mmsi", "name", "split", "last_update", "eta_reference_raw", "eta_reference_dt",
    "target_tte_h", "reference_eta_status", "history_last_at",
}
CAT_COLS = [
    "destination_norm", "ship_type_cat", "nav_status_cat", "flag_cat",
    "motion_state_cat", "stale_risk_cat", "hist_destination_last",
]
CURRENT_FEATURES = [
    "track_lat", "track_lon", "track_sog", "track_cog", "track_heading", "track_draught",
    "track_msg_count", "destination_norm", "ship_type_cat", "nav_status_cat", "flag_cat",
    "last_hour_sin", "last_hour_cos", "last_dow",
]


def bootstrap_gain(y: np.ndarray, pred_base: np.ndarray, pred_model: np.ndarray, n: int = 5000) -> dict[str, float]:
    rng = np.random.default_rng(20260921)
    gains = np.abs(y - pred_base) - np.abs(y - pred_model)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(gains), len(gains))
        vals.append(float(gains[idx].mean()))
    return {
        "mean_gain_h": float(gains.mean()),
        "ci95_low_h": float(np.quantile(vals, 0.025)),
        "ci95_high_h": float(np.quantile(vals, 0.975)),
    }


def fit_ridge(train: pd.DataFrame, features: list[str]):
    cats = [c for c in CAT_COLS if c in features]
    nums = [c for c in features if c not in cats]
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), nums),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
        ]), cats),
    ])
    model = Pipeline([("pre", pre), ("model", Ridge(alpha=10.0))])
    model.fit(train[features], train["target_tte_h"])
    return model


def fit_catboost(train: pd.DataFrame, calibration: pd.DataFrame, features: list[str]):
    cats = [c for c in CAT_COLS if c in features]
    model = CatBoostRegressor(
        loss_function="MAE",
        iterations=700,
        depth=5,
        learning_rate=0.03,
        l2_leaf_reg=10.0,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        train[features], train["target_tte_h"], cat_features=cats,
        eval_set=(calibration[features], calibration["target_tte_h"]),
        early_stopping_rounds=80, verbose=False,
    )
    return model


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    x = df.copy()
    features = [c for c in x.columns if c not in ID_COLS]
    for c in CAT_COLS:
        if c not in x.columns:
            x[c] = "UNKNOWN"
        x[c] = x[c].fillna("UNKNOWN").astype(str)
    return x, features


def candidate_predictions(train: pd.DataFrame, cal: pd.DataFrame, features: list[str]) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    preds: dict[str, np.ndarray] = {}
    objs: dict[str, object] = {}

    global_median = float(train["target_tte_h"].median())
    preds["global_train_median"] = np.repeat(global_median, len(cal))
    objs["global_train_median"] = global_median

    dest_median = train.groupby("destination_norm")["target_tte_h"].median()
    preds["train_destination_median_with_global_fallback"] = (
        cal["destination_norm"].map(dest_median).fillna(global_median).to_numpy(float)
    )
    objs["train_destination_median_with_global_fallback"] = (global_median, dest_median)

    ridge = fit_ridge(train, features)
    preds["ridge_current_plus_history"] = ridge.predict(cal[features])
    objs["ridge_current_plus_history"] = ridge

    cat_current = fit_catboost(train, cal, CURRENT_FEATURES)
    preds["catboost_current_snapshot"] = cat_current.predict(cal[CURRENT_FEATURES])
    objs["catboost_current_snapshot"] = cat_current

    cat_hist = fit_catboost(train, cal, features)
    preds["catboost_current_plus_causal_history"] = cat_hist.predict(cal[features])
    objs["catboost_current_plus_causal_history"] = cat_hist
    return preds, objs


def select_by_calibration(cal: pd.DataFrame, preds: dict[str, np.ndarray]) -> tuple[str, pd.DataFrame]:
    rows = []
    for name, p in preds.items():
        m = error_metrics(cal["target_tte_h"], p)
        rows.append({"model": name, **m})
    table = pd.DataFrame(rows).sort_values(["mae_h", "p90_ae_h", "model"]).reset_index(drop=True)
    return str(table.iloc[0]["model"]), table


def predict_selected(name: str, obj: object, train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    if name == "global_train_median":
        return np.repeat(float(obj), len(test))
    if name == "train_destination_median_with_global_fallback":
        global_median, mapping = obj
        return test["destination_norm"].map(mapping).fillna(global_median).to_numpy(float)
    if name == "ridge_current_plus_history":
        return obj.predict(test[features])
    if name == "catboost_current_snapshot":
        return obj.predict(test[CURRENT_FEATURES])
    if name == "catboost_current_plus_causal_history":
        return obj.predict(test[features])
    raise KeyError(name)


def evaluate_task(df: pd.DataFrame, task: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    train = df.loc[df["split"].eq("train")].copy()
    cal = df.loc[df["split"].eq("calibration")].copy()
    test = df.loc[df["split"].eq("final_test")].copy()
    cal_preds, objs = candidate_predictions(train, cal, FEATURES)
    winner, cal_table = select_by_calibration(cal, cal_preds)

    # The final test is touched only after the winner is fixed from calibration.
    selected_test = predict_selected(winner, objs[winner], train, test, FEATURES)
    global_pred = np.repeat(float(train["target_tte_h"].median()), len(test))
    dest_mapping = train.groupby("destination_norm")["target_tte_h"].median()
    dest_pred = test["destination_norm"].map(dest_mapping).fillna(float(train["target_tte_h"].median())).to_numpy(float)

    test_rows = test[["mmsi", "name", "last_update", "eta_reference_dt", "target_tte_h", "reference_eta_status", "destination_norm"]].copy()
    test_rows["selected_model"] = winner
    test_rows["prediction_tte_h"] = selected_test
    test_rows["prediction_eta_dt"] = test_rows["last_update"] + pd.to_timedelta(test_rows["prediction_tte_h"], unit="h")
    test_rows["abs_error_h"] = (test_rows["prediction_tte_h"] - test_rows["target_tte_h"]).abs()
    test_rows["global_median_prediction_h"] = global_pred
    test_rows["destination_median_prediction_h"] = dest_pred

    selected_metrics = error_metrics(test["target_tte_h"], selected_test)
    global_metrics = error_metrics(test["target_tte_h"], global_pred)
    dest_metrics = error_metrics(test["target_tte_h"], dest_pred)
    gain = bootstrap_gain(test["target_tte_h"].to_numpy(float), dest_pred, selected_test)

    # Save selected CatBoost models for deterministic reproduction.
    if winner.startswith("catboost"):
        M.mkdir(exist_ok=True)
        suffix = "strict_reference" if task == "strict_all_parseable" else "future_reference"
        objs[winner].save_model(str(M / f"m10_{suffix}_catboost.cbm"))

    summary = {
        "task": task,
        "train_n": int(len(train)),
        "calibration_n": int(len(cal)),
        "final_test_n": int(len(test)),
        "selected_model": winner,
        "selection_metric": "calibration MAE hours",
        "selected_final_metrics": selected_metrics,
        "global_median_final_metrics": global_metrics,
        "destination_median_final_metrics": dest_metrics,
        "selected_vs_destination_median_bootstrap": gain,
    }
    return summary, cal_table.assign(task=task), test_rows.assign(task=task)


if __name__ == "__main__":
    if not FREEZE.exists():
        raise RuntimeError("Run 47_m10_freeze.py before benchmarking")
    if OPEN_MARKER.exists():
        raise RuntimeError("M10 final holdout already opened; refusing to score it again")
    raw = pd.read_pickle(D)
    df, FEATURES = prepare(raw)

    strict_summary, strict_cal, strict_test = evaluate_task(df, "strict_all_parseable")
    future_df = df.loc[df["target_tte_h"] >= 0].copy()
    future_summary, future_cal, future_test = evaluate_task(future_df, "future_reference")

    cal_all = pd.concat([strict_cal, future_cal], ignore_index=True)
    test_all = pd.concat([strict_test, future_test], ignore_index=True)
    cal_all.to_csv(R / "m10_calibration_model_comparison.csv", index=False)
    test_all.to_csv(R / "m10_final_predictions.csv", index=False)

    # Future-reference diagnostics by horizon/status, using only the preselected future task model predictions.
    horizon_rows = []
    for status, g in future_test.groupby("reference_eta_status"):
        met = error_metrics(g["target_tte_h"], g["prediction_tte_h"])
        horizon_rows.append({"reference_eta_status": status, **met})
    pd.DataFrame(horizon_rows).sort_values("reference_eta_status").to_csv(R / "m10_future_test_by_reference_status.csv", index=False)

    all_summary = {
        "strict": strict_summary,
        "future_reference": future_summary,
        "interpretation": {
            "strict": "Directly honors every parseable company-provided Tracks ETA label, including stale/past/far-future references.",
            "future_reference": "Operational diagnostic restricted only by ETA semantics: reference ETA is not already in the past at prediction time. No upper horizon cutoff is used.",
            "actual_arrival_claim": False,
        },
        "final_test_opened_once": True,
        "m6_predictor_unchanged": True,
        "m6_holdout_not_reused": True,
    }
    (R / "m10_benchmark_summary.json").write_text(json.dumps(all_summary, indent=2) + "\n")
    OPEN_MARKER.write_text(json.dumps({
        "opened_at": "2026-09-21T16:20:00+02:00",
        "reason": "One-time M10 company-reference ETA benchmark after calibration-only model selection",
        "strict_selected_model": strict_summary["selected_model"],
        "future_selected_model": future_summary["selected_model"],
        "final_test_rows_strict": strict_summary["final_test_n"],
        "final_test_rows_future": future_summary["final_test_n"],
        "no_post_test_tuning": True,
    }, indent=2) + "\n")

    # Serialize strict destination-median mapping when selected, for deterministic reproduction.
    strict_train = df.loc[df["split"].eq("train")]
    mapping = strict_train.groupby("destination_norm")["target_tte_h"].median().sort_index()
    (M / "m10_strict_destination_medians.json").parent.mkdir(exist_ok=True)
    (M / "m10_strict_destination_medians.json").write_text(json.dumps({
        "global_median_h": float(strict_train["target_tte_h"].median()),
        "destination_median_h": {str(k): float(v) for k, v in mapping.items()},
    }, indent=2) + "\n")

    print(json.dumps(all_summary, indent=2))
