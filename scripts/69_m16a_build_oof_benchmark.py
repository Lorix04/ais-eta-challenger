#!/usr/bin/env python3
"""Build the M16A development-only, leakage-safe OOF benchmark ledger."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ais_eta.m16a import (
    M16A_INNER_FOLDS,
    M16A_INNER_SALT,
    M16A_OUTER_FOLDS,
    M16A_OUTER_SALT,
    assert_development_only,
    balanced_hash_folds,
    extended_metrics,
)

D = ROOT / "data/derived/m10_reference_rows.pkl.gz"
R = ROOT / "reports"

CAT_COLS = ["destination_norm", "ship_type_cat", "nav_status_cat", "flag_cat"]
CURRENT_FEATURES = [
    "track_lat", "track_lon", "track_sog", "track_cog", "track_heading", "track_draught",
    "track_msg_count", "destination_norm", "ship_type_cat", "nav_status_cat", "flag_cat",
    "last_hour_sin", "last_hour_cos", "last_dow",
]
MODELS = ["global_train_median", "train_destination_median", "catboost_current_snapshot"]
CATBOOST_BASE_PARAMS = {
    "loss_function": "MAE",
    "depth": 5,
    "learning_rate": 0.03,
    "l2_leaf_reg": 10.0,
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False,
    "thread_count": 1,
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    for c in CAT_COLS:
        x[c] = x[c].fillna("UNKNOWN").astype(str)
    return x


def fit_oof_catboost(outer_train: pd.DataFrame, outer_valid: pd.DataFrame, outer_fold: int) -> tuple[np.ndarray, int]:
    # Inner early-stopping set is derived from outer-train MMSIs only.
    inner_map = balanced_hash_folds(
        outer_train["mmsi"], n_splits=M16A_INNER_FOLDS,
        salt=f"{M16A_INNER_SALT}:outer={outer_fold}",
    )
    inner_fold = outer_train["mmsi"].map(inner_map)
    inner_es = outer_train.loc[inner_fold.eq(0)].copy()
    inner_fit = outer_train.loc[~inner_fold.eq(0)].copy()
    if len(inner_es) == 0 or len(inner_fit) == 0:
        raise AssertionError("empty M16A inner split")

    selector = CatBoostRegressor(iterations=700, **CATBOOST_BASE_PARAMS)
    selector.fit(
        inner_fit[CURRENT_FEATURES], inner_fit["target_tte_h"], cat_features=CAT_COLS,
        eval_set=(inner_es[CURRENT_FEATURES], inner_es["target_tte_h"]),
        early_stopping_rounds=80, verbose=False,
    )
    chosen_trees = max(1, int(selector.tree_count_))

    # Refit with the selected complexity on all outer-train rows.  Outer-valid
    # targets never participate in fitting or early stopping.
    model = CatBoostRegressor(iterations=chosen_trees, **CATBOOST_BASE_PARAMS)
    model.fit(
        outer_train[CURRENT_FEATURES], outer_train["target_tte_h"],
        cat_features=CAT_COLS, verbose=False,
    )
    return np.asarray(model.predict(outer_valid[CURRENT_FEATURES]), dtype=float), chosen_trees


def run(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_pickle(D)
    if len(raw) != 439 or raw["mmsi"].nunique() != 439:
        raise AssertionError("unexpected M10 reference dataset cardinality")

    final = raw.loc[raw["split"].eq("final_test")].copy()
    dev = raw.loc[raw["split"].isin(["train", "calibration"])].copy()
    if len(dev) != 386 or len(final) != 53:
        raise AssertionError(f"unexpected M16A population: dev={len(dev)} final={len(final)}")
    assert_development_only(dev, final["mmsi"])

    dev = prepare(dev).sort_values("mmsi", kind="mergesort").reset_index(drop=True)
    outer_map = balanced_hash_folds(dev["mmsi"], M16A_OUTER_FOLDS, M16A_OUTER_SALT)
    dev["m16a_outer_fold"] = dev["mmsi"].map(outer_map).astype(int)

    manifest = {
        "milestone": "M16A",
        "status": "DEVELOPMENT_ONLY_POPULATION_FROZEN",
        "source_dataset": "data/derived/m10_reference_rows.pkl.gz",
        "source_dataset_sha256": file_sha256(D),
        "allowed_split_labels": ["train", "calibration"],
        "development_rows": int(len(dev)),
        "development_unique_mmsi": int(dev["mmsi"].nunique()),
        "allowed_mmsi": [int(x) for x in sorted(dev["mmsi"].tolist())],
        "blocked_old_final_rows": int(len(final)),
        "blocked_old_final_mmsi": [int(x) for x in sorted(final["mmsi"].tolist())],
        "old_final_may_be_used_for_selection": False,
        "outer_fold_policy": "balanced round-robin over SHA-256(salt:MMSI) ordering; target-free",
        "outer_fold_salt": M16A_OUTER_SALT,
        "outer_folds": M16A_OUTER_FOLDS,
        "outer_fold_counts": {str(int(k)): int(v) for k, v in dev["m16a_outer_fold"].value_counts().sort_index().items()},
        "inner_es_policy": "balanced hash folds inside outer-train only; fold 0 selects CatBoost tree count",
        "inner_es_salt": M16A_INNER_SALT,
        "inner_folds": M16A_INNER_FOLDS,
        "catboost_base_params": CATBOOST_BASE_PARAMS,
        "catboost_max_iterations_for_inner_selection": 700,
        "catboost_early_stopping_rounds": 80,
        "current_features": CURRENT_FEATURES,
        "categorical_features": CAT_COLS,
        "models": MODELS,
        "primary_metric": "pooled OOF MAE hours on all 386 strict-reference development rows",
    }
    (output_dir / "M16A_DEVELOPMENT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")

    ledger_rows: list[pd.DataFrame] = []
    fold_metric_rows: list[dict] = []
    tree_rows: list[dict] = []

    for fold in range(M16A_OUTER_FOLDS):
        tr = dev.loc[dev["m16a_outer_fold"].ne(fold)].copy()
        va = dev.loc[dev["m16a_outer_fold"].eq(fold)].copy()
        assert_development_only(tr, final["mmsi"])
        assert_development_only(va, final["mmsi"])
        if set(tr["mmsi"]) & set(va["mmsi"]):
            raise AssertionError("outer train/validation MMSI overlap")

        global_med = float(tr["target_tte_h"].median())
        dest_med = tr.groupby("destination_norm")["target_tte_h"].median()
        p_global = np.repeat(global_med, len(va))
        p_dest = va["destination_norm"].map(dest_med).fillna(global_med).to_numpy(float)
        p_cat, trees = fit_oof_catboost(tr, va, fold)
        tree_rows.append({"outer_fold": fold, "outer_train_n": len(tr), "outer_valid_n": len(va), "chosen_trees": trees})

        out = va[["mmsi", "split", "reference_eta_status", "destination_norm", "target_tte_h", "m16a_outer_fold"]].copy()
        out["pred_global_train_median_h"] = p_global
        out["pred_train_destination_median_h"] = p_dest
        out["pred_catboost_current_snapshot_h"] = p_cat
        ledger_rows.append(out)

        for name, pred in zip(MODELS, [p_global, p_dest, p_cat]):
            fold_metric_rows.append({"outer_fold": fold, "model": name, **extended_metrics(va["target_tte_h"], pred)})

    ledger = pd.concat(ledger_rows, ignore_index=True).sort_values("mmsi", kind="mergesort").reset_index(drop=True)
    if len(ledger) != 386 or ledger["mmsi"].nunique() != 386:
        raise AssertionError("M16A OOF ledger must contain exactly one row per development MMSI")
    if set(ledger["mmsi"]) & set(final["mmsi"]):
        raise AssertionError("M16A OOF ledger contains old-final MMSI")

    pred_cols = {
        "global_train_median": "pred_global_train_median_h",
        "train_destination_median": "pred_train_destination_median_h",
        "catboost_current_snapshot": "pred_catboost_current_snapshot_h",
    }
    pooled_rows = []
    status_rows = []
    for model, col in pred_cols.items():
        pooled_rows.append({"model": model, **extended_metrics(ledger["target_tte_h"], ledger[col])})
        for status, g in ledger.groupby("reference_eta_status", sort=True):
            status_rows.append({"model": model, "reference_eta_status": status, **extended_metrics(g["target_tte_h"], g[col])})

    pooled = pd.DataFrame(pooled_rows).sort_values(["mae_h", "p90_ae_h", "model"]).reset_index(drop=True)
    folds = pd.DataFrame(fold_metric_rows).sort_values(["outer_fold", "model"]).reset_index(drop=True)
    status = pd.DataFrame(status_rows).sort_values(["model", "reference_eta_status"]).reset_index(drop=True)
    trees = pd.DataFrame(tree_rows).sort_values("outer_fold").reset_index(drop=True)

    # Paired fold gains against the two frozen-recipe baselines.
    pivot = folds.pivot(index="outer_fold", columns="model", values="mae_h")
    gain_rows = []
    for fold in pivot.index:
        gain_rows.append({
            "outer_fold": int(fold),
            "catboost_minus_global_gain_h": float(pivot.loc[fold, "global_train_median"] - pivot.loc[fold, "catboost_current_snapshot"]),
            "catboost_minus_destination_gain_h": float(pivot.loc[fold, "train_destination_median"] - pivot.loc[fold, "catboost_current_snapshot"]),
        })
    gains = pd.DataFrame(gain_rows)

    ledger.to_csv(output_dir / "m16a_oof_predictions.csv", index=False, float_format="%.12g")
    pooled.to_csv(output_dir / "m16a_oof_model_metrics.csv", index=False, float_format="%.12g")
    folds.to_csv(output_dir / "m16a_oof_fold_metrics.csv", index=False, float_format="%.12g")
    status.to_csv(output_dir / "m16a_oof_status_metrics.csv", index=False, float_format="%.12g")
    gains.to_csv(output_dir / "m16a_fold_gains.csv", index=False, float_format="%.12g")
    trees.to_csv(output_dir / "m16a_catboost_tree_counts.csv", index=False)

    summary = {
        "milestone": "M16A",
        "status": "OOF_BENCHMARK_RECONSTRUCTED",
        "development_rows": 386,
        "blocked_old_final_rows": 53,
        "outer_folds": M16A_OUTER_FOLDS,
        "pooled_metrics": {r["model"]: {k: v for k, v in r.items() if k != "model"} for r in pooled.to_dict(orient="records")},
        "catboost_chosen_trees_by_fold": {str(int(r.outer_fold)): int(r.chosen_trees) for r in trees.itertuples()},
        "catboost_fold_wins_vs_global": int((gains["catboost_minus_global_gain_h"] > 0).sum()),
        "catboost_fold_wins_vs_destination": int((gains["catboost_minus_destination_gain_h"] > 0).sum()),
        "final_test_used_for_selection": False,
        "claim": "Development-only OOF baseline reconstruction; not a new untouched final-test result.",
    }
    (output_dir / "m16a_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    frozen_outputs = [
        "M16A_DEVELOPMENT_MANIFEST.json",
        "m16a_oof_predictions.csv",
        "m16a_oof_model_metrics.csv",
        "m16a_oof_fold_metrics.csv",
        "m16a_oof_status_metrics.csv",
        "m16a_fold_gains.csv",
        "m16a_catboost_tree_counts.csv",
        "m16a_summary.json",
    ]
    freeze = {
        "milestone": "M16A",
        "status": "OOF_BENCHMARK_FROZEN_FOR_M16_CHALLENGER_COMPARISONS",
        "development_population": 386,
        "blocked_old_final_population": 53,
        "outer_fold_salt": M16A_OUTER_SALT,
        "outer_folds": M16A_OUTER_FOLDS,
        "inner_es_salt": M16A_INNER_SALT,
        "inner_folds": M16A_INNER_FOLDS,
        "comparison_models": MODELS,
        "comparison_prediction_columns": pred_cols,
        "metric_schema": list(extended_metrics([0.0], [0.0]).keys()),
        "primary_metric": "mae_h",
        "selection_population_rule": "M10 train+calibration only; old M10 final hard-blocked",
        "artifact_sha256": {name: file_sha256(output_dir / name) for name in frozen_outputs},
    }
    (output_dir / "M16A_BENCHMARK_FREEZE.json").write_text(json.dumps(freeze, indent=2) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=R)
    args = parser.parse_args()
    summary = run(args.output_dir)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
