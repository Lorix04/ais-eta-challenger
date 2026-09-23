#!/usr/bin/env python3
"""Build M16C nested-OOF fold-safe hierarchical destination priors."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ais_eta.m16a import assert_development_only, extended_metrics
from ais_eta.m16c import (
    M16C_CLIP_QUANTILES,
    M16C_HIERARCHIES,
    M16C_INNER_FOLDS,
    M16C_INNER_SALT,
    M16C_PARAMETER_PROFILES,
    M16C_VERSION,
    add_m16c_features,
    candidate_configs,
    predict_hierarchical_prior,
    select_config_inner_cv,
)

D = ROOT / "data/derived/m10_reference_rows.pkl.gz"
M16A_LEDGER = ROOT / "reports/m16a_oof_predictions.csv"
M16A_MANIFEST = ROOT / "reports/M16A_DEVELOPMENT_MANIFEST.json"
M16B_LEDGER = ROOT / "reports/m16b_destination_resolutions.csv"
R = ROOT / "reports"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_pickle(D)
    final = raw.loc[raw["split"].eq("final_test")].copy()
    dev = raw.loc[raw["split"].isin(["train", "calibration"])].copy()
    assert_development_only(dev, final["mmsi"])
    if len(dev) != 386 or len(final) != 53:
        raise AssertionError("M16C population drift")

    manifest = json.loads(M16A_MANIFEST.read_text())
    m16a = pd.read_csv(M16A_LEDGER)
    m16b = pd.read_csv(M16B_LEDGER)
    if set(dev["mmsi"].astype(int)) != set(manifest["allowed_mmsi"]):
        raise AssertionError("M16C development population differs from M16A freeze")
    if set(dev["mmsi"].astype(int)) & set(manifest["blocked_old_final_mmsi"]):
        raise AssertionError("M16C old-final contamination")
    if len(m16b) != 386 or m16b["mmsi"].nunique() != 386:
        raise AssertionError("M16C M16B ledger cardinality drift")

    sem_cols = [
        "mmsi", "canonical_destination", "canonical_unlocode", "canonical_port_name",
        "canonical_country", "canonical_lat", "canonical_lon", "resolution_method",
        "resolution_confidence", "is_resolved_port", "is_non_specific",
    ]
    x = dev.merge(m16b[sem_cols], on="mmsi", how="left", validate="one_to_one")
    fold_map = dict(zip(m16a["mmsi"].astype(int), m16a["m16a_outer_fold"].astype(int)))
    x["m16a_outer_fold"] = x["mmsi"].astype(int).map(fold_map)
    if x["m16a_outer_fold"].isna().any():
        raise AssertionError("M16C missing M16A outer fold")
    x = add_m16c_features(x).sort_values("mmsi", kind="mergesort").reset_index(drop=True)

    ledger_parts: list[pd.DataFrame] = []
    search_parts: list[pd.DataFrame] = []
    selection_rows: list[dict] = []
    fold_metric_rows: list[dict] = []

    for outer_fold in sorted(x["m16a_outer_fold"].unique()):
        tr = x.loc[x["m16a_outer_fold"].ne(outer_fold)].copy()
        va = x.loc[x["m16a_outer_fold"].eq(outer_fold)].copy()
        assert_development_only(tr, final["mmsi"])
        assert_development_only(va, final["mmsi"])
        if set(tr["mmsi"]) & set(va["mmsi"]):
            raise AssertionError("M16C outer train/validation overlap")

        best, search = select_config_inner_cv(tr, int(outer_fold))
        search["selected"] = search["config_id"].eq(best.config_id)
        search_parts.append(search)

        pred = predict_hierarchical_prior(tr, va, best)
        fold_metrics = extended_metrics(va["target_tte_h"], pred["prediction_h"])
        fold_metric_rows.append({"outer_fold": int(outer_fold), "model": "m16c_hierarchical_prior", **fold_metrics})
        selection_rows.append({
            "outer_fold": int(outer_fold),
            "outer_train_n": int(len(tr)),
            "outer_valid_n": int(len(va)),
            "selected_config_id": best.config_id,
            "hierarchy": best.hierarchy,
            "min_support": int(best.min_support),
            "shrinkage_alpha": float(best.shrinkage_alpha),
            "clip_quantile": best.clip_quantile,
            "selected_on_inner_only": True,
            "inner_folds": M16C_INNER_FOLDS,
            "inner_best_mae_h": float(search.iloc[0]["mae_h"]),
            "inner_best_p90_ae_h": float(search.iloc[0]["p90_ae_h"]),
        })

        out = va[[
            "mmsi", "split", "reference_eta_status", "destination_norm", "canonical_destination",
            "ship_type_cat", "distance_to_destination_km", "distance_band", "movement_bucket",
            "target_tte_h", "m16a_outer_fold",
        ]].copy()
        out["pred_m16c_hierarchical_prior_h"] = pred["prediction_h"].to_numpy(float)
        out["m16c_deepest_level"] = pred["deepest_level"].to_numpy(int)
        out["m16c_deepest_level_name"] = pred["deepest_level_name"].astype(str).to_numpy()
        out["m16c_deepest_support"] = pred["deepest_support"].to_numpy(float)
        out["m16c_deepest_weight"] = pred["deepest_weight"].to_numpy(float)
        out["m16c_fallback_count"] = pred["fallback_count"].to_numpy(int)
        out["m16c_selected_config_id"] = best.config_id
        ledger_parts.append(out)

    ledger = pd.concat(ledger_parts, ignore_index=True).sort_values("mmsi", kind="mergesort").reset_index(drop=True)
    if len(ledger) != 386 or ledger["mmsi"].nunique() != 386:
        raise AssertionError("M16C OOF ledger must have exactly one row per development MMSI")
    if set(ledger["mmsi"].astype(int)) & set(manifest["blocked_old_final_mmsi"]):
        raise AssertionError("M16C ledger contains old-final MMSI")

    raw_map = dict(zip(m16a["mmsi"].astype(int), m16a["pred_train_destination_median_h"].astype(float)))
    cat_map = dict(zip(m16a["mmsi"].astype(int), m16a["pred_catboost_current_snapshot_h"].astype(float)))
    canon_map = dict(zip(m16b["mmsi"].astype(int), m16b["pred_train_canonical_destination_median_h"].astype(float)))
    ledger["pred_m16a_raw_destination_median_h"] = ledger["mmsi"].astype(int).map(raw_map)
    ledger["pred_m16a_catboost_current_snapshot_h"] = ledger["mmsi"].astype(int).map(cat_map)
    ledger["pred_m16b_naive_canonical_median_h"] = ledger["mmsi"].astype(int).map(canon_map)

    pred_cols = {
        "m16a_raw_destination_median": "pred_m16a_raw_destination_median_h",
        "m16a_catboost_current_snapshot": "pred_m16a_catboost_current_snapshot_h",
        "m16b_naive_canonical_median": "pred_m16b_naive_canonical_median_h",
        "m16c_hierarchical_prior": "pred_m16c_hierarchical_prior_h",
    }
    metric_rows: list[dict] = []
    for model, col in pred_cols.items():
        metric_rows.append({"model": model, **extended_metrics(ledger["target_tte_h"], ledger[col])})
    metrics = pd.DataFrame(metric_rows).sort_values(["mae_h", "p90_ae_h", "model"], kind="mergesort").reset_index(drop=True)

    # Add comparator fold metrics on the exact same outer folds for auditability.
    for outer_fold, g in ledger.groupby("m16a_outer_fold", sort=True):
        for model, col in pred_cols.items():
            if model == "m16c_hierarchical_prior":
                continue
            fold_metric_rows.append({
                "outer_fold": int(outer_fold), "model": model,
                **extended_metrics(g["target_tte_h"], g[col]),
            })
    fold_metrics = pd.DataFrame(fold_metric_rows).sort_values(["outer_fold", "model"], kind="mergesort").reset_index(drop=True)
    selections = pd.DataFrame(selection_rows).sort_values("outer_fold").reset_index(drop=True)
    search_all = pd.concat(search_parts, ignore_index=True).sort_values(["outer_fold", "mae_h", "config_id"], kind="mergesort").reset_index(drop=True)

    fallback = (
        ledger.groupby(["m16c_selected_config_id", "m16c_deepest_level", "m16c_deepest_level_name"], dropna=False)
        .agg(n=("mmsi", "size"), median_support=("m16c_deepest_support", "median"), mean_weight=("m16c_deepest_weight", "mean"))
        .reset_index()
        .sort_values(["m16c_selected_config_id", "m16c_deepest_level"])
    )

    ledger.to_csv(output_dir / "m16c_oof_predictions.csv", index=False, float_format="%.12g")
    search_all.to_csv(output_dir / "m16c_inner_search.csv", index=False, float_format="%.12g")
    selections.to_csv(output_dir / "m16c_outer_selected_configs.csv", index=False, float_format="%.12g")
    metrics.to_csv(output_dir / "m16c_model_metrics.csv", index=False, float_format="%.12g")
    fold_metrics.to_csv(output_dir / "m16c_fold_metrics.csv", index=False, float_format="%.12g")
    fallback.to_csv(output_dir / "m16c_fallback_summary.csv", index=False, float_format="%.12g")

    mm = {r["model"]: r for r in metrics.to_dict(orient="records")}
    m16c_m = mm["m16c_hierarchical_prior"]
    raw_m = mm["m16a_raw_destination_median"]
    cat_m = mm["m16a_catboost_current_snapshot"]
    canon_m = mm["m16b_naive_canonical_median"]
    mae_gain_vs_raw = float(raw_m["mae_h"] - m16c_m["mae_h"])
    mae_gain_vs_cat = float(cat_m["mae_h"] - m16c_m["mae_h"])
    mae_gain_vs_canon = float(canon_m["mae_h"] - m16c_m["mae_h"])
    fold_pivot = fold_metrics.pivot(index="outer_fold", columns="model", values="mae_h")
    fold_wins_vs_raw = int((fold_pivot["m16c_hierarchical_prior"] < fold_pivot["m16a_raw_destination_median"]).sum())
    fold_wins_vs_cat = int((fold_pivot["m16c_hierarchical_prior"] < fold_pivot["m16a_catboost_current_snapshot"]).sum())

    # Fixed, untuned 50/50 blend is a complementarity diagnostic only.  It is
    # not a promoted model; M16G is responsible for fold-safe ensemble fitting.
    equal_blend = 0.5 * ledger["pred_m16c_hierarchical_prior_h"].to_numpy(float) + 0.5 * ledger["pred_m16a_raw_destination_median_h"].to_numpy(float)
    equal_blend_metrics = extended_metrics(ledger["target_tte_h"], equal_blend)
    residual_raw = ledger["target_tte_h"].to_numpy(float) - ledger["pred_m16a_raw_destination_median_h"].to_numpy(float)
    residual_m16c = ledger["target_tte_h"].to_numpy(float) - ledger["pred_m16c_hierarchical_prior_h"].to_numpy(float)
    residual_corr = float(np.corrcoef(residual_raw, residual_m16c)[0, 1])

    standalone_promoted = bool(mae_gain_vs_raw > 0 and fold_wins_vs_raw >= 3 and m16c_m["p90_ae_h"] <= raw_m["p90_ae_h"])
    gate = "PASS_STANDALONE" if standalone_promoted else ("PASS_COMPONENT_NOT_STANDALONE" if mae_gain_vs_raw > 0 else "NO_PROMOTION")

    summary = {
        "milestone": "M16C",
        "status": "FOLD_SAFE_HIERARCHICAL_PRIOR_BUILT",
        "gate": gate,
        "version": M16C_VERSION,
        "development_rows": 386,
        "blocked_old_final_rows": 53,
        "old_final_used_for_selection": False,
        "outer_folds": 5,
        "inner_folds": M16C_INNER_FOLDS,
        "inner_salt": M16C_INNER_SALT,
        "selection_rule": "minimum pooled inner-CV MAE; P90/MedAE/config_id deterministic tie-break",
        "candidate_hierarchies": M16C_HIERARCHIES,
        "parameter_profiles": list(M16C_PARAMETER_PROFILES),
        "clip_quantiles": list(M16C_CLIP_QUANTILES),
        "robust_estimator": "group median shrunk toward hierarchical parent with w=n/(n+alpha)",
        "clip_policy": "optional train-only target quantile guardrail selected by inner CV",
        "m16a_raw_destination_median_oof": raw_m,
        "m16a_catboost_current_snapshot_oof": cat_m,
        "m16b_naive_canonical_median_oof": canon_m,
        "m16c_hierarchical_prior_oof": m16c_m,
        "m16c_mae_gain_h_vs_m16a_raw_destination": mae_gain_vs_raw,
        "m16c_mae_gain_h_vs_m16a_catboost": mae_gain_vs_cat,
        "m16c_mae_gain_h_vs_m16b_naive_canonical": mae_gain_vs_canon,
        "fold_wins_vs_m16a_raw_destination": fold_wins_vs_raw,
        "fold_wins_vs_m16a_catboost": fold_wins_vs_cat,
        "standalone_promoted": standalone_promoted,
        "standalone_promotion_rule": "positive pooled MAE gain + >=3/5 fold wins vs raw destination + no P90 regression",
        "equal_blend_50_50_diagnostic": {
            **equal_blend_metrics,
            "weight_m16c": 0.5,
            "weight_raw_destination": 0.5,
            "weight_selected_or_tuned": False,
            "purpose": "complementarity diagnostic only; M16G owns ensemble selection",
        },
        "residual_correlation_vs_raw_destination": residual_corr,
        "selected_configs_by_outer_fold": {
            str(int(r.outer_fold)): r.selected_config_id for r in selections.itertuples()
        },
        "final_test_used_for_selection": False,
        "claim": "Nested development-only OOF result; not a new untouched final-test result.",
    }
    (output_dir / "M16C_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Diagnostic plot: use exactly the frozen development OOF ledger.
    p = metrics.set_index("model").loc[[
        "m16a_raw_destination_median",
        "m16a_catboost_current_snapshot",
        "m16b_naive_canonical_median",
        "m16c_hierarchical_prior",
    ]]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    xpos = np.arange(len(p))
    ax.bar(xpos - 0.25, p["mae_h"], width=0.25, label="MAE")
    ax.bar(xpos, p["medae_h"], width=0.25, label="MedAE")
    ax.bar(xpos + 0.25, p["p90_ae_h"], width=0.25, label="P90 AE")
    ax.set_xticks(xpos)
    ax.set_xticklabels(["M16A raw dest", "M16A CatBoost", "M16B naive canon", "M16C hierarchy"], rotation=15, ha="right")
    ax.set_ylabel("Hours")
    ax.set_title("M16C development-only OOF comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "m16c_prior_comparison.png", dpi=160)
    plt.close(fig)

    freeze_files = [
        "m16c_oof_predictions.csv", "m16c_inner_search.csv", "m16c_outer_selected_configs.csv",
        "m16c_model_metrics.csv", "m16c_fold_metrics.csv", "m16c_fallback_summary.csv",
        "M16C_SUMMARY.json", "m16c_prior_comparison.png",
    ]
    freeze = {
        "milestone": "M16C",
        "status": "FOLD_SAFE_HIERARCHICAL_PRIOR_FROZEN",
        "version": M16C_VERSION,
        "development_population": 386,
        "blocked_old_final_population": 53,
        "selection_population_rule": "M10 train+calibration only; 53-row old final hard-blocked",
        "outer_fold_source": "M16A frozen outer folds",
        "inner_selection_only": True,
        "final_test_used_for_selection": False,
        "candidate_config_count": len(candidate_configs()),
        "primary_metric": "mae_h",
        "gate": gate,
        "artifact_sha256": {name: sha(output_dir / name) for name in freeze_files},
    }
    (output_dir / "M16C_HIERARCHICAL_PRIOR_FREEZE.json").write_text(json.dumps(freeze, indent=2) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=R)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
