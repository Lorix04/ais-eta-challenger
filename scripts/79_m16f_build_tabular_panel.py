#!/usr/bin/env python3
"""Build M16F nested development-only tabular challenger panel.

Each outer fold is evaluated in a fresh worker process. This keeps native ML
runtime state bounded and makes the build stable on small CI/assessment hosts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m16a import assert_development_only, extended_metrics
from ais_eta.m16f import (
    M16F_CATEGORICAL_FEATURES, M16F_FEATURES, M16F_INNER_FOLDS, M16F_INNER_SALT,
    M16F_NUMERIC_FEATURES, M16F_VERSION, candidate_configs, family_names, fit_predict,
    select_family_config_inner_cv,
)

REF = ROOT / "data/derived/m10_reference_rows.pkl.gz"
R = ROOT / "reports"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_development() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_pickle(REF)
    final = raw.loc[raw["split"].eq("final_test")].copy()
    dev = raw.loc[raw["split"].isin(["train", "calibration"])].copy()
    if len(dev) != 386 or len(final) != 53:
        raise AssertionError("unexpected M16F dev/final cardinality")
    assert_development_only(dev, final["mmsi"])

    b = pd.read_csv(R / "m16b_destination_resolutions.csv")
    e = pd.read_csv(R / "m16e_physics_feature_audit.csv")
    a = pd.read_csv(R / "m16a_oof_predictions.csv")
    d = pd.read_csv(R / "m16d_oof_predictions.csv")
    bcols = [
        "mmsi", "canonical_destination", "canonical_lat", "canonical_lon", "resolution_method",
        "resolution_confidence", "is_resolved_port", "is_non_specific", "is_route_expression",
    ]
    ecols = [
        "mmsi", "physics_distance_gc_nm", "physics_bearing_to_port_deg", "physics_course_alignment_deg",
        "physics_recent_speed_max_kn", "physics_local_sinuosity_180m", "physics_local_sinuosity_360m",
        "physics_eligible",
    ]
    dev = dev.merge(b[bcols], on="mmsi", how="left")
    dev = dev.merge(e[ecols], on="mmsi", how="left")
    dev = dev.merge(a[["mmsi", "m16a_outer_fold", "pred_train_destination_median_h", "pred_catboost_current_snapshot_h"]], on="mmsi", how="left")
    dev = dev.merge(d[["mmsi", "pred_m16d_route_analogue_h"]], on="mmsi", how="left")
    dev = dev.sort_values("mmsi", kind="mergesort").reset_index(drop=True)
    if dev["m16a_outer_fold"].isna().any():
        raise AssertionError("M16F missing frozen M16A outer fold")
    missing = set(M16F_FEATURES) - set(dev.columns)
    if missing:
        raise AssertionError(f"M16F representation missing columns: {sorted(missing)}")
    return dev, final


def run_worker(outer: int, worker_dir: Path) -> int:
    worker_dir.mkdir(parents=True, exist_ok=True)
    dev, final = prepare_development()
    configs = candidate_configs()
    tr = dev.loc[dev["m16a_outer_fold"].ne(outer)].copy()
    va = dev.loc[dev["m16a_outer_fold"].eq(outer)].copy()
    assert_development_only(tr, final["mmsi"])
    assert_development_only(va, final["mmsi"])
    if set(tr["mmsi"]) & set(va["mmsi"]):
        raise AssertionError("M16F outer train/valid overlap")

    pred_frame = va[["mmsi", "target_tte_h", "m16a_outer_fold"]].copy()
    searches = []
    selected_rows = []
    fold_rows = []
    for family in family_names(configs):
        best, search = select_family_config_inner_cv(tr, outer, family, configs=configs)
        p = fit_predict(tr, va, best)
        pred_frame[f"pred_m16f_{family}_h"] = p
        searches.append(search)
        top = search.iloc[0]
        selected_rows.append({
            "outer_fold": outer, "family": family, "selected_config_id": best.config_id,
            "inner_best_mae_h": float(top["mae_h"]), "inner_best_p90_ae_h": float(top["p90_ae_h"]),
            "inner_best_medae_h": float(top["medae_h"]), "selected_on_inner_only": True,
        })
        fold_rows.append({"outer_fold": outer, "model": f"m16f_{family}", **extended_metrics(va["target_tte_h"], p)})
        print(f"M16F worker outer={outer} family={family} selected={best.config_id}", flush=True)

    pred_frame.to_csv(worker_dir / f"outer_{outer}_pred.csv", index=False, float_format="%.12g")
    pd.concat(searches, ignore_index=True).to_csv(worker_dir / f"outer_{outer}_search.csv", index=False, float_format="%.12g")
    pd.DataFrame(selected_rows).to_csv(worker_dir / f"outer_{outer}_selected.csv", index=False, float_format="%.12g")
    pd.DataFrame(fold_rows).to_csv(worker_dir / f"outer_{outer}_fold_metrics.csv", index=False, float_format="%.12g")
    return 0


def _run_workers(tmp: Path) -> None:
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
    for outer in range(5):
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker-fold", str(outer), "--worker-dir", str(tmp)],
            cwd=ROOT, env=env, check=True,
        )


def run(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    dev, final = prepare_development()
    configs = candidate_configs()
    families = family_names(configs)

    manifest = {
        "milestone": "M16F", "version": M16F_VERSION, "status": "TABULAR_PANEL_PREDECLARED",
        "development_population": 386, "blocked_old_final_population": 53,
        "outer_fold_source": "M16A frozen balanced hash folds",
        "inner_folds": M16F_INNER_FOLDS, "inner_salt": M16F_INNER_SALT,
        "family_names": families, "candidate_config_count": len(configs),
        "candidate_configs": [{"family": c.family, "config_id": c.config_id, "params": c.params} for c in configs],
        "feature_policy": "target-free common representation; excludes all M16C/D/E expert predictions and reference_eta_status",
        "numeric_features": M16F_NUMERIC_FEATURES, "categorical_features": M16F_CATEGORICAL_FEATURES,
        "primary_metric": "pooled outer-OOF MAE hours; family config selected by inner OOF MAE only",
        "final_test_used_for_selection": False,
    }
    (output_dir / "M16F_FEATURE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")

    tmp = output_dir / ".m16f_workers"
    _run_workers(tmp)
    worker_pred = pd.concat([pd.read_csv(tmp / f"outer_{o}_pred.csv") for o in range(5)], ignore_index=True)
    inner = pd.concat([pd.read_csv(tmp / f"outer_{o}_search.csv") for o in range(5)], ignore_index=True)
    selected = pd.concat([pd.read_csv(tmp / f"outer_{o}_selected.csv") for o in range(5)], ignore_index=True)
    folds = pd.concat([pd.read_csv(tmp / f"outer_{o}_fold_metrics.csv") for o in range(5)], ignore_index=True)
    shutil.rmtree(tmp)

    ledger = dev[[
        "mmsi", "split", "reference_eta_status", "destination_norm", "canonical_destination",
        "target_tte_h", "m16a_outer_fold",
    ]].merge(worker_pred.drop(columns=["target_tte_h", "m16a_outer_fold"]), on="mmsi", how="left")
    for family in families:
        if not np.isfinite(ledger[f"pred_m16f_{family}_h"]).all():
            raise AssertionError(f"M16F nonfinite/missing OOF prediction family={family}")
    ledger["pred_m16a_raw_destination_median_h"] = dev["pred_train_destination_median_h"].to_numpy(float)
    ledger["pred_m16a_catboost_current_snapshot_h"] = dev["pred_catboost_current_snapshot_h"].to_numpy(float)
    ledger["pred_m16d_route_analogue_h"] = dev["pred_m16d_route_analogue_h"].to_numpy(float)
    if set(ledger["mmsi"].astype(int)) & set(final["mmsi"].astype(int)):
        raise AssertionError("M16F ledger contains old-final MMSI")

    metric_rows = []
    for family in families:
        metric_rows.append({"model": f"m16f_{family}", "source": "M16F nested OOF", **extended_metrics(ledger["target_tte_h"], ledger[f"pred_m16f_{family}_h"])})
    for name, col in [
        ("m16a_raw_destination_median", "pred_m16a_raw_destination_median_h"),
        ("m16a_catboost_current_snapshot", "pred_m16a_catboost_current_snapshot_h"),
        ("m16d_route_analogue", "pred_m16d_route_analogue_h"),
    ]:
        metric_rows.append({"model": name, "source": "frozen prior milestone", **extended_metrics(ledger["target_tte_h"], ledger[col])})
    metrics = pd.DataFrame(metric_rows).sort_values(["mae_h", "p90_ae_h", "model"], kind="mergesort").reset_index(drop=True)
    folds = folds.sort_values(["outer_fold", "model"], kind="mergesort").reset_index(drop=True)
    inner = inner.sort_values(["outer_fold", "family", "mae_h", "config_id"], kind="mergesort").reset_index(drop=True)
    selected = selected.sort_values(["outer_fold", "family"], kind="mergesort").reset_index(drop=True)

    m16f_only = metrics.loc[metrics["source"].eq("M16F nested OOF")].copy()
    best_row = m16f_only.iloc[0]
    best_family = str(best_row["model"]).removeprefix("m16f_")
    best_col = f"pred_m16f_{best_family}_h"
    raw_mae = float(metrics.loc[metrics["model"].eq("m16a_raw_destination_median"), "mae_h"].iloc[0])
    cat_mae = float(metrics.loc[metrics["model"].eq("m16a_catboost_current_snapshot"), "mae_h"].iloc[0])
    route_mae = float(metrics.loc[metrics["model"].eq("m16d_route_analogue"), "mae_h"].iloc[0])
    best_mae = float(best_row["mae_h"])

    fold_compare_rows = []
    for fold, g in ledger.groupby("m16a_outer_fold"):
        best_m = extended_metrics(g["target_tte_h"], g[best_col])
        cat_m = extended_metrics(g["target_tte_h"], g["pred_m16a_catboost_current_snapshot_h"])
        route_m = extended_metrics(g["target_tte_h"], g["pred_m16d_route_analogue_h"])
        fold_compare_rows.append({
            "outer_fold": int(fold), "best_family": best_family,
            "best_mae_h": best_m["mae_h"], "m16a_catboost_mae_h": cat_m["mae_h"], "m16d_route_mae_h": route_m["mae_h"],
            "gain_vs_m16a_catboost_h": cat_m["mae_h"] - best_m["mae_h"], "gain_vs_m16d_route_h": route_m["mae_h"] - best_m["mae_h"],
        })
    fold_compare = pd.DataFrame(fold_compare_rows)
    wins_vs_cat = int((fold_compare["gain_vs_m16a_catboost_h"] > 0).sum())
    wins_vs_route = int((fold_compare["gain_vs_m16d_route_h"] > 0).sum())
    gate = "PASS_TABULAR_COMPLEMENT_FOR_M16G" if (best_mae < cat_mae and wins_vs_cat >= 3) else "NO_STABLE_TABULAR_GAIN_KEEP_FROZEN_BASELINES"

    summary = {
        "milestone": "M16F", "version": M16F_VERSION, "gate": gate,
        "development_rows": 386, "blocked_old_final_rows": 53, "families": families,
        "candidate_config_count": len(configs), "best_family": best_family, "best_model": f"m16f_{best_family}",
        "best_mae_h": best_mae, "best_medae_h": float(best_row["medae_h"]), "best_p90_ae_h": float(best_row["p90_ae_h"]),
        "m16a_raw_destination_mae_h": raw_mae, "m16a_current_catboost_mae_h": cat_mae, "m16d_route_mae_h": route_mae,
        "gain_h_vs_m16a_current_catboost": cat_mae - best_mae, "gain_h_vs_m16a_raw_destination": raw_mae - best_mae,
        "gain_h_vs_m16d_route": route_mae - best_mae, "fold_wins_vs_m16a_current_catboost": wins_vs_cat,
        "fold_wins_vs_m16d_route": wins_vs_route, "final_test_used_for_selection": False,
        "expert_predictions_used_as_features": False,
        "claim": "Development-only nested OOF tabular family comparison; old final remains blocked and no frozen submission is reopened.",
    }

    ledger.to_csv(output_dir / "m16f_oof_predictions.csv", index=False, float_format="%.12g")
    metrics.to_csv(output_dir / "m16f_model_metrics.csv", index=False, float_format="%.12g")
    folds.to_csv(output_dir / "m16f_fold_metrics.csv", index=False, float_format="%.12g")
    inner.to_csv(output_dir / "m16f_inner_search.csv", index=False, float_format="%.12g")
    selected.to_csv(output_dir / "m16f_outer_selected_configs.csv", index=False, float_format="%.12g")
    fold_compare.to_csv(output_dir / "m16f_best_fold_comparison.csv", index=False, float_format="%.12g")
    (output_dir / "M16F_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    plot = metrics.sort_values("mae_h", ascending=True)
    ax.barh(plot["model"], plot["mae_h"])
    ax.set_xlabel("Development-only pooled OOF MAE (hours)")
    ax.set_title("M16F tabular challenger panel vs frozen comparators")
    ax.invert_yaxis()
    for i, v in enumerate(plot["mae_h"]):
        ax.text(v + 0.8, i, f"{v:.2f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "m16f_tabular_panel_comparison.png", dpi=150, metadata={"Software": "AIS ETA M16F"})
    plt.close(fig)

    report = f"""# M16F — Tabular Challenger Panel\n\n## Decision\n\n**{gate}**\n\nM16F compares five predeclared tabular families on the frozen M16A outer folds. Hyperparameters are selected only by {M16F_INNER_FOLDS}-fold inner CV on each outer-train partition. The 53 already-observed M10 final MMSIs are hard-blocked.\n\n## Best nested-OOF tabular family\n\n- best family: **{best_family}**\n- pooled MAE: **{best_mae:.2f} h**\n- MedAE: **{float(best_row['medae_h']):.2f} h**\n- P90 absolute error: **{float(best_row['p90_ae_h']):.2f} h**\n- gain vs frozen M16A current CatBoost: **{cat_mae - best_mae:.2f} h**\n- fold wins vs frozen M16A current CatBoost: **{wins_vs_cat}/5**\n- gain vs M16D route analogue: **{route_mae - best_mae:.2f} h**\n- fold wins vs M16D route analogue: **{wins_vs_route}/5**\n\n## Interpretation\n\nThe panel tests whether a different tabular inductive bias adds stable development-only signal once target-free destination/physics-quality representation is available. M16F does **not** use M16C/M16D/M16E expert predictions as features; those remain reserved for M16G stacking. A family is retained only if it is stable against the frozen M16A current-state CatBoost, not merely because of one pooled score.\n\n## Scope\n\nThis is a post-freeze development experiment, not a reopened final result. No claim about the old 53-row final is made.\n"""
    (output_dir / "M16F_REPORT.md").write_text(report)

    frozen_outputs = [
        "M16F_FEATURE_MANIFEST.json", "m16f_oof_predictions.csv", "m16f_model_metrics.csv", "m16f_fold_metrics.csv",
        "m16f_inner_search.csv", "m16f_outer_selected_configs.csv", "m16f_best_fold_comparison.csv",
        "m16f_tabular_panel_comparison.png", "M16F_SUMMARY.json", "M16F_REPORT.md",
    ]
    freeze = {
        "milestone": "M16F", "status": "TABULAR_CHALLENGER_PANEL_FROZEN", "version": M16F_VERSION,
        "development_population": 386, "blocked_old_final_population": 53,
        "selection_population_rule": "M10 train+calibration only; old M10 final hard-blocked",
        "outer_fold_source": "M16A frozen outer folds", "inner_selection_only": True,
        "final_test_used_for_selection": False, "expert_predictions_used_as_features": False,
        "candidate_config_count": len(configs), "families": families, "gate": gate, "best_family": best_family,
        "primary_metric": "pooled outer-OOF MAE hours",
        "artifact_sha256": {name: sha(output_dir / name) for name in frozen_outputs},
    }
    (output_dir / "M16F_TABULAR_PANEL_FREEZE.json").write_text(json.dumps(freeze, indent=2) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=R)
    parser.add_argument("--worker-fold", type=int)
    parser.add_argument("--worker-dir", type=Path)
    args = parser.parse_args()
    if args.worker_fold is not None:
        if args.worker_dir is None:
            parser.error("--worker-dir required with --worker-fold")
        return run_worker(args.worker_fold, args.worker_dir)
    print(json.dumps(run(args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
