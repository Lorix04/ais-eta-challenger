#!/usr/bin/env python3
"""Build M16G leakage-safe Mixture of Experts / OOF stacking benchmark.

Each outer fold is executed independently.  Inside the current outer-train,
all four base experts are regenerated as inner-OOF predictions before fitting
or selecting a meta strategy.  Outer-validation predictions come only from the
already frozen M16C/D/E/F OOF ledgers for that same outer fold.
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

from ais_eta.m16a import assert_development_only, balanced_hash_folds, extended_metrics
from ais_eta.m16c import add_m16c_features, candidate_configs as c_configs, predict_hierarchical_prior
from ais_eta.m16d import candidate_configs as d_configs, predict_route_analogue
from ais_eta.m16e import candidate_configs as e_configs, predict_physics_expert
from ais_eta.m16f import candidate_configs as f_configs, fit_predict
from ais_eta.m16g import (
    M16G_EXPERT_NAMES, M16G_INNER_FOLDS, M16G_INNER_SALT, M16G_META_CANDIDATES,
    M16G_VERSION, build_gating_features, fit_predict_meta, promotion_gate,
    score_candidates_inner_cv,
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
        raise AssertionError("unexpected M16G dev/final cardinality")
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
    dev = add_m16c_features(dev)
    if dev["m16a_outer_fold"].isna().any():
        raise AssertionError("M16G missing frozen M16A outer fold")
    return dev, final


def _config_maps():
    cm = {c.config_id: c for c in c_configs()}
    dm = {c.config_id: c for c in d_configs()}
    em = {c.config_id: c for c in e_configs()}
    fm = {c.config_id: c for c in f_configs()}
    return cm, dm, em, fm


def _selected_configs(outer: int):
    cm, dm, em, fm = _config_maps()
    sc = pd.read_csv(R / "m16c_outer_selected_configs.csv")
    sd = pd.read_csv(R / "m16d_outer_selected_configs.csv")
    se = pd.read_csv(R / "m16e_outer_selected_configs.csv")
    sf = pd.read_csv(R / "m16f_outer_selected_configs.csv")
    cid = str(sc.loc[sc["outer_fold"].eq(outer), "selected_config_id"].iloc[0])
    did = str(sd.loc[sd["outer_fold"].eq(outer), "selected_config_id"].iloc[0])
    eid = str(se.loc[se["outer_fold"].eq(outer), "selected_config_id"].iloc[0])
    fid = str(sf.loc[sf["outer_fold"].eq(outer) & sf["family"].eq("hist_gb"), "selected_config_id"].iloc[0])
    return cm[cid], dm[did], em[eid], fm[fid]


def _frozen_outer_tables(dev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    c = pd.read_csv(R / "m16c_oof_predictions.csv")
    d = pd.read_csv(R / "m16d_oof_predictions.csv")
    e = pd.read_csv(R / "m16e_oof_predictions.csv")
    f = pd.read_csv(R / "m16f_oof_predictions.csv")
    base = (
        c[["mmsi", "pred_m16c_hierarchical_prior_h"]]
        .merge(d[["mmsi", "pred_m16d_route_analogue_h"]], on="mmsi")
        .merge(e[["mmsi", "pred_m16e_hard_gate_route_fallback_h"]], on="mmsi")
        .merge(f[["mmsi", "pred_m16f_hist_gb_h"]], on="mmsi")
    )
    quality = (
        c[["mmsi", "m16c_deepest_support", "m16c_deepest_weight", "m16c_fallback_count"]]
        .merge(d[["mmsi", "m16d_neighbour_count", "m16d_nearest_distance", "m16d_median_neighbour_distance", "m16d_similarity_gap", "m16d_gate_used"]], on="mmsi")
        .merge(e[["mmsi", "physics_eligible", "resolution_confidence", "physics_distance_gc_nm", "physics_course_alignment_deg", "physics_recent_speed_max_kn", "m16e_effective_speed_kn", "m16e_distance_factor", "m16e_route_distance_proxy_nm"]], on="mmsi")
    )
    base = dev[["mmsi"]].merge(base, on="mmsi", how="left")
    quality = dev[["mmsi"]].merge(quality, on="mmsi", how="left")
    return base, quality


def _quality_from_inner(valid: pd.DataFrame, prior: pd.DataFrame, route: pd.DataFrame, physics: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "m16c_deepest_support": prior["deepest_support"].to_numpy(float),
        "m16c_deepest_weight": prior["deepest_weight"].to_numpy(float),
        "m16c_fallback_count": prior["fallback_count"].to_numpy(float),
        "m16d_neighbour_count": route["neighbour_count"].to_numpy(float),
        "m16d_nearest_distance": route["nearest_distance"].to_numpy(float),
        "m16d_median_neighbour_distance": route["median_neighbour_distance"].to_numpy(float),
        "m16d_similarity_gap": route["similarity_gap"].to_numpy(float),
        "m16d_gate_used": route["gate_used"].astype(str).to_numpy(),
        "physics_eligible": valid["physics_eligible"].astype(bool).to_numpy(),
        "resolution_confidence": pd.to_numeric(valid["resolution_confidence"], errors="coerce").to_numpy(float),
        "physics_distance_gc_nm": pd.to_numeric(valid["physics_distance_gc_nm"], errors="coerce").to_numpy(float),
        "physics_course_alignment_deg": pd.to_numeric(valid["physics_course_alignment_deg"], errors="coerce").to_numpy(float),
        "physics_recent_speed_max_kn": pd.to_numeric(valid["physics_recent_speed_max_kn"], errors="coerce").to_numpy(float),
        "m16e_effective_speed_kn": physics["effective_speed_kn"].to_numpy(float),
        "m16e_distance_factor": physics["distance_factor"].to_numpy(float),
        "m16e_route_distance_proxy_nm": physics["route_distance_proxy_nm"].to_numpy(float),
    })


def run_worker(outer: int, worker_dir: Path) -> int:
    worker_dir.mkdir(parents=True, exist_ok=True)
    dev, final = prepare_development()
    assert_development_only(dev, final["mmsi"])
    y = dev["target_tte_h"].to_numpy(float)
    mmsis = dev["mmsi"].to_numpy(int)
    folds = dev["m16a_outer_fold"].to_numpy(int)
    destinations = dev["canonical_destination"].fillna("UNKNOWN").astype(str).to_numpy()
    matrices = dict(np.load(R / "m16d_route_distance_matrices.npz"))
    cc, dc, ec, fc = _selected_configs(outer)
    outer_train = np.flatnonzero(folds != outer)
    outer_valid = np.flatnonzero(folds == outer)
    if set(mmsis[outer_train]) & set(mmsis[outer_valid]):
        raise AssertionError("M16G outer train/valid overlap")

    inner_map = balanced_hash_folds(mmsis[outer_train], n_splits=M16G_INNER_FOLDS, salt=f"{M16G_INNER_SALT}:outer={outer}")
    inner_fold = np.asarray([inner_map[int(x)] for x in mmsis[outer_train]], dtype=int)
    train_base = np.full((len(outer_train), len(M16G_EXPERT_NAMES)), np.nan, dtype=float)
    quality_parts: list[pd.DataFrame] = []
    quality_local_positions: list[np.ndarray] = []

    for inner in range(M16G_INNER_FOLDS):
        va_local = np.flatnonzero(inner_fold == inner)
        tr_local = np.flatnonzero(inner_fold != inner)
        va_idx = outer_train[va_local]
        tr_idx = outer_train[tr_local]
        tr = dev.iloc[tr_idx].copy()
        va = dev.iloc[va_idx].copy()

        prior = predict_hierarchical_prior(tr, va, cc)
        route = predict_route_analogue(
            train_indices=tr_idx, valid_indices=va_idx,
            distance_matrix=matrices[dc.representation], targets=y,
            destinations=destinations, config=dc,
        )
        physics = predict_physics_expert(tr, va, ec)
        tabular = fit_predict(tr, va, fc)
        route_pred = route["prediction_h"].to_numpy(float)
        physics_pred = physics["prediction_h"].to_numpy(float)
        eligible = va["physics_eligible"].astype(bool).to_numpy()
        physics_gate = np.where(eligible & np.isfinite(physics_pred), physics_pred, route_pred)
        train_base[va_local, :] = np.column_stack([
            prior["prediction_h"].to_numpy(float), route_pred, physics_gate, tabular,
        ])
        quality_parts.append(_quality_from_inner(va, prior, route, physics))
        quality_local_positions.append(va_local)

    if not np.isfinite(train_base).all():
        raise AssertionError("M16G inner OOF expert matrix contains non-finite values")
    train_quality = pd.DataFrame(index=np.arange(len(outer_train)))
    for pos, part in zip(quality_local_positions, quality_parts):
        part = part.reset_index(drop=True)
        for col in part.columns:
            train_quality.loc[pos, col] = part[col].to_numpy()
    train_features = build_gating_features(train_base, train_quality)

    frozen_base, frozen_quality = _frozen_outer_tables(dev)
    fb = frozen_base.iloc[outer_valid]
    valid_base = np.column_stack([
        fb["pred_m16c_hierarchical_prior_h"].to_numpy(float),
        fb["pred_m16d_route_analogue_h"].to_numpy(float),
        fb["pred_m16e_hard_gate_route_fallback_h"].to_numpy(float),
        fb["pred_m16f_hist_gb_h"].to_numpy(float),
    ])
    valid_quality = frozen_quality.iloc[outer_valid].drop(columns=["mmsi"]).reset_index(drop=True)
    valid_features = build_gating_features(valid_base, valid_quality)
    train_features, valid_features = train_features.align(valid_features, join="outer", axis=1, fill_value=0.0)

    search = score_candidates_inner_cv(train_base, y[outer_train], train_features, inner_fold)
    selected_candidate = str(search.iloc[0]["candidate_id"])
    selected = fit_predict_meta(
        selected_candidate, train_base, y[outer_train], train_features,
        valid_base, valid_features,
    )

    pred = dev.iloc[outer_valid][["mmsi", "target_tte_h", "m16a_outer_fold"]].copy().reset_index(drop=True)
    pred["pred_m16c_prior_h"] = valid_base[:, 0]
    pred["pred_m16d_route_h"] = valid_base[:, 1]
    pred["pred_m16e_physics_gate_h"] = valid_base[:, 2]
    pred["pred_m16f_tabular_h"] = valid_base[:, 3]
    pred["pred_m16g_selected_h"] = selected.prediction
    pred["m16g_selected_meta_id"] = selected_candidate
    pred["m16g_selected_expert"] = selected.selected_expert

    candidate_fold_rows = []
    for candidate in M16G_META_CANDIDATES:
        r = fit_predict_meta(candidate, train_base, y[outer_train], train_features, valid_base, valid_features)
        pred[f"pred_m16g_{candidate}_h"] = r.prediction
        candidate_fold_rows.append({"outer_fold": outer, "candidate_id": candidate, **extended_metrics(y[outer_valid], r.prediction)})
    candidate_fold_rows.append({"outer_fold": outer, "candidate_id": "selected_nested_mixture", **extended_metrics(y[outer_valid], selected.prediction)})
    for j, expert in enumerate(M16G_EXPERT_NAMES):
        candidate_fold_rows.append({"outer_fold": outer, "candidate_id": f"expert_{expert}", **extended_metrics(y[outer_valid], valid_base[:, j])})

    search.insert(0, "outer_fold", outer)
    search["selected"] = search["candidate_id"].eq(selected_candidate)
    selected_row = {
        "outer_fold": outer,
        "selected_meta_id": selected_candidate,
        "inner_best_mae_h": float(search.iloc[0]["mae_h"]),
        "inner_best_p90_ae_h": float(search.iloc[0]["p90_ae_h"]),
        "inner_best_medae_h": float(search.iloc[0]["medae_h"]),
        "outer_train_n": int(len(outer_train)),
        "outer_valid_n": int(len(outer_valid)),
        "inner_folds": M16G_INNER_FOLDS,
        "base_predictions_regenerated_inner_oof": True,
        "selected_on_inner_only": True,
    }

    pred.to_csv(worker_dir / f"outer_{outer}_pred.csv", index=False, float_format="%.12g")
    search.to_csv(worker_dir / f"outer_{outer}_search.csv", index=False, float_format="%.12g")
    pd.DataFrame([selected_row]).to_csv(worker_dir / f"outer_{outer}_selected.csv", index=False, float_format="%.12g")
    pd.DataFrame(candidate_fold_rows).to_csv(worker_dir / f"outer_{outer}_fold_metrics.csv", index=False, float_format="%.12g")
    print(f"M16G worker outer={outer} selected={selected_candidate} outer_mae={mean_abs(y[outer_valid], selected.prediction):.6f}", flush=True)
    return 0


def mean_abs(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, dtype=float) - np.asarray(p, dtype=float))))


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


def run(output_dir: Path, *, reuse_workers: Path | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    dev, final = prepare_development()
    if set(dev["mmsi"].astype(int)) & set(final["mmsi"].astype(int)):
        raise AssertionError("M16G development contains old-final MMSI")
    tmp = reuse_workers or (output_dir / ".m16g_workers")
    if reuse_workers is None:
        _run_workers(tmp)

    pred = pd.concat([pd.read_csv(tmp / f"outer_{o}_pred.csv") for o in range(5)], ignore_index=True)
    search = pd.concat([pd.read_csv(tmp / f"outer_{o}_search.csv") for o in range(5)], ignore_index=True)
    selected = pd.concat([pd.read_csv(tmp / f"outer_{o}_selected.csv") for o in range(5)], ignore_index=True)
    fold_metrics = pd.concat([pd.read_csv(tmp / f"outer_{o}_fold_metrics.csv") for o in range(5)], ignore_index=True)
    if reuse_workers is None:
        shutil.rmtree(tmp)

    pred = pred.sort_values("mmsi", kind="mergesort").reset_index(drop=True)
    if len(pred) != 386 or pred["mmsi"].nunique() != 386:
        raise AssertionError("M16G OOF ledger must contain exactly 386 unique development MMSIs")
    if set(pred["mmsi"].astype(int)) & set(final["mmsi"].astype(int)):
        raise AssertionError("M16G ledger contains old-final MMSI")

    y = pred["target_tte_h"].to_numpy(float)
    mixture = pred["pred_m16g_selected_h"].to_numpy(float)
    baseline = pred["pred_m16e_physics_gate_h"].to_numpy(float)
    route = pred["pred_m16d_route_h"].to_numpy(float)
    tab = pred["pred_m16f_tabular_h"].to_numpy(float)
    prior = pred["pred_m16c_prior_h"].to_numpy(float)
    mixture_m = extended_metrics(y, mixture)
    baseline_m = extended_metrics(y, baseline)
    route_m = extended_metrics(y, route)
    tab_m = extended_metrics(y, tab)
    prior_m = extended_metrics(y, prior)
    ae_mix = np.abs(y - mixture)
    ae_base = np.abs(y - baseline)
    delta = ae_base - ae_mix
    changed = np.abs(delta) > 1e-12
    paired_win_share = float(np.mean(delta > 1e-12))
    paired_tie_share = float(np.mean(~changed))
    changed_row_win_share = float(np.mean(delta[changed] > 0)) if np.any(changed) else 0.0

    fold_wins = 0
    fold_compare_rows = []
    for outer in range(5):
        m = pred["m16a_outer_fold"].astype(int).eq(outer).to_numpy()
        mix_mae = mean_abs(y[m], mixture[m])
        base_mae = mean_abs(y[m], baseline[m])
        fold_wins += int(mix_mae < base_mae)
        fold_compare_rows.append({
            "outer_fold": outer, "mixture_mae_h": mix_mae, "best_single_physics_gate_mae_h": base_mae,
            "gain_h": base_mae - mix_mae, "mixture_wins": bool(mix_mae < base_mae),
        })

    gate = promotion_gate(baseline_m, mixture_m, fold_wins=fold_wins, changed_row_win_share=changed_row_win_share)
    gain = float(baseline_m["mae_h"] - mixture_m["mae_h"])

    metric_rows = [
        {"model": "m16g_selected_nested_mixture", "source": "M16G nested OOF", **mixture_m},
        {"model": "m16e_physics_gate_route_fallback", "source": "frozen prior milestone", **baseline_m},
        {"model": "m16d_route_analogue", "source": "frozen prior milestone", **route_m},
        {"model": "m16f_hist_gb", "source": "frozen prior milestone", **tab_m},
        {"model": "m16c_hierarchical_prior", "source": "frozen prior milestone", **prior_m},
    ]
    for candidate in M16G_META_CANDIDATES:
        metric_rows.append({
            "model": f"m16g_{candidate}", "source": "M16G candidate outer OOF",
            **extended_metrics(y, pred[f"pred_m16g_{candidate}_h"]),
        })
    metrics = pd.DataFrame(metric_rows).sort_values(["mae_h", "p90_ae_h", "model"], kind="mergesort").reset_index(drop=True)

    usage = (
        pred.groupby(["m16g_selected_meta_id", "m16g_selected_expert"], dropna=False)
        .size().reset_index(name="rows")
        .sort_values(["m16g_selected_meta_id", "rows"], ascending=[True, False], kind="mergesort")
    )
    usage["share"] = usage["rows"] / len(pred)
    paired = pred[["mmsi", "m16a_outer_fold", "target_tte_h", "m16g_selected_meta_id", "m16g_selected_expert"]].copy()
    paired["ae_m16g_h"] = ae_mix
    paired["ae_best_single_h"] = ae_base
    paired["delta_ae_h"] = ae_base - ae_mix
    paired["m16g_wins"] = paired["delta_ae_h"] > 0

    selected = selected.sort_values("outer_fold", kind="mergesort").reset_index(drop=True)
    search = search.sort_values(["outer_fold", "mae_h", "p90_ae_h", "candidate_id"], kind="mergesort").reset_index(drop=True)
    fold_metrics = fold_metrics.sort_values(["outer_fold", "candidate_id"], kind="mergesort").reset_index(drop=True)
    fold_compare = pd.DataFrame(fold_compare_rows)

    pred.to_csv(output_dir / "m16g_oof_predictions.csv", index=False, float_format="%.12g")
    search.to_csv(output_dir / "m16g_inner_meta_search.csv", index=False, float_format="%.12g")
    selected.to_csv(output_dir / "m16g_outer_selected_meta.csv", index=False, float_format="%.12g")
    metrics.to_csv(output_dir / "m16g_candidate_metrics.csv", index=False, float_format="%.12g")
    fold_metrics.to_csv(output_dir / "m16g_fold_metrics.csv", index=False, float_format="%.12g")
    fold_compare.to_csv(output_dir / "m16g_fold_comparison.csv", index=False, float_format="%.12g")
    usage.to_csv(output_dir / "m16g_expert_usage.csv", index=False, float_format="%.12g")
    paired.to_csv(output_dir / "m16g_paired_error_deltas.csv", index=False, float_format="%.12g")

    fig, ax = plt.subplots(figsize=(11, 6))
    plot = metrics.loc[metrics["model"].isin([
        "m16g_selected_nested_mixture", "m16e_physics_gate_route_fallback",
        "m16d_route_analogue", "m16f_hist_gb", "m16c_hierarchical_prior",
        "m16g_median_blend", "m16g_nnls_convex",
    ])].copy().sort_values("mae_h")
    ax.barh(plot["model"], plot["mae_h"])
    ax.invert_yaxis(); ax.set_xlabel("OOF MAE (hours)"); ax.set_title("M16G mixture vs frozen experts")
    for i, v in enumerate(plot["mae_h"]): ax.text(float(v) + 1, i, f"{v:.1f}", va="center")
    fig.tight_layout(); fig.savefig(output_dir / "m16g_mixture_comparison.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    uc = pred["m16g_selected_expert"].value_counts().sort_values(ascending=False)
    ax.bar(uc.index.astype(str), uc.values)
    ax.set_ylabel("OOF rows"); ax.set_title("M16G selected expert / blend path usage")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout(); fig.savefig(output_dir / "m16g_expert_usage.png", dpi=160); plt.close(fig)

    summary = {
        "milestone": "M16G", "version": M16G_VERSION, "gate": gate,
        "development_rows": 386, "blocked_old_final_rows": 53,
        "final_test_used_for_selection": False,
        "outer_folds": 5, "inner_folds": M16G_INNER_FOLDS,
        "base_experts": list(M16G_EXPERT_NAMES), "meta_candidates": list(M16G_META_CANDIDATES),
        "base_predictions_for_meta_training": "regenerated inner-OOF inside each outer-train; no in-sample base prediction is used to fit meta learner",
        "outer_validation_base_predictions": "frozen M16C/D/E/F OOF predictions for the corresponding M16A outer fold",
        "target_derived_gating_features_used": False,
        "selected_meta_by_outer_fold": {str(int(r.outer_fold)): str(r.selected_meta_id) for r in selected.itertuples()},
        "mixture_mae_h": float(mixture_m["mae_h"]), "mixture_medae_h": float(mixture_m["medae_h"]), "mixture_p90_ae_h": float(mixture_m["p90_ae_h"]),
        "best_single_baseline": "m16e_physics_gate_route_fallback",
        "best_single_mae_h": float(baseline_m["mae_h"]), "gain_h_vs_best_single": gain,
        "fold_wins_vs_best_single": int(fold_wins), "paired_row_win_share_vs_best_single": paired_win_share, "paired_tie_share_vs_best_single": paired_tie_share, "changed_row_win_share_vs_best_single": changed_row_win_share,
        "claim": "Development-only nested OOF mixture evidence; old final remains blocked and M14 submission is not reopened.",
    }
    (output_dir / "M16G_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")

    report = f"""# M16G — Mixture of Experts / OOF Stacking\n\n## Decision\n\n**{gate}**\n\nM16G combines the frozen M16C hierarchical prior, M16D route analogue, M16E physics hard-gate and M16F HistGradientBoosting expert. For every outer fold, the four base experts are **regenerated as inner-OOF predictions using only that outer-train partition** before any meta learner is fitted or selected. Outer-validation expert predictions come from the already frozen prior-milestone OOF ledgers. The 53 old-final MMSIs remain hard-blocked.\n\n## Result\n\n- selected nested-mixture MAE: **{float(mixture_m['mae_h']):.2f} h**\n- MedAE: **{float(mixture_m['medae_h']):.2f} h**\n- P90 absolute error: **{float(mixture_m['p90_ae_h']):.2f} h**\n- best single operational expert (M16E physics-gate/route fallback): **{float(baseline_m['mae_h']):.2f} h MAE**\n- pooled gain vs best single: **{gain:.2f} h**\n- fold wins vs best single: **{fold_wins}/5**\n- paired row-level win share (all rows): **{paired_win_share*100:.1f}%**\n- exact-tie share (gate preserves best single): **{paired_tie_share*100:.1f}%**\n- win share among rows where M16G actually changes the prediction: **{changed_row_win_share*100:.1f}%**\n\n## Meta panel\n\nCompared: row-wise median, equal mean, non-negative normalized NNLS, RandomForest hard gate, ExtraTrees hard gate, HistGradientBoosting hard gate, and logistic soft gating. Candidate selection is performed only inside the current outer-train using the regenerated inner-OOF expert matrix.\n\n## Interpretation\n\nThe gate requires more than a pooled-number improvement: at least {1.0:.1f} h pooled MAE gain, at least 3/5 fold wins, and at least 50% paired wins among rows where the mixture actually changes the best-single prediction; exact ties are neutral. M16G remains development-only evidence and does not alter the official M14 frozen submission.\n"""
    (output_dir / "M16G_REPORT.md").write_text(report)

    artifacts = [
        "M16G_REPORT.md", "M16G_SUMMARY.json", "m16g_oof_predictions.csv", "m16g_inner_meta_search.csv",
        "m16g_outer_selected_meta.csv", "m16g_candidate_metrics.csv", "m16g_fold_metrics.csv",
        "m16g_fold_comparison.csv", "m16g_expert_usage.csv", "m16g_paired_error_deltas.csv",
        "m16g_mixture_comparison.png", "m16g_expert_usage.png",
    ]
    freeze = {
        "milestone": "M16G", "version": M16G_VERSION, "gate": gate,
        "development_population": 386, "blocked_old_final_population": 53,
        "selection_population_rule": "M10 train+calibration only; old M10 final hard-blocked",
        "outer_fold_source": "M16A frozen outer folds", "inner_selection_only": True,
        "final_test_used_for_selection": False,
        "base_predictions_regenerated_inner_oof": True,
        "target_derived_gating_features_used": False,
        "artifact_sha256": {name: sha(output_dir / name) for name in artifacts},
    }
    (output_dir / "M16G_MIXTURE_FREEZE.json").write_text(json.dumps(freeze, indent=2) + "\n")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=R)
    ap.add_argument("--worker-fold", type=int)
    ap.add_argument("--worker-dir", type=Path)
    ap.add_argument("--reuse-workers", type=Path)
    args = ap.parse_args()
    if args.worker_fold is not None:
        if args.worker_dir is None: raise SystemExit("--worker-dir required")
        return run_worker(int(args.worker_fold), args.worker_dir)
    summary = run(args.output_dir, reuse_workers=args.reuse_workers)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
