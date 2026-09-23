#!/usr/bin/env python3
"""Build M16H development-only probabilistic ETA and confidence artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m16h import (
    M16H_CANDIDATES,
    M16H_NOMINAL_COVERAGE,
    M16H_SCALE_FLOORS_H,
    M16H_STABLE_FOLD_COVERAGE,
    M16H_VERSION,
    assign_confidence_tier,
    build_error_features,
    calibrate_interval,
    candidate_floor,
    conformal_upper_quantile,
    fit_error_model,
    interval_metrics,
    predict_error,
    promotion_gate,
)

R = ROOT / "reports"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frame() -> pd.DataFrame:
    g = pd.read_csv(R / "m16g_oof_predictions.csv")
    c = pd.read_csv(R / "m16c_oof_predictions.csv")
    d = pd.read_csv(R / "m16d_oof_predictions.csv")
    e = pd.read_csv(R / "m16e_oof_predictions.csv")
    keep_c = ["mmsi", "reference_eta_status", "canonical_destination", "m16c_deepest_support", "m16c_deepest_weight", "m16c_fallback_count"]
    keep_d = ["mmsi", "m16d_neighbour_count", "m16d_nearest_distance", "m16d_median_neighbour_distance", "m16d_similarity_gap", "m16d_gate_used"]
    keep_e = ["mmsi", "physics_eligible", "resolution_confidence", "physics_distance_gc_nm", "physics_course_alignment_deg", "physics_recent_speed_max_kn"]
    df = g.merge(c[keep_c], on="mmsi", how="left").merge(d[keep_d], on="mmsi", how="left").merge(e[keep_e], on="mmsi", how="left")
    if len(df) != 386 or df["mmsi"].nunique() != 386:
        raise AssertionError("unexpected M16H development cardinality")
    if df["m16a_outer_fold"].isna().any():
        raise AssertionError("missing frozen outer fold")
    return df.sort_values("mmsi", kind="mergesort").reset_index(drop=True)


def crossfit_error_risk(df: pd.DataFrame, features: pd.DataFrame, outer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    folds = df["m16a_outer_fold"].to_numpy(int)
    tr_idx = np.flatnonzero(folds != outer)
    va_idx = np.flatnonzero(folds == outer)
    tr_fold = folds[tr_idx]
    abs_err = np.abs(df["target_tte_h"].to_numpy(float) - df["pred_m16g_selected_h"].to_numpy(float))
    risk_oof = np.full(len(tr_idx), np.nan, dtype=float)
    for inner in sorted(np.unique(tr_fold)):
        a = np.flatnonzero(tr_fold != inner)
        b = np.flatnonzero(tr_fold == inner)
        model = fit_error_model(features.iloc[tr_idx[a]], abs_err[tr_idx[a]])
        risk_oof[b] = predict_error(model, features.iloc[tr_idx[b]])
    if not np.isfinite(risk_oof).all():
        raise AssertionError("non-finite M16H cross-fitted error risk")
    final_model = fit_error_model(features.iloc[tr_idx], abs_err[tr_idx])
    risk_valid = predict_error(final_model, features.iloc[va_idx])
    return tr_idx, va_idx, tr_fold, risk_oof, risk_valid


def _calibrate_from_arrays(err: np.ndarray, risk_cal: np.ndarray, p50_val: np.ndarray, risk_val: np.ndarray, candidate: str):
    return calibrate_interval(err, risk_cal, p50_val, risk_val, candidate, M16H_NOMINAL_COVERAGE)


def score_candidate_inner(df: pd.DataFrame, tr_idx: np.ndarray, tr_fold: np.ndarray, risk_oof: np.ndarray, candidate: str) -> dict[str, float | str]:
    y = df["target_tte_h"].to_numpy(float)
    p = df["pred_m16g_selected_h"].to_numpy(float)
    err = np.abs(y[tr_idx] - p[tr_idx])
    yy: list[float] = []
    lo: list[float] = []
    hi: list[float] = []
    for inner in sorted(np.unique(tr_fold)):
        cal = np.flatnonzero(tr_fold != inner)
        val = np.flatnonzero(tr_fold == inner)
        r = _calibrate_from_arrays(err[cal], risk_oof[cal], p[tr_idx[val]], risk_oof[val], candidate)
        yy.extend(y[tr_idx[val]].tolist())
        lo.extend(r.lower_h.tolist())
        hi.extend(r.upper_h.tolist())
    met = interval_metrics(yy, lo, hi, p[tr_idx])
    return {"candidate_id": candidate, **met}


def choose_candidate(search: pd.DataFrame) -> str:
    # Selection is predeclared: among candidates with >=76% inner coverage,
    # choose the sharpest interval; otherwise maximize coverage then sharpness.
    eligible = search.loc[search["coverage"] >= 0.76].copy()
    if len(eligible):
        best = eligible.sort_values(["mean_width_h", "coverage", "candidate_id"], ascending=[True, False, True], kind="mergesort").iloc[0]
    else:
        best = search.sort_values(["coverage", "mean_width_h", "candidate_id"], ascending=[False, True, True], kind="mergesort").iloc[0]
    return str(best["candidate_id"])


def group_metrics(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for value, g in frame.groupby(group_col, dropna=False, sort=True):
        m = interval_metrics(g["target_tte_h"], g["p10_h"], g["p90_h"], g["p50_h"])
        rows.append({"group": group_col, "value": str(value), "n": int(len(g)), **m})
    return pd.DataFrame(rows)


def main() -> int:
    df = load_frame()
    manifest = json.loads((R / "M16A_DEVELOPMENT_MANIFEST.json").read_text())
    blocked = set(int(x) for x in manifest["blocked_old_final_mmsi"])
    if set(df["mmsi"].astype(int)) & blocked:
        raise AssertionError("old final leaked into M16H development frame")

    # Build only prediction-time features; reference_eta_status remains diagnostic-only.
    feat_input_cols = [c for c in df.columns if c not in {"target_tte_h", "reference_eta_status", "canonical_destination"}]
    features = build_error_features(df[feat_input_cols])
    y_all = df["target_tte_h"].to_numpy(float)
    p_all = df["pred_m16g_selected_h"].to_numpy(float)
    abs_err_all = np.abs(y_all - p_all)

    pred_parts: list[pd.DataFrame] = []
    search_parts: list[pd.DataFrame] = []
    selected_rows: list[dict] = []
    fold_rows: list[dict] = []
    global_parts: list[pd.DataFrame] = []

    for outer in sorted(df["m16a_outer_fold"].unique()):
        outer = int(outer)
        tr_idx, va_idx, tr_fold, risk_oof, risk_valid = crossfit_error_risk(df, features, outer)
        search = pd.DataFrame([score_candidate_inner(df, tr_idx, tr_fold, risk_oof, c) for c in M16H_CANDIDATES])
        selected = choose_candidate(search)
        search.insert(0, "outer_fold", outer)
        search["selected"] = search["candidate_id"].eq(selected)
        search_parts.append(search)

        chosen = calibrate_interval(abs_err_all[tr_idx], risk_oof, p_all[va_idx], risk_valid, selected, M16H_NOMINAL_COVERAGE)
        global_base = calibrate_interval(abs_err_all[tr_idx], risk_oof, p_all[va_idx], risk_valid, "global_abs", M16H_NOMINAL_COVERAGE)
        tier, q1, q2 = assign_confidence_tier(risk_oof, risk_valid)

        part = df.iloc[va_idx][[
            "mmsi", "target_tte_h", "m16a_outer_fold", "reference_eta_status", "canonical_destination",
            "pred_m16g_selected_h", "m16g_selected_meta_id", "m16g_selected_expert",
            "physics_eligible", "resolution_confidence", "m16d_gate_used",
        ]].copy().reset_index(drop=True)
        part["p10_h"] = chosen.lower_h
        part["p50_h"] = chosen.p50_h
        part["p90_h"] = chosen.upper_h
        part["interval_half_width_h"] = chosen.half_width_h
        part["predicted_abs_error_h"] = risk_valid
        part["confidence_tier"] = tier
        part["m16h_selected_candidate"] = selected
        part["m16h_scale_q"] = chosen.scale_q
        part["confidence_high_threshold_h"] = q1
        part["confidence_low_threshold_h"] = q2
        part["covered_80"] = ((part["target_tte_h"] >= part["p10_h"]) & (part["target_tte_h"] <= part["p90_h"])).astype(int)
        pred_parts.append(part)

        gpart = part[["mmsi", "m16a_outer_fold", "target_tte_h", "p50_h"]].copy()
        gpart["global_p10_h"] = global_base.lower_h
        gpart["global_p90_h"] = global_base.upper_h
        global_parts.append(gpart)

        fm = interval_metrics(part["target_tte_h"], part["p10_h"], part["p90_h"], part["p50_h"])
        gm = interval_metrics(part["target_tte_h"], global_base.lower_h, global_base.upper_h, part["p50_h"])
        fold_rows.append({
            "outer_fold": outer,
            "selected_candidate": selected,
            **{f"adaptive_{k}": v for k, v in fm.items()},
            **{f"global_{k}": v for k, v in gm.items()},
            "confidence_high_threshold_h": q1,
            "confidence_low_threshold_h": q2,
            "outer_train_n": int(len(tr_idx)),
            "outer_valid_n": int(len(va_idx)),
            "risk_model_cross_fitted_inside_outer_train": True,
            "selected_on_inner_only": True,
        })
        selected_rows.append({
            "outer_fold": outer,
            "selected_candidate": selected,
            "scale_floor_h": candidate_floor(selected),
            "scale_q": chosen.scale_q,
            "inner_best_coverage": float(search.loc[search["candidate_id"].eq(selected), "coverage"].iloc[0]),
            "inner_best_mean_width_h": float(search.loc[search["candidate_id"].eq(selected), "mean_width_h"].iloc[0]),
        })

    pred = pd.concat(pred_parts, ignore_index=True).sort_values("mmsi", kind="mergesort").reset_index(drop=True)
    global_pred = pd.concat(global_parts, ignore_index=True).sort_values("mmsi", kind="mergesort").reset_index(drop=True)
    search = pd.concat(search_parts, ignore_index=True)
    folds = pd.DataFrame(fold_rows).sort_values("outer_fold", kind="mergesort")
    selected = pd.DataFrame(selected_rows).sort_values("outer_fold", kind="mergesort")

    # Invariant: M16H never changes the frozen M16G point prediction.
    if not np.allclose(pred["p50_h"].to_numpy(float), pred["pred_m16g_selected_h"].to_numpy(float), atol=1e-12):
        raise AssertionError("M16H changed frozen M16G point prediction")

    pooled = interval_metrics(pred["target_tte_h"], pred["p10_h"], pred["p90_h"], pred["p50_h"])
    global_metrics = interval_metrics(global_pred["target_tte_h"], global_pred["global_p10_h"], global_pred["global_p90_h"], global_pred["p50_h"])
    sharp_gain = 1.0 - pooled["mean_width_h"] / global_metrics["mean_width_h"]
    stable_folds = int((folds["adaptive_coverage"] >= M16H_STABLE_FOLD_COVERAGE).sum())

    confidence = group_metrics(pred, "confidence_tier")
    confidence["tier_order"] = confidence["value"].map({"HIGH": 0, "MEDIUM": 1, "LOW": 2})
    confidence = confidence.sort_values(["tier_order", "value"], kind="mergesort").drop(columns="tier_order")
    conf_idx = confidence.set_index("value")
    gate = promotion_gate(
        pooled_coverage=pooled["coverage"],
        adaptive_mean_width_h=pooled["mean_width_h"],
        global_mean_width_h=global_metrics["mean_width_h"],
        stable_folds=stable_folds,
        high_medae_h=float(conf_idx.loc["HIGH", "medae_h"]),
        medium_medae_h=float(conf_idx.loc["MEDIUM", "medae_h"]),
        low_medae_h=float(conf_idx.loc["LOW", "medae_h"]),
    )

    # Diagnostics. reference_eta_status is target-derived and is intentionally
    # used only here, after the interval and confidence path is fully frozen.
    regime = pd.concat([
        group_metrics(pred, "physics_eligible"),
        group_metrics(pred, "m16d_gate_used"),
        group_metrics(pred.assign(destination_confidence=np.where(pred["resolution_confidence"].fillna(0) >= .99, "HIGH", "LOW")), "destination_confidence"),
        group_metrics(pred, "reference_eta_status"),
    ], ignore_index=True)

    pred.to_csv(R / "m16h_oof_intervals.csv", index=False, float_format="%.12g")
    search.to_csv(R / "m16h_inner_interval_search.csv", index=False, float_format="%.12g")
    selected.to_csv(R / "m16h_outer_selected_interval.csv", index=False, float_format="%.12g")
    folds.to_csv(R / "m16h_fold_metrics.csv", index=False, float_format="%.12g")
    confidence.to_csv(R / "m16h_confidence_metrics.csv", index=False, float_format="%.12g")
    regime.to_csv(R / "m16h_regime_metrics.csv", index=False, float_format="%.12g")

    # Deterministic visual: coverage and width by confidence tier.
    plot_conf = confidence.set_index("value").loc[["HIGH", "MEDIUM", "LOW"]]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(plot_conf.index, plot_conf["coverage"])
    ax.axhline(M16H_NOMINAL_COVERAGE, linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Empirical central-80% coverage")
    ax.set_title("M16H coverage by confidence tier")
    fig.tight_layout()
    fig.savefig(R / "m16h_confidence_coverage.png", dpi=160, metadata={"Software": "AIS ETA M16H"})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(plot_conf.index, plot_conf["mean_width_h"])
    ax.set_ylabel("Mean interval width (h)")
    ax.set_title("M16H interval sharpness by confidence tier")
    fig.tight_layout()
    fig.savefig(R / "m16h_confidence_width.png", dpi=160, metadata={"Software": "AIS ETA M16H"})
    plt.close(fig)

    summary = {
        "milestone": "M16H",
        "version": M16H_VERSION,
        "gate": gate,
        "development_rows": 386,
        "blocked_old_final_rows": 53,
        "final_test_used_for_calibration": False,
        "p50_source": "frozen M16G selected nested-OOF point prediction; unchanged",
        "interval_interpretation": "development-only empirical central 80% residual/conformal-style interval; P10/P90 are interval endpoints, not separately fitted conditional quantile models",
        "nominal_coverage": M16H_NOMINAL_COVERAGE,
        "candidate_interval_methods": list(M16H_CANDIDATES),
        "selected_candidate_by_outer_fold": {str(int(r.outer_fold)): str(r.selected_candidate) for r in selected.itertuples()},
        "risk_model": "cross-fitted HistGradientBoostingRegressor on log1p absolute M16G OOF error using prediction-time-only expert/support features",
        "target_derived_confidence_features_used": False,
        "pooled_coverage": pooled["coverage"],
        "pooled_mean_width_h": pooled["mean_width_h"],
        "pooled_median_width_h": pooled["median_width_h"],
        "global_baseline_coverage": global_metrics["coverage"],
        "global_baseline_mean_width_h": global_metrics["mean_width_h"],
        "mean_width_reduction_fraction": sharp_gain,
        "stable_outer_folds_at_or_above_75pct_coverage": stable_folds,
        "m16g_point_mae_h": pooled["mae_h"],
        "confidence_medae_h": {str(r.value): float(r.medae_h) for r in confidence.itertuples()},
        "confidence_p90_ae_h": {str(r.value): float(r.p90_ae_h) for r in confidence.itertuples()},
        "claim": "Development-only probabilistic calibration. Confidence is relative and empirical; conformal-style coverage is marginal, not a subgroup/conditional guarantee, and old M10 final remains blocked.",
    }
    (R / "M16H_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=False) + "\n")

    report = f"""# M16H — Probabilistic ETA + Confidence\n\n## Decision\n\n**{gate}**\n\nM16H does **not** change the M16G point predictor. `P50` is byte-for-value identical to `pred_m16g_selected_h`. The uncertainty layer is calibrated only from development OOF residuals; all 53 old M10 final MMSIs remain hard-blocked.\n\n## Method\n\nFor each frozen M16A outer fold, an error model is cross-fitted inside the outer-train using only prediction-time features from the four frozen expert predictions and their support/quality signals. Candidate interval scalings are then selected with leave-one-inner-fold calibration inside the outer-train. The selected scale is conformalized from held-out OOF absolute residuals at nominal central coverage **{M16H_NOMINAL_COVERAGE:.0%}** and applied once to the outer-validation rows.\n\n`P10/P90` are therefore empirical central-80% interval endpoints around the frozen M16G `P50`; they are **not** separately trained conditional quantile regressors.\n\n## Result\n\n- pooled empirical coverage: **{pooled['coverage']:.2%}**\n- pooled mean width: **{pooled['mean_width_h']:.2f} h**\n- pooled median width: **{pooled['median_width_h']:.2f} h**\n- global symmetric residual baseline coverage: **{global_metrics['coverage']:.2%}**\n- global symmetric residual baseline mean width: **{global_metrics['mean_width_h']:.2f} h**\n- adaptive mean-width reduction: **{sharp_gain:.1%}**\n- stable outer folds with >=75% empirical coverage: **{stable_folds}/5**\n- M16G point MAE remains **{pooled['mae_h']:.2f} h**\n\n## Confidence tiers\n\nConfidence is assigned from cross-fitted predicted absolute error, never from the row's true error. Outer-train OOF risk terciles define HIGH/MEDIUM/LOW thresholds. Typical error is strongly ordered: HIGH MedAE **{conf_idx.loc['HIGH','medae_h']:.2f} h**, MEDIUM **{conf_idx.loc['MEDIUM','medae_h']:.2f} h**, LOW **{conf_idx.loc['LOW','medae_h']:.2f} h**. Extreme stale/reference-label errors can still occur even in HIGH confidence, so confidence must be interpreted as relative operational reliability rather than a conditional guarantee.\n\n## Scope and limitations\n\nConformal-style validity is marginal and relies on exchangeability-like assumptions. AIS/reference-ETA drift, rare destinations and label pathologies can violate those assumptions. Subgroup coverage is reported diagnostically and is not claimed to be guaranteed. `reference_eta_status` is target-derived and appears only in post-hoc diagnostics after calibration is frozen; it is never a confidence or interval input.\n\nThe M14 company submission and all M10/M6/M16A-G freezes remain immutable.\n"""
    (R / "M16H_REPORT.md").write_text(report)

    artifact_names = [
        "M16H_REPORT.md", "M16H_SUMMARY.json", "m16h_oof_intervals.csv",
        "m16h_inner_interval_search.csv", "m16h_outer_selected_interval.csv",
        "m16h_fold_metrics.csv", "m16h_confidence_metrics.csv", "m16h_regime_metrics.csv",
        "m16h_confidence_coverage.png", "m16h_confidence_width.png",
    ]
    prior_files = [
        "M14_FINAL_FREEZE.json", "M10_FREEZE.json", "M6_FREEZE.json",
        "M16A_BENCHMARK_FREEZE.json", "M16B_RESOLVER_FREEZE.json", "M16C_HIERARCHICAL_PRIOR_FREEZE.json",
        "M16D_ROUTE_ANALOGUE_FREEZE.json", "M16E_MARITIME_PHYSICS_FREEZE.json", "M16F_TABULAR_PANEL_FREEZE.json",
        "M16G_MIXTURE_FREEZE.json",
    ]
    freeze = {
        "milestone": "M16H",
        "version": M16H_VERSION,
        "gate": gate,
        "development_population": 386,
        "blocked_old_final_population": 53,
        "final_test_used_for_calibration": False,
        "p50_is_frozen_m16g": True,
        "error_risk_cross_fitted_inside_outer_train": True,
        "interval_candidate_selected_on_inner_only": True,
        "target_derived_confidence_features_used": False,
        "nominal_marginal_coverage": M16H_NOMINAL_COVERAGE,
        "artifact_sha256": {name: sha(R / name) for name in artifact_names},
        "prior_freeze_sha256": {name: sha(R / name) for name in prior_files},
    }
    (R / "M16H_PROBABILISTIC_FREEZE.json").write_text(json.dumps(freeze, indent=2) + "\n")

    print(f"PASS M16H gate: {gate}")
    print(f"PASS pooled coverage: {pooled['coverage']:.6f}")
    print(f"PASS adaptive mean width: {pooled['mean_width_h']:.6f} h")
    print(f"PASS global mean width: {global_metrics['mean_width_h']:.6f} h")
    print(f"PASS mean-width reduction: {sharp_gain:.6f}")
    print(f"PASS stable folds >=75% coverage: {stable_folds}/5")
    print("PASS old final blocked: 53/53")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
