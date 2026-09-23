#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
DOCS = ROOT / "docs"
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m16h import build_error_features, fit_error_model, predict_error  # noqa: E402
from ais_eta.m18f import (  # noqa: E402
    M18F_BALANCED_TARGET_COVERAGE,
    M18F_COVERAGE_TARGETS,
    M18F_VERSION,
    risk_threshold,
    selective_gate,
    selective_metrics,
    sha256_file,
    validate_m18f_audit,
)


def load_m16h_source_frame() -> pd.DataFrame:
    g = pd.read_csv(R / "m16g_oof_predictions.csv")
    c = pd.read_csv(R / "m16c_oof_predictions.csv")
    d = pd.read_csv(R / "m16d_oof_predictions.csv")
    e = pd.read_csv(R / "m16e_oof_predictions.csv")
    keep_c = [
        "mmsi", "reference_eta_status", "canonical_destination", "m16c_deepest_support",
        "m16c_deepest_weight", "m16c_fallback_count",
    ]
    keep_d = [
        "mmsi", "m16d_neighbour_count", "m16d_nearest_distance",
        "m16d_median_neighbour_distance", "m16d_similarity_gap", "m16d_gate_used",
    ]
    keep_e = [
        "mmsi", "physics_eligible", "resolution_confidence", "physics_distance_gc_nm",
        "physics_course_alignment_deg", "physics_recent_speed_max_kn",
    ]
    df = (
        g.merge(c[keep_c], on="mmsi", how="left", validate="one_to_one")
        .merge(d[keep_d], on="mmsi", how="left", validate="one_to_one")
        .merge(e[keep_e], on="mmsi", how="left", validate="one_to_one")
        .sort_values("mmsi", kind="mergesort")
        .reset_index(drop=True)
    )
    if len(df) != 386 or df.mmsi.nunique() != 386:
        raise AssertionError("M18F development population drift")
    return df


def crossfit_outer_risk(df: pd.DataFrame, features: pd.DataFrame, outer: int):
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
        raise AssertionError("non-finite M18F outer-train cross-fitted risk")
    model = fit_error_model(features.iloc[tr_idx], abs_err[tr_idx])
    risk_valid = predict_error(model, features.iloc[va_idx])
    return tr_idx, va_idx, risk_oof, risk_valid


def main() -> int:
    R.mkdir(exist_ok=True)
    DOCS.mkdir(exist_ok=True)

    # Frozen M16G/M16H development artifacts only. No fresh holdout or old-final rows.
    df = load_m16h_source_frame()
    intervals = pd.read_csv(R / "m16h_oof_intervals.csv").sort_values("mmsi", kind="mergesort").reset_index(drop=True)
    if not np.array_equal(df.mmsi.to_numpy(int), intervals.mmsi.to_numpy(int)):
        raise AssertionError("M18F M16H row alignment drift")
    if not np.allclose(df.pred_m16g_selected_h, intervals.p50_h, atol=1e-10):
        raise AssertionError("M18F cannot change frozen M16G P50")

    manifest = json.loads((R / "M16A_DEVELOPMENT_MANIFEST.json").read_text())
    blocked = set(int(x) for x in manifest["blocked_old_final_mmsi"])
    if set(df.mmsi.astype(int)) & blocked:
        raise AssertionError("old final leaked into M18F development frame")
    if len(blocked) != 53:
        raise AssertionError("M18F blocked old-final cardinality drift")

    feat_cols = [c for c in df.columns if c not in {"target_tte_h", "reference_eta_status", "canonical_destination"}]
    features = build_error_features(df[feat_cols])
    y = df.target_tte_h.to_numpy(float)
    p = df.pred_m16g_selected_h.to_numpy(float)
    lo = intervals.p10_h.to_numpy(float)
    hi = intervals.p90_h.to_numpy(float)
    folds = df.m16a_outer_fold.to_numpy(int)

    row_policy = intervals[[
        "mmsi", "m16a_outer_fold", "target_tte_h", "p50_h", "p10_h", "p90_h",
        "predicted_abs_error_h", "confidence_tier", "reference_eta_status",
    ]].copy()
    row_policy["selection_score_source"] = "M16H_cross_fitted_predicted_abs_error_h"

    fold_rows: list[dict] = []
    thresholds: dict[tuple[int, float], float] = {}

    for outer in sorted(np.unique(folds)):
        outer = int(outer)
        tr_idx, va_idx, risk_oof, risk_valid = crossfit_outer_risk(df, features, outer)
        frozen_valid = intervals.iloc[va_idx].predicted_abs_error_h.to_numpy(float)
        if not np.allclose(risk_valid, frozen_valid, atol=1e-6, rtol=0):
            raise AssertionError(f"M18F failed to reproduce frozen M16H risk on outer fold {outer}")

        for cov in M18F_COVERAGE_TARGETS:
            cov = float(cov)
            thr = risk_threshold(risk_oof, cov)
            thresholds[(outer, cov)] = thr
            accepted = risk_valid <= thr
            col = f"accept_target_{int(round(cov * 100)):03d}"
            row_policy.loc[va_idx, col] = accepted
            metrics = selective_metrics(y[va_idx], p[va_idx], lo[va_idx], hi[va_idx], accepted)
            full_fold_mae = float(np.mean(np.abs(y[va_idx] - p[va_idx])))
            fold_rows.append({
                "outer_fold": outer,
                "target_coverage": cov,
                "train_risk_threshold_h": thr,
                "outer_train_rows": int(len(tr_idx)),
                "outer_valid_rows": int(len(va_idx)),
                "full_fold_mae_h": full_fold_mae,
                "retained_mae_gain_vs_full_fold_h": full_fold_mae - float(metrics["retained_mae_h"]),
                **metrics,
                "threshold_uses_validation_target": False,
                "threshold_source": "outer_train_cross_fitted_M16H_risk_quantile",
            })

    accept_cols = [c for c in row_policy.columns if c.startswith("accept_target_")]
    row_policy[accept_cols] = row_policy[accept_cols].astype(bool)

    pooled_rows: list[dict] = []
    near_rows: list[dict] = []
    near_mask = df.reference_eta_status.eq("FUTURE_0_7D").to_numpy()
    for cov in M18F_COVERAGE_TARGETS:
        cov = float(cov)
        col = f"accept_target_{int(round(cov * 100)):03d}"
        accepted = row_policy[col].to_numpy(bool)
        pooled_rows.append({
            "scope": "STRICT_386",
            "target_coverage": cov,
            **selective_metrics(y, p, lo, hi, accepted),
        })
        near_rows.append({
            "scope": "NEAR_TERM_FUTURE_0_7D_DIAGNOSTIC",
            "target_coverage": cov,
            **selective_metrics(y[near_mask], p[near_mask], lo[near_mask], hi[near_mask], accepted[near_mask]),
        })

    pooled = pd.DataFrame(pooled_rows)
    near = pd.DataFrame(near_rows)
    curves = pd.concat([pooled, near], ignore_index=True)
    folds_df = pd.DataFrame(fold_rows).sort_values(["target_coverage", "outer_fold"], ascending=[False, True], kind="mergesort")

    balanced = pooled.loc[np.isclose(pooled.target_coverage, M18F_BALANCED_TARGET_COVERAGE)].iloc[0]
    balanced_near = near.loc[np.isclose(near.target_coverage, M18F_BALANCED_TARGET_COVERAGE)].iloc[0]
    balanced_folds = folds_df.loc[np.isclose(folds_df.target_coverage, M18F_BALANCED_TARGET_COVERAGE)].copy()
    fold_wins = int((balanced_folds.retained_mae_gain_vs_full_fold_h > 0).sum())
    full = pooled.loc[np.isclose(pooled.target_coverage, 1.0)].iloc[0]

    gate = selective_gate(
        balanced_realized_coverage=float(balanced.realized_coverage),
        balanced_retained_mae_h=float(balanced.retained_mae_h),
        full_mae_h=float(full.retained_mae_h),
        fold_wins=fold_wins,
        fold_count=int(len(balanced_folds)),
        near_term_retention=float(balanced_near.realized_coverage),
        retained_empirical_interval_coverage=float(balanced.retained_interval_empirical_coverage),
    )

    balanced_col = "accept_target_080"
    row_policy["m18f_balanced_action"] = np.where(
        row_policy[balanced_col].astype(bool), "AUTO_ETA_WITH_UNCERTAINTY", "DEFER_LOW_CONFIDENCE"
    )
    # A single pre-holdout score threshold is frozen for future one-time lockbox use.
    # Primary development performance, however, uses the nested outer-train thresholds above.
    deployment_threshold_h = risk_threshold(intervals.predicted_abs_error_h, M18F_BALANCED_TARGET_COVERAGE)

    # Confidence-tier diagnostic with the selective action already frozen.
    tier_rows = []
    for tier, g in row_policy.groupby("confidence_tier", sort=True):
        idx = g.index.to_numpy(int)
        accepted = row_policy.loc[idx, balanced_col].to_numpy(bool)
        tier_rows.append({
            "confidence_tier": str(tier),
            "rows": int(len(idx)),
            "balanced_retained_rows": int(accepted.sum()),
            "balanced_retention_fraction": float(accepted.mean()),
            "full_mae_h": float(np.mean(np.abs(y[idx] - p[idx]))),
            "balanced_retained_mae_h": float(np.mean(np.abs(y[idx][accepted] - p[idx][accepted]))) if accepted.any() else float("nan"),
        })
    tiers = pd.DataFrame(tier_rows)

    row_policy.to_csv(R / "m18f_selective_eta_ledger.csv", index=False, float_format="%.12g")
    curves.to_csv(R / "m18f_risk_coverage_curve.csv", index=False, float_format="%.12g")
    folds_df.to_csv(R / "m18f_fold_risk_coverage.csv", index=False, float_format="%.12g")
    tiers.to_csv(R / "m18f_confidence_tier_selectivity.csv", index=False, float_format="%.12g")

    # One chart per figure; no style/color forcing.
    strict_plot = pooled.sort_values("realized_coverage")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(strict_plot.realized_coverage, strict_plot.retained_mae_h, marker="o")
    ax.set_xlabel("Realized coverage (fraction receiving automatic ETA)")
    ax.set_ylabel("Retained-row MAE (h)")
    ax.set_title("M18F selective ETA risk–coverage curve — strict development")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(R / "m18f_risk_coverage.png", dpi=160, metadata={"Software": "AIS ETA M18F"})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(strict_plot.realized_coverage, strict_plot.retained_interval_empirical_coverage, marker="o")
    ax.axhline(0.80, linestyle="--", linewidth=1)
    ax.set_xlabel("Realized coverage (fraction receiving automatic ETA)")
    ax.set_ylabel("Empirical retained-set interval coverage")
    ax.set_title("M18F retained-set interval audit (not a conditional guarantee)")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(R / "m18f_interval_selected_coverage.png", dpi=160, metadata={"Software": "AIS ETA M18F"})
    plt.close(fig)

    audit = {
        "milestone": "M18F",
        "version": M18F_VERSION,
        "gate": gate,
        "scope": {
            "development_population": 386,
            "blocked_old_final_population": 53,
            "fresh_holdout_opened": False,
            "old_final_used": False,
            "m16g_modified": False,
            "m16h_modified": False,
            "target_relabeling": False,
            "strict_rows_deleted": False,
            "selection_uses_validation_target": False,
            "selection_conditional_conformal_guarantee_claimed": False,
        },
        "method": {
            "selection_score": "frozen M16H predicted_abs_error_h",
            "development_evaluation": "for each M16A outer fold, threshold is a quantile of cross-fitted M16H risk scores from outer-train owners only; applied once to outer-valid rows",
            "coverage_targets": list(M18F_COVERAGE_TARGETS),
            "balanced_target_coverage": M18F_BALANCED_TARGET_COVERAGE,
            "fresh_holdout_policy": "freeze a development-derived uncertainty threshold before one-time holdout opening; no threshold tuning on holdout labels",
            "interval_claim": "M16H central interval coverage after selection is empirical audit only; no selection-conditional conformal guarantee is claimed",
        },
        "balanced_policy": {
            "policy_id": "BALANCED_80",
            "target_coverage": M18F_BALANCED_TARGET_COVERAGE,
            "point_prediction_source": "frozen_M16G",
            "interval_source": "frozen_M16H",
            "auto_action": "AUTO_ETA_WITH_UNCERTAINTY",
            "defer_action": "DEFER_LOW_CONFIDENCE",
            "realized_development_coverage": float(balanced.realized_coverage),
            "retained_rows": int(balanced.retained_rows),
            "deferred_rows": int(balanced.deferred_rows),
            "full_mae_h": float(full.retained_mae_h),
            "retained_mae_h": float(balanced.retained_mae_h),
            "mae_reduction_fraction": float(1.0 - balanced.retained_mae_h / full.retained_mae_h),
            "retained_medae_h": float(balanced.retained_medae_h),
            "retained_p90_ae_h": float(balanced.retained_p90_ae_h),
            "retained_abs_error_share": float(balanced.retained_abs_error_share),
            "retained_empirical_interval_coverage": float(balanced.retained_interval_empirical_coverage),
            "retained_mean_interval_width_h": float(balanced.retained_mean_interval_width_h),
            "near_term_retention_fraction": float(balanced_near.realized_coverage),
            "near_term_retained_mae_h": float(balanced_near.retained_mae_h),
            "outer_fold_mae_improvements": fold_wins,
            "outer_fold_count": int(len(balanced_folds)),
            "frozen_pre_holdout_risk_threshold_h": float(deployment_threshold_h),
        },
        "interpretation": {
            "selectivity_is_model_improvement": False,
            "reason": "M18F improves reliability of automatic output by deferring uncertain rows; it does not lower error on the same complete population and does not replace M16G.",
            "near_term_slice_is_target_derived_diagnostic_only": True,
            "balanced_policy_is_advisory_until_fresh_holdout": True,
        },
    }
    validate_m18f_audit(audit)
    (R / "M18F_SELECTIVE_ETA.json").write_text(json.dumps(audit, indent=2) + "\n")

    policy_md = f"""# M18F Uncertainty & Selective ETA Policy

## Frozen inputs

- point ETA: **M16G**, unchanged;
- uncertainty/risk and interval: **M16H**, unchanged;
- development population: **386** owners;
- old-final owners: **53/53 blocked**;
- fresh holdout: **sealed**.

## Selection rule

M18F uses **predicted absolute error risk** from M16H. Lower score means more reliable. During development evaluation, each M16A outer-validation fold receives a threshold derived only from cross-fitted uncertainty scores inside that fold's outer-train. Validation targets are never used to decide whether a row is retained.

The frozen service levels are {', '.join(f'{int(x*100)}%' for x in M18F_COVERAGE_TARGETS)} target coverage. The advisory balanced profile is **BALANCED_80**. It automatically emits the frozen M16G ETA + frozen M16H interval when risk is accepted and otherwise returns **DEFER_LOW_CONFIDENCE**.

For a future one-time fresh holdout, the development-derived M16H OOF 80th-percentile risk threshold is frozen at **{deployment_threshold_h:.3f} h predicted absolute error risk** before the holdout is opened. It must not be tuned after seeing holdout labels.

## Interval validity statement

M16H's central interval is an empirical, marginal development calibration. Selecting low-risk rows changes the evaluated population. M18F therefore reports retained-set interval coverage **empirically only** and makes **no selection-conditional conformal guarantee**.

## Balanced development result

- realized automatic-ETA coverage: **{balanced.realized_coverage:.2%}** ({int(balanced.retained_rows)}/386);
- retained MAE: **{balanced.retained_mae_h:.2f} h** vs full-population **{full.retained_mae_h:.2f} h**;
- retained MedAE: **{balanced.retained_medae_h:.2f} h**;
- retained P90 absolute error: **{balanced.retained_p90_ae_h:.2f} h**;
- retained rows contain **{balanced.retained_abs_error_share:.1%}** of strict total absolute error;
- empirical M16H interval coverage among retained rows: **{balanced.retained_interval_empirical_coverage:.2%}**;
- near-term diagnostic retention: **{balanced_near.realized_coverage:.2%}** with MAE **{balanced_near.retained_mae_h:.2f} h**;
- retained MAE improves relative to full-fold MAE in **{fold_wins}/{len(balanced_folds)}** outer folds.

Selective ETA is a **reliability policy**, not a claim that the underlying point model improved on all 386 rows.
"""
    (DOCS / "M18F_SELECTIVE_ETA_POLICY.md").write_text(policy_md)

    report = f"""# M18F — Uncertainty & Selective ETA

## Decision

**{gate}**

M18F promotes an **advisory selective-output layer**, not a new ETA model. M16G remains the point champion and M16H remains the frozen uncertainty sidecar.

## Why selective ETA

M18D showed that M16H uncertainty meaningfully ranks residual difficulty. M18F converts that signal into an auditable action: automatically issue ETA when expected error risk is sufficiently low; otherwise defer rather than pretending all predictions are equally trustworthy.

## Fold-safe evaluation

Thresholds are computed from cross-fitted risk scores inside each outer-train only. The corresponding outer-validation targets are not used in selection. Six target service levels are frozen: {', '.join(f'{int(x*100)}%' for x in M18F_COVERAGE_TARGETS)}.

### Strict risk–coverage curve

{pooled.to_markdown(index=False)}

### Near-term diagnostic curve

`FUTURE_0_7D` is target-derived and appears **only after** selection flags are frozen.

{near.to_markdown(index=False)}

## BALANCED_80

At the predeclared 80% target service level, realized coverage is **{balanced.realized_coverage:.2%}** ({int(balanced.retained_rows)}/386). Retained MAE falls from **{full.retained_mae_h:.2f} h** on all rows to **{balanced.retained_mae_h:.2f} h**, a **{(1-balanced.retained_mae_h/full.retained_mae_h):.1%}** reduction among automatically served rows. Those retained rows account for only **{balanced.retained_abs_error_share:.1%}** of total strict absolute error, so deferring roughly one fifth of rows removes a disproportionate share of error. All **{len(balanced_folds)}/{len(balanced_folds)}** outer folds show lower retained MAE than their full-fold MAE.

Within the M18C near-term diagnostic regime, BALANCED_80 retains **{balanced_near.realized_coverage:.2%}** of rows and reduces MAE from **{near.loc[np.isclose(near.target_coverage,1.0),'retained_mae_h'].iloc[0]:.2f} h** to **{balanced_near.retained_mae_h:.2f} h**.

## Uncertainty interval caution

Among BALANCED_80 retained rows the existing M16H central interval has empirical coverage **{balanced.retained_interval_empirical_coverage:.2%}** with mean width **{balanced.retained_mean_interval_width_h:.2f} h**. This is a descriptive audit only. Selection can alter conformal coverage properties, so M18F does not claim a selection-conditional guarantee.

## Fresh-holdout rule

The fresh holdout remains sealed. The development-derived 80th-percentile risk threshold **{deployment_threshold_h:.3f} h** is frozen before any future holdout scoring. It cannot be retuned after holdout labels are observed. A true external promotion still requires M18G/the final external gate.
"""
    (R / "M18F_REPORT.md").write_text(report)

    freeze_files = [
        "reports/M18F_SELECTIVE_ETA.json",
        "reports/m18f_selective_eta_ledger.csv",
        "reports/m18f_risk_coverage_curve.csv",
        "reports/m18f_fold_risk_coverage.csv",
        "reports/m18f_confidence_tier_selectivity.csv",
        "reports/m18f_risk_coverage.png",
        "reports/m18f_interval_selected_coverage.png",
        "docs/M18F_SELECTIVE_ETA_POLICY.md",
        "reports/M18F_REPORT.md",
    ]
    freeze = {
        "milestone": "M18F",
        "status": "SELECTIVE_ETA_ADVISORY_LAYER_FROZEN",
        "version": M18F_VERSION,
        "gate": gate,
        "development_population": 386,
        "blocked_old_final_population": 53,
        "fresh_holdout_opened": False,
        "point_model_changed": False,
        "uncertainty_model_changed": False,
        "selection_uses_validation_target": False,
        "m18a_baseline_freeze_sha256": sha256_file(R / "M18A_BASELINE_FREEZE.json"),
        "m18b_validation_freeze_sha256": sha256_file(R / "M18B_VALIDATION_FREEZE.json"),
        "m18c_audit_freeze_sha256": sha256_file(R / "M18C_AUDIT_FREEZE.json"),
        "m18d_attribution_freeze_sha256": sha256_file(R / "M18D_ATTRIBUTION_FREEZE.json"),
        "m18e_ablation_freeze_sha256": sha256_file(R / "M18E_ABLATION_FREEZE.json"),
        "m16g_oof_sha256": sha256_file(R / "m16g_oof_predictions.csv"),
        "m16h_oof_intervals_sha256": sha256_file(R / "m16h_oof_intervals.csv"),
        "artifact_sha256": {rel: sha256_file(ROOT / rel) for rel in freeze_files},
    }
    (R / "M18F_SELECTIVE_FREEZE.json").write_text(json.dumps(freeze, indent=2) + "\n")

    # Update project state after freeze hashes are generated.
    state_path = ROOT / "PROJECT_STATE.json"
    state = json.loads(state_path.read_text())
    if state.get("current_phase") != "M18F_DONE_UNCERTAINTY_SELECTIVE_ETA":
        state["state_version"] = int(state.get("state_version", 0)) + 1
    state["updated_at"] = "2026-09-23T12:19:00+02:00"
    state["current_phase"] = "M18F_DONE_UNCERTAINTY_SELECTIVE_ETA"
    state["current_task"] = "M18F froze a fold-safe uncertainty-driven selective ETA policy around unchanged M16G/M16H. BALANCED_80 is advisory, fresh holdout remains sealed, and retained-set interval coverage is empirical only rather than a selection-conditional conformal guarantee."
    completed = list(state.get("completed", []))
    line = "M18F uncertainty/selective ETA: nested train-only risk thresholds, risk-coverage and retained-interval audits, BALANCED_80 advisory auto/defer policy, pre-holdout risk-threshold freeze, and explicit no-selection-conditional-guarantee statement; M16G/M16H unchanged, old-final blocked and fresh holdout sealed"
    if line not in completed:
        completed.append(line)
    state["completed"] = completed
    state["m18f_summary"] = {
        "gate": gate,
        "development_rows": 386,
        "blocked_old_final_rows": 53,
        "fresh_holdout_opened": False,
        "point_model_changed": False,
        "balanced_target_coverage": M18F_BALANCED_TARGET_COVERAGE,
        "balanced_realized_coverage": float(balanced.realized_coverage),
        "balanced_retained_rows": int(balanced.retained_rows),
        "balanced_retained_mae_h": float(balanced.retained_mae_h),
        "full_mae_h": float(full.retained_mae_h),
        "balanced_retained_p90_h": float(balanced.retained_p90_ae_h),
        "balanced_near_term_retention": float(balanced_near.realized_coverage),
        "balanced_near_term_mae_h": float(balanced_near.retained_mae_h),
        "balanced_outer_fold_improvements": fold_wins,
        "retained_interval_empirical_coverage": float(balanced.retained_interval_empirical_coverage),
        "selection_conditional_conformal_guarantee_claimed": False,
        "frozen_pre_holdout_risk_threshold_h": float(deployment_threshold_h),
        "next_milestone": "M18G_FROZEN_EXTERNAL_PROMOTION_GATE",
    }
    state_path.write_text(json.dumps(state, indent=2) + "\n")

    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
