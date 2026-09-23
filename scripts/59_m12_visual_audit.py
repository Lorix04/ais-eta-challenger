#!/usr/bin/env python3
"""Create M12 frozen-model explainability visuals. No model fitting."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"


def main() -> int:
    pvc = pd.read_csv(R / "m12_prediction_values_change.csv")
    shap = pd.read_csv(R / "m12_mean_abs_shap.csv")
    group = pd.read_csv(R / "m12_group_shap_importance.csv")
    mask = pd.read_csv(R / "m12_calibration_group_masking_stress.csv")
    local = pd.read_csv(R / "m12_local_top_contributors.csv")
    summary = json.loads((R / "m12_summary.json").read_text())

    # 1) Frozen feature importance: native PVC and calibration mean |SHAP| normalized for display.
    s = shap.loc[shap["split"].eq("calibration"), ["feature", "mean_abs_shap_h"]]
    plot = pvc.merge(s, on="feature")
    plot["native_pct"] = plot["prediction_values_change"] / plot["prediction_values_change"].sum() * 100
    plot["shap_pct"] = plot["mean_abs_shap_h"] / plot["mean_abs_shap_h"].sum() * 100
    plot = plot.sort_values("shap_pct", ascending=True)
    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    ax.barh(y - 0.18, plot["native_pct"], height=0.34, label="CatBoost PredictionValuesChange")
    ax.barh(y + 0.18, plot["shap_pct"], height=0.34, label="Calibration mean |SHAP|")
    ax.set_yticks(y, plot["feature"])
    ax.set_xlabel("Normalized importance (%)")
    ax.set_title("M12 — frozen M10 CatBoost feature attribution")
    ax.legend(loc="lower right")
    ax.text(0.01, 0.99, "Diagnostic attribution only — not causal importance", transform=ax.transAxes, va="top", fontsize=9)
    fig.tight_layout()
    fig.savefig(R / "m12_feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 2) Group SHAP stability across train/cal/final.
    piv = group.pivot(index="feature_group", columns="split", values="mean_abs_group_shap_h")
    order = piv["calibration"].sort_values().index
    piv = piv.loc[order]
    y = np.arange(len(piv))
    fig, ax = plt.subplots(figsize=(10.2, 6.2))
    width = 0.24
    for i, col in enumerate(["train", "calibration", "final_test"]):
        ax.barh(y + (i - 1) * width, piv[col], height=width, label=col)
    ax.set_yticks(y, piv.index)
    ax.set_xlabel("Mean absolute grouped SHAP contribution (hours)")
    ax.set_title("M12 — feature-group attribution is broadly stable across splits")
    ax.legend()
    fig.tight_layout()
    fig.savefig(R / "m12_group_shap_stability.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 3) Frozen calibration masking stress.
    m = mask.sort_values("mean_abs_prediction_shift_h", ascending=True)
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    ax.barh(m["item"], m["mean_abs_prediction_shift_h"])
    ax.set_xlabel("Mean absolute change in frozen prediction (hours)")
    ax.set_title("M12 — train-baseline group masking stress on calibration")
    ax.text(0.01, 0.96, "Sensitivity only; no retraining and no feature selection", transform=ax.transAxes, va="top", fontsize=9)
    fig.tight_layout()
    fig.savefig(R / "m12_group_masking_stress.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 4) Local explanation panel for the five largest final errors and one median-error example.
    top = local.loc[local["abs_error_rank"] <= 5].drop_duplicates("mmsi").copy()
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.axis("off")
    lines = [
        "M12 — Frozen Model Explainability & Ablation",
        "",
        "NO RETRAINING / NO POST-FINAL MODEL SELECTION",
        f"  frozen model: 47-tree CatBoost, {summary['feature_count']} current-snapshot features",
        f"  SHAP expected value: {summary['expected_value_h']:.1f} h",
        "",
        "GLOBAL ATTRIBUTION",
        f"  native top feature:       {summary['top_native_importance_feature']}",
        f"  calibration SHAP top:     {summary['top_calibration_mean_abs_shap_feature']} ({summary['top_calibration_mean_abs_shap_h']:.1f} h mean |SHAP|)",
        f"  calibration top group:    {summary['top_calibration_shap_group']}",
        "",
        "FROZEN CALIBRATION ABLATION",
        f"  adding causal history in M10: {summary['strict_calibration_history_delta_mae_h']:+.1f} h MAE vs current snapshot",
        f"  future-reference destination median vs current CatBoost: {summary['future_calibration_destination_median_delta_mae_h_vs_current']:+.1f} h MAE",
        "",
        "PREDICTION RANGE VS REFERENCE RANGE (STRICT FINAL)",
        f"  frozen predictions: {summary['final_prediction_min_h']:.1f} to {summary['final_prediction_max_h']:.1f} h",
        f"  company references: {summary['final_target_min_h']:.1f} to {summary['final_target_max_h']:.1f} h",
        "",
        "TOP FIVE STRICT-FINAL ERRORS",
    ]
    for rr in top.sort_values("abs_error_rank").itertuples(index=False):
        contrib = local.loc[local["mmsi"].eq(rr.mmsi)].sort_values("contributor_rank").iloc[0]
        lines.append(f"  #{int(rr.abs_error_rank)} {rr.name[:20]:20s} target={rr.target_tte_h:8.1f} h pred={rr.prediction_tte_h:7.1f} h | top SHAP={contrib.feature} {contrib.shap_h:+.1f} h")
    lines += [
        "",
        "INTERPRETATION",
        "  SHAP explains the frozen model, not maritime causality.",
        "  The model shifts a ~40 h baseline by tens of hours; it cannot represent multi-month stale/far ETA references.",
        "  M10's history-heavy candidate did not improve calibration, so M12 does not revive or retune it.",
    ]
    ax.text(0.025, 0.975, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9.6)
    fig.tight_layout()
    fig.savefig(R / "m12_result_panel.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("PASS M12 visual explainability figures written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
