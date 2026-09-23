#!/usr/bin/env python3
"""Create M3 diagnostic figures without touching the locked final holdout."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

LABELS = {
    "pred_m3_route_physics_h": "Route physics",
    "pred_m3_direct_ridge_h": "Direct Ridge",
    "pred_m3_residual_ridge_h": "Residual Ridge",
    "pred_m3_residual_huber_h": "Residual Huber",
    "pred_m3_residual_lightgbm_h": "Residual LightGBM",
    "pred_m3_residual_lightgbm_portstate_h": "+ port-state",
}


def model_ablation():
    m = pd.read_csv(REPORTS / "m3_eta_metrics.csv")
    q = m[m.scope.eq("under24h") & m.baseline.isin(LABELS)].copy()
    q["label"] = q.baseline.map(LABELS)
    q["mae_min"] = q.voyage_balanced_mae_h * 60
    q = q.sort_values("mae_min")
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.barh(q.label, q.mae_min)
    ax.set_xlabel("Voyage-balanced MAE at ≤24 h (minutes)")
    ax.set_title("M3 ablation: added ML complexity does not beat route physics")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(REPORTS / "m3_model_ablation.png", dpi=160)
    plt.close(fig)


def fold_consistency():
    f = pd.read_csv(REPORTS / "m3_fold_metrics.csv")
    f = f[f.scope.eq("under24h")]
    keep = ["pred_m3_route_physics_h", "pred_m3_residual_lightgbm_h", "pred_m3_residual_lightgbm_portstate_h"]
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for model in keep:
        q = f[f.baseline.eq(model)].sort_values("m3_temporal_fold")
        ax.plot(q.m3_temporal_fold, q.voyage_balanced_mae_h * 60, marker="o", label=LABELS[model])
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("Expanding temporal stack fold")
    ax.set_ylabel("Voyage-balanced MAE at ≤24 h (minutes)")
    ax.set_title("M3 fold consistency")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "m3_fold_consistency.png", dpi=160)
    plt.close(fig)


def call_gain():
    g = pd.read_csv(REPORTS / "m3_selected_model_call_gains.csv").sort_values("gain_h")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(g.session_id, g.gain_h * 60)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Route-physics MAE − residual-LightGBM MAE (minutes); positive is better")
    ax.set_title("M3 call-level gain: improvements are outweighed by degradations")
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(REPORTS / "m3_call_gain.png", dpi=160)
    plt.close(fig)


def result_panel():
    s = json.loads((REPORTS / "m3_summary.json").read_text())
    lines = [
        "M3 — Physics + residual ML",
        "",
        f"OOF: {s['m3_oof_calls']} calls / {s['m3_oof_unique_vessels']} vessels / {s['m3_oof_rows']} prediction points",
        "16 locked chronological calls: untouched",
        "",
        f"Route physics ≤24h MAE: {s['route_physics_under24_mae_h']*60:.1f} min",
        f"Best residual model: {s['selected_core_model'].replace('pred_m3_', '').replace('_h','')}",
        f"Residual ≤24h MAE: {s['selected_under24_mae_h']*60:.1f} min",
        f"Relative change: {s['selected_under24_improvement_fraction']*100:.1f}%",
        f"Cold-vessel: {s['cold_route_physics_under24_mae_h']*60:.1f} → {s['cold_selected_under24_mae_h']*60:.1f} min",
        f"Fold wins: {s['fold_wins']}/{s['fold_total']}",
        f"Bootstrap gain 95% CI: [{s['bootstrap']['ci95_low_h']*60:.1f}, {s['bootstrap']['ci95_high_h']*60:.1f}] min",
        "",
        f"Port-state gate: {s['port_state_gate']['status']}",
        f"M3 gate: {s['status']}",
        "",
        "Decision: retain M2 route-physics baseline; reject residual ML and port-state complexity for now.",
    ]
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.axis("off")
    ax.text(0.04, 0.96, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=12)
    fig.tight_layout()
    fig.savefig(REPORTS / "m3_result_panel.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    model_ablation(); fold_consistency(); call_gain(); result_panel()
    print("M3 visual audit complete")
