#!/usr/bin/env python3
"""Create M5 port/generalization stress-test visual evidence."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def port_comparison():
    x = pd.read_csv(REPORTS / "m5_port_metrics.csv").sort_values("port")
    pos = np.arange(len(x)); width = 0.36
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    ax.bar(pos - width/2, x.geodesic_voyage_mae_h * 60, width, label="Geodesic physics")
    ax.bar(pos + width/2, x.route_voyage_mae_h * 60, width, label="Route-kNN physics")
    ax.set_xticks(pos, x.port)
    ax.set_ylabel("Voyage-balanced MAE <=24 h (minutes)")
    ax.set_title("M5: route signal survives separately in both seen ports")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    for i, r in enumerate(x.itertuples()):
        ax.text(i, max(r.geodesic_voyage_mae_h, r.route_voyage_mae_h)*60 + 1.5,
                f"{r.route_improvement_fraction*100:.1f}% gain\n{int(r.calls)} calls",
                ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(REPORTS / "m5_port_comparison.png", dpi=160); plt.close(fig)


def route_family_stress():
    x = pd.read_csv(REPORTS / "m5_route_family_metrics.csv")
    x = x[x.calls >= 3].copy()
    labels = [f"{p}-{f}" for p, f in zip(x.port, x.initial_route_family)]
    vals = x.route_improvement_fraction * 100
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    ax.barh(np.arange(len(x)), vals)
    ax.axvline(0, linewidth=1)
    ax.set_yticks(np.arange(len(x)), labels)
    ax.set_xlabel("Route-kNN improvement vs geodesic at <=24 h (%)")
    ax.set_title("M5: route value is heterogeneous by initial route family")
    ax.grid(axis="x", alpha=0.25)
    for i, (v, n) in enumerate(zip(vals, x.calls)):
        ax.text(v + (0.8 if v >= 0 else -0.8), i, f"{v:.1f}% / {int(n)} calls",
                va="center", ha="left" if v >= 0 else "right", fontsize=9)
    fig.tight_layout(); fig.savefig(REPORTS / "m5_route_family_stress.png", dpi=160); plt.close(fig)


def interval_port_diag():
    x = pd.read_csv(REPORTS / "m5_port_interval_diagnostics.csv")
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    pos = np.arange(len(x)); width = 0.36
    ax.bar(pos - width/2, x.pooled_m4_half_width_h * 60, width, label="Pooled M4 half-width")
    ax.bar(pos + width/2, x.port_specific_q_h * 60, width, label="Port-only 90% diagnostic q")
    ax.set_xticks(pos, x.port)
    ax.set_ylabel("Interval half-width / diagnostic q (minutes)")
    ax.set_title("M5: port-only conformal calibration is small-N and unstable")
    ax.grid(axis="y", alpha=0.25); ax.legend()
    for i, r in enumerate(x.itertuples()):
        ax.text(i, max(r.pooled_m4_half_width_h, r.port_specific_q_h)*60 + 8,
                f"cal={int(r.calibration_calls)}, eval={int(r.temporal_high_calls)}\nk={int(r.port_specific_conformal_order_k)}/{int(r.calibration_calls)}",
                ha="center", fontsize=8.5)
    fig.tight_layout(); fig.savefig(REPORTS / "m5_port_interval_diagnostics.png", dpi=160); plt.close(fig)


def result_panel():
    s = json.loads((REPORTS / "m5_summary.json").read_text())
    p = s["port_route_under24"]; c = s["cold_vessel_under24"]
    warnings = json.loads((REPORTS / "m5_gate_result.json").read_text())["warnings"]
    lines = [
        "M5 — port generalization / stress tests",
        "",
        f"Gate: {s['status']}",
        f"OOF calls / vessels: {s['oof_calls']} / {s['oof_unique_vessels']}",
        "Locked final chronological calls: 0 scored",
        "",
        f"CATANIA <=24h: {p['CATANIA']['geodesic_mae_min']:.1f} -> {p['CATANIA']['route_mae_min']:.1f} min ({p['CATANIA']['route_improvement_fraction']*100:.1f}% gain)",
        f"AUGUSTA <=24h: {p['AUGUSTA']['geodesic_mae_min']:.1f} -> {p['AUGUSTA']['route_mae_min']:.1f} min ({p['AUGUSTA']['route_improvement_fraction']*100:.1f}% gain)",
        "",
        f"Cold vessel CATANIA: {c['CATANIA']['route_improvement_fraction']*100:.1f}% gain ({c['CATANIA']['calls']} calls)",
        f"Cold vessel AUGUSTA: {c['AUGUSTA']['route_improvement_fraction']*100:.1f}% gain ({c['AUGUSTA']['calls']} calls)",
        "",
        f"Warnings: {len(warnings)}",
        "- Augusta E-family route-kNN degrades vs geodesic",
        "- Augusta seen-vessel subgroup degrades; repeated MMSI != stable route",
        "- Catania later interval evaluation has only 2 calls",
        "",
        "NOT ESTABLISHED:",
        "unseen-port transfer / Augusta Scirocco / Siracusa-Santa Panagia",
        "",
        "Next: M6 final deliverable + one-time frozen final-holdout evaluation.",
    ]
    fig, ax = plt.subplots(figsize=(11, 7.4)); ax.axis("off")
    ax.text(0.04, 0.96, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=11.2)
    fig.tight_layout(); fig.savefig(REPORTS / "m5_result_panel.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    port_comparison(); route_family_stress(); interval_port_diag(); result_panel()
    print("M5 visual audit complete")
