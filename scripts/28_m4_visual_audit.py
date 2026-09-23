#!/usr/bin/env python3
"""Create M4 calibration/reliability/stability visual evidence."""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def interval_calibration():
    x = pd.read_csv(REPORTS / "m4_interval_sensitivity.csv")
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    ax.plot(x.confidence_level * 100, x.trajectory_coverage * 100, marker="o", label="Temporal eval trajectory coverage")
    ax.plot(x.confidence_level * 100, x.confidence_level * 100, linestyle="--", label="Nominal")
    ax.set_xlabel("Calibration confidence level (%)")
    ax.set_ylabel("Whole-voyage coverage (%)")
    ax.set_ylim(0, 105)
    ax.set_title("M4 high-reliability call-level interval calibration")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    for r in x.itertuples():
        ax.annotate(f"±{r.half_width_h*60:.0f} min", (r.confidence_level*100, r.trajectory_coverage*100),
                    textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(REPORTS / "m4_interval_calibration.png", dpi=160)
    plt.close(fig)


def reliability_panel():
    x = pd.read_csv(REPORTS / "m4_reliability_metrics.csv")
    order = [v for v in ["HIGH", "MEDIUM", "LOW"] if v in set(x.reliability_tier)]
    x = x.set_index("reliability_tier").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    ax.bar(x.reliability_tier, x.voyage_balanced_mae_h * 60)
    ax.set_yscale("log")
    ax.set_ylabel("Voyage-balanced MAE, all observed horizons (minutes, log scale)")
    ax.set_title("M4 reliability diagnostics separate supported from ambiguous states")
    ax.grid(axis="y", alpha=0.25)
    for i, r in enumerate(x.itertuples()):
        ax.text(i, r.voyage_balanced_mae_h * 60 * 1.12, f"{r.calls} calls", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(REPORTS / "m4_reliability_diagnostics.png", dpi=160)
    plt.close(fig)


def stability_tradeoff():
    x = pd.read_csv(REPORTS / "m4_stability_metrics.csv")
    x = x[x.scope.eq("all")].copy()
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    ax.scatter(x.p90_revision_min, x.voyage_balanced_mae_under24_h * 60, s=100)
    for r in x.itertuples():
        label = "Raw route physics" if r.model == "raw_route_physics" else "Causal ETA smoother"
        ax.annotate(label, (r.p90_revision_min, r.voyage_balanced_mae_under24_h*60),
                    textcoords="offset points", xytext=(7, 7), fontsize=9)
    ax.set_xlabel("P90 absolute ETA timestamp revision (minutes)")
    ax.set_ylabel("Voyage-balanced MAE at ≤24 h (minutes)")
    ax.set_title("M4 accuracy–stability trade-off on later temporal calls")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(REPORTS / "m4_stability_tradeoff.png", dpi=160)
    plt.close(fig)


def result_panel():
    s = json.loads((REPORTS / "m4_summary.json").read_text())
    p = s["primary_interval"]
    st = s["stability"]
    rp = s["reported_eta"]
    lines = [
        "M4 — uncertainty, reliability, stability",
        "",
        f"Point model retained: {s['retained_point_model']}",
        f"Calibration / temporal eval: {s['calibration_calls']} / {s['temporal_evaluation_calls']} calls",
        "16 locked final chronological calls: untouched",
        "",
        f"HIGH-reliability 90% simultaneous band: ±{p['half_width_h']*60:.0f} min",
        f"Temporal whole-voyage coverage: {p['trajectory_coverage']*100:.1f}% ({p['calls']} calls)",
        f"HIGH-reliability ≤24h MAE: {s['reliability']['high_under24_voyage_balanced_mae_h']*60:.1f} min",
        "",
        f"Causal ETA smoother alpha: {st['selected_alpha_from_calibration']:.1f}",
        f"≤24h MAE: {st['raw_eval_under24_mae_h']*60:.1f} → {st['stabilized_eval_under24_mae_h']*60:.1f} min",
        f"P90 revision: {st['raw_eval_p90_revision_min']:.1f} → {st['stabilized_eval_p90_revision_min']:.1f} min",
        f"Revision improvement: {st['eval_p90_revision_improvement_fraction']*100:.1f}%",
        "",
        f"Reported AIS ETA alignment: {rp['status']}",
        f"Closest same-MMSI snapshot: {rp['closest_snapshot_offset_min']:.1f} min away",
        "",
        f"M4 gate: {s['status']}",
        "Primary caveat: calibrated interval is intentionally withheld outside HIGH reliability.",
    ]
    fig, ax = plt.subplots(figsize=(11, 7.4))
    ax.axis("off")
    ax.text(0.04, 0.96, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(REPORTS / "m4_result_panel.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    interval_calibration(); reliability_panel(); stability_tradeoff(); result_panel()
    print("M4 visual audit complete")
