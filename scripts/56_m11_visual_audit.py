#!/usr/bin/env python3
"""Create M11 diagnostic figures. No model fitting."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"


def main() -> int:
    f = pd.read_csv(R / "m11_reference_eta_forensics.csv")
    strict = pd.read_csv(R / "m11_strict_final_error_forensics.csv")
    summary = json.loads((R / "m11_summary.json").read_text())

    # 1) Signed reference horizon sorted; highlights the heavy temporal tails.
    vals = np.sort(f["target_tte_h"].to_numpy(dtype=float)) / 24.0
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.plot(np.arange(1, len(vals) + 1), vals, linewidth=1.5)
    ax.axhline(0, linewidth=1)
    ax.axhline(7, linestyle="--", linewidth=1)
    ax.axhline(-1, linestyle="--", linewidth=1)
    ax.set_xlabel("Reference rows sorted by signed ETA horizon")
    ax.set_ylabel("Tracks reference ETA − last_update (days)")
    ax.set_title("M11 — company reference ETA temporal distribution")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.text(0.01, 0.98, "64 references are already past; 56 future references are >7 days away.", transform=ax.transAxes, va="top", fontsize=9)
    fig.tight_layout()
    fig.savefig(R / "m11_reference_horizon_distribution.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 2) Pareto concentration of strict final absolute error.
    e = strict.sort_values("abs_error_h", ascending=False)["abs_error_h"].to_numpy(dtype=float)
    cum = np.cumsum(e) / np.sum(e)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.bar(np.arange(1, len(e) + 1), e)
    ax.set_xlabel("Strict final-test rows sorted by absolute error")
    ax.set_ylabel("Absolute error (hours)")
    ax.set_title("M11 — a few extreme references dominate strict MAE")
    ax2 = ax.twinx()
    ax2.plot(np.arange(1, len(e) + 1), cum, linewidth=2)
    ax2.set_ylabel("Cumulative share of total absolute error")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(0.8, linestyle="--", linewidth=1)
    ax.text(0.02, 0.93, f"Top 5 rows = {summary['strict_final_top5_error_share']*100:.1f}% of total absolute error", transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    fig.savefig(R / "m11_final_error_concentration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 3) ETA minute granularity.
    minute_counts = f["eta_minute"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(minute_counts.index.astype(int), minute_counts.values)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 59])
    ax.set_xlabel("Minute component of Tracks.eta")
    ax.set_ylabel("Reference rows")
    ax.set_title("M11 — reference ETA time granularity")
    ax.text(0.02, 0.92, f":00 = {summary['eta_exact_hour_fraction']*100:.1f}% | :00/:30 = {summary['eta_00_or_30_fraction']*100:.1f}% | 5-min grid = {summary['eta_5min_grid_fraction']*100:.1f}%", transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    fig.savefig(R / "m11_eta_minute_granularity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 4) Executive diagnostic panel.
    fig, ax = plt.subplots(figsize=(12, 7.2))
    ax.axis("off")
    lines = [
        "M11 — Reference ETA Quality & Error Forensics",
        "",
        "NO MODEL CHANGES",
        "  M10 model, split and final-test choice remain frozen",
        "  0 company-reference rows removed from the strict benchmark",
        "",
        "REFERENCE QUALITY",
        f"  past at last_update: {summary['past_reference_rows']} / {summary['reference_rows']} ({summary['past_reference_fraction']*100:.1f}%)",
        f"  future >7 days:      {summary['future_gt7d_rows']} / {summary['reference_rows']}",
        f"  abs horizon >90 d:   {summary['abs_reference_gt90d_rows']}",
        f"  year assignment margin <90 d: {summary['year_assignment_margin_lt90d_rows']}",
        f"  placeholder-like 01/01 patterns: {summary['placeholder_like_rows']}",
        "",
        "TEMPORAL GRANULARITY",
        f"  exact hour (:00):    {summary['eta_exact_hour_fraction']*100:.1f}%",
        f"  :00 or :30:          {summary['eta_00_or_30_fraction']*100:.1f}%",
        f"  5-minute grid:       {summary['eta_5min_grid_fraction']*100:.1f}%",
        "",
        "STRICT FINAL ERROR FORENSICS",
        f"  all parseable MAE:   {summary['strict_final_mae_h']:.1f} h",
        f"  future 0–7 d MAE:    {summary['strict_final_future_0_7d_mae_h']:.1f} h",
        f"  top 1 / 3 / 5 / 10 rows explain: {summary['strict_final_top1_error_share']*100:.1f}% / {summary['strict_final_top3_error_share']*100:.1f}% / {summary['strict_final_top5_error_share']*100:.1f}% / {summary['strict_final_top10_error_share']*100:.1f}%",
        "",
        "INTERPRETATION",
        "  Tracks.eta remains the company exercise reference exactly as instructed.",
        "  The strict mean error is tail-dominated; this is reference-target forensics, not a reason to delete labels.",
        "  AIS Message 5 ETA has no year and is manually maintained voyage-related information.",
    ]
    ax.text(0.03, 0.97, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10.2)
    fig.tight_layout()
    fig.savefig(R / "m11_result_panel.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("PASS M11 visual forensic figures written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
