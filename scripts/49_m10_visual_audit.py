#!/usr/bin/env python3
"""Create M10 audit and benchmark figures."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"

STATUS_ORDER = [
    "PAST_GT24H", "PAST_1_24H", "PAST_LT1H",
    "FUTURE_0_7D", "FUTURE_7_14D", "FUTURE_14_30D", "FUTURE_GT30D",
]


def main() -> int:
    counts = pd.read_csv(R / "m10_reference_eta_status_counts.csv").set_index("reference_eta_status").reindex(STATUS_ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(np.arange(len(counts)), counts["rows"].to_numpy())
    ax.set_xticks(np.arange(len(counts)))
    ax.set_xticklabels([x.replace("_", "\n") for x in counts.index], fontsize=8)
    ax.set_ylabel("Ship Tracks rows")
    ax.set_title("M10 — company reference ETA quality audit")
    for i, v in enumerate(counts["rows"]):
        ax.text(i, float(v) + 4, str(int(v)), ha="center", va="bottom", fontsize=9)
    ax.text(0.01, -0.18, "Tracks.eta is the exercise reference label; status is measured relative to Tracks.last_update.", transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    fig.savefig(R / "m10_reference_eta_quality.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    cal = pd.read_csv(R / "m10_calibration_model_comparison.csv")
    # Cap only the chart axis? No: plot raw calibration MAE with separate panels per task.
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
    for ax, task, title in zip(
        axes,
        ["strict_all_parseable", "future_reference"],
        ["Strict benchmark — all parseable reference ETA", "Operational diagnostic — future reference ETA"],
    ):
        g = cal[cal["task"].eq(task)].sort_values("mae_h")
        ax.barh(g["model"], g["mae_h"])
        ax.invert_yaxis()
        ax.set_xlabel("Calibration MAE (hours)")
        ax.set_title(title)
        for y, v in enumerate(g["mae_h"]):
            ax.text(float(v) + max(g["mae_h"]) * 0.01, y, f"{v:.1f} h", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(R / "m10_calibration_ablation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    s = json.loads((R / "m10_benchmark_summary.json").read_text())
    by = pd.read_csv(R / "m10_future_test_by_reference_status.csv")
    zero7 = by.loc[by["reference_eta_status"].eq("FUTURE_0_7D")].iloc[0]
    strict = s["strict"]; fut = s["future_reference"]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")
    lines = [
        "M10 — Company Reference-ETA Benchmark",
        "",
        "DATA CONTRACT",
        "  439 parseable ship ETA references = 439 supervised MMSI rows",
        "  ETA is target only; historical Positions are cut at/before Tracks.last_update",
        "  deterministic target-free MMSI split; previous M6 holdout is not reused",
        "",
        "REFERENCE-LABEL AUDIT",
        "  64 / 439 reference ETAs are already in the past at last_update",
        "  8 / 439 are >30 days in the future",
        "  this is treated as label/reference quality, not silently cleaned away",
        "",
        "STRICT FINAL TEST — all parseable labels",
        f"  selected on calibration: {strict['selected_model']}",
        f"  n={strict['final_test_n']} | MAE={strict['selected_final_metrics']['mae_h']:.1f} h | MedAE={strict['selected_final_metrics']['medae_h']:.1f} h",
        "",
        "FUTURE-REFERENCE FINAL TEST",
        f"  selected on calibration: {fut['selected_model']}",
        f"  n={fut['final_test_n']} | MAE={fut['selected_final_metrics']['mae_h']:.1f} h | MedAE={fut['selected_final_metrics']['medae_h']:.1f} h",
        f"  0–7 day subset: n={int(zero7['n'])} | MAE={zero7['mae_h']:.1f} h | within ±24 h={zero7['within_24h']*100:.1f}% | within ±48 h={zero7['within_48h']*100:.1f}%",
        "",
        "INTERPRETATION",
        "  The company-defined task is now implemented exactly as a Tracks.eta reference benchmark.",
        "  Current/history ML did not clearly dominate simple destination priors on this small/noisy snapshot target.",
        "  These scores are NOT actual-arrival accuracy and NOT a comparison with the external existing model.",
    ]
    ax.text(0.03, 0.97, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(R / "m10_result_panel.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("PASS M10 visual audit figures written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
