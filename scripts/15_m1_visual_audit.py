#!/usr/bin/env python3
"""Generate M1 diagnostic figures without scoring the locked final test."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def main() -> None:
    splits = pd.read_csv(REPORTS / "catania_m1_call_splits.csv")
    panel = pd.read_csv(REPORTS / "catania_m1_decision_panel.csv")
    horizon = pd.read_csv(REPORTS / "catania_m1_horizon_metrics.csv")

    # 1) Repeated-vessel concentration.
    counts = (
        splits.groupby(["mmsi", "name"])["session_id"]
        .size()
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(11, 5.5))
    labels = [name for _, name in counts.index]
    ax.bar(range(len(counts)), counts.values)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(labels, rotation=55, ha="right")
    ax.set_ylabel("Validated ETA calls")
    ax.set_title("M1 — Catania primary cohort is concentrated in repeated vessels")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(REPORTS / "m1_call_concentration.png", dpi=160)
    plt.close(fig)

    # 2) Horizon coverage: number of independent development calls.
    dev = panel[panel["m1_split"].eq("development")].copy()
    rows = []
    for scope, label in [
        ("scope_conditional_target_known", "Target known externally"),
        ("scope_declared_destination", "AIS destination says Catania"),
        ("scope_inbound_approach", "Causal inbound-approach state"),
    ]:
        q = dev[dev[scope].astype(bool)]
        for band in ["<6h", "6-12h", "12-24h", "24-48h", "48-72h", ">72h"]:
            rows.append({"scope": label, "band": band, "calls": q[q["horizon_band"].eq(band)]["session_id"].nunique()})
    cov = pd.DataFrame(rows)
    pivot = cov.pivot(index="band", columns="scope", values="calls").reindex(["<6h", "6-12h", "12-24h", "24-48h", "48-72h", ">72h"])
    fig, ax = plt.subplots(figsize=(10, 5.5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Independent development calls")
    ax.set_xlabel("True horizon (post-hoc reporting only)")
    ax.set_title("M1 — usable long-horizon evidence collapses under causal scopes")
    ax.legend(title="Causal scope")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(REPORTS / "m1_horizon_coverage.png", dpi=160)
    plt.close(fig)

    # 3) Robust physical-baseline errors on approach states, development only.
    q = dev[
        dev["scope_inbound_approach"].astype(bool)
        & dev["pred_b2_geodesic_median30_sog_floor_h"].notna()
    ].copy()
    q["abs_error_h"] = (q["pred_b2_geodesic_median30_sog_floor_h"] - q["true_tta_h"]).abs()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(q["true_tta_h"], q["abs_error_h"], alpha=0.7)
    ax.set_xlabel("True time to research-gate arrival (h)")
    ax.set_ylabel("Absolute error (h)")
    ax.set_title("M1 — physical baseline is strong in final approach, brittle on rare long waits")
    ax.grid(alpha=0.25)
    for session_id in ["CT-041", "CT-019"]:
        z = q[q["session_id"].eq(session_id)]
        if len(z):
            row = z.iloc[z["abs_error_h"].argmax()]
            ax.annotate(session_id, (row["true_tta_h"], row["abs_error_h"]), xytext=(6, 6), textcoords="offset points")
    fig.tight_layout()
    fig.savefig(REPORTS / "m1_baseline_error_vs_horizon.png", dpi=160)
    plt.close(fig)

    print("wrote M1 diagnostic figures")


if __name__ == "__main__":
    main()
