#!/usr/bin/env python3
"""Create M2 visual evidence: route-value ablation, fold consistency and maps."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

LABELS = {
    "pred_m2_geodesic_h": "Geodesic",
    "pred_m2_sector_prototype_h": "Sector prototype",
    "pred_m2_route_knn_h": "Route-kNN",
}


def main() -> None:
    metrics = pd.read_csv(REPORTS / "m2_eta_metrics.csv")
    folds = pd.read_csv(REPORTS / "m2_fold_metrics.csv")
    models = json.loads((REPORTS / "m2_route_models.json").read_text())

    # 1) Scope-specific improvement relative to geodesic.
    scopes = ["under6h", "under12h", "under24h", "all_oof"]
    rows = []
    for scope in scopes:
        q = metrics[metrics.scope.eq(scope)].set_index("baseline")
        geo = float(q.loc["pred_m2_geodesic_h", "voyage_balanced_mae_h"])
        for pred in ["pred_m2_sector_prototype_h", "pred_m2_route_knn_h"]:
            mae = float(q.loc[pred, "voyage_balanced_mae_h"])
            rows.append({
                "scope": scope,
                "model": LABELS[pred],
                "improvement_pct": 100.0 * (geo - mae) / geo,
            })
    imp = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=170)
    x = np.arange(len(scopes))
    width = 0.36
    for j, model in enumerate(["Sector prototype", "Route-kNN"]):
        vals = [float(imp[(imp.scope.eq(s)) & imp.model.eq(model)].improvement_pct.iloc[0]) for s in scopes]
        ax.bar(x + (j - 0.5) * width, vals, width=width, label=model)
    ax.axhline(0, linewidth=1)
    ax.set_xticks(x, ["<=6h", "<=12h", "<=24h", "all OOF"])
    ax.set_ylabel("Voyage-balanced MAE improvement vs geodesic (%)")
    ax.set_title("M2 train-only route ablation")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(REPORTS / "m2_route_ablation.png")
    plt.close(fig)

    # 2) Fold consistency for route-kNN.
    all_fold = folds[folds.scope.eq("all")].copy()
    pivot = all_fold.pivot_table(
        index=["port", "m2_temporal_fold"], columns="baseline", values="voyage_balanced_mae_h"
    ).reset_index()
    pivot["improvement_pct"] = 100.0 * (
        pivot["pred_m2_geodesic_h"] - pivot["pred_m2_route_knn_h"]
    ) / pivot["pred_m2_geodesic_h"]
    pivot["fold_label"] = pivot["port"] + " F" + pivot["m2_temporal_fold"].astype(int).astype(str)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=170)
    ax.bar(pivot["fold_label"], pivot["improvement_pct"])
    ax.axhline(0, linewidth=1)
    ax.set_ylabel("Route-kNN MAE improvement vs geodesic (%)")
    ax.set_title("M2 expanding-fold consistency")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(REPORTS / "m2_fold_consistency.png")
    plt.close(fig)

    # 3) Train-only model geometry for the last fold of each port.  These are
    # diagnostic maps only; they are not nautical charts.
    for port in ["CATANIA", "AUGUSTA"]:
        model = models[f"{port}_fold3"]
        fig, ax = plt.subplots(figsize=(7, 7), dpi=170)
        for path in model["individual_paths"].values():
            p = pd.DataFrame(path["points"])
            ax.plot(p.lon, p.lat, linewidth=0.7, alpha=0.22)
        pp = pd.DataFrame(model["port_prototype"]["points"])
        ax.plot(pp.lon, pp.lat, linewidth=2.5, label="Port-wide prototype")
        for fam, proto in model["family_prototypes"].items():
            p = pd.DataFrame(proto["points"])
            ax.plot(p.lon, p.lat, linewidth=2.0, label=f"{fam} prototype")
        ax.scatter([model["gate_lon"]], [model["gate_lat"]], marker="x", s=70, label="Research gate")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"{port} M2 fold-3 train-only historical routes")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()
        fig.savefig(REPORTS / f"m2_{port.lower()}_fold3_route_map.png")
        plt.close(fig)

    print("wrote M2 route ablation, fold consistency, and train-only route maps")


if __name__ == "__main__":
    main()
