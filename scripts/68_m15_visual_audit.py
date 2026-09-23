#!/usr/bin/env python3
"""Generate an internal M15 technical-defense map from frozen facts."""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"


def main() -> int:
    m = json.loads((R / "M15_DEFENSE_MANIFEST.json").read_text(encoding="utf-8"))
    b = m["primary_benchmark"]
    d = m["diagnostic_facts"]

    fig = plt.figure(figsize=(13, 8))
    fig.text(0.055, 0.93, "M15 — Interview / Technical Defense Map", fontsize=21, weight="bold")
    fig.text(0.055, 0.885, "Internal post-freeze preparation — M14 submission remains immutable", fontsize=12)

    sections = [
        ("1  TARGET", [
            "Company reference: Tracks.eta (not observed ATA)",
            "One supervised row per MMSI; causal cutoff at Tracks.last_update",
        ]),
        ("2  PROTOCOL", [
            f"Split: {b['split']['train']} train / {b['split']['calibration']} calibration / {b['split']['final_test']} final",
            "Selection on calibration; final opened once; no post-final tuning",
        ]),
        ("3  STRICT FINAL", [
            f"MAE {b['final_mae_h']:.1f} h | MedAE {b['final_medae_h']:.1f} h | P90 {b['final_p90_abs_error_h']:.1f} h",
            f"Frozen CatBoost: {b['tree_count']} trees | {b['feature_count']} features",
        ]),
        ("4  FORENSICS", [
            f"Past references: {d['past_reference_rows']} | >7d future: {d['future_gt7d_rows']}",
            f"Top-5 final errors: {100*d['top5_final_abs_error_share']:.1f}% of total absolute error",
            f"0–7d same-model diagnostic: {d['future_0_7d_final_rows']} rows | {d['future_0_7d_same_model_mae_h']:.1f} h MAE",
        ]),
        ("5  DEFENSE RULE", [
            "Explain SHAP as model attribution, not causality",
            "Do not claim external-model superiority, ATA equivalence or production readiness",
            "Lead with leakage control, evaluation discipline and reproducibility",
        ]),
    ]

    y = 0.80
    for title, lines in sections:
        fig.text(0.07, y, title, fontsize=14, weight="bold")
        y -= 0.04
        for line in lines:
            fig.text(0.095, y, "• " + line, fontsize=11.5)
            y -= 0.044
        y -= 0.022

    fig.text(0.055, 0.055, "Status: TECHNICAL DEFENSE READY — scientific freeze unchanged", fontsize=15, weight="bold")
    plt.axis("off")
    out = R / "m15_defense_map.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"PASS wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
