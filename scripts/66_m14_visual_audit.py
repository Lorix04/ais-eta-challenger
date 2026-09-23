#!/usr/bin/env python3
"""Create a concise M14 final-freeze result panel."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"


def main() -> int:
    freeze = json.loads((R / "M14_FINAL_FREEZE.json").read_text())
    strict = freeze["strict_final"]
    fig = plt.figure(figsize=(12, 7.2))
    fig.text(0.06, 0.91, "M14 — Final Reproducibility & Submission Freeze", fontsize=20, weight="bold")
    fig.text(0.06, 0.85, "Scientific state unchanged; packaging/integrity only", fontsize=12)
    lines = [
        f"Strict final: {strict['rows']} MMSIs | MAE {strict['mae_h']:.1f} h | MedAE {strict['medae_h']:.1f} h | P90 {strict['p90_abs_error_h']:.1f} h",
        f"Frozen M10 model: {freeze['frozen_artifacts']['m10_model_sha256'][:16]}…",
        f"Frozen M10 reference dataset: {freeze['frozen_artifacts']['m10_reference_dataset_sha256'][:16]}…",
        f"Frozen M10 final predictions: {freeze['frozen_artifacts']['m10_final_predictions_sha256'][:16]}…",
        "No model/split/final-prediction change; no post-final tuning",
        "Final ZIP: deterministic order + fixed metadata + per-file SHA-256 manifest",
        "Clean-room: package integrity verifier + pytest + compileall",
        "Raw AIS / high-volume row-level data / internal evidence excluded",
    ]
    y = 0.75
    for line in lines:
        fig.text(0.08, y, "• " + line, fontsize=12)
        y -= 0.075
    fig.text(0.06, 0.08, "Status: FINAL SUBMISSION FREEZE", fontsize=16, weight="bold")
    plt.axis("off")
    out = R / "m14_result_panel.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"PASS wrote {out.relative_to(ROOT)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
