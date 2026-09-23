#!/usr/bin/env python3
"""Build the M10 one-row-per-vessel company reference-ETA dataset."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m10 import M10Config, build_reference_rows

TRACKS = ROOT / "data" / "vessel_tracks.csv"
STATES = ROOT / "data" / "derived" / "m0c_ship_states.pkl.gz"
OUT = ROOT / "data" / "derived" / "m10_reference_rows.pkl.gz"
REPORTS = ROOT / "reports"

STATUS_ORDER = [
    "PAST_GT24H", "PAST_1_24H", "PAST_LT1H",
    "FUTURE_0_7D", "FUTURE_7_14D", "FUTURE_14_30D", "FUTURE_GT30D",
]


def main() -> int:
    REPORTS.mkdir(exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not TRACKS.exists():
        raise FileNotFoundError(f"Missing {TRACKS}. Place the company Tracks CSV under data/.")
    if not STATES.exists():
        raise FileNotFoundError(f"Missing {STATES}. Run M0C first.")

    tracks = pd.read_csv(TRACKS, low_memory=False)
    states = pd.read_pickle(STATES)
    rows = build_reference_rows(tracks, states, M10Config())
    rows.to_pickle(OUT, compression="gzip", protocol=5)

    # Compact, non-row-level audit tables only.
    status = (
        rows["reference_eta_status"].value_counts().reindex(STATUS_ORDER, fill_value=0)
        .rename_axis("reference_eta_status").reset_index(name="rows")
    )
    status["pct"] = status["rows"] / len(rows)
    status.to_csv(REPORTS / "m10_reference_eta_status_counts.csv", index=False)

    split = rows["split"].value_counts().rename_axis("split").reset_index(name="rows")
    split.to_csv(REPORTS / "m10_split_counts.csv", index=False)

    target_summary = rows["target_tte_h"].describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
    summary = {
        "tracks_rows_total": int(len(tracks)),
        "ship_tracks": int(tracks["category"].eq("ship").sum()),
        "ship_eta_nonnull": int((tracks["category"].eq("ship") & tracks["eta"].notna()).sum()),
        "supervised_parseable_ship_eta_rows": int(len(rows)),
        "supervised_unique_mmsi": int(rows["mmsi"].nunique()),
        "one_row_per_mmsi": bool(len(rows) == rows["mmsi"].nunique()),
        "status_counts": {str(r.reference_eta_status): int(r.rows) for r in status.itertuples()},
        "split_counts": {str(r.split): int(r.rows) for r in split.itertuples()},
        "future_reference_rows": int((rows["target_tte_h"] >= 0).sum()),
        "past_reference_rows": int((rows["target_tte_h"] < 0).sum()),
        "target_tte_h_summary": {str(k): float(v) for k, v in target_summary.items()},
        "median_position_age_min": float(rows["position_age_min"].median()),
        "p95_position_age_min": float(rows["position_age_min"].quantile(0.95)),
        "rows_without_history": int(rows["history_n"].isna().sum()),
        "target_is_feature": False,
        "eta_reference_interpretation": "company exercise reference label; not observed ATA",
        "split_policy": "deterministic SHA-256 hash of MMSI only, 70/15/15",
        "derived_file": str(OUT.relative_to(ROOT)),
    }
    (REPORTS / "m10_reference_dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
