#!/usr/bin/env python3
"""Build the M0C causal vessel-state layer and audit reports."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m0c import M0CConfig, engineer_semantic_states, sentinel_summary

INPUTS = [ROOT / "data/vessel_positions_part1.csv", ROOT / "data/vessel_positions_part2.csv"]
DERIVED = ROOT / "data/derived/m0c_ship_states.pkl.gz"
REPORTS = ROOT / "reports"

USECOLS = [
    "id", "mmsi", "name", "category", "nav_status", "nav_status_str", "sog", "cog",
    "heading", "lat", "lon", "destination", "draught", "manoeuvre_str",
    "recorded_at", "created_at", "updated_at",
]


def load_ships() -> pd.DataFrame:
    parts = []
    for path in INPUTS:
        d = pd.read_csv(path, usecols=USECOLS, low_memory=False)
        parts.append(d.loc[d["category"].eq("ship")])
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    DERIVED.parent.mkdir(parents=True, exist_ok=True)

    raw = load_ships()
    states = engineer_semantic_states(raw, M0CConfig())
    sentinels = sentinel_summary(raw, states)

    # Persist the complete layer locally (data/ is intentionally gitignored).
    states.to_pickle(DERIVED, compression="gzip", protocol=5)

    state_counts = states["motion_state"].value_counts(dropna=False).rename_axis("motion_state").reset_index(name="rows")
    state_counts["pct"] = state_counts["rows"] / len(states)
    state_counts.to_csv(REPORTS / "m0c_motion_state_counts.csv", index=False)

    stale_counts = states["stale_risk_level"].value_counts(dropna=False).rename_axis("stale_risk_level").reset_index(name="rows")
    stale_counts["pct"] = stale_counts["rows"] / len(states)
    stale_counts.to_csv(REPORTS / "m0c_stale_risk_counts.csv", index=False)

    event_cols = ["stop_start", "stop_end", "slow_start", "slow_end", "turn_candidate", "observation_gap_candidate", "kinematic_conflict", "position_jump_candidate"]
    event_counts = {c: int(states[c].sum()) for c in event_cols}

    # Prefix-invariance spot check on several vessels: core causal columns should be unchanged
    # when future observations are removed.
    causal_cols = [
        "observation_dt_s", "position_delta_m", "implied_speed_kn", "cog_delta_deg",
        "dynamic_state_changed", "dynamic_state_age_s", "stop_candidate", "slow_motion_candidate",
        "moving_candidate", "turn_candidate", "observation_gap_candidate", "kinematic_conflict",
        "stale_risk_level", "motion_state", "stop_duration_s", "slow_duration_s",
    ]
    prefix_checks = []
    for mmsi, g in raw.groupby("mmsi"):
        if len(g) < 40:
            continue
        g = g.sort_values(["recorded_at", "id"], kind="mergesort").head(80)
        cut = len(g) // 2
        a = engineer_semantic_states(g.iloc[:cut].copy())
        b = engineer_semantic_states(g.copy()).iloc[:cut].reset_index(drop=True)
        ok = True
        for c in causal_cols:
            x, y = a[c].reset_index(drop=True), b[c]
            if pd.api.types.is_numeric_dtype(x):
                ok &= bool(np.allclose(pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce"), equal_nan=True))
            else:
                ok &= bool(x.fillna("<NA>").astype(str).equals(y.fillna("<NA>").astype(str)))
        prefix_checks.append({"mmsi": int(mmsi), "rows_checked": cut, "pass": bool(ok)})
        if len(prefix_checks) >= 12:
            break

    # Representative vessel: prefer one with multiple semantic states + turns + stops.
    vessel_summary = states.groupby("mmsi").agg(
        rows=("mmsi", "size"),
        n_motion_states=("motion_state", "nunique"),
        stops=("stop_start", "sum"),
        turns=("turn_candidate", "sum"),
        conflicts=("kinematic_conflict", "sum"),
        gaps=("observation_gap_candidate", "sum"),
        start=("recorded_at", "min"),
        end=("recorded_at", "max"),
        name=("name", "last"),
    ).reset_index()
    candidates = vessel_summary.query("rows >= 100 and n_motion_states >= 3 and stops >= 1 and turns >= 1")
    if candidates.empty:
        candidates = vessel_summary.query("rows >= 100 and n_motion_states >= 2")
    candidates = candidates.assign(score=candidates["n_motion_states"] * 10 + candidates["stops"].clip(upper=10) + candidates["turns"].clip(upper=10) - candidates["conflicts"].clip(upper=20) * 0.1)
    rep = candidates.sort_values(["score", "rows"], ascending=False).iloc[0]
    rep_mmsi = int(rep["mmsi"])
    timeline = states.loc[states["mmsi"].eq(rep_mmsi)].copy()
    # Keep a meaningful window around the first semantic event, capped for audit readability.
    event_mask = timeline["semantic_events"].fillna("").ne("")
    if event_mask.any():
        first_event_idx = timeline.index[event_mask][0]
        pos = timeline.index.get_loc(first_event_idx)
        timeline = timeline.iloc[max(0, pos - 60): min(len(timeline), pos + 180)]
    else:
        timeline = timeline.head(240)

    sample_cols = [
        "mmsi", "name", "recorded_at", "lat_clean", "lon_clean", "sog_clean", "cog_clean",
        "nav_status_name", "observation_dt_s", "position_delta_m", "implied_speed_kn",
        "dynamic_state_age_s", "stale_risk_level", "motion_state", "stop_duration_s",
        "slow_duration_s", "semantic_events",
    ]
    timeline[sample_cols].to_csv(REPORTS / "m0c_representative_timeline.csv", index=False)

    # Diverse event sample for human audit.
    event_sample = states.loc[states["semantic_events"].fillna("").ne(""), sample_cols].copy()
    event_sample = event_sample.groupby("mmsi", group_keys=False).head(8).head(500)
    event_sample.to_csv(REPORTS / "m0c_semantic_event_sample.csv", index=False)

    summary = {
        "rows": int(len(states)),
        "mmsi": int(states["mmsi"].nunique()),
        "time_min": str(states["recorded_at"].min()),
        "time_max": str(states["recorded_at"].max()),
        "sentinels": sentinels,
        "motion_state_counts": {str(r.motion_state): int(r.rows) for r in state_counts.itertuples()},
        "stale_risk_counts": {str(r.stale_risk_level): int(r.rows) for r in stale_counts.itertuples()},
        "event_counts": event_counts,
        "dynamic_state_repeated_pct": float(states["dynamic_state_repeated"].mean()),
        "position_repeated_pct": float(states["position_repeated"].mean()),
        "state_age_s_quantiles": {str(q): float(v) for q, v in states["dynamic_state_age_s"].quantile([0.5, 0.9, 0.95, 0.99, 1.0]).items()},
        "implied_speed_kn_quantiles_nonjump": {str(q): float(v) for q, v in states.loc[~states["position_jump_candidate"], "implied_speed_kn"].dropna().quantile([0.5, 0.9, 0.95, 0.99]).items()},
        "prefix_invariance_checks": prefix_checks,
        "prefix_invariance_all_pass": bool(all(x["pass"] for x in prefix_checks)),
        "representative_mmsi": rep_mmsi,
        "representative_name": None if pd.isna(rep["name"]) else str(rep["name"]),
        "derived_file": str(DERIVED.relative_to(ROOT)),
    }
    (REPORTS / "m0c_state_summary.json").write_text(json.dumps(summary, indent=2))
    (REPORTS / "m0c_sentinel_report.json").write_text(json.dumps(sentinels, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
