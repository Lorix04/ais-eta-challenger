#!/usr/bin/env python3
"""Verify M1 causal panel and locked-test contract."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m1 import M1Config, assign_call_splits, build_decision_panel  # noqa: E402


def main() -> None:
    reports = ROOT / "reports"
    panel = pd.read_csv(
        reports / "catania_m1_decision_panel.csv",
        parse_dates=[
            "ground_truth_time",
            "decision_time",
            "voyage_start_time",
            "source_observation_time",
        ],
    )
    splits = pd.read_csv(
        reports / "catania_m1_call_splits.csv",
        parse_dates=["ground_truth_time"],
    )
    metrics = pd.read_csv(reports / "catania_m1_baseline_metrics.csv")
    summary = json.loads((reports / "m1_summary.json").read_text(encoding="utf-8"))

    checks: list[tuple[str, bool]] = []
    checks.append(("panel rows unique by session/decision", not panel.duplicated(["session_id", "decision_time"]).any()))
    checks.append(("source observation never after decision", bool((panel["source_observation_time"] <= panel["decision_time"]).all())))
    checks.append(("decision always before ground truth", bool((panel["decision_time"] < panel["ground_truth_time"]).all())))
    checks.append(("voyage start not after decision", bool((panel["voyage_start_time"] <= panel["decision_time"]).all())))
    checks.append(("provider age nonnegative", bool((panel["provider_observation_age_s"] >= 0).all())))
    checks.append(("15-minute wall-clock grid", bool(panel["decision_time"].dt.minute.mod(15).eq(0).all() and panel["decision_time"].dt.second.eq(0).all())))
    final = splits[splits["m1_split"].eq("final_chronological_test")]
    dev = splits[splits["m1_split"].eq("development")]
    checks.append(("final chronological test has 6 calls", len(final) == 6))
    checks.append(("final test later than development", bool(final["ground_truth_time"].min() > dev["ground_truth_time"].max())))
    checks.append(("final test has no CV folds", final["session_cv_fold"].isna().all() and final["vessel_cv_fold"].isna().all()))
    checks.append(("development has CV folds", dev["session_cv_fold"].notna().all() and dev["vessel_cv_fold"].notna().all()))
    checks.append(("metrics are development-only", metrics["development_only"].astype(bool).all()))
    checks.append(("no final-test rows used in reported metrics", not metrics.empty))
    checks.append(("M1 gate expands ground truth", summary["m1_gate"] == "EXPAND_GROUND_TRUTH_BEFORE_COMPLEXITY"))
    checks.append(("under6h approach covers >=10 calls", int(summary["approach_under6h_development_calls"]) >= 10))
    checks.append(("long-horizon approach evidence remains sparse", int(summary["approach_over6h_development_calls"]) < 8))

    # Prefix-invariance spot checks: for several development decision points,
    # rebuild using only states observed up to that decision and confirm that
    # the prediction-time features at that timestamp are unchanged.
    states = pd.read_pickle(ROOT / "data" / "derived" / "m0c_ship_states.pkl.gz")
    cohort = pd.read_csv(reports / "catania_m0e_eta_primary_cohort.csv")
    events = pd.read_csv(reports / "catania_m0d_gate_events.csv")
    cfg = M1Config(decision_interval_min=15)
    split_table = assign_call_splits(cohort, cfg)
    candidates = panel[
        panel["m1_split"].eq("development") & panel["scope_inbound_approach"].astype(bool)
    ].drop_duplicates("session_id").head(8)
    invariant = 0
    keys = [
        "source_observation_time",
        "provider_observation_age_s",
        "distance_to_gate_mid_km",
        "sog_current_kn",
        "sog_median_30m_kn",
        "progress_to_gate_30m_kn",
        "destination_supports_catania",
        "stale_risk_level",
        "scope_inbound_approach",
        "pred_b2_geodesic_median30_sog_floor_h",
    ]
    for row in candidates.itertuples(index=False):
        one_call = cohort[cohort["session_id"].eq(row.session_id)].copy()
        one_split = split_table[split_table["session_id"].eq(row.session_id)].copy()
        prefix_states = states[
            (states["mmsi"].astype(int).eq(int(row.mmsi)))
            & (pd.to_datetime(states["recorded_at"]) <= pd.Timestamp(row.decision_time))
        ].copy()
        prefix_events = events[
            ~(
                events["mmsi"].astype(int).eq(int(row.mmsi))
                & (pd.to_datetime(events["crossing_time_proxy"]) > pd.Timestamp(row.decision_time))
            )
        ].copy()
        rebuilt = build_decision_panel(prefix_states, one_call, prefix_events, one_split, cfg)
        match = rebuilt[rebuilt["decision_time"].eq(pd.Timestamp(row.decision_time))]
        if len(match) != 1:
            continue
        got = match.iloc[0]
        original = panel[
            panel["session_id"].eq(row.session_id)
            & panel["decision_time"].eq(pd.Timestamp(row.decision_time))
        ].iloc[0]
        same = True
        for key in keys:
            a, b = original[key], got[key]
            if isinstance(a, str) or isinstance(b, str):
                same &= str(a) == str(b)
            elif pd.isna(a) and pd.isna(b):
                pass
            elif isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
                same &= bool(a) == bool(b)
            elif key.endswith("time"):
                same &= pd.Timestamp(a) == pd.Timestamp(b)
            else:
                same &= bool(np.isclose(float(a), float(b), rtol=1e-10, atol=1e-10))
        invariant += int(same)
    checks.append((f"prefix invariance {invariant}/{len(candidates)}", invariant == len(candidates)))

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(("PASS" if ok else "FAIL"), name)
    if failed:
        raise SystemExit(f"M1 verification failed: {failed}")
    print(f"M1 verification PASS: {len(checks)} checks")


if __name__ == "__main__":
    main()
