from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m0d import build_call_sessions, detect_gate_crossings, geometry_manifest, label_catania_geometry


def main() -> int:
    source = ROOT / "data" / "derived" / "m0c_ship_states.pkl.gz"
    if not source.exists():
        raise SystemExit(f"missing M0C dataset: {source}")

    df = pd.read_pickle(source)
    geo = label_catania_geometry(df)
    crossings = detect_gate_crossings(df)
    sessions = build_call_sessions(df, crossings=crossings)

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    crossings.to_csv(reports / "catania_m0d_gate_events.csv", index=False)
    sessions.to_csv(reports / "catania_m0d_candidate_calls.csv", index=False)

    local = geo[geo["catania_analysis_envelope"]]
    summary = {
        "geometry": geometry_manifest(),
        "rows_in_catania_analysis_envelope": int(len(local)),
        "vessels_in_catania_analysis_envelope": int(local["mmsi"].nunique()),
        "gate_events": int(len(crossings)),
        "gate_event_vessels": int(crossings["mmsi"].nunique()) if len(crossings) else 0,
        "inbound_events": int(crossings["crossing_direction"].eq("INBOUND").sum()) if len(crossings) else 0,
        "outbound_events": int(crossings["crossing_direction"].eq("OUTBOUND").sum()) if len(crossings) else 0,
        "candidate_sessions": int(len(sessions)),
        "candidate_session_vessels": int(sessions["mmsi"].nunique()) if len(sessions) else 0,
        "stop_confirmed_call_candidates": int(sessions["candidate_class"].eq("STOP_CONFIRMED_CALL_CANDIDATE").sum()) if len(sessions) else 0,
        "right_censored_sessions": int(sessions["right_censored"].sum()) if len(sessions) else 0,
        "unresolved_missing_outbound_sessions": int(sessions["unresolved_missing_outbound"].sum()) if len(sessions) else 0,
        "roadstead_wait_before_entry": int(sessions["roadstead_wait_before_entry"].sum()) if len(sessions) else 0,
        "destination_support_catania": int(sessions["destination_support_catania"].sum()) if len(sessions) else 0,
        "candidate_quality": sessions["m0d_candidate_quality"].value_counts().to_dict() if len(sessions) else {},
        "candidate_class": sessions["candidate_class"].value_counts().to_dict() if len(sessions) else {},
        "warning": "M0D outputs are candidate events only; M0E manual audit is required before any row becomes ground truth.",
    }
    (reports / "m0d_catania_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())