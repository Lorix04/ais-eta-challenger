from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m0d import CataniaGeometryV1, detect_gate_crossings, label_catania_geometry


def main() -> int:
    df = pd.read_pickle(ROOT / "data" / "derived" / "m0c_ship_states.pkl.gz")
    events = pd.read_csv(ROOT / "reports" / "catania_m0d_gate_events.csv", parse_dates=[
        "crossing_last_before", "crossing_first_after", "crossing_time_proxy"
    ])
    calls = pd.read_csv(ROOT / "reports" / "catania_m0d_candidate_calls.csv", parse_dates=[
        "inbound_time", "inbound_last_outside", "inbound_first_inside", "outbound_time"
    ])
    geom = CataniaGeometryV1()

    assert len(events) > 0, "no Catania gate events"
    assert len(calls) > 0, "no Catania candidate sessions"
    assert set(events["crossing_direction"].unique()) <= {"INBOUND", "OUTBOUND"}
    assert events["crossing_uncertainty_s"].between(0, 300).all()
    assert (events["crossing_first_after"] >= events["crossing_last_before"]).all()
    assert calls["final_ground_truth"].eq(False).all(), "M0D must not create final ground truth"
    assert calls["requires_manual_audit"].eq(True).all()
    assert calls["geometry_version"].eq(geom.version).all()

    # Every crossing point must lie on the finite gate bounding box within rounded-coordinate tolerance.
    lon_min, lon_max = sorted([geom.gate_west[0], geom.gate_east[0]])
    lat_min, lat_max = sorted([geom.gate_west[1], geom.gate_east[1]])
    assert events["crossing_lon"].between(lon_min - 1e-6, lon_max + 1e-6).all()
    assert events["crossing_lat"].between(lat_min - 1e-6, lat_max + 1e-6).all()

    # Prefix invariance: geometry-state columns must not change when future rows are removed.
    sample_mmsi = (
        events[events["crossing_direction"].eq("INBOUND")]["mmsi"].drop_duplicates().head(12).tolist()
    )
    checks = 0
    cols = [
        "catania_gate_side",
        "catania_port_side",
        "catania_inner_harbour_proxy",
        "catania_distance_to_gate_mid_km",
        "catania_approach_proxy",
        "catania_roadstead_wait_proxy",
    ]
    for mmsi in sample_mmsi:
        vessel = df[df["mmsi"].eq(mmsi)].sort_values(["recorded_at", "id"]).reset_index(drop=True)
        if len(vessel) < 20:
            continue
        cut = min(len(vessel) - 1, max(10, len(vessel) // 3))
        prefix = label_catania_geometry(vessel.iloc[:cut].copy()).reset_index(drop=True)
        full = label_catania_geometry(vessel.copy()).iloc[:cut].reset_index(drop=True)
        pd.testing.assert_frame_equal(prefix[cols], full[cols], check_dtype=False)
        checks += 1

    # Re-run crossing detector on prefixes ending just after selected inbound events.
    crossing_prefix_checks = 0
    for row in events[events["crossing_direction"].eq("INBOUND")].head(10).itertuples(index=False):
        vessel = df[df["mmsi"].eq(row.mmsi)].sort_values(["recorded_at", "id"])
        prefix = vessel[vessel["recorded_at"] <= row.crossing_first_after].copy()
        got = detect_gate_crossings(prefix)
        match = got[
            (got["crossing_direction"].eq("INBOUND"))
            & (got["crossing_first_after"].eq(row.crossing_first_after))
        ]
        assert len(match) == 1, f"crossing changed with future truncation for MMSI {row.mmsi}"
        crossing_prefix_checks += 1

    result = {
        "events": len(events),
        "candidate_sessions": len(calls),
        "geometry_prefix_invariance_checks": checks,
        "crossing_prefix_invariance_checks": crossing_prefix_checks,
        "all_candidate_rows_require_manual_audit": True,
        "final_ground_truth_rows": int(calls["final_ground_truth"].sum()),
        "status": "PASS",
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())