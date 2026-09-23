from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m0c import engineer_semantic_states
from ais_eta.m0d import CataniaGeometryV1, build_call_sessions, detect_gate_crossings, label_catania_geometry


def frame(points):
    rows = []
    for i, point in enumerate(points, 1):
        ts, lat, lon, sog = point[:4]
        destination = point[4] if len(point) > 4 else "ITCTA"
        rows.append(
            {
                "id": i,
                "mmsi": 247000001,
                "name": "TEST VESSEL",
                "category": "ship",
                "nav_status": 0,
                "nav_status_str": "",
                "sog": sog,
                "cog": 350.0 if sog > 0.5 else 0.0,
                "heading": 350.0 if sog > 0.5 else 0.0,
                "lat": lat,
                "lon": lon,
                "destination": destination,
                "draught": 6.0,
                "manoeuvre_str": "",
                "recorded_at": ts,
                "created_at": ts,
                "updated_at": ts,
            }
        )
    return engineer_semantic_states(pd.DataFrame(rows))


def test_gate_side_places_inner_reference_port_side():
    d = frame([
        ("2026-04-01 00:00:00", 37.4960, 15.0940, 0.0),
        ("2026-04-01 00:01:00", 37.4700, 15.1050, 0.0),
    ])
    x = label_catania_geometry(d)
    assert bool(x.loc[0, "catania_port_side"])
    assert not bool(x.loc[1, "catania_port_side"])


def test_inbound_crossing_has_bracket_and_direction():
    d = frame([
        ("2026-04-01 00:00:00", 37.4840, 15.0980, 5.0),
        ("2026-04-01 00:01:00", 37.4870, 15.0972, 5.0),
    ])
    e = detect_gate_crossings(d)
    assert len(e) == 1
    assert e.loc[0, "crossing_direction"] == "INBOUND"
    assert e.loc[0, "crossing_uncertainty_s"] == 60.0
    assert e.loc[0, "crossing_last_before"] == pd.Timestamp("2026-04-01 00:00:00")
    assert e.loc[0, "crossing_first_after"] == pd.Timestamp("2026-04-01 00:01:00")


def test_parallel_or_off_gate_motion_is_not_crossing():
    d = frame([
        ("2026-04-01 00:00:00", 37.4700, 15.1200, 6.0),
        ("2026-04-01 00:01:00", 37.5000, 15.1200, 6.0),
    ])
    assert detect_gate_crossings(d).empty


def test_stop_after_entry_confirms_candidate_but_not_final_ground_truth():
    points = [
        ("2026-04-01 00:00:00", 37.4840, 15.0980, 5.0),
        ("2026-04-01 00:01:00", 37.4870, 15.0972, 5.0),
        ("2026-04-01 00:02:00", 37.4950, 15.0940, 0.0),
    ]
    # 20 minutes of stationary observations build a causal stop duration.
    for minute in range(3, 23):
        points.append((f"2026-04-01 00:{minute:02d}:00", 37.4950, 15.0940, 0.0))
    points.extend([
        ("2026-04-01 00:23:00", 37.4870, 15.0972, 5.0),
        ("2026-04-01 00:24:00", 37.4840, 15.0980, 5.0),
    ])
    d = frame(points)
    s = build_call_sessions(d)
    assert len(s) == 1
    assert s.loc[0, "candidate_class"] == "STOP_CONFIRMED_CALL_CANDIDATE"
    assert bool(s.loc[0, "requires_manual_audit"])
    assert not bool(s.loc[0, "final_ground_truth"])
    assert bool(s.loc[0, "session_closed"])
    assert not bool(s.loc[0, "unresolved_missing_outbound"])


def test_short_exit_reentry_is_hysteresis_merged():
    points = [
        ("2026-04-01 00:00:00", 37.4840, 15.0980, 5.0),
        ("2026-04-01 00:01:00", 37.4870, 15.0972, 5.0),  # inbound
        ("2026-04-01 00:02:00", 37.4840, 15.0980, 5.0),  # outbound
        ("2026-04-01 00:05:00", 37.4870, 15.0972, 5.0),  # re-entry within hysteresis
        ("2026-04-01 00:06:00", 37.4950, 15.0940, 0.0),
    ]
    for minute in range(7, 24):
        points.append((f"2026-04-01 00:{minute:02d}:00", 37.4950, 15.0940, 0.0))
    points.extend([
        ("2026-04-01 00:24:00", 37.4870, 15.0972, 5.0),
        ("2026-04-01 00:25:00", 37.4840, 15.0980, 5.0),
    ])
    d = frame(points)
    s = build_call_sessions(d)
    assert len(s) == 1
    assert s.loc[0, "hysteresis_merges"] == 1


def test_prefix_invariance_for_geometry_state():
    d = frame([
        ("2026-04-01 00:00:00", 37.4700, 15.1050, 5.0),
        ("2026-04-01 00:01:00", 37.4800, 15.1010, 5.0),
        ("2026-04-01 00:02:00", 37.4870, 15.0972, 5.0),
        ("2026-04-01 00:03:00", 37.4950, 15.0940, 0.0),
    ])
    p = label_catania_geometry(d.iloc[:3].copy()).reset_index(drop=True)
    f = label_catania_geometry(d.copy()).iloc[:3].reset_index(drop=True)
    cols = [
        "catania_gate_side",
        "catania_port_side",
        "catania_inner_harbour_proxy",
        "catania_distance_to_gate_mid_km",
        "catania_approach_proxy",
        "catania_roadstead_wait_proxy",
    ]
    pd.testing.assert_frame_equal(p[cols], f[cols])

def test_destination_support_does_not_use_future_post_entry_update():
    points = [
        ("2026-04-01 00:00:00", 37.4840, 15.0980, 5.0, "FOR ORDERS"),
        ("2026-04-01 00:01:00", 37.4870, 15.0972, 5.0, "FOR ORDERS"),
        ("2026-04-01 00:02:00", 37.4950, 15.0940, 0.0, "ITCTA"),
    ]
    for minute in range(3, 23):
        points.append((f"2026-04-01 00:{minute:02d}:00", 37.4950, 15.0940, 0.0, "ITCTA"))
    points.extend([
        ("2026-04-01 00:23:00", 37.4870, 15.0972, 5.0, "ITCTA"),
        ("2026-04-01 00:24:00", 37.4840, 15.0980, 5.0, "ITCTA"),
    ])
    d = frame(points)
    s = build_call_sessions(d)
    assert len(s) == 1
    assert s.loc[0, "candidate_class"] == "STOP_CONFIRMED_CALL_CANDIDATE"
    # The ITCTA update happened only after the entry event and must not be used
    # as entry-time destination support.
    assert not bool(s.loc[0, "destination_support_catania"])
    assert s.loc[0, "inbound_destination"] == "FOR ORDERS"

