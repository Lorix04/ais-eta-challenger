from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m0c import M0CConfig, circular_delta_deg, engineer_semantic_states, normalize_ais_fields


def frame(rows):
    base = []
    for i, r in enumerate(rows, 1):
        x = {
            "id": i, "mmsi": 111000111, "name": "TEST", "category": "ship",
            "nav_status": 0, "nav_status_str": "", "sog": 0.0, "cog": 0.0,
            "heading": 0.0, "lat": 37.0, "lon": 15.0, "destination": " TEST ",
            "draught": 5.0, "manoeuvre_str": "", "recorded_at": "2026-04-01 00:00:00",
            "created_at": "2026-04-01 00:00:00", "updated_at": "2026-04-01 00:00:00",
        }
        x.update(r)
        base.append(x)
    return pd.DataFrame(base)


def test_ais_sentinels_normalize_to_missing_and_raw_is_preserved():
    d = frame([{"sog": 102.3, "cog": 360.0, "heading": 511, "draught": 0.0, "nav_status": 15}])
    c = normalize_ais_fields(d)
    assert c.loc[0, "sog"] == 102.3
    assert pd.isna(c.loc[0, "sog_clean"])
    assert pd.isna(c.loc[0, "cog_clean"])
    assert pd.isna(c.loc[0, "heading_clean"])
    assert pd.isna(c.loc[0, "draught_clean"])
    assert pd.isna(c.loc[0, "nav_status_clean"])
    assert c.loc[0, "destination_clean"] == "TEST"



def test_sog_102_2_is_valid_capped_value_not_unavailable():
    d = frame([{"sog": 102.2}])
    c = normalize_ais_fields(d)
    assert c.loc[0, "sog_clean"] == 102.2
    assert bool(c.loc[0, "sog_capped_high"])

def test_circular_delta_wraps_across_north():
    current = pd.Series([1.0, 359.0, 180.0])
    previous = pd.Series([359.0, 1.0, 0.0])
    got = circular_delta_deg(current, previous)
    assert got.tolist() == [2.0, 2.0, 180.0]


def test_state_age_resets_only_on_past_observed_change():
    d = frame([
        {"recorded_at": "2026-04-01 00:00:00", "sog": 10.0},
        {"recorded_at": "2026-04-01 00:01:00", "sog": 10.0},
        {"recorded_at": "2026-04-01 00:02:00", "sog": 11.0},
        {"recorded_at": "2026-04-01 00:03:00", "sog": 11.0},
    ])
    x = engineer_semantic_states(d)
    assert x["dynamic_state_age_s"].tolist() == [0.0, 60.0, 0.0, 60.0]
    assert x["dynamic_state_changed"].tolist() == [True, False, True, False]


def test_stop_gap_turn_and_kinematic_conflict_candidates():
    # ~111 m latitude step in 60 s is ~3.6 kn, sufficient for MOVING.
    d = frame([
        {"recorded_at": "2026-04-01 00:00:00", "lat": 37.0000, "sog": 4.0, "cog": 359.0},
        {"recorded_at": "2026-04-01 00:01:00", "lat": 37.0010, "sog": 4.0, "cog": 1.0},
        {"recorded_at": "2026-04-01 00:02:00", "lat": 37.0010, "sog": 0.0, "cog": 1.0},
        {"recorded_at": "2026-04-01 00:03:00", "lat": 37.0010, "sog": 0.0, "cog": 1.0},
        {"recorded_at": "2026-04-01 00:07:00", "lat": 37.0010, "sog": 5.0, "cog": 40.0},
        {"recorded_at": "2026-04-01 00:08:00", "lat": 37.0010, "sog": 5.0, "cog": 80.0},
    ])
    x = engineer_semantic_states(d)
    assert x.loc[2, "stop_start"]
    assert x.loc[3, "stop_duration_s"] == 60.0
    assert x.loc[4, "observation_gap_candidate"]
    assert x.loc[5, "kinematic_conflict"]
    # Turn requires previous/current speed but conflict can coexist with turn candidate.
    assert x.loc[5, "turn_candidate"]



def test_observation_gap_breaks_stop_episode_duration():
    d = frame([
        {"recorded_at": "2026-04-01 00:00:00", "sog": 0.0},
        {"recorded_at": "2026-04-01 00:01:00", "sog": 0.0},
        {"recorded_at": "2026-04-01 00:10:00", "sog": 0.0},
        {"recorded_at": "2026-04-01 00:11:00", "sog": 0.0},
    ])
    x = engineer_semantic_states(d)
    assert x.loc[2, "observation_gap_candidate"]
    assert x.loc[2, "stop_start"]
    assert x.loc[2, "stop_duration_s"] == 0.0
    assert x.loc[3, "stop_duration_s"] == 60.0

def test_long_moored_repeat_is_not_high_stale_risk():
    d = frame([
        {"recorded_at": "2026-04-01 00:00:00", "nav_status": 5, "sog": 0.0},
        {"recorded_at": "2026-04-01 00:05:00", "nav_status": 5, "sog": 0.0},
        {"recorded_at": "2026-04-01 00:15:00", "nav_status": 5, "sog": 0.0},
    ])
    x = engineer_semantic_states(d)
    assert x.loc[2, "dynamic_state_age_s"] == 900.0
    assert x.loc[2, "stale_risk_level"] == "LOW"


def test_prefix_invariance_no_future_information():
    d = frame([
        {"recorded_at": f"2026-04-01 00:0{i}:00", "lat": 37.0 + i * 0.001, "sog": 4.0 + (i % 2), "cog": (350 + i * 7) % 360}
        for i in range(8)
    ])
    prefix = engineer_semantic_states(d.iloc[:5].copy()).reset_index(drop=True)
    full = engineer_semantic_states(d.copy()).iloc[:5].reset_index(drop=True)
    cols = ["observation_dt_s", "position_delta_m", "implied_speed_kn", "cog_delta_deg", "dynamic_state_age_s", "motion_state", "stale_risk_level", "semantic_events"]
    for c in cols:
        if pd.api.types.is_numeric_dtype(prefix[c]):
            assert np.allclose(prefix[c], full[c], equal_nan=True)
        else:
            assert prefix[c].fillna("<NA>").astype(str).equals(full[c].fillna("<NA>").astype(str))


def test_repeated_rows_are_preserved_not_deduplicated():
    d = frame([
        {"recorded_at": "2026-04-01 00:00:00"},
        {"recorded_at": "2026-04-01 00:01:00"},
        {"recorded_at": "2026-04-01 00:02:00"},
    ])
    x = engineer_semantic_states(d)
    assert len(x) == 3
    assert x.loc[1, "dynamic_state_repeated"]
    assert x.loc[2, "dynamic_state_age_s"] == 120.0
