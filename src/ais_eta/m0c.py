"""M0C: causal AIS normalization and semantic motion-state features.

The functions in this module operate only on information available at or before
an observation timestamp.  They intentionally preserve repeated provider
snapshots because dwell time and stale-state diagnostics are part of the signal.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

KNOTS_PER_MPS = 1.9438444924406
EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class M0CConfig:
    """Auditable thresholds for candidate semantic events.

    These are engineering defaults for this dataset, not universal maritime
    constants.  They are deliberately named *candidate* thresholds and will be
    sensitivity-tested before port-call ground truth is frozen.
    """

    stop_sog_kn: float = 0.5
    stop_implied_kn: float = 0.5
    slow_sog_kn: float = 3.0
    moving_sog_kn: float = 3.0
    moving_implied_kn: float = 0.5
    conflict_sog_kn: float = 2.0
    conflict_implied_kn: float = 0.2
    turn_min_sog_kn: float = 1.5
    turn_delta_deg: float = 25.0
    max_turn_dt_s: float = 180.0
    observation_gap_s: float = 180.0
    position_jump_kn: float = 80.0
    stale_medium_age_s: float = 180.0
    stale_high_age_s: float = 600.0


AIS_NAV_STATUS = {
    0: "UNDERWAY_ENGINE",
    1: "AT_ANCHOR",
    2: "NOT_UNDER_COMMAND",
    3: "RESTRICTED_MANOEUVRABILITY",
    4: "CONSTRAINED_BY_DRAUGHT",
    5: "MOORED",
    6: "AGROUND",
    7: "FISHING",
    8: "SAILING",
    9: "RESERVED_HSC",
    10: "RESERVED_WIG",
    11: "TOWING_ASTERN",
    12: "PUSHING_OR_TOWING_ALONGSIDE",
    13: "RESERVED",
    14: "AIS_SART_ACTIVE",
}


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalize_destination(series: pd.Series) -> pd.Series:
    """Light normalization only; this is not a port resolver."""
    out = series.astype("string")
    out = out.str.strip().str.upper().str.replace(r"\s+", " ", regex=True)
    unavailable = out.isna() | out.eq("") | out.str.fullmatch(r"@+")
    return out.mask(unavailable)


def normalize_ais_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Create normalized fields while preserving raw columns.

    Decoded sentinel semantics follow the AIS Class-A / Message-5 conventions:
    SOG 102.3 -> unavailable, COG 360.0 -> unavailable, heading 511 ->
    unavailable, draught 0 -> unavailable, latitude +/-91 and longitude +/-181
    are unavailable/invalid.  Nav status 15 is retained in the raw field but is
    represented as missing in nav_status_clean because it means undefined.
    """
    out = df.copy()

    for col in ["sog", "cog", "heading", "lat", "lon", "draught", "nav_status"]:
        if col in out:
            out[col] = _numeric(out[col])

    # Decoded 102.3 corresponds to raw 1023 = unavailable; decoded 102.2 is the valid capped value "102.2 kn or higher".
    out["sog_capped_high"] = out["sog"].eq(102.2)
    out["sog_clean"] = out["sog"].mask(out["sog"].gt(102.2))
    out["cog_clean"] = out["cog"].mask(out["cog"].ge(360.0) | out["cog"].lt(0.0))
    out["heading_clean"] = out["heading"].mask(out["heading"].eq(511) | out["heading"].lt(0) | out["heading"].gt(359))
    out["lat_clean"] = out["lat"].mask(out["lat"].abs().gt(90.0) | out["lat"].abs().eq(91.0))
    out["lon_clean"] = out["lon"].mask(out["lon"].abs().gt(180.0) | out["lon"].abs().eq(181.0))
    out["draught_clean"] = out["draught"].mask(out["draught"].le(0.0))

    nav = out["nav_status"]
    out["nav_status_clean"] = nav.where(nav.isin(AIS_NAV_STATUS.keys()))
    out["nav_status_name"] = out["nav_status_clean"].map(AIS_NAV_STATUS).astype("string")
    out["destination_clean"] = normalize_destination(out.get("destination", pd.Series(pd.NA, index=out.index)))

    out["position_valid"] = out["lat_clean"].notna() & out["lon_clean"].notna()
    out["sog_valid"] = out["sog_clean"].notna()
    out["cog_valid"] = out["cog_clean"].notna()
    out["heading_valid"] = out["heading_clean"].notna()
    out["draught_valid"] = out["draught_clean"].notna()
    return out


def _haversine_m(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1.astype(float))
    lon1 = np.radians(lon1.astype(float))
    lat2 = np.radians(lat2.astype(float))
    lon2 = np.radians(lon2.astype(float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def circular_delta_deg(current: pd.Series, previous: pd.Series) -> pd.Series:
    """Absolute smallest angular difference in [0, 180]."""
    return ((current - previous + 180.0) % 360.0 - 180.0).abs()


def _episode_duration(timestamp: pd.Series, active: pd.Series, group: pd.Series, break_mask: pd.Series | None = None) -> pd.Series:
    prev_active = active.groupby(group, sort=False).shift(fill_value=False)
    episode_start = active & ~prev_active
    if break_mask is not None:
        episode_start = episode_start | (active & break_mask.fillna(False))
    start = timestamp.where(episode_start)
    start = start.groupby(group, sort=False).ffill()
    duration = (timestamp - start).dt.total_seconds()
    return duration.where(active, 0.0)


def engineer_semantic_states(df: pd.DataFrame, config: M0CConfig | None = None) -> pd.DataFrame:
    """Add past-only motion, state-age, candidate-event and stale-risk features."""
    cfg = config or M0CConfig()
    out = normalize_ais_fields(df)
    out["recorded_at"] = pd.to_datetime(out["recorded_at"], errors="coerce")
    out = out.sort_values(["mmsi", "recorded_at", "id"], kind="mergesort").reset_index(drop=True)
    g = out.groupby("mmsi", sort=False)

    out["observation_dt_s"] = g["recorded_at"].diff().dt.total_seconds()
    prev_lat = g["lat_clean"].shift()
    prev_lon = g["lon_clean"].shift()
    valid_pair = out["position_valid"] & prev_lat.notna() & prev_lon.notna()
    dist = pd.Series(np.nan, index=out.index, dtype=float)
    if valid_pair.any():
        idx = valid_pair
        dist.loc[idx] = _haversine_m(prev_lat.loc[idx], prev_lon.loc[idx], out.loc[idx, "lat_clean"], out.loc[idx, "lon_clean"])
    out["position_delta_m"] = dist
    dt = out["observation_dt_s"]
    out["implied_speed_kn"] = (out["position_delta_m"] / dt * KNOTS_PER_MPS).where(dt.gt(0))

    prev_cog = g["cog_clean"].shift()
    out["cog_delta_deg"] = circular_delta_deg(out["cog_clean"], prev_cog).where(out["cog_clean"].notna() & prev_cog.notna())
    out["sog_delta_kn"] = (out["sog_clean"] - g["sog_clean"].shift()).abs()

    # Deterministic hashes are used as compact state identifiers, not as model features.
    dynamic_cols = ["lat_clean", "lon_clean", "sog_clean", "cog_clean", "heading_clean", "nav_status_clean"]
    out["dynamic_state_hash"] = pd.util.hash_pandas_object(out[dynamic_cols], index=False).astype("uint64")
    prev_hash = g["dynamic_state_hash"].shift()
    first = g.cumcount().eq(0)
    out["dynamic_state_changed"] = first | out["dynamic_state_hash"].ne(prev_hash)
    out["dynamic_state_repeated"] = ~out["dynamic_state_changed"]
    out["last_dynamic_state_change_at"] = out["recorded_at"].where(out["dynamic_state_changed"]).groupby(out["mmsi"], sort=False).ffill()
    out["dynamic_state_age_s"] = (out["recorded_at"] - out["last_dynamic_state_change_at"]).dt.total_seconds().clip(lower=0)

    out["position_repeated"] = out["position_delta_m"].fillna(np.inf).le(1.0) & ~first

    # Candidate semantic states.  These are not port-call labels.
    out["kinematic_conflict"] = (
        out["sog_clean"].ge(cfg.conflict_sog_kn)
        & out["implied_speed_kn"].le(cfg.conflict_implied_kn)
        & dt.gt(0)
        & dt.le(cfg.max_turn_dt_s)
    )
    out["position_jump_candidate"] = out["implied_speed_kn"].gt(cfg.position_jump_kn) & dt.gt(0) & dt.le(600)
    out["stop_candidate"] = (
        out["sog_clean"].le(cfg.stop_sog_kn)
        & out["implied_speed_kn"].le(cfg.stop_implied_kn)
        & dt.gt(0)
        & ~out["position_jump_candidate"]
    )
    out["slow_motion_candidate"] = (
        out["sog_clean"].gt(cfg.stop_sog_kn)
        & out["sog_clean"].lt(cfg.slow_sog_kn)
        & out["implied_speed_kn"].lt(cfg.slow_sog_kn)
        & dt.gt(0)
        & ~out["position_jump_candidate"]
    )
    out["moving_candidate"] = (
        out["sog_clean"].ge(cfg.moving_sog_kn)
        & out["implied_speed_kn"].ge(cfg.moving_implied_kn)
        & dt.gt(0)
        & ~out["position_jump_candidate"]
    )
    prev_sog = g["sog_clean"].shift()
    out["turn_candidate"] = (
        out["cog_delta_deg"].ge(cfg.turn_delta_deg)
        & out["sog_clean"].ge(cfg.turn_min_sog_kn)
        & prev_sog.ge(cfg.turn_min_sog_kn)
        & dt.gt(0)
        & dt.le(cfg.max_turn_dt_s)
        & ~out["position_jump_candidate"]
    )
    out["observation_gap_candidate"] = dt.gt(cfg.observation_gap_s)

    prev_stop = out["stop_candidate"].groupby(out["mmsi"], sort=False).shift(fill_value=False)
    prev_slow = out["slow_motion_candidate"].groupby(out["mmsi"], sort=False).shift(fill_value=False)
    # A provider observation gap breaks episode continuity: we do not assume the vessel
    # remained stopped/slow while it was unobserved.
    out["stop_start"] = out["stop_candidate"] & (~prev_stop | out["observation_gap_candidate"])
    out["stop_end"] = ~out["stop_candidate"] & prev_stop
    out["slow_start"] = out["slow_motion_candidate"] & (~prev_slow | out["observation_gap_candidate"])
    out["slow_end"] = ~out["slow_motion_candidate"] & prev_slow
    out["stop_duration_s"] = _episode_duration(out["recorded_at"], out["stop_candidate"], out["mmsi"], out["observation_gap_candidate"])
    out["slow_duration_s"] = _episode_duration(out["recorded_at"], out["slow_motion_candidate"], out["mmsi"], out["observation_gap_candidate"])

    # Stationary anchor/moored states can legitimately remain unchanged for hours.
    stationary_context = out["sog_clean"].le(cfg.stop_sog_kn) & out["nav_status_clean"].isin([1, 5])
    high = ~stationary_context & (
        out["kinematic_conflict"]
        | (out["dynamic_state_age_s"].ge(cfg.stale_high_age_s) & out["sog_clean"].gt(1.0))
    )
    medium = ~stationary_context & ~high & (
        (out["dynamic_state_repeated"] & out["dynamic_state_age_s"].ge(cfg.stale_medium_age_s))
        | (out["position_repeated"] & out["sog_clean"].gt(cfg.stop_sog_kn))
    )
    out["stale_risk_level"] = np.select([high, medium], ["HIGH", "MEDIUM"], default="LOW")

    # Human-readable current motion state.  Kinematic conflicts/jumps have priority.
    conditions = [
        out["position_jump_candidate"],
        out["kinematic_conflict"],
        out["stop_candidate"],
        out["slow_motion_candidate"],
        out["moving_candidate"],
    ]
    choices = ["POSITION_JUMP", "KINEMATIC_CONFLICT", "STOPPED", "SLOW_MOTION", "MOVING"]
    out["motion_state"] = np.select(conditions, choices, default="UNKNOWN")

    labels = np.full(len(out), "", dtype=object)
    events = [
        ("STOP_START", out["stop_start"]),
        ("STOP_END", out["stop_end"]),
        ("SLOW_START", out["slow_start"]),
        ("SLOW_END", out["slow_end"]),
        ("TURN", out["turn_candidate"]),
        ("OBS_GAP", out["observation_gap_candidate"]),
        ("KIN_CONFLICT", out["kinematic_conflict"]),
        ("POS_JUMP", out["position_jump_candidate"]),
    ]
    for name, mask in events:
        idx = np.flatnonzero(mask.to_numpy())
        for i in idx:
            labels[i] = name if labels[i] == "" else labels[i] + ";" + name
    out["semantic_events"] = pd.Series(labels, index=out.index, dtype="string")
    return out


def sentinel_summary(raw: pd.DataFrame, clean: pd.DataFrame) -> dict:
    """Counts of AIS special values and post-normalization validity."""
    n = len(raw)
    def count(mask): return int(mask.fillna(False).sum())
    def pct(v): return float(v / n) if n else 0.0

    raw_num = {c: _numeric(raw[c]) for c in ["sog", "cog", "heading", "lat", "lon", "draught", "nav_status"]}
    counts = {
        "sog_unavailable_102_3": count(raw_num["sog"].gt(102.2)),
        "sog_capped_102_2": count(raw_num["sog"].eq(102.2)),
        "cog_unavailable_ge_360": count(raw_num["cog"].ge(360.0)),
        "heading_unavailable_511": count(raw_num["heading"].eq(511)),
        "draught_zero_or_negative": count(raw_num["draught"].le(0.0)),
        "nav_status_undefined_15": count(raw_num["nav_status"].eq(15)),
        "lat_invalid_or_unavailable": count(raw_num["lat"].abs().gt(90.0) | raw_num["lat"].abs().eq(91.0)),
        "lon_invalid_or_unavailable": count(raw_num["lon"].abs().gt(180.0) | raw_num["lon"].abs().eq(181.0)),
        "destination_missing_or_blank": int(normalize_destination(raw.get("destination", pd.Series(pd.NA, index=raw.index))).isna().sum()),
    }
    return {
        "rows": n,
        "sentinel_counts": counts,
        "sentinel_pct": {k: pct(v) for k, v in counts.items()},
        "clean_valid_pct": {
            "position": float(clean["position_valid"].mean()),
            "sog": float(clean["sog_valid"].mean()),
            "cog": float(clean["cog_valid"].mean()),
            "heading": float(clean["heading_valid"].mean()),
            "draught": float(clean["draught_valid"].mean()),
        },
    }
