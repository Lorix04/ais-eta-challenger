"""M0D: Catania research geometry and auditable port-call reconstruction.

This module intentionally does **not** claim to reconstruct an official ATA.
The primary event is named ``ACTUAL_VALIDATED_PORT_ENTRANCE`` only after the
manual audit in M0E.  M0D creates candidate gate crossings and candidate call
sessions using a documented research gate.

The Catania gate is a research cross-section anchored to published hydrographic
aid coordinates, not an official regulatory port-limit line.  Geometry must
therefore remain versioned, reviewable and independent from downstream ETA
performance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from shapely.geometry import Polygon
import shapely


EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class CataniaGeometryV1:
    """Versioned research geometry for the first Catania audit.

    Coordinates are ``(lon, lat)`` WGS84.

    ``gate_west`` is the published rounded position of the light at the end of
    the new ferry dock / berth 31 (Istituto Idrografico light list 2804).
    ``gate_east`` is the published rounded position of the outer Molo di
    Levante light (light list 2802).

    Connecting them is a *research gate proxy*.  It is not asserted to be an
    official port-limit, PBP or harbour-master arrival line.
    """

    version: str = "catania_research_geometry_v1"
    gate_west: tuple[float, float] = (15.0950, 37.4883333333)
    gate_east: tuple[float, float] = (15.1000, 37.4833333333)
    inside_reference: tuple[float, float] = (15.0940, 37.4960)

    # Conservative analysis-only harbour envelope derived from the official
    # port-plan layout and frozen before model development.  It is used to
    # confirm port-side dwell, not presented as an official water boundary.
    inner_harbour_vertices: tuple[tuple[float, float], ...] = (
        (15.0950, 37.4883333333),
        (15.0890, 37.4870),
        (15.0890, 37.5035),
        (15.1025, 37.5035),
        (15.1020, 37.4890),
        (15.1000, 37.4833333333),
    )

    # Broad local analysis envelope; deliberately much larger than the inner
    # harbour to preserve pre-entry waiting and approach behaviour.
    analysis_bbox: tuple[float, float, float, float] = (
        15.0600,
        37.4400,
        15.1400,
        37.5300,
    )

    @property
    def inner_harbour_polygon(self) -> Polygon:
        return Polygon(self.inner_harbour_vertices)

    @property
    def gate_midpoint(self) -> tuple[float, float]:
        return (
            (self.gate_west[0] + self.gate_east[0]) / 2.0,
            (self.gate_west[1] + self.gate_east[1]) / 2.0,
        )


@dataclass(frozen=True)
class M0DConfig:
    max_crossing_dt_s: float = 300.0
    gate_hysteresis_s: float = 15 * 60.0
    stop_confirmation_s: float = 15 * 60.0
    approach_radius_km: float = 6.0
    roadstead_wait_radius_km: float = 4.0
    pre_entry_wait_lookback_h: float = 24.0
    pre_entry_wait_min_s: float = 30 * 60.0
    high_quality_crossing_dt_s: float = 90.0
    medium_quality_crossing_dt_s: float = 180.0


def _signed_side(lon: pd.Series | np.ndarray, lat: pd.Series | np.ndarray, geometry: CataniaGeometryV1) -> np.ndarray:
    """Signed side of the infinite research-gate line.

    The sign is oriented so that the configured inside reference is positive.
    """
    ax, ay = geometry.gate_west
    bx, by = geometry.gate_east
    raw = (bx - ax) * (np.asarray(lat, dtype=float) - ay) - (by - ay) * (np.asarray(lon, dtype=float) - ax)
    rx, ry = geometry.inside_reference
    ref = (bx - ax) * (ry - ay) - (by - ay) * (rx - ax)
    if ref == 0:
        raise ValueError("inside_reference lies on research gate")
    return raw if ref > 0 else -raw


def _segment_intersection(
    p1: tuple[float, float],
    p2: tuple[float, float],
    q1: tuple[float, float],
    q2: tuple[float, float],
) -> tuple[float, float] | None:
    """Return segment intersection in lon/lat coordinates, else ``None``."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-14:
        return None
    det12 = x1 * y2 - y1 * x2
    det34 = x3 * y4 - y3 * x4
    px = (det12 * (x3 - x4) - (x1 - x2) * det34) / den
    py = (det12 * (y3 - y4) - (y1 - y2) * det34) / den
    tol = 1e-8
    in_p = min(x1, x2) - tol <= px <= max(x1, x2) + tol and min(y1, y2) - tol <= py <= max(y1, y2) + tol
    in_q = min(x3, x4) - tol <= px <= max(x3, x4) + tol and min(y3, y4) - tol <= py <= max(y3, y4) + tol
    return (float(px), float(py)) if in_p and in_q else None


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M / 1000.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def label_catania_geometry(
    df: pd.DataFrame,
    geometry: CataniaGeometryV1 | None = None,
    config: M0DConfig | None = None,
) -> pd.DataFrame:
    """Add research geometry states without using future observations."""
    geom = geometry or CataniaGeometryV1()
    cfg = config or M0DConfig()
    out = df.copy()
    min_lon, min_lat, max_lon, max_lat = geom.analysis_bbox
    out["catania_analysis_envelope"] = (
        out["lon_clean"].between(min_lon, max_lon)
        & out["lat_clean"].between(min_lat, max_lat)
        & out["position_valid"]
    )
    out["catania_gate_side"] = np.nan
    valid = out["position_valid"]
    if valid.any():
        out.loc[valid, "catania_gate_side"] = _signed_side(
            out.loc[valid, "lon_clean"], out.loc[valid, "lat_clean"], geom
        )
    out["catania_port_side"] = out["catania_gate_side"].gt(0)

    poly = geom.inner_harbour_polygon
    inside = np.zeros(len(out), dtype=bool)
    if valid.any():
        idx = np.flatnonzero(valid.to_numpy())
        inside[idx] = shapely.contains_xy(
            poly,
            out.loc[valid, "lon_clean"].to_numpy(),
            out.loc[valid, "lat_clean"].to_numpy(),
        )
    out["catania_inner_harbour_proxy"] = inside

    mid_lon, mid_lat = geom.gate_midpoint
    dist = np.full(len(out), np.nan, dtype=float)
    if valid.any():
        idx = np.flatnonzero(valid.to_numpy())
        dist[idx] = _haversine_km(
            out.loc[valid, "lat_clean"],
            out.loc[valid, "lon_clean"],
            mid_lat,
            mid_lon,
        )
    out["catania_distance_to_gate_mid_km"] = dist
    out["catania_approach_proxy"] = (
        out["catania_analysis_envelope"]
        & ~out["catania_port_side"]
        & out["catania_distance_to_gate_mid_km"].le(cfg.approach_radius_km)
    )
    out["catania_roadstead_wait_proxy"] = (
        out["catania_approach_proxy"]
        & out["catania_distance_to_gate_mid_km"].le(cfg.roadstead_wait_radius_km)
        & (out["stop_candidate"] | out["nav_status_clean"].eq(1))
    )
    return out


def detect_gate_crossings(
    df: pd.DataFrame,
    geometry: CataniaGeometryV1 | None = None,
    config: M0DConfig | None = None,
) -> pd.DataFrame:
    """Detect direction-aware crossings of the finite research gate.

    Crossing rows are based only on the immediately previous and current
    observations.  The crossing timestamp is represented by an uncertainty
    bracket and a midpoint proxy; no interpolation-derived position is used as
    a predictive feature.
    """
    geom = geometry or CataniaGeometryV1()
    cfg = config or M0DConfig()
    x = label_catania_geometry(df, geom, cfg)
    x = x[x["catania_analysis_envelope"]].copy()
    x = x.sort_values(["mmsi", "recorded_at", "id"], kind="mergesort").reset_index(drop=True)
    g = x.groupby("mmsi", sort=False)
    x["prev_lon"] = g["lon_clean"].shift()
    x["prev_lat"] = g["lat_clean"].shift()
    x["prev_recorded_at"] = g["recorded_at"].shift()
    x["prev_gate_side"] = g["catania_gate_side"].shift()
    x["prev_stale_risk_level"] = g["stale_risk_level"].shift()
    x["prev_position_jump_candidate"] = g["position_jump_candidate"].shift(fill_value=False)

    side_change = (
        (x["prev_gate_side"].lt(0) & x["catania_gate_side"].ge(0))
        | (x["prev_gate_side"].ge(0) & x["catania_gate_side"].lt(0))
    )
    candidate = x[
        x["prev_lon"].notna()
        & x["observation_dt_s"].gt(0)
        & x["observation_dt_s"].le(cfg.max_crossing_dt_s)
        & ~x["position_jump_candidate"]
        & ~x["prev_position_jump_candidate"]
        & side_change
    ].copy()

    intersections: list[tuple[float, float] | None] = []
    for row in candidate.itertuples(index=False):
        intersections.append(
            _segment_intersection(
                (float(row.prev_lon), float(row.prev_lat)),
                (float(row.lon_clean), float(row.lat_clean)),
                geom.gate_west,
                geom.gate_east,
            )
        )
    candidate["_intersection"] = intersections
    candidate = candidate[candidate["_intersection"].notna()].copy()
    candidate["crossing_lon"] = [p[0] for p in candidate["_intersection"]]
    candidate["crossing_lat"] = [p[1] for p in candidate["_intersection"]]
    candidate["crossing_direction"] = np.where(
        candidate["prev_gate_side"].lt(0) & candidate["catania_gate_side"].ge(0),
        "INBOUND",
        "OUTBOUND",
    )
    candidate["crossing_last_before"] = candidate["prev_recorded_at"]
    candidate["crossing_first_after"] = candidate["recorded_at"]
    candidate["crossing_time_proxy"] = candidate["prev_recorded_at"] + (
        candidate["recorded_at"] - candidate["prev_recorded_at"]
    ) / 2
    candidate["crossing_uncertainty_s"] = candidate["observation_dt_s"]

    stale_high = candidate["stale_risk_level"].eq("HIGH") | candidate["prev_stale_risk_level"].eq("HIGH")
    candidate["crossing_quality"] = np.select(
        [
            candidate["observation_dt_s"].le(cfg.high_quality_crossing_dt_s) & ~stale_high,
            candidate["observation_dt_s"].le(cfg.medium_quality_crossing_dt_s) & ~stale_high,
        ],
        ["A", "B"],
        default="C",
    )
    candidate["geometry_version"] = geom.version
    candidate["event_name"] = np.where(
        candidate["crossing_direction"].eq("INBOUND"),
        "CATANIA_RESEARCH_GATE_INBOUND",
        "CATANIA_RESEARCH_GATE_OUTBOUND",
    )
    cols = [
        "mmsi",
        "name",
        "crossing_direction",
        "event_name",
        "crossing_last_before",
        "crossing_first_after",
        "crossing_time_proxy",
        "crossing_uncertainty_s",
        "crossing_quality",
        "crossing_lat",
        "crossing_lon",
        "sog_clean",
        "cog_clean",
        "destination_clean",
        "stale_risk_level",
        "geometry_version",
    ]
    return candidate[cols].sort_values(["mmsi", "crossing_time_proxy"]).reset_index(drop=True)


def _merge_short_excursions(sessions: list[dict], hysteresis_s: float) -> list[dict]:
    if not sessions:
        return []
    merged: list[dict] = []
    cur = sessions[0].copy()
    for nxt in sessions[1:]:
        same_vessel = nxt["mmsi"] == cur["mmsi"]
        can_merge = False
        if same_vessel and pd.notna(cur.get("outbound_time")):
            gap = (nxt["inbound_time"] - cur["outbound_time"]).total_seconds()
            can_merge = 0 <= gap <= hysteresis_s
        if can_merge:
            cur["outbound_time"] = nxt.get("outbound_time")
            cur["outbound_quality"] = nxt.get("outbound_quality")
            cur["outbound_last_before"] = nxt.get("outbound_last_before")
            cur["outbound_first_after"] = nxt.get("outbound_first_after")
            cur["hysteresis_merges"] = int(cur.get("hysteresis_merges", 0)) + 1
        else:
            merged.append(cur)
            cur = nxt.copy()
    merged.append(cur)
    return merged


def build_call_sessions(
    df: pd.DataFrame,
    crossings: pd.DataFrame | None = None,
    geometry: CataniaGeometryV1 | None = None,
    config: M0DConfig | None = None,
) -> pd.DataFrame:
    """Pair gate events and add ex-post confirmation evidence.

    Future observations may be used here to *confirm the label candidate* (e.g.
    a later stop inside the harbour).  They are not exported as prediction-time
    features.  Final ground-truth confidence is reserved for M0E manual audit.
    """
    geom = geometry or CataniaGeometryV1()
    cfg = config or M0DConfig()
    geo = label_catania_geometry(df, geom, cfg)
    events = crossings if crossings is not None else detect_gate_crossings(df, geom, cfg)
    if events.empty:
        return pd.DataFrame()

    raw_sessions: list[dict] = []
    for mmsi, group in events.groupby("mmsi", sort=False):
        opened: dict | None = None
        for event in group.sort_values("crossing_time_proxy").itertuples(index=False):
            if event.crossing_direction == "INBOUND":
                if opened is None:
                    opened = {
                        "mmsi": int(mmsi),
                        "name": event.name,
                        "inbound_time": event.crossing_time_proxy,
                        "inbound_last_outside": event.crossing_last_before,
                        "inbound_first_inside": event.crossing_first_after,
                        "inbound_uncertainty_s": event.crossing_uncertainty_s,
                        "inbound_quality": event.crossing_quality,
                        "inbound_lat": event.crossing_lat,
                        "inbound_lon": event.crossing_lon,
                        "inbound_destination": event.destination_clean,
                        "inbound_sog": event.sog_clean,
                        "inbound_cog": event.cog_clean,
                        "geometry_version": event.geometry_version,
                        "hysteresis_merges": 0,
                    }
            elif opened is not None:
                opened.update(
                    {
                        "outbound_time": event.crossing_time_proxy,
                        "outbound_last_before": event.crossing_last_before,
                        "outbound_first_after": event.crossing_first_after,
                        "outbound_quality": event.crossing_quality,
                    }
                )
                raw_sessions.append(opened)
                opened = None
        if opened is not None:
            opened.update(
                {
                    "outbound_time": pd.NaT,
                    "outbound_last_before": pd.NaT,
                    "outbound_first_after": pd.NaT,
                    "outbound_quality": pd.NA,
                }
            )
            raw_sessions.append(opened)

    raw_sessions.sort(key=lambda d: (d["mmsi"], d["inbound_time"]))
    merged: list[dict] = []
    for _, grp in pd.DataFrame(raw_sessions).groupby("mmsi", sort=False):
        merged.extend(_merge_short_excursions(grp.to_dict("records"), cfg.gate_hysteresis_s))

    dataset_end = geo["recorded_at"].max()
    rows: list[dict] = []
    for session_id, sess in enumerate(merged, 1):
        start = pd.Timestamp(sess["inbound_time"])
        closed = pd.notna(sess.get("outbound_time"))
        vessel_all_after_entry = geo[
            (geo["mmsi"] == sess["mmsi"])
            & (geo["recorded_at"] >= start)
        ]
        last_observed_at = vessel_all_after_entry["recorded_at"].max() if len(vessel_all_after_entry) else start
        end = pd.Timestamp(sess["outbound_time"]) if closed else pd.Timestamp(last_observed_at)
        vessel = geo[
            (geo["mmsi"] == sess["mmsi"])
            & (geo["recorded_at"] >= start)
            & (geo["recorded_at"] <= end)
        ]
        inner = vessel[vessel["catania_inner_harbour_proxy"]]
        max_stop_s = float(inner["stop_duration_s"].max()) if len(inner) else 0.0
        stop_confirmed = max_stop_s >= cfg.stop_confirmation_s
        moored_rows = int(inner["nav_status_clean"].eq(5).sum())
        anchor_rows = int(inner["nav_status_clean"].eq(1).sum())

        lookback_start = start - pd.Timedelta(hours=cfg.pre_entry_wait_lookback_h)
        pre = geo[
            (geo["mmsi"] == sess["mmsi"])
            & (geo["recorded_at"] >= lookback_start)
            & (geo["recorded_at"] < start)
            & (geo["catania_roadstead_wait_proxy"])
        ]
        pre_wait_max_s = float(pre["stop_duration_s"].max()) if len(pre) else 0.0
        roadstead_wait = pre_wait_max_s >= cfg.pre_entry_wait_min_s

        # Destination is supporting evidence only.  Keep it strictly as-of the
        # entry event / pre-entry window; using destination values observed after
        # entry could accidentally capture a Message-5 update for the *next* leg.
        inbound_dest = pd.Series([sess.get("inbound_destination")], dtype="object")
        dest_values = pd.concat(
            [
                pre["destination_clean"],
                inbound_dest,
            ],
            ignore_index=True,
        ).dropna().astype(str)
        destination_support = bool(
            dest_values.str.replace(" ", "", regex=False).str.contains("CATANIA|ITCTA|ITCAT", regex=True).any()
        ) if len(dest_values) else False
        destination_mode = dest_values.mode().iloc[0] if len(dest_values) and len(dest_values.mode()) else pd.NA

        last_row = vessel_all_after_entry.sort_values("recorded_at").tail(1)
        last_on_port_side = bool(last_row["catania_port_side"].iloc[0]) if len(last_row) else False
        near_dataset_end = bool((dataset_end - pd.Timestamp(last_observed_at)).total_seconds() <= 10 * 60)
        right_censored = (not closed) and near_dataset_end and last_on_port_side
        unresolved_missing_outbound = (not closed) and not right_censored

        if stop_confirmed and sess["inbound_quality"] == "A" and not unresolved_missing_outbound:
            candidate_quality = "A"
        elif stop_confirmed and not unresolved_missing_outbound:
            candidate_quality = "B"
        else:
            candidate_quality = "C"

        if stop_confirmed:
            candidate_class = "STOP_CONFIRMED_CALL_CANDIDATE"
        elif len(inner) > 0:
            candidate_class = "ENTRY_WITHOUT_CONFIRMED_STOP"
        else:
            candidate_class = "GATE_CROSSING_ONLY"

        rows.append(
            {
                **sess,
                "session_closed": bool(closed),
                "right_censored": bool(right_censored),
                "unresolved_missing_outbound": bool(unresolved_missing_outbound),
                "last_observed_at": last_observed_at,
                "last_observed_port_side": last_on_port_side,
                "session_duration_min": float((end - start).total_seconds() / 60.0),
                "observations_in_session": int(len(vessel)),
                "inner_harbour_rows": int(len(inner)),
                "first_inner_harbour_time": inner["recorded_at"].min() if len(inner) else pd.NaT,
                "max_inner_stop_duration_min": max_stop_s / 60.0,
                "inner_moored_rows": moored_rows,
                "inner_anchor_rows": anchor_rows,
                "roadstead_wait_before_entry": bool(roadstead_wait),
                "max_pre_entry_wait_min": pre_wait_max_s / 60.0,
                "destination_support_catania": destination_support,
                "destination_mode": destination_mode,
                "candidate_class": candidate_class,
                "m0d_candidate_quality": candidate_quality,
                "requires_manual_audit": True,
                "final_ground_truth": False,
            }
        )
    out = pd.DataFrame(rows).sort_values(["inbound_time", "mmsi"]).reset_index(drop=True)
    out.insert(0, "session_id", [f"CT-{i:03d}" for i in range(1, len(out) + 1)])
    return out


def geometry_manifest(geometry: CataniaGeometryV1 | None = None) -> dict:
    geom = geometry or CataniaGeometryV1()
    return {
        "version": geom.version,
        "crs": "EPSG:4326",
        "semantics": {
            "research_gate": "proxy cross-section; NOT official port limit/PBP/ATA line",
            "inner_harbour_polygon": "analysis-only envelope; NOT official legal water boundary",
        },
        "gate": {
            "west": {"lon": geom.gate_west[0], "lat": geom.gate_west[1], "anchor": "IIM light 2804 rounded position"},
            "east": {"lon": geom.gate_east[0], "lat": geom.gate_east[1], "anchor": "IIM light 2802 rounded position"},
        },
        "inside_reference": {"lon": geom.inside_reference[0], "lat": geom.inside_reference[1]},
        "inner_harbour_vertices": [list(p) for p in geom.inner_harbour_vertices],
        "analysis_bbox": list(geom.analysis_bbox),
    }