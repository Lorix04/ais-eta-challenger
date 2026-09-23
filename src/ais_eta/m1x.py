"""M1X: Augusta multi-entrance research geometry and audited ground-truth expansion.

This module intentionally labels *research-gate* crossings, not an official ATA,
PBP, berth-arrival or all-fast event.  Augusta has two operational entrances to
Porto Megarese.  The Levante cross-section is anchored to Istituto Idrografico
light-list positions.  The Scirocco cross-section is a versioned research proxy
anchored to official breakwater reference coordinates and validated against AIS
tracks; it must not be represented as an official regulatory entrance line.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from shapely.geometry import Polygon
import shapely

from .m0d import _segment_intersection, _haversine_km


@dataclass(frozen=True)
class AugustaGate:
    name: str
    a: tuple[float, float]  # lon, lat
    b: tuple[float, float]
    inside_reference: tuple[float, float]
    provenance: str

    @property
    def midpoint(self) -> tuple[float, float]:
        return ((self.a[0] + self.b[0]) / 2.0, (self.a[1] + self.b[1]) / 2.0)


@dataclass(frozen=True)
class AugustaGeometryV1:
    version: str = "augusta_research_geometry_v1"

    # Istituto Idrografico light list, rounded to 0.1 minute:
    # EF 2844 Diga settentrionale 37 11.9 N, 15 13.9 E
    # EF 2850 Diga centrale, estremita N 37 11.7 N, 15 13.9 E
    levante_a: tuple[float, float] = (15.2316666667, 37.1983333333)
    levante_b: tuple[float, float] = (15.2316666667, 37.1950000000)
    levante_inside_reference: tuple[float, float] = (15.2150, 37.1970)

    # Scirocco is a RESEARCH PROXY, not an official gate.  Point A uses the
    # southern Diga Centrale works reference 37 10.456 N, 15 13.016 E from an
    # IIM notice. Point B uses the IIM Diga meridionale monitoring mark at
    # 37 10.4 N, 15 12.6 E.  AIS trajectories are used only to validate that
    # the frozen segment intersects the observed passage, not to optimise ETA.
    scirocco_a: tuple[float, float] = (15.2169333333, 37.1742666667)
    scirocco_b: tuple[float, float] = (15.2100000000, 37.1733333333)
    scirocco_inside_reference: tuple[float, float] = (15.2070, 37.1830)

    # Analysis-only proxy for Porto Megarese water-side occupancy.  The polygon
    # is intentionally conservative and is used only to confirm a port-side
    # stop after a gate crossing. It is not a legal/official port boundary.
    inner_vertices: tuple[tuple[float, float], ...] = (
        (15.1880, 37.1760),
        (15.2095, 37.1720),
        (15.2180, 37.1740),
        (15.2225, 37.1885),
        (15.2322, 37.1948),
        (15.2323, 37.1986),
        (15.2200, 37.2050),
        (15.2165, 37.2425),
        (15.1920, 37.2450),
        (15.1880, 37.1760),
    )

    analysis_bbox: tuple[float, float, float, float] = (15.15, 37.15, 15.31, 37.255)

    @property
    def gates(self) -> tuple[AugustaGate, AugustaGate]:
        return (
            AugustaGate(
                "LEVANTE",
                self.levante_a,
                self.levante_b,
                self.levante_inside_reference,
                "IIM light-list EF2844/EF2850 rounded positions",
            ),
            AugustaGate(
                "SCIROCCO",
                self.scirocco_a,
                self.scirocco_b,
                self.scirocco_inside_reference,
                "research proxy from official Diga Centrale/Diga meridionale references",
            ),
        )

    @property
    def inner_polygon(self) -> Polygon:
        return Polygon(self.inner_vertices)


@dataclass(frozen=True)
class M1XConfig:
    max_crossing_dt_s: float = 300.0
    gate_hysteresis_s: float = 20 * 60.0
    stop_confirmation_s: float = 20 * 60.0
    pre_entry_wait_lookback_h: float = 36.0
    pre_entry_wait_min_s: float = 30 * 60.0
    roadstead_radius_km: float = 8.0
    high_quality_crossing_dt_s: float = 90.0
    medium_quality_crossing_dt_s: float = 180.0


def _signed_side(lon, lat, gate: AugustaGate) -> np.ndarray:
    ax, ay = gate.a
    bx, by = gate.b
    raw = (bx - ax) * (np.asarray(lat, dtype=float) - ay) - (by - ay) * (np.asarray(lon, dtype=float) - ax)
    rx, ry = gate.inside_reference
    ref = (bx - ax) * (ry - ay) - (by - ay) * (rx - ax)
    if ref == 0:
        raise ValueError(f"inside reference on gate {gate.name}")
    return raw if ref > 0 else -raw


def label_augusta_geometry(df: pd.DataFrame, geometry: AugustaGeometryV1 | None = None) -> pd.DataFrame:
    geom = geometry or AugustaGeometryV1()
    out = df.copy()
    min_lon, min_lat, max_lon, max_lat = geom.analysis_bbox
    out["augusta_analysis_envelope"] = (
        out["position_valid"]
        & out["lon_clean"].between(min_lon, max_lon)
        & out["lat_clean"].between(min_lat, max_lat)
    )
    valid = out["position_valid"]
    poly_inside = np.zeros(len(out), dtype=bool)
    if valid.any():
        idx = np.flatnonzero(valid.to_numpy())
        poly_inside[idx] = shapely.contains_xy(
            geom.inner_polygon,
            out.loc[valid, "lon_clean"].to_numpy(),
            out.loc[valid, "lat_clean"].to_numpy(),
        )
    out["augusta_inner_megarese_proxy"] = poly_inside

    gate_distances = []
    for gate in geom.gates:
        side_col = f"augusta_{gate.name.lower()}_side"
        out[side_col] = np.nan
        if valid.any():
            out.loc[valid, side_col] = _signed_side(out.loc[valid, "lon_clean"], out.loc[valid, "lat_clean"], gate)
        mlon, mlat = gate.midpoint
        d = np.full(len(out), np.nan)
        if valid.any():
            idx = np.flatnonzero(valid.to_numpy())
            d[idx] = _haversine_km(out.loc[valid, "lat_clean"], out.loc[valid, "lon_clean"], mlat, mlon)
        out[f"augusta_distance_to_{gate.name.lower()}_gate_km"] = d
        gate_distances.append(d)
    out["augusta_distance_to_nearest_gate_km"] = np.nanmin(np.vstack(gate_distances), axis=0)
    out["augusta_external_wait_proxy"] = (
        out["augusta_analysis_envelope"]
        & ~out["augusta_inner_megarese_proxy"]
        & out["augusta_distance_to_nearest_gate_km"].le(8.0)
        & (out["stop_candidate"] | out["nav_status_clean"].eq(1))
    )
    return out


def detect_augusta_crossings(
    df: pd.DataFrame,
    geometry: AugustaGeometryV1 | None = None,
    config: M1XConfig | None = None,
) -> pd.DataFrame:
    geom = geometry or AugustaGeometryV1()
    cfg = config or M1XConfig()
    x = label_augusta_geometry(df, geom)
    x = x[x["augusta_analysis_envelope"]].copy()
    x = x.sort_values(["mmsi", "recorded_at", "id"], kind="mergesort").reset_index(drop=True)
    g = x.groupby("mmsi", sort=False)
    x["prev_lon"] = g["lon_clean"].shift()
    x["prev_lat"] = g["lat_clean"].shift()
    x["prev_recorded_at"] = g["recorded_at"].shift()
    x["prev_stale_risk_level"] = g["stale_risk_level"].shift()
    x["prev_position_jump_candidate"] = g["position_jump_candidate"].shift(fill_value=False)

    rows: list[dict] = []
    for gate in geom.gates:
        side_col = f"augusta_{gate.name.lower()}_side"
        prev_side = g[side_col].shift()
        for idx in x.index:
            if pd.isna(x.at[idx, "prev_lon"]):
                continue
            dt = x.at[idx, "observation_dt_s"]
            if not pd.notna(dt) or dt <= 0 or dt > cfg.max_crossing_dt_s:
                continue
            if bool(x.at[idx, "position_jump_candidate"]) or bool(x.at[idx, "prev_position_jump_candidate"]):
                continue
            ps = prev_side.loc[idx]
            cs = x.at[idx, side_col]
            if pd.isna(ps) or pd.isna(cs) or not ((ps < 0 <= cs) or (ps >= 0 > cs)):
                continue
            inter = _segment_intersection(
                (float(x.at[idx, "prev_lon"]), float(x.at[idx, "prev_lat"])),
                (float(x.at[idx, "lon_clean"]), float(x.at[idx, "lat_clean"])),
                gate.a,
                gate.b,
            )
            if inter is None:
                continue
            inbound = ps < 0 <= cs
            stale_high = x.at[idx, "stale_risk_level"] == "HIGH" or x.at[idx, "prev_stale_risk_level"] == "HIGH"
            if dt <= cfg.high_quality_crossing_dt_s and not stale_high:
                quality = "A"
            elif dt <= cfg.medium_quality_crossing_dt_s and not stale_high:
                quality = "B"
            else:
                quality = "C"
            t0 = x.at[idx, "prev_recorded_at"]
            t1 = x.at[idx, "recorded_at"]
            rows.append({
                "mmsi": int(x.at[idx, "mmsi"]),
                "name": x.at[idx, "name"],
                "entrance_name": gate.name,
                "crossing_direction": "INBOUND" if inbound else "OUTBOUND",
                "event_name": f"AUGUSTA_{gate.name}_RESEARCH_GATE_{'INBOUND' if inbound else 'OUTBOUND'}",
                "crossing_last_before": t0,
                "crossing_first_after": t1,
                "crossing_time_proxy": t0 + (t1 - t0) / 2,
                "crossing_uncertainty_s": float(dt),
                "crossing_quality": quality,
                "crossing_lon": inter[0],
                "crossing_lat": inter[1],
                "sog_clean": x.at[idx, "sog_clean"],
                "cog_clean": x.at[idx, "cog_clean"],
                "destination_clean": x.at[idx, "destination_clean"],
                "stale_risk_level": x.at[idx, "stale_risk_level"],
                "geometry_version": geom.version,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # If a single movement segment intersects both research proxies (rare edge
    # case), keep separate events but deterministic ordering by entrance name.
    return out.sort_values(["mmsi", "crossing_time_proxy", "entrance_name"]).reset_index(drop=True)


def _merge_short_excursions(sessions: list[dict], hysteresis_s: float) -> list[dict]:
    if not sessions:
        return []
    merged = []
    cur = sessions[0].copy()
    for nxt in sessions[1:]:
        can = False
        if nxt["mmsi"] == cur["mmsi"] and pd.notna(cur.get("outbound_time")):
            gap = (nxt["inbound_time"] - cur["outbound_time"]).total_seconds()
            can = 0 <= gap <= hysteresis_s
        if can:
            cur["outbound_time"] = nxt.get("outbound_time")
            cur["outbound_entrance"] = nxt.get("outbound_entrance")
            cur["outbound_quality"] = nxt.get("outbound_quality")
            cur["outbound_last_before"] = nxt.get("outbound_last_before")
            cur["outbound_first_after"] = nxt.get("outbound_first_after")
            cur["hysteresis_merges"] = int(cur.get("hysteresis_merges", 0)) + 1
        else:
            merged.append(cur)
            cur = nxt.copy()
    merged.append(cur)
    return merged


def _destination_support_augusta(values: pd.Series) -> bool:
    if values.empty:
        return False
    z = values.dropna().astype(str).str.upper().str.replace(" ", "", regex=False).str.replace("-", "", regex=False)
    return bool(z.str.contains(r"ITAUG|AUGUSTA|^AUG$|>ITAUG", regex=True).any())


def build_augusta_sessions(
    df: pd.DataFrame,
    crossings: pd.DataFrame | None = None,
    geometry: AugustaGeometryV1 | None = None,
    config: M1XConfig | None = None,
) -> pd.DataFrame:
    geom = geometry or AugustaGeometryV1()
    cfg = config or M1XConfig()
    geo = label_augusta_geometry(df, geom)
    events = crossings if crossings is not None else detect_augusta_crossings(df, geom, cfg)
    if events.empty:
        return pd.DataFrame()

    raw: list[dict] = []
    for mmsi, group in events.groupby("mmsi", sort=False):
        opened = None
        for ev in group.sort_values(["crossing_time_proxy", "entrance_name"]).itertuples(index=False):
            if ev.crossing_direction == "INBOUND":
                if opened is None:
                    opened = {
                        "mmsi": int(mmsi), "name": ev.name,
                        "inbound_time": ev.crossing_time_proxy,
                        "inbound_entrance": ev.entrance_name,
                        "inbound_last_outside": ev.crossing_last_before,
                        "inbound_first_inside": ev.crossing_first_after,
                        "inbound_uncertainty_s": ev.crossing_uncertainty_s,
                        "inbound_quality": ev.crossing_quality,
                        "inbound_lat": ev.crossing_lat, "inbound_lon": ev.crossing_lon,
                        "inbound_destination": ev.destination_clean,
                        "inbound_sog": ev.sog_clean, "inbound_cog": ev.cog_clean,
                        "geometry_version": ev.geometry_version,
                        "hysteresis_merges": 0,
                    }
            elif opened is not None:
                opened.update({
                    "outbound_time": ev.crossing_time_proxy,
                    "outbound_entrance": ev.entrance_name,
                    "outbound_last_before": ev.crossing_last_before,
                    "outbound_first_after": ev.crossing_first_after,
                    "outbound_quality": ev.crossing_quality,
                })
                raw.append(opened); opened = None
        if opened is not None:
            opened.update({
                "outbound_time": pd.NaT, "outbound_entrance": pd.NA,
                "outbound_last_before": pd.NaT, "outbound_first_after": pd.NaT,
                "outbound_quality": pd.NA,
            })
            raw.append(opened)

    raw.sort(key=lambda d: (d["mmsi"], d["inbound_time"]))
    merged: list[dict] = []
    raw_df = pd.DataFrame(raw)
    for _, grp in raw_df.groupby("mmsi", sort=False):
        merged.extend(_merge_short_excursions(grp.to_dict("records"), cfg.gate_hysteresis_s))

    dataset_end = geo["recorded_at"].max()
    rows = []
    for sess in merged:
        start = pd.Timestamp(sess["inbound_time"])
        closed = pd.notna(sess.get("outbound_time"))
        all_after = geo[(geo["mmsi"] == sess["mmsi"]) & (geo["recorded_at"] >= start)]
        last_obs = all_after["recorded_at"].max() if len(all_after) else start
        end = pd.Timestamp(sess["outbound_time"]) if closed else pd.Timestamp(last_obs)
        vessel = geo[(geo["mmsi"] == sess["mmsi"]) & (geo["recorded_at"] >= start) & (geo["recorded_at"] <= end)]
        inner = vessel[vessel["augusta_inner_megarese_proxy"]]
        max_stop_s = float(inner["stop_duration_s"].max()) if len(inner) else 0.0
        stop_confirmed = max_stop_s >= cfg.stop_confirmation_s

        pre = geo[(geo["mmsi"] == sess["mmsi"]) & (geo["recorded_at"] >= start - pd.Timedelta(hours=cfg.pre_entry_wait_lookback_h)) & (geo["recorded_at"] < start)]
        wait = pre[pre["augusta_external_wait_proxy"]]
        pre_wait_max_s = float(wait["stop_duration_s"].max()) if len(wait) else 0.0

        dest_values = pd.concat([pre["destination_clean"], pd.Series([sess.get("inbound_destination")])], ignore_index=True).dropna()
        dest_support = _destination_support_augusta(dest_values)
        dest_mode = dest_values.mode().iloc[0] if len(dest_values) and len(dest_values.mode()) else pd.NA

        last_row = all_after.sort_values("recorded_at").tail(1)
        last_inside = bool(last_row["augusta_inner_megarese_proxy"].iloc[0]) if len(last_row) else False
        near_end = bool((dataset_end - pd.Timestamp(last_obs)).total_seconds() <= 10 * 60)
        right_censored = (not closed) and near_end and last_inside
        unresolved = (not closed) and not right_censored

        if stop_confirmed and sess["inbound_quality"] == "A" and not unresolved:
            quality = "A"
        elif stop_confirmed and not unresolved:
            quality = "B"
        else:
            quality = "C"
        candidate_class = "STOP_CONFIRMED_CALL_CANDIDATE" if stop_confirmed else ("ENTRY_WITHOUT_CONFIRMED_STOP" if len(inner) else "GATE_CROSSING_ONLY")

        rows.append({
            **sess,
            "session_closed": bool(closed),
            "right_censored": bool(right_censored),
            "unresolved_missing_outbound": bool(unresolved),
            "last_observed_at": last_obs,
            "last_observed_inner_proxy": last_inside,
            "session_duration_min": float((end-start).total_seconds()/60.0),
            "observations_in_session": int(len(vessel)),
            "inner_megarese_rows": int(len(inner)),
            "first_inner_megarese_time": inner["recorded_at"].min() if len(inner) else pd.NaT,
            "max_inner_stop_duration_min": max_stop_s/60.0,
            "inner_moored_rows": int(inner["nav_status_clean"].eq(5).sum()) if len(inner) else 0,
            "inner_anchor_rows": int(inner["nav_status_clean"].eq(1).sum()) if len(inner) else 0,
            "external_wait_before_entry": bool(pre_wait_max_s >= cfg.pre_entry_wait_min_s),
            "max_pre_entry_wait_min": pre_wait_max_s/60.0,
            "destination_support_augusta": bool(dest_support),
            "destination_mode": dest_mode,
            "candidate_class": candidate_class,
            "m1x_candidate_quality": quality,
            "requires_manual_audit": True,
            "final_ground_truth": False,
        })
    out = pd.DataFrame(rows).sort_values(["inbound_time", "mmsi"]).reset_index(drop=True)
    out.insert(0, "session_id", [f"AU-{i:03d}" for i in range(1, len(out)+1)])
    return out


def geometry_manifest(geometry: AugustaGeometryV1 | None = None) -> dict:
    g = geometry or AugustaGeometryV1()
    return {
        "version": g.version,
        "crs": "EPSG:4326",
        "semantics": {
            "LEVANTE": "research cross-section anchored to IIM light-list endpoints EF2844/EF2850; not official ATA/PBP",
            "SCIROCCO": "research cross-section proxy anchored to official breakwater reference coordinates; not an official regulatory entrance line",
            "inner_megarese_proxy": "analysis-only stop-confirmation envelope; not legal port limits",
        },
        "gates": [
            {"name": gate.name, "a": list(gate.a), "b": list(gate.b), "inside_reference": list(gate.inside_reference), "provenance": gate.provenance}
            for gate in g.gates
        ],
        "inner_vertices": [list(p) for p in g.inner_vertices],
        "analysis_bbox": list(g.analysis_bbox),
    }
