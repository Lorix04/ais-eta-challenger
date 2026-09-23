"""M19 fresh external holdout utilities.

M19 defines a reproducible adapter for the public MMDEC v1 AIS dataset
(DOI 10.5281/zenodo.17491518) and keeps the external-evaluation order fixed:
cohort/provenance -> label-free feature rows -> M18G1 blind scoring -> sealed
prediction ledger -> label reveal -> one-shot M18G evaluation.

The public source co-locates AIS Message-5 ETA fields with static/voyage fields,
so M19 can enforce *software* blinding but cannot claim provider-enforced label
separation.  ETA component values are never copied to the target-free scoring
rows; only a boolean availability check is permitted during cohort construction.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

from .m0c import AIS_NAV_STATUS, engineer_semantic_states
from .m10 import haversine_km, normalize_destination, reference_eta_status
from .m18g import canonical_decision_time, sample_key, sha256_file

M19_VERSION = "m19-mmdec-fresh-external-holdout-v1-20260923"
M19_SOURCE_DOI = "10.5281/zenodo.17491518"
M19_SOURCE_RECORD = "https://zenodo.org/records/17491518"
M19_SOURCE_PERIOD_START = "2023-07-01T00:00:00Z"
M19_SOURCE_PERIOD_END = "2023-10-01T00:00:00Z"
M19_MAX_ROWS = 500
M19_PRESELECT_ROWS = 2000
M19_HISTORY_HOURS = 6
M19_MAX_POSITION_AGE_MIN = 60.0
M19_MIN_HISTORY_POINTS = 2
M19_SELECTION_SALT = "M19_MMDEC_V1_20260923"

M19_SOURCE_FILES = {
    "Dataset_AIS_POS.parquet": {
        "md5": "12b824d26488e381680b2de90090cc2c",
        "size_bytes_approx": 450_900_000,
    },
    "Dataset_AIS_SPEC.parquet": {
        "md5": "f1dd53064868fa078c714986991c3941",
        "size_bytes_approx": 210_000_000,
    },
}


@dataclass(frozen=True)
class M19Policy:
    max_rows: int = M19_MAX_ROWS
    preselect_rows: int = M19_PRESELECT_ROWS
    history_hours: int = M19_HISTORY_HOURS
    max_position_age_min: float = M19_MAX_POSITION_AGE_MIN
    min_history_points: int = M19_MIN_HISTORY_POINTS


def _key(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _resolve(df: pd.DataFrame, *aliases: str, required: bool = True) -> str | None:
    cmap = {_key(c): c for c in df.columns}
    for a in aliases:
        k = _key(a)
        if k in cmap:
            return cmap[k]
    if required:
        raise ValueError(f"missing MMDEC column; expected one of {aliases}")
    return None


def _selection_hash(mmsi: int, decision_time) -> str:
    payload = f"{M19_SELECTION_SALT}|{int(mmsi)}|{canonical_decision_time(decision_time)}".encode()
    return hashlib.sha256(payload).hexdigest()


def _eta_components_available(month, day, hour, minute) -> bool:
    try:
        m, d, h, mi = int(month), int(day), int(hour), int(minute)
    except Exception:
        return False
    return 1 <= m <= 12 and 1 <= d <= 31 and 0 <= h <= 23 and 0 <= mi <= 59


def parse_mmdec_eta(month, day, hour, minute, decision_time) -> pd.Timestamp:
    """Resolve AIS Message-5 MMDDHHMM to the nearest calendar year."""
    if not _eta_components_available(month, day, hour, minute):
        return pd.NaT
    ref = pd.Timestamp(decision_time)
    if ref.tzinfo is not None:
        ref = ref.tz_convert("UTC").tz_localize(None)
    candidates: list[pd.Timestamp] = []
    for year in (ref.year - 1, ref.year, ref.year + 1):
        try:
            candidates.append(pd.Timestamp(year, int(month), int(day), int(hour), int(minute)))
        except ValueError:
            pass
    return min(candidates, key=lambda x: abs((x - ref).total_seconds())) if candidates else pd.NaT


def standardize_mmdec_spec(df: pd.DataFrame) -> pd.DataFrame:
    c_date = _resolve(df, "Date")
    c_msg = _resolve(df, "MessageType")
    c_mmsi = _resolve(df, "Mmsi", "MMSI")
    c_ship = _resolve(df, "ShipType", required=False)
    c_draught = _resolve(df, "Draught10thMetres", "Draught", required=False)
    c_dest = _resolve(df, "Destination", required=False)
    c_month = _resolve(df, "EtaMonth")
    c_day = _resolve(df, "EtaDay")
    c_hour = _resolve(df, "EtaHour")
    c_min = _resolve(df, "EtaMinute")
    out = pd.DataFrame({
        "decision_time": pd.to_datetime(df[c_date], errors="coerce", utc=True),
        "message_type": pd.to_numeric(df[c_msg], errors="coerce"),
        "mmsi": pd.to_numeric(df[c_mmsi], errors="coerce"),
        "ship_type_code": pd.to_numeric(df[c_ship], errors="coerce") if c_ship else np.nan,
        "draught_tenths_m": pd.to_numeric(df[c_draught], errors="coerce") if c_draught else np.nan,
        "destination": df[c_dest].astype("string") if c_dest else pd.Series(pd.NA, index=df.index, dtype="string"),
        "eta_month": pd.to_numeric(df[c_month], errors="coerce"),
        "eta_day": pd.to_numeric(df[c_day], errors="coerce"),
        "eta_hour": pd.to_numeric(df[c_hour], errors="coerce"),
        "eta_minute": pd.to_numeric(df[c_min], errors="coerce"),
    })
    out = out.loc[out["decision_time"].notna() & out["mmsi"].notna()].copy()
    out["mmsi"] = out["mmsi"].astype("int64")
    out["eta_available"] = [
        _eta_components_available(m, d, h, mi)
        for m, d, h, mi in zip(out.eta_month, out.eta_day, out.eta_hour, out.eta_minute)
    ]
    return out


def standardize_mmdec_positions(df: pd.DataFrame) -> pd.DataFrame:
    c_date = _resolve(df, "Date")
    c_msg = _resolve(df, "MessageType", required=False)
    c_mmsi = _resolve(df, "Mmsi", "MMSI")
    c_nav = _resolve(df, "NavigationStatus", required=False)
    c_lat = _resolve(df, "Latitude")
    c_lon = _resolve(df, "Longitude")
    c_cog = _resolve(df, "CourseOverGroundDegrees", "COG", required=False)
    c_sog = _resolve(df, "SpeedOverGround", "SOG", required=False)
    c_head = _resolve(df, "TrueHeadingDegrees", "Heading", required=False)
    out = pd.DataFrame({
        "recorded_at": pd.to_datetime(df[c_date], errors="coerce", utc=True),
        "mmsi": pd.to_numeric(df[c_mmsi], errors="coerce"),
        "message_type": pd.to_numeric(df[c_msg], errors="coerce") if c_msg else np.nan,
        "nav_status": pd.to_numeric(df[c_nav], errors="coerce") if c_nav else np.nan,
        "lat": pd.to_numeric(df[c_lat], errors="coerce"),
        "lon": pd.to_numeric(df[c_lon], errors="coerce"),
        "cog": pd.to_numeric(df[c_cog], errors="coerce") if c_cog else np.nan,
        "sog": pd.to_numeric(df[c_sog], errors="coerce") if c_sog else np.nan,
        "heading": pd.to_numeric(df[c_head], errors="coerce") if c_head else np.nan,
    })
    out = out.loc[out["recorded_at"].notna() & out["mmsi"].notna()].copy()
    out["mmsi"] = out["mmsi"].astype("int64")
    out["id"] = np.arange(len(out), dtype=np.int64)
    out["draught"] = np.nan
    out["destination"] = pd.NA
    return out


def select_mmdec_candidates(spec: pd.DataFrame, policy: M19Policy = M19Policy()) -> pd.DataFrame:
    s = standardize_mmdec_spec(spec)
    start = pd.Timestamp(M19_SOURCE_PERIOD_START)
    end = pd.Timestamp(M19_SOURCE_PERIOD_END)
    s = s.loc[
        s["message_type"].eq(5)
        & s["eta_available"]
        & s["decision_time"].ge(start)
        & s["decision_time"].lt(end)
    ].copy()
    if not len(s):
        raise ValueError("MMDEC spec contains no eligible Message-5 rows with available ETA components")
    # The ETA magnitude is never used here; only outcome availability is checked.
    s["selection_hash"] = [_selection_hash(m, t) for m, t in zip(s.mmsi, s.decision_time)]
    s = s.sort_values(["mmsi", "selection_hash", "decision_time"], kind="mergesort")
    s = s.drop_duplicates("mmsi", keep="first")
    s = s.sort_values(["selection_hash", "mmsi"], kind="mergesort").head(int(policy.preselect_rows)).reset_index(drop=True)
    return s


def _draught_m(v) -> float:
    if pd.isna(v):
        return float("nan")
    x = float(v)
    if x <= 0:
        return float("nan")
    return 25.5 if x >= 255 else x / 10.0


def _window_features(g: pd.DataFrame, decision: pd.Timestamp, minutes: int) -> dict[str, float | int]:
    w = g.loc[g["recorded_at"] >= decision - pd.Timedelta(minutes=minutes)]
    sog = w["sog_clean"].dropna()
    ww = w.dropna(subset=["lat_clean", "lon_clean"])
    return {
        f"sog_median_{minutes}m": float(sog.median()) if len(sog) else np.nan,
        f"sog_std_{minutes}m": float(sog.std()) if len(sog) > 1 else (0.0 if len(sog) == 1 else np.nan),
        f"moving_fraction_{minutes}m": float(w["moving_candidate"].mean()) if len(w) else np.nan,
        f"stopped_fraction_{minutes}m": float(w["stop_candidate"].mean()) if len(w) else np.nan,
        f"turn_count_{minutes}m": int(w["turn_candidate"].sum()) if len(w) else 0,
        f"distance_travelled_km_{minutes}m": float(w["position_delta_m"].fillna(0).sum() / 1000) if len(w) else np.nan,
        f"net_displacement_km_{minutes}m": (
            haversine_km(ww["lat_clean"].iloc[0], ww["lon_clean"].iloc[0], ww["lat_clean"].iloc[-1], ww["lon_clean"].iloc[-1])
            if len(ww) >= 2 else np.nan
        ),
    }


def build_mmdec_target_free_holdout(
    spec: pd.DataFrame,
    positions: pd.DataFrame,
    policy: M19Policy = M19Policy(),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Build target-free rows/states and a target-free cohort locator.

    Returns ``(rows, states, cohort_locator, audit)``.  ``cohort_locator`` has
    sample keys / source timestamps only and deliberately contains no ETA fields.
    """
    candidates = select_mmdec_candidates(spec, policy)
    p = standardize_mmdec_positions(positions)
    p = p.loc[p["mmsi"].isin(set(candidates.mmsi.astype(int)))].copy()
    if not len(p):
        raise ValueError("MMDEC positions contain no rows for preselected candidate MMSIs")
    states = engineer_semantic_states(p)
    by_mmsi = {int(m): g.sort_values("recorded_at", kind="mergesort") for m, g in states.groupby("mmsi", sort=False)}

    rows: list[dict] = []
    selected_states: list[pd.DataFrame] = []
    locators: list[dict] = []
    rejects = {"insufficient_history": 0, "stale_position": 0}
    for rec in candidates.itertuples(index=False):
        decision = pd.Timestamp(rec.decision_time)
        g0 = by_mmsi.get(int(rec.mmsi))
        if g0 is None:
            rejects["insufficient_history"] += 1
            continue
        g = g0.loc[
            g0["recorded_at"].between(
                decision - pd.Timedelta(hours=int(policy.history_hours)), decision, inclusive="both"
            )
        ].copy()
        valid = g.loc[g["position_valid"].fillna(False).astype(bool)].dropna(subset=["lat_clean", "lon_clean"])
        if valid["recorded_at"].nunique() < int(policy.min_history_points):
            rejects["insufficient_history"] += 1
            continue
        last = valid.iloc[-1]
        position_age_min = float((decision - pd.Timestamp(last["recorded_at"])).total_seconds() / 60)
        if position_age_min < -1e-9 or position_age_min > float(policy.max_position_age_min):
            rejects["stale_position"] += 1
            continue
        dest_norm = normalize_destination(rec.destination)
        nav_status = last.get("nav_status_clean")
        nav_status_cat = AIS_NAV_STATUS.get(int(nav_status), "UNKNOWN") if pd.notna(nav_status) else "UNKNOWN"
        row = {
            "mmsi": int(rec.mmsi),
            "last_update": decision,
            "history_last_at": pd.Timestamp(last["recorded_at"]),
            "track_lat": last.get("lat_clean"),
            "track_lon": last.get("lon_clean"),
            "track_sog": last.get("sog_clean"),
            "track_cog": last.get("cog_clean"),
            "track_heading": last.get("heading_clean"),
            "track_draught": _draught_m(rec.draught_tenths_m),
            "track_msg_count": int(len(g)),
            "destination_norm": dest_norm,
            "ship_type_cat": f"AIS_TYPE_{int(rec.ship_type_code)}" if pd.notna(rec.ship_type_code) else "UNKNOWN",
            "nav_status_cat": nav_status_cat,
            "flag_cat": "EXTERNAL_MMDEC",
            "last_hour_sin": math.sin(2 * math.pi * decision.hour / 24),
            "last_hour_cos": math.cos(2 * math.pi * decision.hour / 24),
            "last_dow": int(decision.dayofweek),
            "position_age_min": position_age_min,
            "history_n": int(len(g)),
            "history_span_h": float((g["recorded_at"].max() - g["recorded_at"].min()).total_seconds() / 3600),
            "dynamic_state_age_min": float(last["dynamic_state_age_s"] / 60) if pd.notna(last.get("dynamic_state_age_s")) else np.nan,
            "motion_state_cat": str(last.get("motion_state", "UNKNOWN")),
            "stale_risk_cat": str(last.get("stale_risk_level", "UNKNOWN")),
            # MMDEC position messages do not carry voyage destination.  We do
            # not backfill the decision-time destination into past states.
            "hist_destination_unique": 0,
            "hist_destination_matches_track": 0,
            "track_vs_hist_position_gap_km": haversine_km(
                last.get("lat_clean"), last.get("lon_clean"), last.get("lat_clean"), last.get("lon_clean")
            ),
        }
        row.update(_window_features(g, decision, 180))
        row.update(_window_features(g, decision, 360))
        sk = sample_key(int(rec.mmsi), decision)
        row["m19_selection_hash"] = rec.selection_hash
        rows.append(row)
        selected_states.append(g)
        locators.append({
            "sample_key": sk,
            "mmsi": int(rec.mmsi),
            "decision_time": canonical_decision_time(decision),
            "selection_hash": rec.selection_hash,
        })
        if len(rows) >= int(policy.max_rows):
            break

    if not rows:
        raise ValueError("MMDEC cohort construction produced zero scorable rows")
    rowdf = pd.DataFrame(rows).sort_values(["m19_selection_hash", "mmsi"], kind="mergesort").reset_index(drop=True)
    statedf = pd.concat(selected_states, ignore_index=True)
    keep = set(rowdf.mmsi.astype(int))
    statedf = statedf.loc[statedf.mmsi.astype(int).isin(keep)].copy()
    locator = pd.DataFrame(locators)
    locator = locator.loc[locator.mmsi.astype(int).isin(keep)].copy()
    locator = locator.set_index("mmsi").loc[rowdf.mmsi.astype(int)].reset_index()
    audit = {
        "version": M19_VERSION,
        "source_doi": M19_SOURCE_DOI,
        "candidate_rows_preselected": int(len(candidates)),
        "selected_rows": int(len(rowdf)),
        "selected_unique_mmsi": int(rowdf.mmsi.nunique()),
        "history_hours": int(policy.history_hours),
        "max_position_age_min": float(policy.max_position_age_min),
        "rejections": rejects,
        "eta_values_exposed_to_scoring_rows": False,
        "prelabel_eta_use": "availability/range mask only; target magnitude not computed",
        "provider_enforced_blindness": False,
        "public_source_blinding": "software/procedural only",
    }
    return rowdf, statedf, locator, audit


def reveal_mmdec_label_ledger(spec: pd.DataFrame, cohort_locator: pd.DataFrame) -> pd.DataFrame:
    """Reveal Message-5 ETA values only for the already frozen cohort."""
    s = standardize_mmdec_spec(spec)
    s = s.loc[s["message_type"].eq(5)].copy()
    lookup: dict[tuple[int, str], pd.DataFrame] = {}
    for (m, dt), g in s.groupby(["mmsi", "decision_time"], sort=False):
        lookup[(int(m), canonical_decision_time(dt))] = g
    out: list[dict] = []
    for rec in cohort_locator.itertuples(index=False):
        key = (int(rec.mmsi), canonical_decision_time(rec.decision_time))
        g = lookup.get(key)
        if g is None or not len(g):
            raise ValueError(f"cannot reveal MMDEC label for {key}")
        vals = g[["eta_month", "eta_day", "eta_hour", "eta_minute"]].drop_duplicates()
        vals = vals.loc[[
            _eta_components_available(a, b, c, d)
            for a, b, c, d in vals.itertuples(index=False, name=None)
        ]]
        if len(vals) != 1:
            raise ValueError(f"ambiguous or unavailable MMDEC ETA components for {key}")
        month, day, hour, minute = vals.iloc[0].tolist()
        eta_dt = parse_mmdec_eta(month, day, hour, minute, rec.decision_time)
        decision = pd.Timestamp(rec.decision_time)
        if decision.tzinfo is not None:
            decision = decision.tz_convert("UTC").tz_localize(None)
        target_h = float((eta_dt - decision).total_seconds() / 3600)
        out.append({
            "sample_key": str(rec.sample_key),
            "target_tte_h": target_h,
            "reference_eta_status": reference_eta_status(target_h),
        })
    frame = pd.DataFrame(out)
    if len(frame) != len(cohort_locator) or frame.sample_key.duplicated().any():
        raise AssertionError("M19 revealed label ledger mismatch")
    if not np.isfinite(frame.target_tte_h.to_numpy(float)).all():
        raise AssertionError("M19 revealed non-finite target")
    return frame


def md5_file(path: str | Path) -> str:
    h = hashlib.md5()  # nosec - source-integrity check against published Zenodo MD5
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_mmdec_source_files(pos_path: str | Path, spec_path: str | Path) -> dict:
    pos_path, spec_path = Path(pos_path), Path(spec_path)
    expected = M19_SOURCE_FILES
    observed = {
        pos_path.name: {"md5": md5_file(pos_path), "sha256": sha256_file(pos_path), "bytes": pos_path.stat().st_size},
        spec_path.name: {"md5": md5_file(spec_path), "sha256": sha256_file(spec_path), "bytes": spec_path.stat().st_size},
    }
    for p in (pos_path, spec_path):
        if p.name in expected and observed[p.name]["md5"] != expected[p.name]["md5"]:
            raise ValueError(f"MMDEC published MD5 mismatch: {p.name}")
    return observed
