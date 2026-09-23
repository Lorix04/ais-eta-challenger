"""M10 company reference-ETA benchmark utilities.

The company clarified that ``Tracks.eta`` is the exercise reference/ground-truth
value.  M10 therefore treats that field as a *reference label*, not as an
observed Actual Time of Arrival (ATA).

Important invariants:
- one supervised record per MMSI / Tracks row;
- ETA is never used as a feature;
- historical Positions features are cut at or before ``Tracks.last_update``;
- split assignment is deterministic from MMSI only and therefore target-free.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Iterable

import numpy as np
import pandas as pd

ETA_RE = re.compile(r"^(\d{2})/(\d{2}) (\d{2}):(\d{2})$")
M10_SPLIT_SALT = "M10_V1_20260921"


@dataclass(frozen=True)
class M10Config:
    split_salt: str = M10_SPLIT_SALT
    train_pct: int = 70
    calibration_pct: int = 15
    recent_windows_min: tuple[int, ...] = (15, 30, 60, 180, 360)


def parse_ais_eta_reference(value: object, reference_time: object) -> pd.Timestamp:
    """Parse AIS-style ``MM/DD HH:MM`` and infer the nearest calendar year.

    AIS Message 5 does not transmit a year.  We therefore evaluate the
    previous/current/next year around the observation time and choose the
    temporally closest representation.  This does *not* make the ETA accurate;
    it only resolves the missing year deterministically.
    """
    if value is None or pd.isna(value) or reference_time is None or pd.isna(reference_time):
        return pd.NaT
    match = ETA_RE.match(str(value).strip())
    if not match:
        return pd.NaT
    month, day, hour, minute = map(int, match.groups())
    if not (1 <= month <= 12 and 1 <= day <= 31 and 0 <= hour <= 23 and 0 <= minute <= 59):
        return pd.NaT
    ref = pd.Timestamp(reference_time)
    candidates: list[pd.Timestamp] = []
    for year in (ref.year - 1, ref.year, ref.year + 1):
        try:
            candidates.append(pd.Timestamp(year, month, day, hour, minute))
        except ValueError:
            continue
    if not candidates:
        return pd.NaT
    return min(candidates, key=lambda x: abs((x - ref).total_seconds()))


def reference_eta_status(tte_hours: float) -> str:
    """Transparent diagnostics for the company-provided ETA reference."""
    if pd.isna(tte_hours):
        return "UNPARSEABLE"
    x = float(tte_hours)
    if x < -24:
        return "PAST_GT24H"
    if x < -1:
        return "PAST_1_24H"
    if x < 0:
        return "PAST_LT1H"
    if x < 24 * 7:
        return "FUTURE_0_7D"
    if x < 24 * 14:
        return "FUTURE_7_14D"
    if x < 24 * 30:
        return "FUTURE_14_30D"
    return "FUTURE_GT30D"


def deterministic_split(mmsi: int | str, cfg: M10Config = M10Config()) -> str:
    """Assign split from MMSI only; no label or date information enters."""
    key = f"{cfg.split_salt}:{int(mmsi)}".encode()
    value = int(hashlib.sha256(key).hexdigest()[:8], 16) % 100
    if value < cfg.train_pct:
        return "train"
    if value < cfg.train_pct + cfg.calibration_pct:
        return "calibration"
    return "final_test"


def normalize_destination(value: object) -> str:
    if value is None or pd.isna(value):
        return "UNKNOWN"
    s = str(value).upper().strip()
    s = re.sub(r"[^A-Z0-9>]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or "UNKNOWN"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    vals = [lat1, lon1, lat2, lon2]
    if any(pd.isna(x) for x in vals):
        return float("nan")
    r = 6371.0088
    p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
    dp = p2 - p1
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return float(2 * r * math.asin(math.sqrt(max(0.0, min(1.0, a)))))


def _safe_category(value: object) -> str:
    return "UNKNOWN" if value is None or pd.isna(value) else str(value)


def build_reference_rows(
    tracks: pd.DataFrame,
    states: pd.DataFrame,
    cfg: M10Config = M10Config(),
) -> pd.DataFrame:
    """Build one causal supervised row per parseable ship ETA in Tracks."""
    t = tracks.copy()
    t["last_update"] = pd.to_datetime(t["last_update"], errors="coerce")
    t["eta_reference_dt"] = [
        parse_ais_eta_reference(v, ts) for v, ts in zip(t["eta"], t["last_update"])
    ]
    t = t.loc[t["category"].eq("ship") & t["eta_reference_dt"].notna()].copy()
    t["target_tte_h"] = (t["eta_reference_dt"] - t["last_update"]).dt.total_seconds() / 3600
    t["reference_eta_status"] = t["target_tte_h"].map(reference_eta_status)
    t["split"] = t["mmsi"].map(lambda x: deterministic_split(x, cfg))
    t["destination_norm"] = t["destination"].map(normalize_destination)

    cutoff = t.set_index("mmsi")["last_update"]
    s = states.loc[states["mmsi"].isin(t["mmsi"])].copy()
    s["m10_cutoff"] = s["mmsi"].map(cutoff)
    s = s.loc[s["recorded_at"] <= s["m10_cutoff"]].copy()

    rows: list[dict[str, object]] = []
    by_mmsi = {int(k): g.sort_values("recorded_at", kind="mergesort") for k, g in s.groupby("mmsi")}
    for r in t.sort_values("mmsi").itertuples(index=False):
        mmsi = int(r.mmsi)
        g = by_mmsi.get(mmsi, pd.DataFrame())
        row: dict[str, object] = {
            "mmsi": mmsi,
            "name": _safe_category(r.name),
            "split": r.split,
            "last_update": r.last_update,
            "eta_reference_raw": str(r.eta),
            "eta_reference_dt": r.eta_reference_dt,
            "target_tte_h": float(r.target_tte_h),
            "reference_eta_status": r.reference_eta_status,
            # Current Tracks snapshot features (ETA deliberately excluded).
            "track_lat": r.lat,
            "track_lon": r.lon,
            "track_sog": r.sog,
            "track_cog": np.nan if pd.isna(r.cog) or float(r.cog) >= 360 else float(r.cog),
            "track_heading": np.nan if pd.isna(r.heading) or float(r.heading) == 511 else float(r.heading),
            "track_draught": np.nan if pd.isna(r.draught) or float(r.draught) <= 0 else float(r.draught),
            "track_msg_count": r.msg_count,
            "destination_norm": r.destination_norm,
            "ship_type_cat": _safe_category(r.ship_type_str),
            "nav_status_cat": _safe_category(r.nav_status_str),
            "flag_cat": _safe_category(r.flag),
            "last_hour_sin": math.sin(2 * math.pi * r.last_update.hour / 24),
            "last_hour_cos": math.cos(2 * math.pi * r.last_update.hour / 24),
            "last_dow": int(r.last_update.dayofweek),
        }
        if len(g):
            last = g.iloc[-1]
            row.update({
                "history_last_at": last["recorded_at"],
                "position_age_min": (r.last_update - last["recorded_at"]).total_seconds() / 60,
                "history_n": int(len(g)),
                "history_span_h": (g["recorded_at"].max() - g["recorded_at"].min()).total_seconds() / 3600,
                "dynamic_state_age_min": float(last["dynamic_state_age_s"] / 60) if pd.notna(last["dynamic_state_age_s"]) else np.nan,
                "motion_state_cat": _safe_category(last["motion_state"]),
                "stale_risk_cat": _safe_category(last["stale_risk_level"]),
                "hist_destination_unique": int(g["destination_clean"].dropna().nunique()),
                "hist_destination_last": normalize_destination(last["destination_clean"]),
                "hist_destination_matches_track": int(normalize_destination(last["destination_clean"]) == r.destination_norm),
                "track_vs_hist_position_gap_km": haversine_km(r.lat, r.lon, last["lat_clean"], last["lon_clean"]),
            })
            for minutes in cfg.recent_windows_min:
                w = g.loc[g["recorded_at"] >= r.last_update - pd.Timedelta(minutes=minutes)]
                sog = w["sog_clean"].dropna()
                row[f"sog_median_{minutes}m"] = float(sog.median()) if len(sog) else np.nan
                row[f"sog_mean_{minutes}m"] = float(sog.mean()) if len(sog) else np.nan
                row[f"sog_std_{minutes}m"] = float(sog.std()) if len(sog) > 1 else (0.0 if len(sog) == 1 else np.nan)
                row[f"moving_fraction_{minutes}m"] = float(w["moving_candidate"].mean()) if len(w) else np.nan
                row[f"stopped_fraction_{minutes}m"] = float(w["stop_candidate"].mean()) if len(w) else np.nan
                row[f"turn_count_{minutes}m"] = int(w["turn_candidate"].sum()) if len(w) else 0
                row[f"distance_travelled_km_{minutes}m"] = float(w["position_delta_m"].fillna(0).sum() / 1000) if len(w) else np.nan
                ww = w.dropna(subset=["lat_clean", "lon_clean"])
                row[f"net_displacement_km_{minutes}m"] = (
                    haversine_km(
                        ww["lat_clean"].iloc[0], ww["lon_clean"].iloc[0],
                        ww["lat_clean"].iloc[-1], ww["lon_clean"].iloc[-1],
                    ) if len(ww) >= 2 else np.nan
                )
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("mmsi").reset_index(drop=True)
    if out["mmsi"].duplicated().any():
        raise AssertionError("M10 requires exactly one supervised row per MMSI")
    return out


def error_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    y = np.asarray(list(y_true), dtype=float)
    p = np.asarray(list(y_pred), dtype=float)
    ae = np.abs(y - p)
    return {
        "n": int(len(y)),
        "mae_h": float(np.mean(ae)),
        "medae_h": float(np.median(ae)),
        "rmse_h": float(np.sqrt(np.mean((y - p) ** 2))),
        "p90_ae_h": float(np.quantile(ae, 0.9)),
        "within_6h": float(np.mean(ae <= 6)),
        "within_12h": float(np.mean(ae <= 12)),
        "within_24h": float(np.mean(ae <= 24)),
        "within_48h": float(np.mean(ae <= 48)),
    }
