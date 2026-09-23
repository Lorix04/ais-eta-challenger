from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


PLACEHOLDER_LIKE_ETA_STRINGS = {"01/01 00:00", "01/01 01:01"}
NON_SPECIFIC_DESTINATIONS = {
    "UNKNOWN", "FOR ORDER", "FOR ORDERS", "ORDER", "ORDERS", "IN PORT",
    "MARE", "PESCA", "DRIFTING", "WAITING", "SEA", "OPEN SEA",
}


@dataclass(frozen=True)
class EtaYearResolution:
    chosen_year: int
    chosen_delta_days: float
    second_year: int
    second_delta_days: float
    assignment_margin_days: float


def eta_year_resolution(raw_eta: str, last_update: pd.Timestamp) -> EtaYearResolution:
    """Audit the year inference implied by AIS MM/DD HH:MM ETA.

    AIS Message 5 ETA does not encode a year. M10 resolves the closest valid
    candidate across previous/current/next year. This function does not alter
    that policy; it quantifies how separated the chosen candidate is from the
    next-best year assignment.
    """
    if pd.isna(raw_eta) or pd.isna(last_update):
        raise ValueError("raw_eta and last_update are required")
    text = str(raw_eta).strip()
    mmdd, hm = text.split()
    month, day = (int(x) for x in mmdd.split("/"))
    hour, minute = (int(x) for x in hm.split(":"))
    t0 = pd.Timestamp(last_update)
    candidates: list[tuple[float, float, int]] = []
    for year in (t0.year - 1, t0.year, t0.year + 1):
        try:
            cand = pd.Timestamp(year=year, month=month, day=day, hour=hour, minute=minute)
        except ValueError:
            continue
        delta_days = (cand - t0).total_seconds() / 86400.0
        candidates.append((abs(delta_days), delta_days, year))
    if len(candidates) < 2:
        raise ValueError(f"not enough valid year candidates for {raw_eta!r}")
    candidates.sort(key=lambda x: (x[0], x[2]))
    first, second = candidates[:2]
    return EtaYearResolution(
        chosen_year=int(first[2]),
        chosen_delta_days=float(first[1]),
        second_year=int(second[2]),
        second_delta_days=float(second[1]),
        assignment_margin_days=float(second[0] - first[0]),
    )


def eta_granularity_bucket(dt: pd.Timestamp) -> str:
    minute = int(pd.Timestamp(dt).minute)
    if minute == 0:
        return "EXACT_HOUR"
    if minute == 30:
        return "HALF_HOUR"
    if minute % 5 == 0:
        return "OTHER_5MIN_MULTIPLE"
    return "NON_5MIN"


def feature_freshness_bucket(position_age_min: float) -> str:
    if pd.isna(position_age_min):
        return "NO_HISTORY"
    x = float(position_age_min)
    if x <= 5:
        return "LE_5MIN"
    if x <= 60:
        return "GT5_LE60MIN"
    if x <= 180:
        return "GT60_LE180MIN"
    return "GT180MIN"


def destination_specificity(destination: object) -> str:
    if pd.isna(destination):
        return "NON_SPECIFIC"
    text = str(destination).strip().upper()
    if text in NON_SPECIFIC_DESTINATIONS or not text:
        return "NON_SPECIFIC"
    if "FOR ORDER" in text:
        return "NON_SPECIFIC"
    return "SPECIFIC_OR_STRUCTURED"


def reference_diagnostic_regime(row: pd.Series) -> str:
    """Descriptive regime only; it is not a relabeling of the company target."""
    raw = str(row["eta_reference_raw"]).strip()
    status = str(row["reference_eta_status"])
    margin = float(row["year_assignment_margin_days"])
    if raw in PLACEHOLDER_LIKE_ETA_STRINGS:
        return "PLACEHOLDER_LIKE_PATTERN"
    if margin < 90:
        return "YEAR_ASSIGNMENT_SENSITIVE"
    if status.startswith("PAST_"):
        return "PAST_REFERENCE"
    if status == "FUTURE_GT30D":
        return "FAR_FUTURE_GT30D"
    if status == "FUTURE_14_30D":
        return "FUTURE_14_30D"
    if status == "FUTURE_7_14D":
        return "FUTURE_7_14D"
    return "FUTURE_0_7D"


def build_reference_forensics(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    audits = [eta_year_resolution(r.eta_reference_raw, pd.Timestamp(r.last_update)) for r in out.itertuples(index=False)]
    out["chosen_eta_year"] = [a.chosen_year for a in audits]
    out["chosen_eta_delta_days"] = [a.chosen_delta_days for a in audits]
    out["second_eta_year"] = [a.second_year for a in audits]
    out["second_eta_delta_days"] = [a.second_delta_days for a in audits]
    out["year_assignment_margin_days"] = [a.assignment_margin_days for a in audits]
    out["year_assignment_margin_lt30d"] = out["year_assignment_margin_days"] < 30
    out["year_assignment_margin_lt60d"] = out["year_assignment_margin_days"] < 60
    out["year_assignment_margin_lt90d"] = out["year_assignment_margin_days"] < 90
    out["abs_reference_horizon_days"] = out["target_tte_h"].abs() / 24.0
    out["reference_abs_gt90d"] = out["abs_reference_horizon_days"] > 90
    out["reference_abs_gt120d"] = out["abs_reference_horizon_days"] > 120
    out["reference_abs_gt150d"] = out["abs_reference_horizon_days"] > 150
    out["eta_minute"] = pd.to_datetime(out["eta_reference_dt"]).dt.minute.astype(int)
    out["eta_granularity"] = pd.to_datetime(out["eta_reference_dt"]).map(eta_granularity_bucket)
    out["eta_on_5min_grid"] = out["eta_minute"].mod(5).eq(0)
    out["eta_on_00_or_30"] = out["eta_minute"].isin([0, 30])
    out["eta_placeholder_like_pattern"] = out["eta_reference_raw"].astype(str).isin(PLACEHOLDER_LIKE_ETA_STRINGS)
    out["feature_freshness_bucket"] = out["position_age_min"].map(feature_freshness_bucket)
    out["destination_specificity"] = out["destination_norm"].map(destination_specificity)
    out["reference_diagnostic_regime"] = out.apply(reference_diagnostic_regime, axis=1)
    return out


def top_error_concentration(abs_errors: Iterable[float], ks: Iterable[int] = (1, 3, 5, 10)) -> pd.DataFrame:
    x = pd.Series(list(abs_errors), dtype=float).dropna().sort_values(ascending=False).reset_index(drop=True)
    total = float(x.sum())
    rows = []
    for k in ks:
        k2 = min(int(k), len(x))
        share = float(x.iloc[:k2].sum() / total) if total > 0 and k2 else np.nan
        rows.append({"top_k": int(k), "rows_available": len(x), "share_of_total_absolute_error": share})
    return pd.DataFrame(rows)
