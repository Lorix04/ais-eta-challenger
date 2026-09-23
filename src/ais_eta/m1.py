"""M1: ETA baseline feasibility under a limited-data Catania cohort.

This module intentionally stays below the complexity ceiling established by M0E.
It builds a *causal* decision-time panel on a UTC-aligned schedule and evaluates
simple geodesic/speed baselines.  It does not fit high-capacity ML and it does
not use future-informed horizon sampling.

The ground-truth event remains the M0E research event:
``ACTUAL_VALIDATED_CATANIA_RESEARCH_GATE_ENTRANCE``.  Nothing here upgrades
that proxy to an official PBP, port-limit ATA, berth arrival or all-fast time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .m0d import CataniaGeometryV1, label_catania_geometry

KM_PER_NM = 1.852


@dataclass(frozen=True)
class M1Config:
    """Frozen M1 design choices.

    ``decision_interval_min`` is anchored to the wall clock rather than to ATA,
    preventing fixed-horizon sample construction from encoding the label.
    ``max_provider_observation_age_s`` controls whether a scheduled decision
    point has a sufficiently recent provider snapshot.  The M0C stale-risk flag
    remains a separate quality check.
    """

    decision_interval_min: int = 30
    robust_speed_window_min: int = 30
    max_provider_observation_age_s: float = 180.0
    min_speed_kn: float = 0.5
    speed_floor_kn: float = 1.0
    approach_max_distance_km: float = 120.0
    approach_min_progress_kn: float = 0.5
    final_chronological_test_calls: int = 6
    session_cv_folds: int = 4
    vessel_cv_folds: int = 4


HORIZON_BINS_H = [-np.inf, 6.0, 12.0, 24.0, 48.0, 72.0, np.inf]
HORIZON_LABELS = ["<6h", "6-12h", "12-24h", "24-48h", "48-72h", ">72h"]


def destination_supports_catania(value: object) -> bool:
    """Conservative current-value resolver used only as a scope flag."""
    if value is None or pd.isna(value):
        return False
    text = str(value).upper().replace(" ", "")
    return any(token in text for token in ("CATANIA", "ITCTA", "ITCAT"))


def add_horizon_band(values_h: pd.Series) -> pd.Categorical:
    return pd.cut(
        values_h.astype(float),
        bins=HORIZON_BINS_H,
        labels=HORIZON_LABELS,
        include_lowest=True,
        right=True,
    )


def _balanced_group_assignment(group_sizes: pd.Series, n_folds: int) -> dict[int, int]:
    """Deterministically assign whole groups to folds while balancing call count."""
    if n_folds < 2:
        raise ValueError("n_folds must be >=2")
    loads = [0] * n_folds
    mapping: dict[int, int] = {}
    # Largest groups first; MMSI is a deterministic tie-breaker.
    ordered = sorted(((int(k), int(v)) for k, v in group_sizes.items()), key=lambda kv: (-kv[1], kv[0]))
    for group, size in ordered:
        fold = min(range(n_folds), key=lambda i: (loads[i], i))
        mapping[group] = fold
        loads[fold] += size
    return mapping


def assign_call_splits(cohort: pd.DataFrame, config: M1Config | None = None) -> pd.DataFrame:
    """Freeze development/final-test and grouped-CV assignments at call level.

    The final chronological test is intentionally small and *composition-only*
    during M1.  Its baseline metrics are not used for model selection.
    """
    cfg = config or M1Config()
    c = cohort.copy()
    c["ground_truth_time"] = pd.to_datetime(c["ground_truth_time"], errors="raise")
    c = c.sort_values(["ground_truth_time", "session_id"], kind="mergesort").reset_index(drop=True)
    if len(c) <= cfg.final_chronological_test_calls:
        raise ValueError("cohort too small for requested chronological test size")

    n_final = cfg.final_chronological_test_calls
    c["m1_split"] = "development"
    c.loc[c.index[-n_final:], "m1_split"] = "final_chronological_test"
    c["chronological_call_index"] = np.arange(len(c), dtype=int)
    c["vessel_call_count_total"] = c.groupby("mmsi")["session_id"].transform("size").astype(int)
    c["vessel_call_number"] = c.groupby("mmsi").cumcount().add(1).astype(int)
    c["is_first_observed_call_of_vessel"] = c["vessel_call_number"].eq(1)

    dev_mask = c["m1_split"].eq("development")
    dev_idx = c.index[dev_mask].to_numpy()
    # Session-level folds are assigned chronologically in round-robin order.
    c["session_cv_fold"] = pd.Series(pd.NA, index=c.index, dtype="Int64")
    c.loc[dev_idx, "session_cv_fold"] = np.arange(len(dev_idx)) % cfg.session_cv_folds

    # Vessel-level stress folds keep each MMSI intact.
    vessel_sizes = c.loc[dev_mask].groupby("mmsi")["session_id"].size()
    vessel_map = _balanced_group_assignment(vessel_sizes, cfg.vessel_cv_folds)
    c["vessel_cv_fold"] = pd.Series(pd.NA, index=c.index, dtype="Int64")
    c.loc[dev_mask, "vessel_cv_fold"] = c.loc[dev_mask, "mmsi"].astype(int).map(vessel_map).astype("Int64")

    dev_vessels = set(c.loc[dev_mask, "mmsi"].astype(int))
    c["vessel_seen_in_development"] = c["mmsi"].astype(int).isin(dev_vessels)
    return c


def _rolling_median_speed(vessel: pd.DataFrame, window_min: int) -> pd.Series:
    x = vessel.sort_values(["recorded_at", "id"], kind="mergesort").copy()
    s = x.set_index("recorded_at")["sog_clean"].rolling(
        f"{int(window_min)}min", min_periods=1, closed="both"
    ).median()
    return pd.Series(s.to_numpy(), index=x.index, dtype=float)


def _latest_outbound_before(events: pd.DataFrame, mmsi: int, target_time: pd.Timestamp) -> pd.Timestamp | pd.NaT:
    mask = (
        events["mmsi"].astype(int).eq(int(mmsi))
        & events["crossing_direction"].eq("OUTBOUND")
        & (events["crossing_time_proxy"] < target_time)
    )
    if not mask.any():
        return pd.NaT
    return pd.Timestamp(events.loc[mask, "crossing_time_proxy"].max())


def _decision_grid(start: pd.Timestamp, target: pd.Timestamp, interval_min: int) -> pd.DatetimeIndex:
    freq = f"{int(interval_min)}min"
    first = pd.Timestamp(start).ceil(freq)
    last = pd.Timestamp(target).floor(freq)
    if first >= target or last < first:
        return pd.DatetimeIndex([], dtype="datetime64[ns]")
    grid = pd.date_range(first, last, freq=freq)
    return grid[grid < target]


def build_decision_panel(
    states: pd.DataFrame,
    cohort: pd.DataFrame,
    gate_events: pd.DataFrame,
    split_table: pd.DataFrame | None = None,
    config: M1Config | None = None,
    geometry: CataniaGeometryV1 | None = None,
) -> pd.DataFrame:
    """Build a causal scheduled-decision panel for the M1 cohort.

    Each row represents a UTC wall-clock decision time.  The feature snapshot is
    the latest provider observation at or before that time.  No row after the
    decision timestamp is used.  The next-arrival truth is used only to form the
    supervised label and to stop generating rows after the event.
    """
    cfg = config or M1Config()
    geom = geometry or CataniaGeometryV1()
    splits = split_table if split_table is not None else assign_call_splits(cohort, cfg)
    splits = splits.copy()
    splits["ground_truth_time"] = pd.to_datetime(splits["ground_truth_time"], errors="raise")

    events = gate_events.copy()
    events["crossing_time_proxy"] = pd.to_datetime(events["crossing_time_proxy"], errors="raise")

    mmsi_set = set(splits["mmsi"].astype(int))
    geo = label_catania_geometry(states[states["mmsi"].astype(int).isin(mmsi_set)].copy(), geom)
    geo["recorded_at"] = pd.to_datetime(geo["recorded_at"], errors="raise")

    prepared: dict[int, pd.DataFrame] = {}
    for mmsi, group in geo.groupby("mmsi", sort=False):
        v = group.sort_values(["recorded_at", "id"], kind="mergesort").copy()
        v["sog_median_30m"] = _rolling_median_speed(v, cfg.robust_speed_window_min)
        prepared[int(mmsi)] = v.reset_index(drop=True)

    rows: list[dict] = []
    for call in splits.sort_values("ground_truth_time").itertuples(index=False):
        mmsi = int(call.mmsi)
        v = prepared[mmsi]
        target = pd.Timestamp(call.ground_truth_time)
        prev_outbound = _latest_outbound_before(events, mmsi, target)
        voyage_start = prev_outbound if pd.notna(prev_outbound) else pd.Timestamp(v["recorded_at"].min())
        grid = _decision_grid(voyage_start, target, cfg.decision_interval_min)
        if len(grid) == 0:
            continue

        times = v["recorded_at"].to_numpy(dtype="datetime64[ns]")
        for decision_time in grid:
            idx = int(np.searchsorted(times, np.datetime64(decision_time), side="right") - 1)
            if idx < 0:
                continue
            obs = v.iloc[idx]
            if pd.Timestamp(obs.recorded_at) < voyage_start:
                continue

            provider_age_s = float((decision_time - pd.Timestamp(obs.recorded_at)).total_seconds())
            lag_time = decision_time - pd.Timedelta(minutes=cfg.robust_speed_window_min)
            lag_idx = int(np.searchsorted(times, np.datetime64(lag_time), side="right") - 1)
            progress_kn = np.nan
            if lag_idx >= 0:
                lag = v.iloc[lag_idx]
                if pd.Timestamp(lag.recorded_at) >= voyage_start:
                    elapsed_h = (pd.Timestamp(obs.recorded_at) - pd.Timestamp(lag.recorded_at)).total_seconds() / 3600.0
                    if elapsed_h > 0 and pd.notna(lag.catania_distance_to_gate_mid_km) and pd.notna(obs.catania_distance_to_gate_mid_km):
                        progress_kmh = (
                            float(lag.catania_distance_to_gate_mid_km)
                            - float(obs.catania_distance_to_gate_mid_km)
                        ) / elapsed_h
                        progress_kn = progress_kmh / KM_PER_NM

            destination_support = destination_supports_catania(obs.destination_clean)
            provider_snapshot_recent = 0 <= provider_age_s <= cfg.max_provider_observation_age_s
            input_quality_ok = (
                provider_snapshot_recent
                and str(obs.stale_risk_level) != "HIGH"
                and not bool(obs.kinematic_conflict)
                and not bool(obs.position_jump_candidate)
                and bool(obs.position_valid)
            )
            outside_gate = not bool(obs.catania_port_side)
            conditional_scope = bool(input_quality_ok and outside_gate)
            declared_destination_scope = bool(conditional_scope and destination_support)
            approach_scope = bool(
                conditional_scope
                and pd.notna(progress_kn)
                and progress_kn >= cfg.approach_min_progress_kn
                and float(obs.catania_distance_to_gate_mid_km) <= cfg.approach_max_distance_km
            )

            true_tta_h = float((target - decision_time).total_seconds() / 3600.0)
            distance_km = float(obs.catania_distance_to_gate_mid_km)
            distance_nm = distance_km / KM_PER_NM
            sog = float(obs.sog_clean) if pd.notna(obs.sog_clean) else np.nan
            med_sog = float(obs.sog_median_30m) if pd.notna(obs.sog_median_30m) else np.nan

            pred_current = distance_nm / sog if np.isfinite(sog) and sog >= cfg.min_speed_kn else np.nan
            pred_floor = distance_nm / max(sog, cfg.speed_floor_kn) if np.isfinite(sog) else np.nan
            pred_robust = distance_nm / max(med_sog, cfg.speed_floor_kn) if np.isfinite(med_sog) else np.nan

            rows.append(
                {
                    "session_id": call.session_id,
                    "mmsi": mmsi,
                    "name": call.name,
                    "ground_truth_time": target,
                    "ground_truth_confidence": call.ground_truth_confidence,
                    "m1_split": call.m1_split,
                    "session_cv_fold": call.session_cv_fold,
                    "vessel_cv_fold": call.vessel_cv_fold,
                    "vessel_call_count_total": int(call.vessel_call_count_total),
                    "vessel_call_number": int(call.vessel_call_number),
                    "decision_time": pd.Timestamp(decision_time),
                    "voyage_start_time": pd.Timestamp(voyage_start),
                    "source_observation_time": pd.Timestamp(obs.recorded_at),
                    "provider_observation_age_s": provider_age_s,
                    "true_tta_h": true_tta_h,
                    "horizon_band": None,  # assigned after DataFrame construction
                    "distance_to_gate_mid_km": distance_km,
                    "distance_to_gate_mid_nm": distance_nm,
                    "sog_current_kn": sog,
                    "sog_median_30m_kn": med_sog,
                    "progress_to_gate_30m_kn": progress_kn,
                    "destination_current": obs.destination_clean,
                    "destination_supports_catania": destination_support,
                    "stale_risk_level": obs.stale_risk_level,
                    "motion_state": obs.motion_state,
                    "dynamic_state_age_s": float(obs.dynamic_state_age_s),
                    "kinematic_conflict": bool(obs.kinematic_conflict),
                    "position_jump_candidate": bool(obs.position_jump_candidate),
                    "provider_snapshot_recent": provider_snapshot_recent,
                    "input_quality_ok": input_quality_ok,
                    "outside_research_gate": outside_gate,
                    "scope_conditional_target_known": conditional_scope,
                    "scope_declared_destination": declared_destination_scope,
                    "scope_inbound_approach": approach_scope,
                    "pred_b0_geodesic_current_sog_h": pred_current,
                    "pred_b1_geodesic_current_sog_floor_h": pred_floor,
                    "pred_b2_geodesic_median30_sog_floor_h": pred_robust,
                }
            )

    panel = pd.DataFrame(rows)
    if panel.empty:
        return panel
    panel["horizon_band"] = add_horizon_band(panel["true_tta_h"])
    return panel.sort_values(["ground_truth_time", "session_id", "decision_time"]).reset_index(drop=True)


def _weighted_quantile(values: Iterable[float], weights: Iterable[float], q: float) -> float:
    v = np.asarray(list(values), dtype=float)
    w = np.asarray(list(weights), dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not mask.any():
        return np.nan
    v = v[mask]
    w = w[mask]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cdf = np.cumsum(w) / np.sum(w)
    return float(v[min(np.searchsorted(cdf, q, side="left"), len(v) - 1)])


def _equal_call_weights(df: pd.DataFrame) -> pd.Series:
    counts = df.groupby("session_id")["session_id"].transform("size").astype(float)
    return 1.0 / counts


def baseline_metrics(
    panel: pd.DataFrame,
    scope_column: str,
    prediction_columns: Iterable[str] | None = None,
    development_only: bool = True,
) -> pd.DataFrame:
    """Voyage-balanced metrics for simple deterministic baselines."""
    preds = list(
        prediction_columns
        or [
            "pred_b0_geodesic_current_sog_h",
            "pred_b1_geodesic_current_sog_floor_h",
            "pred_b2_geodesic_median30_sog_floor_h",
        ]
    )
    base = panel[panel[scope_column].astype(bool)].copy()
    if development_only:
        base = base[base["m1_split"].eq("development")].copy()

    rows: list[dict] = []
    for pred in preds:
        q = base[base[pred].notna()].copy()
        if q.empty:
            continue
        q["error_h"] = q[pred] - q["true_tta_h"]
        q["abs_error_h"] = q["error_h"].abs()
        q["weight"] = _equal_call_weights(q)
        w = q["weight"].to_numpy(dtype=float)
        e = q["error_h"].to_numpy(dtype=float)
        ae = q["abs_error_h"].to_numpy(dtype=float)
        rows.append(
            {
                "scope": scope_column,
                "baseline": pred,
                "development_only": development_only,
                "prediction_points": int(len(q)),
                "calls_covered": int(q["session_id"].nunique()),
                "row_coverage": float(len(q) / len(base)) if len(base) else np.nan,
                "voyage_balanced_mae_h": float(np.average(ae, weights=w)),
                "voyage_balanced_rmse_h": float(np.sqrt(np.average(e**2, weights=w))),
                "voyage_balanced_median_ae_h": _weighted_quantile(ae, w, 0.50),
                "voyage_balanced_p90_ae_h": _weighted_quantile(ae, w, 0.90),
                "voyage_balanced_p95_ae_h": _weighted_quantile(ae, w, 0.95),
                "within_30m": float(np.average(ae <= 0.5, weights=w)),
                "within_60m": float(np.average(ae <= 1.0, weights=w)),
                "within_120m": float(np.average(ae <= 2.0, weights=w)),
            }
        )
    return pd.DataFrame(rows)


def horizon_metrics(
    panel: pd.DataFrame,
    scope_column: str,
    prediction_column: str,
    development_only: bool = True,
) -> pd.DataFrame:
    base = panel[
        panel[scope_column].astype(bool) & panel[prediction_column].notna()
    ].copy()
    if development_only:
        base = base[base["m1_split"].eq("development")]
    rows: list[dict] = []
    for band in HORIZON_LABELS:
        q = base[base["horizon_band"].astype(str).eq(band)].copy()
        if q.empty:
            rows.append(
                {
                    "scope": scope_column,
                    "baseline": prediction_column,
                    "horizon_band": band,
                    "prediction_points": 0,
                    "calls_covered": 0,
                    "voyage_balanced_mae_h": np.nan,
                    "voyage_balanced_median_ae_h": np.nan,
                    "voyage_balanced_p90_ae_h": np.nan,
                }
            )
            continue
        q["abs_error_h"] = (q[prediction_column] - q["true_tta_h"]).abs()
        q["weight"] = _equal_call_weights(q)
        ae = q["abs_error_h"].to_numpy(dtype=float)
        w = q["weight"].to_numpy(dtype=float)
        rows.append(
            {
                "scope": scope_column,
                "baseline": prediction_column,
                "horizon_band": band,
                "prediction_points": int(len(q)),
                "calls_covered": int(q["session_id"].nunique()),
                "voyage_balanced_mae_h": float(np.average(ae, weights=w)),
                "voyage_balanced_median_ae_h": _weighted_quantile(ae, w, 0.50),
                "voyage_balanced_p90_ae_h": _weighted_quantile(ae, w, 0.90),
            }
        )
    return pd.DataFrame(rows)


def call_level_metrics(
    panel: pd.DataFrame,
    scope_column: str,
    prediction_column: str,
    development_only: bool = True,
) -> pd.DataFrame:
    q = panel[panel[scope_column].astype(bool) & panel[prediction_column].notna()].copy()
    if development_only:
        q = q[q["m1_split"].eq("development")]
    if q.empty:
        return pd.DataFrame()
    q["abs_error_h"] = (q[prediction_column] - q["true_tta_h"]).abs()
    return (
        q.groupby(["session_id", "mmsi", "name"], as_index=False)
        .agg(
            prediction_points=("decision_time", "size"),
            min_true_tta_h=("true_tta_h", "min"),
            max_true_tta_h=("true_tta_h", "max"),
            mae_h=("abs_error_h", "mean"),
            median_ae_h=("abs_error_h", "median"),
            p90_ae_h=("abs_error_h", lambda s: float(s.quantile(0.90))),
            max_distance_km=("distance_to_gate_mid_km", "max"),
        )
        .sort_values("mae_h", ascending=False)
        .reset_index(drop=True)
    )


def split_diagnostics(split_table: pd.DataFrame) -> dict:
    s = split_table.copy()
    dev = s[s["m1_split"].eq("development")]
    final = s[s["m1_split"].eq("final_chronological_test")]
    counts = s.groupby(["mmsi", "name"])["session_id"].size().sort_values(ascending=False)
    top5_share = float(counts.head(5).sum() / len(s)) if len(s) else np.nan
    singleton_vessels = int((counts == 1).sum())
    return {
        "calls_total": int(len(s)),
        "unique_vessels_total": int(s["mmsi"].nunique()),
        "development_calls": int(len(dev)),
        "development_unique_vessels": int(dev["mmsi"].nunique()),
        "final_test_calls": int(len(final)),
        "final_test_unique_vessels": int(final["mmsi"].nunique()),
        "final_test_unseen_vessels": int((~final["vessel_seen_in_development"]).sum()),
        "max_calls_single_vessel": int(counts.max()) if len(counts) else 0,
        "top5_vessel_call_share": top5_share,
        "singleton_vessels": singleton_vessels,
        "session_cv_fold_calls": dev["session_cv_fold"].value_counts().sort_index().astype(int).to_dict(),
        "vessel_cv_fold_calls": dev["vessel_cv_fold"].value_counts().sort_index().astype(int).to_dict(),
        "chronological_test_start": final["ground_truth_time"].min().isoformat() if len(final) else None,
        "chronological_test_end": final["ground_truth_time"].max().isoformat() if len(final) else None,
    }


def scope_coverage(panel: pd.DataFrame, development_only: bool = True) -> pd.DataFrame:
    p = panel.copy()
    if development_only:
        p = p[p["m1_split"].eq("development")]
    rows = []
    for scope in [
        "scope_conditional_target_known",
        "scope_declared_destination",
        "scope_inbound_approach",
    ]:
        q = p[p[scope].astype(bool)]
        rows.append(
            {
                "scope": scope,
                "decision_points": int(len(q)),
                "calls_covered": int(q["session_id"].nunique()),
                "unique_vessels": int(q["mmsi"].nunique()),
                "median_true_tta_h": float(q["true_tta_h"].median()) if len(q) else np.nan,
                "max_true_tta_h": float(q["true_tta_h"].max()) if len(q) else np.nan,
            }
        )
        for band in HORIZON_LABELS:
            rows[-1][f"calls_{band}"] = int(q[q["horizon_band"].astype(str).eq(band)]["session_id"].nunique())
    return pd.DataFrame(rows)
