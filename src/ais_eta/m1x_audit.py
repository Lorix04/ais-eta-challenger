"""M1X audit and combined evidence gate for Augusta expansion.

This layer keeps three ideas separate:

1. a research-gate crossing candidate;
2. a manually audited AIS-derived Augusta port-call entry;
3. an ETA-feasibility cohort with enough observed pre-entry approach to support
   a causal benchmark.

Post-entry data may be used to validate the *label*, but never as prediction-time
features.  The Augusta target remains a versioned research event and is not
claimed to be an official ATA/PBP/berth/all-fast timestamp.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .m1 import KM_PER_NM, M1Config, _decision_grid, _rolling_median_speed, add_horizon_band
from .m1x import AugustaGeometryV1, label_augusta_geometry

AUDIT_REQUIRED_COLUMNS = {
    "session_id",
    "keep_for_port_call_truth",
    "ground_truth_confidence",
    "audited_event_type",
    "audit_reason_code",
    "audit_note",
}


@dataclass(frozen=True)
class M1XAuditConfig:
    confirmation_stop_s: float = 20 * 60.0
    major_gap_s: float = 10 * 60.0
    eta_min_pre_entry_distance_km: float = 15.0
    eta_min_pre_entry_observations_6h: int = 30


@dataclass(frozen=True)
class M1XGateCriteria:
    """Pre-specified evidence gate before evaluating the combined panel.

    These are engineering sufficiency thresholds, not literature constants.
    They are intentionally about *independent calls/vessels and horizon
    support*, not AIS row count or point-model performance.
    """

    min_combined_eta_calls: int = 50
    min_combined_unique_vessels: int = 25
    min_calls_with_over_6h_approach: int = 10
    min_calls_with_over_12h_approach: int = 5


def _to_bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)
    mapping = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
    out = s.astype(str).str.strip().str.lower().map(mapping)
    if out.isna().any():
        raise ValueError(f"cannot parse booleans: {s[out.isna()].unique().tolist()}")
    return out.astype(bool)


def validate_augusta_manual_audit(candidates: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    missing = AUDIT_REQUIRED_COLUMNS.difference(audit.columns)
    if missing:
        raise ValueError(f"Augusta audit missing columns: {sorted(missing)}")
    if audit["session_id"].duplicated().any():
        raise ValueError("Augusta audit has duplicate session_id")
    expected = set(candidates["session_id"].astype(str))
    observed = set(audit["session_id"].astype(str))
    if expected != observed:
        raise ValueError(
            f"Augusta audit coverage mismatch missing={sorted(expected-observed)} extra={sorted(observed-expected)}"
        )
    out = audit.copy()
    out["keep_for_port_call_truth"] = _to_bool_series(out["keep_for_port_call_truth"])
    out["ground_truth_confidence"] = out["ground_truth_confidence"].astype(str).str.upper().str.strip()
    if not set(out["ground_truth_confidence"]).issubset({"A", "B", "C"}):
        raise ValueError("ground_truth_confidence must be A/B/C")
    bad = out[(~out["keep_for_port_call_truth"]) & out["audited_event_type"].eq("VALIDATED_PORT_ENTRANCE")]
    if len(bad):
        raise ValueError("excluded candidates cannot be VALIDATED_PORT_ENTRANCE")
    return out.sort_values("session_id").reset_index(drop=True)


def build_augusta_confirmation_diagnostics(
    candidates: pd.DataFrame,
    states: pd.DataFrame,
    config: M1XAuditConfig | None = None,
    geometry: AugustaGeometryV1 | None = None,
) -> pd.DataFrame:
    """Ex-post diagnostics only for human label validation."""
    cfg = config or M1XAuditConfig()
    geom = geometry or AugustaGeometryV1()
    geo = label_augusta_geometry(states, geom)
    rows: list[dict] = []
    c = candidates.copy()
    c["inbound_time"] = pd.to_datetime(c["inbound_time"], errors="raise")
    c["outbound_time"] = pd.to_datetime(c["outbound_time"], errors="coerce")
    c["last_observed_at"] = pd.to_datetime(c["last_observed_at"], errors="coerce")
    for r in c.itertuples(index=False):
        v = geo[geo["mmsi"].eq(int(r.mmsi))].sort_values(["recorded_at", "id"], kind="mergesort")
        start = pd.Timestamp(r.inbound_time)
        end = pd.Timestamp(r.outbound_time) if pd.notna(r.outbound_time) else pd.Timestamp(r.last_observed_at)
        sess = v[(v["recorded_at"] >= start) & (v["recorded_at"] <= end)].copy()
        inner = sess[sess["augusta_inner_megarese_proxy"]]
        confirmed = inner[inner["stop_duration_s"].ge(cfg.confirmation_stop_s)]
        if len(confirmed):
            confirmation_time = pd.Timestamp(confirmed.iloc[0]["recorded_at"])
            to_conf = sess[sess["recorded_at"] <= confirmation_time].copy()
            gaps = to_conf["recorded_at"].diff().dt.total_seconds()
            time_to_conf = (confirmation_time - start).total_seconds() / 60.0
            max_gap = float(gaps.max() / 60.0) if len(gaps) else np.nan
            gaps_major = int(gaps.gt(cfg.major_gap_s).sum())
            confirmation_obs = int(len(to_conf))
            confirmation_high_stale = float(to_conf["stale_risk_level"].eq("HIGH").mean()) if len(to_conf) else np.nan
        else:
            confirmation_time = pd.NaT
            time_to_conf = np.nan
            max_gap = np.nan
            gaps_major = 0
            confirmation_obs = 0
            confirmation_high_stale = np.nan

        # Causal pre-entry observability diagnostic. This does not use future data.
        pre = v[(v["recorded_at"] >= start - pd.Timedelta(hours=6)) & (v["recorded_at"] < start)]
        gate_col = f"augusta_distance_to_{str(r.inbound_entrance).lower()}_gate_km"
        pre_max = float(pre[gate_col].max()) if len(pre) else np.nan
        rows.append(
            {
                "session_id": r.session_id,
                "confirmation_time": confirmation_time,
                "confirmation_time_min": float(time_to_conf) if pd.notna(time_to_conf) else np.nan,
                "confirmation_observations": confirmation_obs,
                "confirmation_max_gap_min": max_gap,
                "confirmation_gaps_gt_10m": gaps_major,
                "confirmation_high_stale_fraction": confirmation_high_stale,
                "pre_entry_observations_6h": int(len(pre)),
                "pre_entry_max_distance_to_target_gate_km": pre_max,
            }
        )
    return pd.DataFrame(rows).sort_values("session_id").reset_index(drop=True)


def assemble_augusta_audited_calls(
    candidates: pd.DataFrame,
    audit: pd.DataFrame,
    confirmation_diagnostics: pd.DataFrame,
    known_non_primary_mmsi: set[int] | None = None,
    config: M1XAuditConfig | None = None,
) -> pd.DataFrame:
    cfg = config or M1XAuditConfig()
    a = validate_augusta_manual_audit(candidates, audit)
    out = candidates.copy().merge(a, on="session_id", how="left", validate="one_to_one")
    out = out.merge(confirmation_diagnostics, on="session_id", how="left", validate="one_to_one")
    out["final_ground_truth"] = out["keep_for_port_call_truth"].astype(bool)
    out["ground_truth_event_name"] = np.where(
        out["final_ground_truth"], "ACTUAL_VALIDATED_AUGUSTA_RESEARCH_GATE_ENTRANCE", pd.NA
    )
    out["ground_truth_time"] = pd.to_datetime(out["inbound_time"], errors="raise")
    out["ground_truth_last_outside"] = pd.to_datetime(out["inbound_last_outside"], errors="coerce")
    out["ground_truth_first_inside"] = pd.to_datetime(out["inbound_first_inside"], errors="coerce")
    out["ground_truth_uncertainty_s"] = out["inbound_uncertainty_s"].astype(float)

    known_non_primary_mmsi = set() if known_non_primary_mmsi is None else {int(x) for x in known_non_primary_mmsi}
    named = out["name"].notna() & out["name"].astype(str).str.strip().ne("")
    long_range_observed = out["pre_entry_max_distance_to_target_gate_km"].ge(cfg.eta_min_pre_entry_distance_km)
    enough_pre_obs = out["pre_entry_observations_6h"].ge(cfg.eta_min_pre_entry_observations_6h)
    known_primary = ~out["mmsi"].astype(int).isin(known_non_primary_mmsi)
    out["eta_scope_method"] = "behavioral_long_range_approach_v1"
    out["eta_primary_eligible"] = (
        out["final_ground_truth"]
        & out["ground_truth_confidence"].isin(["A", "B"])
        & named
        & long_range_observed
        & enough_pre_obs
        & known_primary
    )
    out["eta_primary_exclusion_reason"] = pd.NA
    out.loc[~out["final_ground_truth"], "eta_primary_exclusion_reason"] = "event_not_trustworthy"
    out.loc[
        out["final_ground_truth"] & ~out["ground_truth_confidence"].isin(["A", "B"]),
        "eta_primary_exclusion_reason",
    ] = "ground_truth_confidence_too_low"
    out.loc[out["final_ground_truth"] & ~named, "eta_primary_exclusion_reason"] = "unnamed_vessel_scope_uncertain"
    out.loc[
        out["final_ground_truth"] & named & ~long_range_observed,
        "eta_primary_exclusion_reason",
    ] = "insufficient_long_range_pre_entry_observation"
    out.loc[
        out["final_ground_truth"] & named & long_range_observed & ~enough_pre_obs,
        "eta_primary_exclusion_reason",
    ] = "insufficient_pre_entry_observations"
    out.loc[
        out["final_ground_truth"] & out["mmsi"].astype(int).isin(known_non_primary_mmsi),
        "eta_primary_exclusion_reason",
    ] = "known_local_or_non_primary_role_from_prior_audit"
    out.loc[out["eta_primary_eligible"], "eta_primary_exclusion_reason"] = pd.NA
    return out.sort_values("session_id").reset_index(drop=True)


def _latest_augusta_outbound_before(events: pd.DataFrame, mmsi: int, target_time: pd.Timestamp) -> pd.Timestamp | pd.NaT:
    q = events[
        events["mmsi"].astype(int).eq(int(mmsi))
        & events["crossing_direction"].eq("OUTBOUND")
        & (pd.to_datetime(events["crossing_time_proxy"]) < target_time)
    ]
    return pd.NaT if q.empty else pd.Timestamp(pd.to_datetime(q["crossing_time_proxy"]).max())


def build_augusta_decision_panel(
    states: pd.DataFrame,
    cohort: pd.DataFrame,
    gate_events: pd.DataFrame,
    config: M1Config | None = None,
    geometry: AugustaGeometryV1 | None = None,
) -> pd.DataFrame:
    """Causal 15/30-minute panel for audited Augusta research-gate calls."""
    cfg = config or M1Config()
    geom = geometry or AugustaGeometryV1()
    calls = cohort.copy()
    calls["ground_truth_time"] = pd.to_datetime(calls["ground_truth_time"], errors="raise")
    events = gate_events.copy()
    events["crossing_time_proxy"] = pd.to_datetime(events["crossing_time_proxy"], errors="raise")
    mmsis = set(calls["mmsi"].astype(int))
    geo = label_augusta_geometry(states[states["mmsi"].astype(int).isin(mmsis)].copy(), geom)
    geo["recorded_at"] = pd.to_datetime(geo["recorded_at"], errors="raise")

    prepared: dict[int, pd.DataFrame] = {}
    for mmsi, group in geo.groupby("mmsi", sort=False):
        v = group.sort_values(["recorded_at", "id"], kind="mergesort").copy()
        v["sog_median_30m"] = _rolling_median_speed(v, cfg.robust_speed_window_min)
        prepared[int(mmsi)] = v.reset_index(drop=True)

    rows: list[dict] = []
    for call in calls.sort_values("ground_truth_time").itertuples(index=False):
        mmsi = int(call.mmsi)
        v = prepared[mmsi]
        target = pd.Timestamp(call.ground_truth_time)
        gate_name = str(call.inbound_entrance).upper()
        gate = next(g for g in geom.gates if g.name == gate_name)
        dist_col = f"augusta_distance_to_{gate_name.lower()}_gate_km"
        side_col = f"augusta_{gate_name.lower()}_side"
        prev_out = _latest_augusta_outbound_before(events, mmsi, target)
        voyage_start = prev_out if pd.notna(prev_out) else pd.Timestamp(v["recorded_at"].min())
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
                elapsed_h = (pd.Timestamp(obs.recorded_at) - pd.Timestamp(lag.recorded_at)).total_seconds() / 3600.0
                if pd.Timestamp(lag.recorded_at) >= voyage_start and elapsed_h > 0 and pd.notna(lag[dist_col]) and pd.notna(obs[dist_col]):
                    progress_kn = (float(lag[dist_col]) - float(obs[dist_col])) / elapsed_h / KM_PER_NM

            provider_recent = 0 <= provider_age_s <= cfg.max_provider_observation_age_s
            quality_ok = bool(
                provider_recent
                and str(obs.stale_risk_level) != "HIGH"
                and not bool(obs.kinematic_conflict)
                and not bool(obs.position_jump_candidate)
                and bool(obs.position_valid)
            )
            outside_target_gate = bool(pd.notna(obs[side_col]) and float(obs[side_col]) < 0)
            distance_km = float(obs[dist_col]) if pd.notna(obs[dist_col]) else np.nan
            approach_scope = bool(
                quality_ok
                and outside_target_gate
                and pd.notna(progress_kn)
                and progress_kn >= cfg.approach_min_progress_kn
                and np.isfinite(distance_km)
                and distance_km <= cfg.approach_max_distance_km
            )
            tta_h = float((target - decision_time).total_seconds() / 3600.0)
            med_sog = float(obs.sog_median_30m) if pd.notna(obs.sog_median_30m) else np.nan
            pred_h = (distance_km / KM_PER_NM) / max(med_sog, cfg.speed_floor_kn) if np.isfinite(distance_km) and np.isfinite(med_sog) else np.nan
            rows.append(
                {
                    "port": "AUGUSTA",
                    "session_id": call.session_id,
                    "mmsi": mmsi,
                    "name": call.name,
                    "inbound_entrance": gate_name,
                    "ground_truth_time": target,
                    "ground_truth_confidence": call.ground_truth_confidence,
                    "decision_time": pd.Timestamp(decision_time),
                    "voyage_start_time": pd.Timestamp(voyage_start),
                    "source_observation_time": pd.Timestamp(obs.recorded_at),
                    "provider_observation_age_s": provider_age_s,
                    "true_tta_h": tta_h,
                    "distance_to_target_gate_km": distance_km,
                    "sog_median_30m_kn": med_sog,
                    "progress_to_gate_30m_kn": progress_kn,
                    "input_quality_ok": quality_ok,
                    "outside_target_research_gate": outside_target_gate,
                    "scope_inbound_approach": approach_scope,
                    "pred_geodesic_median30_sog_floor_h": pred_h,
                }
            )
    panel = pd.DataFrame(rows)
    if panel.empty:
        return panel
    panel["horizon_band"] = add_horizon_band(panel["true_tta_h"])
    return panel.sort_values(["ground_truth_time", "session_id", "decision_time"]).reset_index(drop=True)


def call_horizon_support(panel: pd.DataFrame, scope_column: str = "scope_inbound_approach") -> pd.DataFrame:
    q = panel[panel[scope_column].astype(bool)].copy()
    if q.empty:
        return pd.DataFrame(columns=["port", "session_id", "mmsi", "max_supported_horizon_h"])
    return (
        q.groupby(["port", "session_id", "mmsi"], as_index=False)["true_tta_h"]
        .max()
        .rename(columns={"true_tta_h": "max_supported_horizon_h"})
    )


def evaluate_m1x_evidence_gate(
    combined_calls: pd.DataFrame,
    horizon_support: pd.DataFrame,
    criteria: M1XGateCriteria | None = None,
) -> dict:
    c = criteria or M1XGateCriteria()
    total_calls = int(len(combined_calls))
    unique_vessels = int(combined_calls["mmsi"].astype(int).nunique())
    over6 = int(horizon_support["max_supported_horizon_h"].gt(6.0).sum())
    over12 = int(horizon_support["max_supported_horizon_h"].gt(12.0).sum())
    checks = {
        "combined_eta_calls": total_calls >= c.min_combined_eta_calls,
        "combined_unique_vessels": unique_vessels >= c.min_combined_unique_vessels,
        "calls_with_over_6h_approach": over6 >= c.min_calls_with_over_6h_approach,
        "calls_with_over_12h_approach": over12 >= c.min_calls_with_over_12h_approach,
    }
    return {
        "criteria": {
            "min_combined_eta_calls": c.min_combined_eta_calls,
            "min_combined_unique_vessels": c.min_combined_unique_vessels,
            "min_calls_with_over_6h_approach": c.min_calls_with_over_6h_approach,
            "min_calls_with_over_12h_approach": c.min_calls_with_over_12h_approach,
        },
        "observed": {
            "combined_eta_calls": total_calls,
            "combined_unique_vessels": unique_vessels,
            "calls_with_over_6h_approach": over6,
            "calls_with_over_12h_approach": over12,
        },
        "checks": checks,
        "status": "GO_M2" if all(checks.values()) else "HOLD_M2",
    }
