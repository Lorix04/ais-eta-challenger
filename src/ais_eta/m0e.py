"""M0E: manual Catania event audit and effective-sample-size accounting.

M0E is a *label-validation* layer.  It is intentionally allowed to inspect
post-entry trajectory behaviour in order to confirm whether an M0D candidate
behaves like a real port call.  Nothing produced here is a prediction-time
feature.

Two concepts are kept separate:

1. ``keep_for_port_call_truth``: whether the reconstructed Catania research-gate
   entry is trustworthy enough to retain as an AIS-derived arrival event.
2. ``eta_primary_eligible``: whether the retained event belongs to the primary
   operational-vessel cohort for the ETA challenger.  Port tugs, fishing,
   sailing/pleasure craft and small reserved craft remain useful Port
   Intelligence labels but are not mixed into the primary ETA benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .m0d import label_catania_geometry


AUDIT_REQUIRED_COLUMNS = {
    "session_id",
    "keep_for_port_call_truth",
    "ground_truth_confidence",
    "audited_event_type",
    "audit_reason_code",
    "audit_note",
}

ROLE_REQUIRED_COLUMNS = {
    "mmsi",
    "audited_vessel_role",
    "role_confidence",
    "role_source",
    "eta_primary_scope",
}


@dataclass(frozen=True)
class M0EAuditConfig:
    nominal_snapshot_s: float = 61.0
    pre_entry_window_h: float = 3.0
    post_exit_window_h: float = 6.0
    confirmation_stop_s: float = 15 * 60.0
    sparse_session_coverage_threshold: float = 0.50
    major_gap_s: float = 10 * 60.0


def _to_bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(bool)
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    out = s.astype(str).str.strip().str.lower().map(mapping)
    if out.isna().any():
        bad = s[out.isna()].unique().tolist()
        raise ValueError(f"cannot parse boolean values: {bad}")
    return out.astype(bool)


def validate_manual_audit(candidates: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    """Validate one explicit human-review decision for every M0D candidate."""
    missing = AUDIT_REQUIRED_COLUMNS.difference(audit.columns)
    if missing:
        raise ValueError(f"manual audit missing columns: {sorted(missing)}")
    if audit["session_id"].duplicated().any():
        dups = audit.loc[audit["session_id"].duplicated(), "session_id"].tolist()
        raise ValueError(f"duplicate manual audit session ids: {dups}")

    expected = set(candidates["session_id"].astype(str))
    observed = set(audit["session_id"].astype(str))
    if expected != observed:
        raise ValueError(
            "manual audit coverage mismatch; "
            f"missing={sorted(expected-observed)}, extra={sorted(observed-expected)}"
        )

    out = audit.copy()
    out["keep_for_port_call_truth"] = _to_bool_series(out["keep_for_port_call_truth"])
    out["ground_truth_confidence"] = out["ground_truth_confidence"].astype(str).str.upper().str.strip()
    allowed_conf = {"A", "B", "C"}
    invalid_conf = sorted(set(out["ground_truth_confidence"]) - allowed_conf)
    if invalid_conf:
        raise ValueError(f"invalid ground-truth confidence: {invalid_conf}")
    # Excluded candidates are retained in the audit ledger but may never be
    # presented as final ground truth.
    bad_keep = out[~out["keep_for_port_call_truth"] & out["audited_event_type"].eq("VALIDATED_PORT_ENTRANCE")]
    if len(bad_keep):
        raise ValueError("excluded rows cannot be labelled VALIDATED_PORT_ENTRANCE")
    return out.sort_values("session_id").reset_index(drop=True)


def validate_vessel_roles(candidates: pd.DataFrame, roles: pd.DataFrame) -> pd.DataFrame:
    missing = ROLE_REQUIRED_COLUMNS.difference(roles.columns)
    if missing:
        raise ValueError(f"vessel role table missing columns: {sorted(missing)}")
    if roles["mmsi"].duplicated().any():
        raise ValueError("vessel role table contains duplicate MMSI")
    expected = set(candidates["mmsi"].astype(int))
    observed = set(roles["mmsi"].astype(int))
    if expected != observed:
        raise ValueError(
            "vessel role coverage mismatch; "
            f"missing={sorted(expected-observed)}, extra={sorted(observed-expected)}"
        )
    out = roles.copy()
    out["mmsi"] = out["mmsi"].astype(int)
    out["eta_primary_scope"] = _to_bool_series(out["eta_primary_scope"])
    return out.sort_values("mmsi").reset_index(drop=True)


def build_session_diagnostics(
    candidates: pd.DataFrame,
    states: pd.DataFrame,
    config: M0EAuditConfig | None = None,
) -> pd.DataFrame:
    """Build ex-post audit diagnostics for each M0D candidate.

    These diagnostics intentionally use the complete candidate session to
    validate the *label*. They must never be merged into prediction-time model
    features.
    """
    cfg = config or M0EAuditConfig()
    c = candidates.copy()
    for col in ["inbound_time", "outbound_time", "last_observed_at"]:
        c[col] = pd.to_datetime(c[col], errors="coerce")

    geo = label_catania_geometry(states)
    rows: list[dict] = []
    for r in c.itertuples(index=False):
        vessel = geo[geo["mmsi"].eq(int(r.mmsi))].sort_values(["recorded_at", "id"], kind="mergesort")
        start = pd.Timestamp(r.inbound_time)
        end = pd.Timestamp(r.outbound_time) if pd.notna(r.outbound_time) else pd.Timestamp(r.last_observed_at)
        sess = vessel[(vessel["recorded_at"] >= start) & (vessel["recorded_at"] <= end)].copy()
        pre = vessel[
            (vessel["recorded_at"] >= start - pd.Timedelta(hours=cfg.pre_entry_window_h))
            & (vessel["recorded_at"] < start)
        ]
        post = vessel[
            (vessel["recorded_at"] > end)
            & (vessel["recorded_at"] <= end + pd.Timedelta(hours=cfg.post_exit_window_h))
        ]

        duration_s = max((end - start).total_seconds(), 0.0)
        expected_obs = max(duration_s / cfg.nominal_snapshot_s, 1.0)
        coverage = len(sess) / expected_obs
        gaps = sess["recorded_at"].diff().dt.total_seconds()
        inner = sess[sess["catania_inner_harbour_proxy"]]
        confirmed_stop_rows = inner[inner["stop_duration_s"].ge(cfg.confirmation_stop_s)]
        if len(confirmed_stop_rows):
            time_to_confirmed_stop_min = (
                confirmed_stop_rows.iloc[0]["recorded_at"] - start
            ).total_seconds() / 60.0
        else:
            time_to_confirmed_stop_min = np.nan

        pre_start_km = float(pre["catania_distance_to_gate_mid_km"].iloc[0]) if len(pre) else np.nan
        pre_end_km = float(pre["catania_distance_to_gate_mid_km"].iloc[-1]) if len(pre) else np.nan
        pre_max_km = float(pre["catania_distance_to_gate_mid_km"].max()) if len(pre) else np.nan
        stale_window = vessel[
            (vessel["recorded_at"] >= start - pd.Timedelta(minutes=10))
            & (vessel["recorded_at"] <= start + pd.Timedelta(minutes=10))
        ]
        stale_high_fraction = (
            float(stale_window["stale_risk_level"].eq("HIGH").mean()) if len(stale_window) else np.nan
        )

        rows.append(
            {
                "session_id": r.session_id,
                "session_observation_coverage": float(coverage),
                "session_sparse": bool(coverage < cfg.sparse_session_coverage_threshold),
                "max_session_observation_gap_min": float(gaps.max() / 60.0) if len(gaps) else np.nan,
                "session_gaps_gt_10m": int(gaps.gt(cfg.major_gap_s).sum()),
                "time_to_confirmed_stop_min": float(time_to_confirmed_stop_min)
                if pd.notna(time_to_confirmed_stop_min)
                else np.nan,
                "pre_entry_observations_3h": int(len(pre)),
                "pre_entry_start_distance_km": pre_start_km,
                "pre_entry_end_distance_km": pre_end_km,
                "pre_entry_max_distance_km": pre_max_km,
                "post_session_observations_6h": int(len(post)),
                "session_port_side_fraction": float(sess["catania_port_side"].mean()) if len(sess) else np.nan,
                "session_inner_harbour_fraction": float(sess["catania_inner_harbour_proxy"].mean()) if len(sess) else np.nan,
                "entry_window_high_stale_fraction": stale_high_fraction,
            }
        )
    return pd.DataFrame(rows).sort_values("session_id").reset_index(drop=True)


def assemble_audited_calls(
    candidates: pd.DataFrame,
    audit: pd.DataFrame,
    roles: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Return the complete M0E audit ledger and final entry-truth flags."""
    a = validate_manual_audit(candidates, audit)
    r = validate_vessel_roles(candidates, roles)
    out = candidates.copy()
    out = out.merge(a, on="session_id", how="left", validate="one_to_one")
    out = out.merge(r, on="mmsi", how="left", validate="many_to_one")
    out = out.merge(diagnostics, on="session_id", how="left", validate="one_to_one")

    out["final_ground_truth"] = out["keep_for_port_call_truth"].astype(bool)
    out["ground_truth_event_name"] = np.where(
        out["final_ground_truth"],
        "ACTUAL_VALIDATED_CATANIA_RESEARCH_GATE_ENTRANCE",
        pd.NA,
    )
    out["ground_truth_time"] = pd.to_datetime(out["inbound_time"], errors="coerce")
    out["ground_truth_last_outside"] = pd.to_datetime(out["inbound_last_outside"], errors="coerce")
    out["ground_truth_first_inside"] = pd.to_datetime(out["inbound_first_inside"], errors="coerce")
    out["ground_truth_uncertainty_s"] = out["inbound_uncertainty_s"].astype(float)

    # Primary ETA cohort is intentionally narrower than the Port Intelligence
    # truth set.  This protects the benchmark from being dominated by local
    # service craft and small/recreational vessels with qualitatively different
    # operating processes.
    out["eta_primary_eligible"] = (
        out["final_ground_truth"]
        & out["eta_primary_scope"].astype(bool)
        & out["ground_truth_confidence"].isin(["A", "B"])
    )
    out["eta_primary_exclusion_reason"] = pd.NA
    excluded_event = ~out["final_ground_truth"]
    out.loc[excluded_event, "eta_primary_exclusion_reason"] = "event_not_trustworthy"
    non_scope = out["final_ground_truth"] & ~out["eta_primary_scope"].astype(bool)
    out.loc[non_scope, "eta_primary_exclusion_reason"] = "outside_primary_operational_vessel_scope"
    low_conf = out["final_ground_truth"] & out["eta_primary_scope"].astype(bool) & ~out["ground_truth_confidence"].isin(["A", "B"])
    out.loc[low_conf, "eta_primary_exclusion_reason"] = "ground_truth_confidence_too_low"

    return out.sort_values("session_id").reset_index(drop=True)


def audit_summary(audited: pd.DataFrame) -> dict:
    kept = audited[audited["final_ground_truth"]]
    core = audited[audited["eta_primary_eligible"]]
    excluded = audited[~audited["final_ground_truth"]]
    return {
        "candidate_sessions": int(len(audited)),
        "validated_port_call_entries": int(len(kept)),
        "excluded_ambiguous_or_non_call": int(len(excluded)),
        "validated_unique_vessels": int(kept["mmsi"].nunique()),
        "ground_truth_confidence": kept["ground_truth_confidence"].value_counts().sort_index().to_dict(),
        "primary_eta_eligible_calls": int(len(core)),
        "primary_eta_unique_vessels": int(core["mmsi"].nunique()),
        "primary_eta_roles": core["audited_vessel_role"].value_counts().to_dict(),
        "right_censored_validated_entries": int(kept["right_censored"].sum()),
        "unresolved_missing_outbound_validated_entries": int(kept["unresolved_missing_outbound"].sum()),
        "destination_support_fraction_validated": float(kept["destination_support_catania"].mean()) if len(kept) else np.nan,
        "median_ground_truth_uncertainty_s": float(kept["ground_truth_uncertainty_s"].median()) if len(kept) else np.nan,
        "p95_ground_truth_uncertainty_s": float(kept["ground_truth_uncertainty_s"].quantile(0.95)) if len(kept) else np.nan,
        "sparse_validated_sessions": int(kept["session_sparse"].sum()),
        "audit_reason_counts": audited["audit_reason_code"].value_counts().to_dict(),
    }
