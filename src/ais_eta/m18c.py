"""M18C target/label reliability audit utilities.

M18C is diagnostic-only. It does not train, tune, relabel, filter, or promote a
model. It audits the company-provided ``Tracks.eta`` reference under the frozen
M18A prediction contract and M18B development population.

Key principles:
* preserve every parseable company reference in the strict benchmark;
* distinguish label-semantics diagnostics from authoritative ATA/berth truth;
* keep old-final owners and the fresh external holdout sealed;
* use historical AIS-derived Catania/Augusta research-gate events only as a
  post-hoc temporal-alignment diagnostic, never as replacement ground truth.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .m11 import build_reference_forensics

M18C_VERSION = "m18c-target-label-reliability-v1-20260923"
M18C_EXPECTED_DEV_ROWS = 386
M18C_EXPECTED_OLD_FINAL_ROWS = 53


def diagnostic_reliability_tier(row: pd.Series) -> str:
    """Assign a *diagnostic-only* label-reliability tier.

    The tier uses only properties of the company reference itself. It must not
    be exposed to a predictor or used to delete/reweight rows without a new,
    separately frozen protocol.
    """
    status = str(row["reference_eta_status"])
    if bool(row.get("eta_placeholder_like_pattern", False)):
        return "R0_REFERENCE_AMBIGUOUS"
    if bool(row.get("year_assignment_margin_lt90d", False)):
        return "R0_REFERENCE_AMBIGUOUS"
    if status in {"PAST_GT24H", "FUTURE_GT30D"}:
        return "R0_REFERENCE_AMBIGUOUS"
    if status in {"PAST_1_24H", "PAST_LT1H", "FUTURE_14_30D"}:
        return "R1_REFERENCE_QUESTIONABLE"
    if status == "FUTURE_7_14D":
        return "R2_REFERENCE_EXTENDED_HORIZON"
    if status == "FUTURE_0_7D":
        return "R3_REFERENCE_NEAR_TERM"
    return "R0_REFERENCE_AMBIGUOUS"


def build_label_forensics(rows: pd.DataFrame, development_mmsi: Sequence[int]) -> pd.DataFrame:
    ids = {int(x) for x in development_mmsi}
    out = rows[rows["mmsi"].astype(int).isin(ids)].copy()
    if len(out) != M18C_EXPECTED_DEV_ROWS or out["mmsi"].astype(int).nunique() != M18C_EXPECTED_DEV_ROWS:
        raise AssertionError("M18C requires exactly 386 unique development MMSIs")
    out = build_reference_forensics(out)
    out["m18c_reliability_tier"] = out.apply(diagnostic_reliability_tier, axis=1)
    out["m18c_near_term_reference"] = out["reference_eta_status"].eq("FUTURE_0_7D")
    out["m18c_future_reference"] = ~out["reference_eta_status"].str.startswith("PAST_")
    return out.sort_values("mmsi").reset_index(drop=True)


def attach_frozen_prediction_errors(forensics: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    required = {"mmsi", "target_tte_h", "pred_m16g_selected_h"}
    missing = required - set(oof.columns)
    if missing:
        raise KeyError(f"M18C missing M16G OOF columns: {sorted(missing)}")
    p = oof[["mmsi", "target_tte_h", "pred_m16g_selected_h"]].copy()
    if len(p) != M18C_EXPECTED_DEV_ROWS or p["mmsi"].astype(int).nunique() != M18C_EXPECTED_DEV_ROWS:
        raise AssertionError("M18C expected 386 M16G OOF rows")
    out = forensics.merge(p, on="mmsi", how="left", validate="one_to_one", suffixes=("", "_oof"))
    if out["pred_m16g_selected_h"].isna().any():
        raise AssertionError("M18C missing frozen M16G predictions")
    if not np.allclose(out["target_tte_h"], out["target_tte_h_oof"], rtol=0, atol=1e-8):
        raise AssertionError("M18C target mismatch versus frozen M16G ledger")
    out["m16g_signed_error_h"] = out["pred_m16g_selected_h"] - out["target_tte_h"]
    out["m16g_abs_error_h"] = out["m16g_signed_error_h"].abs()
    return out.drop(columns=["target_tte_h_oof"])


def grouped_error_metrics(rows: pd.DataFrame, group_col: str) -> pd.DataFrame:
    def q90(x: pd.Series) -> float:
        return float(x.quantile(0.90))
    g = (
        rows.groupby(group_col, dropna=False)
        .agg(
            rows=("mmsi", "size"),
            mae_h=("m16g_abs_error_h", "mean"),
            medae_h=("m16g_abs_error_h", "median"),
            p90_ae_h=("m16g_abs_error_h", q90),
            max_ae_h=("m16g_abs_error_h", "max"),
            total_abs_error_h=("m16g_abs_error_h", "sum"),
        )
        .reset_index()
    )
    total = float(rows["m16g_abs_error_h"].sum())
    g["row_fraction"] = g["rows"] / len(rows)
    g["absolute_error_share"] = g["total_abs_error_h"] / total if total else np.nan
    return g.sort_values("mae_h", ascending=False).reset_index(drop=True)


def error_concentration(abs_errors: Sequence[float], ks: Sequence[int] = (1, 3, 5, 10, 20, 50, 107)) -> pd.DataFrame:
    x = pd.Series(abs_errors, dtype=float).dropna().sort_values(ascending=False).reset_index(drop=True)
    total = float(x.sum())
    records = []
    for k in ks:
        k2 = min(int(k), len(x))
        records.append({
            "top_k": int(k),
            "rows_available": int(len(x)),
            "absolute_error_share": float(x.iloc[:k2].sum() / total) if total and k2 else np.nan,
        })
    return pd.DataFrame(records)


def build_ais_proxy_alignment(forensics: pd.DataFrame, calls: pd.DataFrame) -> pd.DataFrame:
    """Post-hoc alignment to historical research-gate events.

    This is deliberately *not* a ground-truth replacement. The Catania/Augusta
    events are manually validated research-gate entrance proxies and all were
    derived from the same finite AIS observation period. We only quantify
    whether a company's reference timestamp resembles a historically observed
    event for the same MMSI.
    """
    c = calls.copy()
    required = {"mmsi", "ground_truth_time", "port"}
    missing = required - set(c.columns)
    if missing:
        raise KeyError(f"M18C missing AIS proxy columns: {sorted(missing)}")
    c["mmsi"] = c["mmsi"].astype(int)
    c["ground_truth_time"] = pd.to_datetime(c["ground_truth_time"], format="mixed", errors="coerce")
    if c["ground_truth_time"].isna().any():
        raise AssertionError("invalid M18C AIS proxy timestamps")

    records: list[dict[str, object]] = []
    for row in forensics.itertuples(index=False):
        sub = c[c["mmsi"].eq(int(row.mmsi))].copy()
        if sub.empty:
            continue
        eta = pd.Timestamp(row.eta_reference_dt)
        decision = pd.Timestamp(row.last_update)
        sub["call_minus_eta_h"] = (sub["ground_truth_time"] - eta).dt.total_seconds() / 3600.0
        sub["call_minus_decision_h"] = (sub["ground_truth_time"] - decision).dt.total_seconds() / 3600.0
        closest = sub.loc[sub["call_minus_eta_h"].abs().idxmin()]
        future_count = int((sub["ground_truth_time"] >= decision).sum())
        records.append({
            "mmsi": int(row.mmsi),
            "name": str(row.name),
            "last_update": decision,
            "eta_reference_dt": eta,
            "reference_eta_status": str(row.reference_eta_status),
            "m18c_reliability_tier": str(row.m18c_reliability_tier),
            "historical_proxy_calls_for_owner": int(len(sub)),
            "closest_proxy_port": str(closest["port"]),
            "closest_proxy_time": pd.Timestamp(closest["ground_truth_time"]),
            "closest_proxy_minus_eta_h": float(closest["call_minus_eta_h"]),
            "closest_proxy_abs_eta_delta_h": float(abs(closest["call_minus_eta_h"])),
            "closest_proxy_minus_decision_h": float(closest["call_minus_decision_h"]),
            "proxy_calls_on_or_after_decision": future_count,
            "proxy_alignment_is_posthoc_only": True,
            "proxy_is_authoritative_ata": False,
        })
    return pd.DataFrame(records).sort_values(["closest_proxy_abs_eta_delta_h", "mmsi"]).reset_index(drop=True)


def validate_audit_dict(audit: Mapping[str, object]) -> None:
    if audit.get("milestone") != "M18C" or audit.get("version") != M18C_VERSION:
        raise AssertionError("invalid M18C audit identity")
    scope = audit.get("scope", {})
    if scope.get("model_change") is not False or scope.get("relabeling") is not False:
        raise AssertionError("M18C must remain diagnostic-only")
    if scope.get("fresh_holdout_opened") is not False:
        raise AssertionError("M18C must not open fresh holdout")
    pop = audit.get("population", {})
    if int(pop.get("development_rows", -1)) != M18C_EXPECTED_DEV_ROWS:
        raise AssertionError("invalid M18C development population")
    if int(pop.get("blocked_old_final_rows", -1)) != M18C_EXPECTED_OLD_FINAL_ROWS:
        raise AssertionError("invalid M18C old-final block")
    findings = audit.get("findings", {})
    if int(findings.get("near_term_future_0_7d_rows", -1)) <= 0:
        raise AssertionError("M18C missing near-term reference findings")
    if float(findings.get("near_term_absolute_error_share", 2.0)) >= 1.0:
        raise AssertionError("invalid M18C error share")
    policy = audit.get("policy", {})
    if policy.get("strict_benchmark_rows_deleted") != 0:
        raise AssertionError("M18C must not delete strict benchmark rows")
    if policy.get("ais_proxy_replaces_company_target") is not False:
        raise AssertionError("M18C cannot replace company target with AIS proxy")
