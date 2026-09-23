"""M18D residual error attribution utilities.

M18D is diagnostic-only. It decomposes the already-frozen M16G out-of-fold
residuals under the M18A prediction contract and M18B/M18C governance. It must
not train, tune, relabel, filter, promote, or open a fresh holdout.

The analysis deliberately separates:
* the strict 386-row company-reference benchmark; and
* the M18C `FUTURE_0_7D` diagnostic slice, used only to reduce obvious label-
  regime confounding while studying model residual structure.

Associations reported by M18D are not causal effects. Their purpose is to rank
which missing-information hypotheses deserve controlled ablation in M18E.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

M18D_VERSION = "m18d-residual-error-attribution-v1-20260923"
M18D_EXPECTED_DEV_ROWS = 386
M18D_EXPECTED_OLD_FINAL_ROWS = 53
M18D_NEAR_TERM_STATUS = "FUTURE_0_7D"

EXPERT_COLUMNS: Mapping[str, str] = {
    "prior": "pred_m16c_prior_h",
    "route": "pred_m16d_route_h",
    "physics_gate": "pred_m16e_physics_gate_h",
    "tabular": "pred_m16f_tabular_h",
}


def rank_correlation(x: pd.Series, y: pd.Series) -> float:
    """Deterministic Spearman-style rank correlation without scipy dependency."""
    z = pd.concat([x, y], axis=1).dropna()
    if len(z) < 3 or z.iloc[:, 0].nunique() < 2 or z.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(z.iloc[:, 0].rank(method="average").corr(z.iloc[:, 1].rank(method="average")))


def attach_residual_context(
    forensics: pd.DataFrame,
    m16g: pd.DataFrame,
    destination: pd.DataFrame,
    route: pd.DataFrame,
    physics: pd.DataFrame,
    uncertainty: pd.DataFrame,
) -> pd.DataFrame:
    """Build the one-row-per-owner M18D diagnostic ledger."""
    if len(forensics) != M18D_EXPECTED_DEV_ROWS or forensics["mmsi"].nunique() != M18D_EXPECTED_DEV_ROWS:
        raise AssertionError("M18D requires exactly 386 unique development rows")

    out = forensics.copy()
    # M18C already contains frozen M16G selected prediction and residuals. Add
    # only expert and decision-time diagnostics from earlier frozen milestones.
    additions = [
        (
            m16g,
            ["mmsi", "m16a_outer_fold", *EXPERT_COLUMNS.values(), "m16g_selected_meta_id", "m16g_selected_expert"],
        ),
        (
            destination,
            [
                "mmsi", "canonical_destination", "resolution_method", "resolution_confidence",
                "is_resolved_port", "is_non_specific", "is_route_expression",
            ],
        ),
        (
            route,
            [
                "mmsi", "m16d_neighbour_count", "m16d_nearest_distance",
                "m16d_median_neighbour_distance", "m16d_similarity_gap", "m16d_gate_used",
            ],
        ),
        (
            physics,
            [
                "mmsi", "physics_distance_gc_nm", "physics_course_alignment_deg",
                "physics_recent_speed_max_kn", "physics_eligible",
                "physics_local_sinuosity_180m", "physics_local_sinuosity_360m",
            ],
        ),
        (
            uncertainty,
            [
                "mmsi", "predicted_abs_error_h", "confidence_tier",
                "interval_half_width_h", "covered_80",
            ],
        ),
    ]
    for frame, cols in additions:
        missing = set(cols) - set(frame.columns)
        if missing:
            raise KeyError(f"M18D missing columns: {sorted(missing)}")
        out = out.merge(frame[cols], on="mmsi", how="left", validate="one_to_one")

    if len(out) != M18D_EXPECTED_DEV_ROWS or out["mmsi"].nunique() != M18D_EXPECTED_DEV_ROWS:
        raise AssertionError("M18D context merge changed development population")
    if out[list(EXPERT_COLUMNS.values())].isna().any().any():
        raise AssertionError("M18D expert predictions incomplete")

    expert_matrix = out[list(EXPERT_COLUMNS.values())].astype(float)
    out["expert_spread_h"] = expert_matrix.max(axis=1) - expert_matrix.min(axis=1)
    out["expert_std_h"] = expert_matrix.std(axis=1, ddof=1)

    expert_abs_cols: list[str] = []
    for name, col in EXPERT_COLUMNS.items():
        ae_col = f"{name}_abs_error_h"
        out[ae_col] = (out[col].astype(float) - out["target_tte_h"].astype(float)).abs()
        expert_abs_cols.append(ae_col)
    ae = out[expert_abs_cols].to_numpy(float)
    names = list(EXPERT_COLUMNS)
    oracle_idx = np.argmin(ae, axis=1)
    out["oracle_expert"] = [names[i] for i in oracle_idx]
    out["oracle_expert_abs_error_h"] = ae[np.arange(len(out)), oracle_idx]
    out["m16g_selection_regret_h"] = out["m16g_abs_error_h"] - out["oracle_expert_abs_error_h"]
    out["selected_expert_is_oracle"] = out["m16g_selected_expert"].astype(str).eq(out["oracle_expert"])
    out["m18d_scope_near_term"] = out["reference_eta_status"].eq(M18D_NEAR_TERM_STATUS)
    return out.sort_values("mmsi").reset_index(drop=True)


def error_metrics(rows: pd.DataFrame) -> dict[str, float | int]:
    ae = rows["m16g_abs_error_h"].astype(float)
    se = rows["m16g_signed_error_h"].astype(float)
    return {
        "rows": int(len(rows)),
        "mae_h": float(ae.mean()),
        "medae_h": float(ae.median()),
        "p90_ae_h": float(ae.quantile(0.90)),
        "p95_ae_h": float(ae.quantile(0.95)),
        "mean_signed_error_h": float(se.mean()),
        "underprediction_fraction": float(se.lt(0).mean()),
        "overprediction_fraction": float(se.gt(0).mean()),
        "total_abs_error_h": float(ae.sum()),
    }


def categorical_slice_metrics(
    rows: pd.DataFrame,
    columns: Sequence[str],
    scope_name: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    scope_total = float(rows["m16g_abs_error_h"].sum())
    for col in columns:
        if col not in rows.columns:
            continue
        for value, sub in rows.groupby(col, dropna=False):
            m = error_metrics(sub)
            n = int(m["rows"])
            support = "ROBUST_N_GE20" if n >= 20 else ("SMALL_N_8_19" if n >= 8 else "TINY_N_LT8")
            records.append({
                "scope": scope_name,
                "slice_family": col,
                "slice_value": "<NA>" if pd.isna(value) else str(value),
                **m,
                "row_fraction_within_scope": float(n / len(rows)) if len(rows) else np.nan,
                "absolute_error_share_within_scope": float(sub["m16g_abs_error_h"].sum() / scope_total) if scope_total else np.nan,
                "support_grade": support,
            })
    return pd.DataFrame(records)


def numeric_attribution(
    rows: pd.DataFrame,
    columns: Sequence[str],
    scope_name: str,
    min_n: int = 20,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for col in columns:
        if col not in rows.columns:
            continue
        z = rows[[col, "m16g_abs_error_h", "m16g_signed_error_h"]].dropna()
        if len(z) < min_n or z[col].nunique() < 3:
            continue
        records.append({
            "scope": scope_name,
            "feature": col,
            "rows": int(len(z)),
            "unique_values": int(z[col].nunique()),
            "rank_corr_abs_error": rank_correlation(z[col], z["m16g_abs_error_h"]),
            "rank_corr_signed_error": rank_correlation(z[col], z["m16g_signed_error_h"]),
        })
    out = pd.DataFrame(records)
    if len(out):
        out["abs_rank_corr_abs_error"] = out["rank_corr_abs_error"].abs()
        out = out.sort_values(["scope", "abs_rank_corr_abs_error"], ascending=[True, False]).reset_index(drop=True)
    return out


def quantile_slice_metrics(
    rows: pd.DataFrame,
    columns: Sequence[str],
    scope_name: str,
    q: int = 4,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for col in columns:
        if col not in rows.columns:
            continue
        z = rows[[col, "m16g_abs_error_h", "m16g_signed_error_h"]].dropna().copy()
        if len(z) < 20 or z[col].nunique() < 4:
            continue
        try:
            z["qbin"] = pd.qcut(z[col], q=q, duplicates="drop")
        except ValueError:
            continue
        for bin_value, sub in z.groupby("qbin", observed=True):
            m = error_metrics(sub)
            records.append({
                "scope": scope_name,
                "feature": col,
                "quantile_bin": str(bin_value),
                "feature_min": float(sub[col].min()),
                "feature_max": float(sub[col].max()),
                **m,
            })
    return pd.DataFrame(records)


def expert_diagnostics(rows: pd.DataFrame, scope_name: str) -> dict[str, object]:
    selected_mae = float(rows["m16g_abs_error_h"].mean())
    oracle_mae = float(rows["oracle_expert_abs_error_h"].mean())
    regret = rows["m16g_selection_regret_h"].astype(float)
    counts = rows["oracle_expert"].value_counts().to_dict()
    return {
        "scope": scope_name,
        "rows": int(len(rows)),
        "m16g_selected_mae_h": selected_mae,
        "oracle_existing_expert_mae_h": oracle_mae,
        "oracle_gap_h": float(selected_mae - oracle_mae),
        "selected_expert_is_oracle_fraction": float(rows["selected_expert_is_oracle"].mean()),
        "positive_selection_regret_fraction": float(regret.gt(1e-9).mean()),
        "mean_selection_regret_h": float(regret.mean()),
        "median_selection_regret_h": float(regret.median()),
        "p90_selection_regret_h": float(regret.quantile(0.90)),
        "oracle_expert_counts": {str(k): int(v) for k, v in counts.items()},
        "diagnostic_only": True,
    }


def validate_attribution_dict(audit: Mapping[str, object]) -> None:
    if audit.get("milestone") != "M18D" or audit.get("version") != M18D_VERSION:
        raise AssertionError("invalid M18D identity")
    scope = audit.get("scope", {})
    for key in ("model_change", "training", "tuning", "relabeling", "row_filtering", "fresh_holdout_opened"):
        if scope.get(key) is not False:
            raise AssertionError(f"M18D must remain diagnostic-only: {key}")
    pop = audit.get("population", {})
    if int(pop.get("development_rows", -1)) != M18D_EXPECTED_DEV_ROWS:
        raise AssertionError("invalid M18D development population")
    if int(pop.get("blocked_old_final_rows", -1)) != M18D_EXPECTED_OLD_FINAL_ROWS:
        raise AssertionError("invalid M18D old-final block")
    findings = audit.get("findings", {})
    if int(findings.get("near_term_rows", -1)) <= 0:
        raise AssertionError("M18D near-term diagnostic scope missing")
    if float(findings.get("strict_m16g_mae_h", -1)) <= 0:
        raise AssertionError("M18D strict MAE missing")
    if float(findings.get("near_term_m16g_mae_h", -1)) <= 0:
        raise AssertionError("M18D near-term MAE missing")
    if audit.get("policy", {}).get("attribution_is_causal_proof") is not False:
        raise AssertionError("M18D associations cannot be labelled causal proof")
