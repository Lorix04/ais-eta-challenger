#!/usr/bin/env python3
"""M11 — Reference ETA quality and frozen-model error forensics.

This milestone is diagnostic only. It does not fit, tune, select, or alter a model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m11 import build_reference_forensics, top_error_concentration

R = ROOT / "reports"
D = ROOT / "data" / "derived" / "m10_reference_rows.pkl.gz"


def error_metrics(g: pd.DataFrame) -> dict[str, float | int]:
    e = pd.to_numeric(g["abs_error_h"], errors="coerce").dropna()
    return {
        "n": int(len(e)),
        "mae_h": float(e.mean()) if len(e) else np.nan,
        "medae_h": float(e.median()) if len(e) else np.nan,
        "p90_ae_h": float(e.quantile(0.9)) if len(e) else np.nan,
        "max_ae_h": float(e.max()) if len(e) else np.nan,
    }


def main() -> int:
    rows = pd.read_pickle(D)
    f = build_reference_forensics(rows)
    f.to_csv(R / "m11_reference_eta_forensics.csv", index=False)

    # Reference temporal / granularity / year-inference diagnostics.
    status = (
        f.groupby("reference_eta_status", dropna=False)
        .agg(rows=("mmsi", "size"), median_tte_h=("target_tte_h", "median"), min_tte_h=("target_tte_h", "min"), max_tte_h=("target_tte_h", "max"))
        .reset_index()
    )
    status.to_csv(R / "m11_reference_status_diagnostics.csv", index=False)

    gran = (
        f.groupby("eta_granularity", dropna=False)
        .agg(rows=("mmsi", "size"))
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    gran["fraction"] = gran["rows"] / len(f)
    gran.to_csv(R / "m11_eta_granularity.csv", index=False)

    raw_dupes = (
        f.groupby("eta_reference_raw", dropna=False)
        .agg(rows=("mmsi", "size"))
        .reset_index()
        .sort_values(["rows", "eta_reference_raw"], ascending=[False, True])
    )
    raw_dupes.to_csv(R / "m11_eta_reference_multiplicity.csv", index=False)

    year_sensitive = f.loc[f["year_assignment_margin_lt90d"], [
        "mmsi", "name", "last_update", "eta_reference_raw", "eta_reference_dt", "target_tte_h",
        "reference_eta_status", "chosen_eta_year", "second_eta_year", "year_assignment_margin_days",
        "destination_norm",
    ]].sort_values("year_assignment_margin_days")
    year_sensitive.to_csv(R / "m11_year_assignment_sensitive_rows.csv", index=False)

    placeholders = f.loc[f["eta_placeholder_like_pattern"], [
        "mmsi", "name", "last_update", "eta_reference_raw", "eta_reference_dt", "target_tte_h",
        "reference_eta_status", "destination_norm", "track_sog", "nav_status_cat",
    ]].sort_values("mmsi")
    placeholders.to_csv(R / "m11_placeholder_like_eta_rows.csv", index=False)

    # Strict frozen M10 final-test model error forensics only.
    pred = pd.read_csv(R / "m10_final_predictions.csv")
    strict = pred.loc[pred["task"].eq("strict_all_parseable")].copy()
    strict["signed_error_h"] = strict["prediction_tte_h"] - strict["target_tte_h"]
    strict = strict.merge(
        f[[
            "mmsi", "eta_reference_raw", "eta_minute", "eta_granularity",
            "eta_placeholder_like_pattern", "year_assignment_margin_days",
            "year_assignment_margin_lt90d", "abs_reference_horizon_days",
            "feature_freshness_bucket", "position_age_min", "destination_specificity",
            "reference_diagnostic_regime",
        ]], on="mmsi", how="left", validate="one_to_one"
    )
    strict.to_csv(R / "m11_strict_final_error_forensics.csv", index=False)

    by_status = (
        strict.groupby("reference_eta_status", dropna=False)
        .apply(lambda x: pd.Series(error_metrics(x)), include_groups=False)
        .reset_index()
    )
    by_status.to_csv(R / "m11_final_error_by_reference_status.csv", index=False)

    by_regime = (
        strict.groupby("reference_diagnostic_regime", dropna=False)
        .apply(lambda x: pd.Series(error_metrics(x)), include_groups=False)
        .reset_index()
        .sort_values("mae_h", ascending=False)
    )
    by_regime.to_csv(R / "m11_final_error_by_diagnostic_regime.csv", index=False)

    by_fresh = (
        strict.groupby("feature_freshness_bucket", dropna=False)
        .apply(lambda x: pd.Series(error_metrics(x)), include_groups=False)
        .reset_index()
        .sort_values("mae_h", ascending=False)
    )
    by_fresh.to_csv(R / "m11_final_error_by_feature_freshness.csv", index=False)

    concentration = top_error_concentration(strict["abs_error_h"], ks=(1, 3, 5, 10, 20))
    concentration.to_csv(R / "m11_final_error_concentration.csv", index=False)

    top_errors = strict.nlargest(15, "abs_error_h")[[
        "mmsi", "name", "last_update", "eta_reference_raw", "eta_reference_dt", "target_tte_h",
        "reference_eta_status", "destination_norm", "prediction_tte_h", "abs_error_h", "signed_error_h",
        "reference_diagnostic_regime", "year_assignment_margin_days", "position_age_min",
    ]]
    top_errors.to_csv(R / "m11_final_top15_errors.csv", index=False)

    # Post-hoc sensitivity only. These are NOT alternative model-selection results.
    filters = {
        "ALL_PARSEABLE": pd.Series(True, index=strict.index),
        "FUTURE_ONLY": strict["target_tte_h"].ge(0),
        "FUTURE_0_7D": strict["target_tte_h"].between(0, 24 * 7),
        "FUTURE_0_14D": strict["target_tte_h"].between(0, 24 * 14),
        "FUTURE_0_30D": strict["target_tte_h"].between(0, 24 * 30),
        "ABS_TTE_LE_7D": strict["target_tte_h"].abs().le(24 * 7),
        "ABS_TTE_LE_30D": strict["target_tte_h"].abs().le(24 * 30),
    }
    sens_rows = []
    for label, mask in filters.items():
        m = error_metrics(strict.loc[mask])
        sens_rows.append({"diagnostic_filter": label, **m})
    sensitivity = pd.DataFrame(sens_rows)
    sensitivity.to_csv(R / "m11_frozen_model_posthoc_sensitivity.csv", index=False)

    minute00 = float(f["eta_minute"].eq(0).mean())
    minute00or30 = float(f["eta_on_00_or_30"].mean())
    minute5 = float(f["eta_on_5min_grid"].mean())
    dup_counts = f["eta_reference_dt"].value_counts()
    conc = concentration.set_index("top_k")["share_of_total_absolute_error"].to_dict()

    summary = {
        "scope": "diagnostic-only; no training, tuning or model selection",
        "reference_rows": int(len(f)),
        "past_reference_rows": int(f["reference_eta_status"].str.startswith("PAST_").sum()),
        "past_reference_fraction": float(f["reference_eta_status"].str.startswith("PAST_").mean()),
        "future_0_7d_rows": int(f["reference_eta_status"].eq("FUTURE_0_7D").sum()),
        "future_gt7d_rows": int(f["reference_eta_status"].isin(["FUTURE_7_14D", "FUTURE_14_30D", "FUTURE_GT30D"]).sum()),
        "future_gt30d_rows": int(f["reference_eta_status"].eq("FUTURE_GT30D").sum()),
        "abs_reference_gt90d_rows": int(f["reference_abs_gt90d"].sum()),
        "year_assignment_margin_lt90d_rows": int(f["year_assignment_margin_lt90d"].sum()),
        "year_assignment_margin_lt30d_rows": int(f["year_assignment_margin_lt30d"].sum()),
        "placeholder_like_rows": int(f["eta_placeholder_like_pattern"].sum()),
        "eta_exact_hour_fraction": minute00,
        "eta_00_or_30_fraction": minute00or30,
        "eta_5min_grid_fraction": minute5,
        "unique_reference_timestamps": int(f["eta_reference_dt"].nunique()),
        "reference_timestamp_values_repeated_gt1": int((dup_counts > 1).sum()),
        "max_reference_timestamp_multiplicity": int(dup_counts.max()),
        "strict_final_n": int(len(strict)),
        "strict_final_mae_h": float(strict["abs_error_h"].mean()),
        "strict_final_medae_h": float(strict["abs_error_h"].median()),
        "strict_final_top1_error_share": float(conc[1]),
        "strict_final_top3_error_share": float(conc[3]),
        "strict_final_top5_error_share": float(conc[5]),
        "strict_final_top10_error_share": float(conc[10]),
        "strict_final_future_0_7d_mae_h": float(sensitivity.loc[sensitivity["diagnostic_filter"].eq("FUTURE_0_7D"), "mae_h"].iloc[0]),
        "strict_final_future_only_mae_h": float(sensitivity.loc[sensitivity["diagnostic_filter"].eq("FUTURE_ONLY"), "mae_h"].iloc[0]),
        "interpretation": {
            "company_reference_is_preserved": True,
            "rows_deleted_from_strict_benchmark": 0,
            "m10_model_changed": False,
            "m10_split_changed": False,
            "final_test_reselected": False,
        },
    }
    (R / "m11_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"PASS M11 forensic table written: {len(f)} reference rows")
    print(f"PASS strict final diagnostics written: {len(strict)} frozen predictions")
    print(f"PASS past references retained and flagged: {summary['past_reference_rows']}")
    print(f"PASS year-assignment sensitivity quantified: {summary['year_assignment_margin_lt90d_rows']} rows <90d margin")
    print(f"PASS placeholder-like ETA patterns flagged, not deleted: {summary['placeholder_like_rows']}")
    print(f"PASS top-5 strict final errors explain {summary['strict_final_top5_error_share']*100:.1f}% of total absolute error")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
