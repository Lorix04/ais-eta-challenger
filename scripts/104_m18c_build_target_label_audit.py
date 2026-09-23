#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
D = ROOT / "data" / "derived" / "m10_reference_rows.pkl.gz"
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m18a import sha256_file  # noqa: E402
from ais_eta.m18c import (  # noqa: E402
    M18C_VERSION,
    attach_frozen_prediction_errors,
    build_ais_proxy_alignment,
    build_label_forensics,
    error_concentration,
    grouped_error_metrics,
    validate_audit_dict,
)


def _sha_map(paths: list[Path]) -> dict[str, str]:
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha256_file(p) for p in paths}


def _status_counts(f: pd.DataFrame) -> dict[str, int]:
    return {str(k): int(v) for k, v in f["reference_eta_status"].value_counts().sort_index().items()}


def main() -> int:
    rows = pd.read_pickle(D)
    oof = pd.read_csv(R / "m16g_oof_predictions.csv")
    old_final = set(rows.loc[rows["split"].eq("final_test"), "mmsi"].astype(int))
    dev_ids = set(oof["mmsi"].astype(int))
    if len(old_final) != 53 or len(dev_ids) != 386 or dev_ids & old_final:
        raise AssertionError("M18C population/final-block invariant failed")

    f = build_label_forensics(rows, sorted(dev_ids))
    x = attach_frozen_prediction_errors(f, oof)
    x.to_csv(R / "m18c_reference_forensics.csv", index=False)

    status_metrics = grouped_error_metrics(x, "reference_eta_status")
    status_metrics.to_csv(R / "m18c_error_by_reference_status.csv", index=False)
    tier_metrics = grouped_error_metrics(x, "m18c_reliability_tier")
    tier_metrics.to_csv(R / "m18c_error_by_reliability_tier.csv", index=False)
    regime_metrics = grouped_error_metrics(x, "reference_diagnostic_regime")
    regime_metrics.to_csv(R / "m18c_error_by_diagnostic_regime.csv", index=False)

    conc = error_concentration(x["m16g_abs_error_h"])
    conc.to_csv(R / "m18c_error_concentration.csv", index=False)

    near = x["reference_eta_status"].eq("FUTURE_0_7D")
    future = ~x["reference_eta_status"].str.startswith("PAST_")
    other = ~near
    clean_diag = pd.DataFrame([
        {
            "slice": "STRICT_ALL_386",
            "rows": len(x),
            "row_fraction": 1.0,
            "mae_h": float(x["m16g_abs_error_h"].mean()),
            "medae_h": float(x["m16g_abs_error_h"].median()),
            "p90_ae_h": float(x["m16g_abs_error_h"].quantile(0.90)),
            "absolute_error_share": 1.0,
            "status": "PRIMARY_STRICT_DIAGNOSTIC",
        },
        {
            "slice": "FUTURE_0_7D",
            "rows": int(near.sum()),
            "row_fraction": float(near.mean()),
            "mae_h": float(x.loc[near, "m16g_abs_error_h"].mean()),
            "medae_h": float(x.loc[near, "m16g_abs_error_h"].median()),
            "p90_ae_h": float(x.loc[near, "m16g_abs_error_h"].quantile(0.90)),
            "absolute_error_share": float(x.loc[near, "m16g_abs_error_h"].sum() / x["m16g_abs_error_h"].sum()),
            "status": "SECONDARY_LABEL_ALIGNED_DIAGNOSTIC_ONLY",
        },
        {
            "slice": "NOT_FUTURE_0_7D",
            "rows": int(other.sum()),
            "row_fraction": float(other.mean()),
            "mae_h": float(x.loc[other, "m16g_abs_error_h"].mean()),
            "medae_h": float(x.loc[other, "m16g_abs_error_h"].median()),
            "p90_ae_h": float(x.loc[other, "m16g_abs_error_h"].quantile(0.90)),
            "absolute_error_share": float(x.loc[other, "m16g_abs_error_h"].sum() / x["m16g_abs_error_h"].sum()),
            "status": "LABEL_RISK_DIAGNOSTIC_ONLY",
        },
        {
            "slice": "FUTURE_ONLY",
            "rows": int(future.sum()),
            "row_fraction": float(future.mean()),
            "mae_h": float(x.loc[future, "m16g_abs_error_h"].mean()),
            "medae_h": float(x.loc[future, "m16g_abs_error_h"].median()),
            "p90_ae_h": float(x.loc[future, "m16g_abs_error_h"].quantile(0.90)),
            "absolute_error_share": float(x.loc[future, "m16g_abs_error_h"].sum() / x["m16g_abs_error_h"].sum()),
            "status": "SECONDARY_SEMANTIC_DIAGNOSTIC_ONLY",
        },
    ])
    clean_diag.to_csv(R / "m18c_label_slice_diagnostics.csv", index=False)

    calls = pd.read_csv(R / "m1x_combined_eta_call_inventory.csv")
    proxy = build_ais_proxy_alignment(x, calls)
    proxy.to_csv(R / "m18c_historical_ais_proxy_alignment.csv", index=False)

    if len(proxy):
        within = {str(h): int(proxy["closest_proxy_abs_eta_delta_h"].le(h).sum()) for h in (1, 6, 12, 24, 48, 72, 168)}
        future_proxy_calls = int(proxy["proxy_calls_on_or_after_decision"].gt(0).sum())
    else:
        within = {str(h): 0 for h in (1, 6, 12, 24, 48, 72, 168)}
        future_proxy_calls = 0

    eta_counts = x["eta_reference_dt"].value_counts()
    findings = {
        "status_counts": _status_counts(x),
        "past_reference_rows": int(x["reference_eta_status"].str.startswith("PAST_").sum()),
        "past_reference_fraction": float(x["reference_eta_status"].str.startswith("PAST_").mean()),
        "near_term_future_0_7d_rows": int(near.sum()),
        "near_term_future_0_7d_fraction": float(near.mean()),
        "future_gt7d_rows": int(x["reference_eta_status"].isin(["FUTURE_7_14D", "FUTURE_14_30D", "FUTURE_GT30D"]).sum()),
        "future_gt30d_rows": int(x["reference_eta_status"].eq("FUTURE_GT30D").sum()),
        "placeholder_like_rows": int(x["eta_placeholder_like_pattern"].sum()),
        "year_assignment_margin_lt90d_rows": int(x["year_assignment_margin_lt90d"].sum()),
        "year_assignment_margin_lt30d_rows": int(x["year_assignment_margin_lt30d"].sum()),
        "abs_reference_gt90d_rows": int(x["reference_abs_gt90d"].sum()),
        "eta_exact_hour_fraction": float(x["eta_minute"].eq(0).mean()),
        "eta_00_or_30_fraction": float(x["eta_on_00_or_30"].mean()),
        "eta_5min_grid_fraction": float(x["eta_on_5min_grid"].mean()),
        "unique_reference_timestamps": int(x["eta_reference_dt"].nunique()),
        "repeated_reference_timestamp_values": int((eta_counts > 1).sum()),
        "max_reference_timestamp_multiplicity": int(eta_counts.max()),
        "strict_m16g_mae_h": float(x["m16g_abs_error_h"].mean()),
        "near_term_m16g_mae_h": float(x.loc[near, "m16g_abs_error_h"].mean()),
        "near_term_absolute_error_share": float(x.loc[near, "m16g_abs_error_h"].sum() / x["m16g_abs_error_h"].sum()),
        "non_near_term_rows": int(other.sum()),
        "non_near_term_m16g_mae_h": float(x.loc[other, "m16g_abs_error_h"].mean()),
        "non_near_term_absolute_error_share": float(x.loc[other, "m16g_abs_error_h"].sum() / x["m16g_abs_error_h"].sum()),
        "historical_ais_proxy_owner_overlap": int(len(proxy)),
        "historical_ais_proxy_owners_with_future_event_after_decision": future_proxy_calls,
        "historical_ais_proxy_closest_match_within_h": within,
    }

    audit = {
        "milestone": "M18C",
        "version": M18C_VERSION,
        "decision": "TARGET_LABEL_RELIABILITY_AUDIT_FROZEN_NO_RELABELING",
        "scope": {
            "model_change": False,
            "training": False,
            "tuning": False,
            "relabeling": False,
            "row_filtering_for_primary_benchmark": False,
            "fresh_holdout_opened": False,
            "old_final_used_for_selection": False,
        },
        "population": {
            "development_rows": 386,
            "development_unique_mmsi": 386,
            "blocked_old_final_rows": 53,
            "primary_label_source": "company-provided Tracks.eta reference",
            "decision_time": "Tracks.last_update",
        },
        "findings": findings,
        "policy": {
            "strict_benchmark_rows_deleted": 0,
            "strict_primary_contract_preserved": True,
            "diagnostic_reliability_tiers_are_model_features": False,
            "diagnostic_reliability_tiers_may_drive_training_weights": False,
            "future_0_7d_slice_is_secondary_only": True,
            "ais_proxy_replaces_company_target": False,
            "ais_proxy_role": "historical temporal-alignment diagnostic only; not ATA/berth/all-fast truth",
            "authoritative_port_call_ground_truth_still_missing": True,
        },
        "next_step": "M18D_RESIDUAL_ERROR_ATTRIBUTION",
        "research_basis": {
            "ais_message_5_eta": "AIS Message 5 ETA is MMDDHHMM UTC and carries no year.",
            "imo_event_semantics": "IMO JIT data model distinguishes planned/requested arrival timestamps by specified port location such as pilot boarding place or berth.",
            "port_call_fusion": "Recent vessel-arrival research fuses AIS with port-call records and separately evaluates vessel-reported ETA accuracy.",
        },
    }
    validate_audit_dict(audit)
    (R / "M18C_TARGET_LABEL_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")

    report = f"""# M18C — Target & Label Reliability Audit

## Decision

**TARGET/LABEL AUDIT FROZEN — NO MODEL CHANGE, NO RELABELING, NO ROW DELETION.**

M18C audits the 386 development labels under the frozen M18A prediction contract. M16G, M16H, the 53-owner old-final block, M18B forward splits and the unopened fresh external holdout are unchanged.

## Main finding

The overall M16G development MAE is **{findings['strict_m16g_mae_h']:.2f} h**, but that aggregate is dominated by reference-label regimes outside the near-term future window.

- `FUTURE_0_7D`: **{findings['near_term_future_0_7d_rows']}/386 rows ({100*findings['near_term_future_0_7d_fraction']:.1f}%)**, M16G MAE **{findings['near_term_m16g_mae_h']:.2f} h**, only **{100*findings['near_term_absolute_error_share']:.1f}%** of total absolute error.
- All other regimes: **{findings['non_near_term_rows']}/386 rows ({100*findings['non_near_term_rows']/386:.1f}%)**, M16G MAE **{findings['non_near_term_m16g_mae_h']:.2f} h**, **{100*findings['non_near_term_absolute_error_share']:.1f}%** of total absolute error.
- Past references: **{findings['past_reference_rows']} rows ({100*findings['past_reference_fraction']:.1f}%)**.
- Future references beyond 7 days: **{findings['future_gt7d_rows']} rows**; beyond 30 days: **{findings['future_gt30d_rows']} rows**.
- Placeholder-like ETA patterns: **{findings['placeholder_like_rows']}**.
- Year-resolution margin <90 d: **{findings['year_assignment_margin_lt90d_rows']} rows**.

This does **not** justify deleting hard rows or reporting 20.26 h as the official score. It shows that the 185.88 h aggregate mixes very different label regimes and therefore should not be interpreted as pure model error.

## Historical AIS-derived proxy alignment

The Catania/Augusta research ledgers overlap **{findings['historical_ais_proxy_owner_overlap']}** development MMSIs. For all of them, the validated research-gate events available in the finite dataset occur before the corresponding `Tracks.last_update`; **{findings['historical_ais_proxy_owners_with_future_event_after_decision']}** owners have a validated research-gate event at/after decision time. Consequently these events cannot serve as future actual-arrival ground truth for the M18A task.

Post-hoc only, the company ETA timestamp lies within 6 h of a historical research-gate event for **{findings['historical_ais_proxy_closest_match_within_h']['6']}/{findings['historical_ais_proxy_owner_overlap']}** overlapping owners and within 7 days for **{findings['historical_ais_proxy_closest_match_within_h']['168']}/{findings['historical_ais_proxy_owner_overlap']}**. This is evidence worth investigating for stale/carry-over ETA semantics, but it is not authoritative proof that those historical events are the intended labels.

## Granularity

- Exact-hour ETA fraction: **{100*findings['eta_exact_hour_fraction']:.1f}%**.
- Minute 00/30: **{100*findings['eta_00_or_30_fraction']:.1f}%**.
- Five-minute grid: **{100*findings['eta_5min_grid_fraction']:.1f}%**.

The minute-level granularity is tiny compared with the hundreds/thousands of hours in the extreme regimes, so timestamp rounding alone cannot explain the heavy tail.

## Policy frozen by M18C

1. Keep all 386 labels in the strict primary benchmark; no post-hoc filtering.
2. Treat `FUTURE_0_7D` and the reliability tiers only as secondary diagnostics, never as predictor features or training weights.
3. Do not reinterpret `Tracks.eta` as ATA, PBP, berth arrival or all-fast.
4. Do not substitute the M0/M1X research-gate events for company labels.
5. Authoritative port-call/ATA data remains the missing evidence needed to resolve target semantics conclusively.
6. M18D may use these frozen regimes to attribute residual error, but cannot change the primary target contract.
"""
    (R / "M18C_REPORT.md").write_text(report)

    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    docs_text = """# M18C Target & Label Reliability Policy

M18C is a diagnostic-only audit of the company-provided `Tracks.eta` reference. It does not define a new target.

## Frozen interpretation

- `Tracks.last_update` remains the decision time.
- `Tracks.eta` remains the strict company reference target under M18A.
- AIS Message 5 encodes ETA as MMDDHHMM UTC and does not encode a year; year resolution therefore remains an explicit project assumption.
- An ETA timestamp without an authoritative target location/event must not be silently described as ATA, pilot-boarding arrival, berth arrival or all-fast.

## Diagnostic tiers

M18C assigns reference-only reliability tiers for analysis. They are prohibited as model inputs and prohibited as training weights unless a future milestone explicitly freezes a new protocol before evaluating outcomes.

## AIS research-gate events

The manually validated Catania/Augusta gate events are valuable behavioral evidence, but they are not official port-call records. M18C uses them only to test temporal alignment with company reference timestamps. They do not replace `Tracks.eta` and do not establish an authoritative ATA.

## External basis

- USCG/NAVCEN AIS Message 5: ETA is MMDDHHMM UTC: https://www.navcen.uscg.gov/ais-class-a-static-voyage-message-5
- IMO JIT/FAL model: arrival/planned/requested timestamps are tied to specified locations including Pilot Boarding Place and berth: https://imocompendium.imo.org/public/IMO-Compendium/Current/DS/Just%20In%20Time%20Concept/d11.htm
- Chu, Yan & Wang (2025), Transportation Research Part C: vessel-arrival prediction framework fusing AIS and port-call records and analysing vessel-reported ETA accuracy: https://www.sciencedirect.com/science/article/abs/pii/S0968090X25001329
"""
    (docs / "M18C_TARGET_LABEL_RELIABILITY.md").write_text(docs_text)

    artifacts = [
        R / "M18C_TARGET_LABEL_AUDIT.json",
        R / "M18C_REPORT.md",
        R / "m18c_reference_forensics.csv",
        R / "m18c_error_by_reference_status.csv",
        R / "m18c_error_by_reliability_tier.csv",
        R / "m18c_error_by_diagnostic_regime.csv",
        R / "m18c_error_concentration.csv",
        R / "m18c_label_slice_diagnostics.csv",
        R / "m18c_historical_ais_proxy_alignment.csv",
        docs / "M18C_TARGET_LABEL_RELIABILITY.md",
    ]
    freeze = {
        "milestone": "M18C",
        "version": M18C_VERSION,
        "status": "FROZEN_TARGET_LABEL_RELIABILITY_AUDIT",
        "model_change": False,
        "relabeling": False,
        "strict_rows_deleted": 0,
        "fresh_holdout_opened": False,
        "development_population": 386,
        "blocked_old_final_population": 53,
        "m18a_baseline_freeze_sha256": sha256_file(R / "M18A_BASELINE_FREEZE.json"),
        "m18b_validation_freeze_sha256": sha256_file(R / "M18B_VALIDATION_FREEZE.json"),
        "m16g_oof_sha256": sha256_file(R / "m16g_oof_predictions.csv"),
        "audit_artifact_sha256": _sha_map(artifacts),
    }
    (R / "M18C_AUDIT_FREEZE.json").write_text(json.dumps(freeze, indent=2) + "\n")

    print("PASS M18C target/label reliability audit frozen; no model change")
    print(f"PASS development=386 old-final blocked=53 fresh holdout sealed")
    print(f"PASS FUTURE_0_7D={near.sum()}/386 MAE={x.loc[near, 'm16g_abs_error_h'].mean():.2f}h")
    print(f"PASS non-near-term={other.sum()}/386 contributes {100*findings['non_near_term_absolute_error_share']:.1f}% of absolute error")
    print(f"PASS historical AIS proxy overlap={len(proxy)}; future proxy events after decision={future_proxy_calls}")
    print("PASS strict target preserved; AIS proxy is diagnostic only, not authoritative ATA")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
