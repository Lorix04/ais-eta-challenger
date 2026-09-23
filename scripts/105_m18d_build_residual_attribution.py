#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
DOCS = ROOT / "docs"
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m18a import sha256_file  # noqa: E402
from ais_eta.m18d import (  # noqa: E402
    M18D_VERSION,
    attach_residual_context,
    categorical_slice_metrics,
    error_metrics,
    expert_diagnostics,
    numeric_attribution,
    quantile_slice_metrics,
    validate_attribution_dict,
)

CAT_SLICES = [
    "reference_eta_status",
    "m18c_reliability_tier",
    "destination_specificity",
    "resolution_confidence",
    "is_non_specific",
    "is_route_expression",
    "motion_state_cat",
    "stale_risk_cat",
    "feature_freshness_bucket",
    "ship_type_cat",
    "nav_status_cat",
    "m16d_gate_used",
    "physics_eligible",
    "confidence_tier",
    "m16g_selected_expert",
    "m16g_selected_meta_id",
]

NUMERIC_FEATURES = [
    "target_tte_h",  # target-derived diagnostic only
    "track_sog",
    "position_age_min",
    "history_n",
    "history_span_h",
    "dynamic_state_age_min",
    "hist_destination_unique",
    "track_vs_hist_position_gap_km",
    "sog_std_360m",
    "moving_fraction_360m",
    "stopped_fraction_360m",
    "distance_travelled_km_360m",
    "net_displacement_km_360m",
    "m16d_neighbour_count",
    "m16d_nearest_distance",
    "m16d_median_neighbour_distance",
    "m16d_similarity_gap",
    "physics_distance_gc_nm",
    "physics_course_alignment_deg",
    "physics_recent_speed_max_kn",
    "physics_local_sinuosity_360m",
    "expert_spread_h",
    "expert_std_h",
    "predicted_abs_error_h",  # frozen M16H diagnostic, not M16G feature
    "interval_half_width_h",
]

QUANTILE_FEATURES = [
    "track_sog",
    "position_age_min",
    "history_span_h",
    "track_vs_hist_position_gap_km",
    "m16d_neighbour_count",
    "m16d_nearest_distance",
    "physics_distance_gc_nm",
    "physics_course_alignment_deg",
    "expert_spread_h",
    "predicted_abs_error_h",
    "interval_half_width_h",
]


def _sha_map(paths: list[Path]) -> dict[str, str]:
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha256_file(p) for p in paths}


def _slice_value(table: pd.DataFrame, scope: str, family: str, value: str, field: str) -> float:
    z = table[(table.scope.eq(scope)) & (table.slice_family.eq(family)) & (table.slice_value.eq(value))]
    if len(z) != 1:
        return float("nan")
    return float(z.iloc[0][field])


def _corr_value(table: pd.DataFrame, scope: str, feature: str) -> float:
    z = table[(table.scope.eq(scope)) & (table.feature.eq(feature))]
    if len(z) != 1:
        return float("nan")
    return float(z.iloc[0].rank_corr_abs_error)


def main() -> int:
    f = pd.read_csv(R / "m18c_reference_forensics.csv")
    g = pd.read_csv(R / "m16g_oof_predictions.csv")
    b = pd.read_csv(R / "m16b_destination_resolutions.csv")
    d = pd.read_csv(R / "m16d_oof_predictions.csv")
    e = pd.read_csv(R / "m16e_oof_predictions.csv")
    h = pd.read_csv(R / "m16h_oof_intervals.csv")

    x = attach_residual_context(f, g, b, d, e, h)
    x.to_csv(R / "m18d_residual_ledger.csv", index=False)

    near = x[x["m18d_scope_near_term"]].copy()
    strict = x.copy()
    if len(near) != 279:
        raise AssertionError(f"M18D expected frozen M18C near-term n=279, got {len(near)}")

    slices = pd.concat([
        categorical_slice_metrics(strict, CAT_SLICES, "STRICT_386"),
        categorical_slice_metrics(near, CAT_SLICES, "NEAR_TERM_FUTURE_0_7D"),
    ], ignore_index=True)
    slices.to_csv(R / "m18d_categorical_slice_metrics.csv", index=False)

    numeric = pd.concat([
        numeric_attribution(strict, NUMERIC_FEATURES, "STRICT_386"),
        numeric_attribution(near, NUMERIC_FEATURES, "NEAR_TERM_FUTURE_0_7D"),
    ], ignore_index=True)
    numeric.to_csv(R / "m18d_numeric_attribution.csv", index=False)

    qmetrics = pd.concat([
        quantile_slice_metrics(strict, QUANTILE_FEATURES, "STRICT_386"),
        quantile_slice_metrics(near, QUANTILE_FEATURES, "NEAR_TERM_FUTURE_0_7D"),
    ], ignore_index=True)
    qmetrics.to_csv(R / "m18d_quantile_slice_metrics.csv", index=False)

    expert_summary = {
        "strict": expert_diagnostics(strict, "STRICT_386"),
        "near_term": expert_diagnostics(near, "NEAR_TERM_FUTURE_0_7D"),
    }
    (R / "m18d_expert_diagnostics.json").write_text(json.dumps(expert_summary, indent=2) + "\n")

    strict_metrics = error_metrics(strict)
    near_metrics = error_metrics(near)

    # Evidence signals used to build a deliberately conservative missing-info matrix.
    near_phys_true = _slice_value(slices, "NEAR_TERM_FUTURE_0_7D", "physics_eligible", "True", "mae_h")
    near_phys_false = _slice_value(slices, "NEAR_TERM_FUTURE_0_7D", "physics_eligible", "False", "mae_h")
    near_non_specific_n = _slice_value(slices, "NEAR_TERM_FUTURE_0_7D", "is_non_specific", "True", "rows")
    near_non_specific_mae = _slice_value(slices, "NEAR_TERM_FUTURE_0_7D", "is_non_specific", "True", "mae_h")
    near_conf_low = _slice_value(slices, "NEAR_TERM_FUTURE_0_7D", "confidence_tier", "LOW", "mae_h")
    near_conf_high = _slice_value(slices, "NEAR_TERM_FUTURE_0_7D", "confidence_tier", "HIGH", "mae_h")
    near_moving = _slice_value(slices, "NEAR_TERM_FUTURE_0_7D", "motion_state_cat", "MOVING", "mae_h")
    near_conflict = _slice_value(slices, "NEAR_TERM_FUTURE_0_7D", "motion_state_cat", "KINEMATIC_CONFLICT", "mae_h")
    rho_uncert = _corr_value(numeric, "NEAR_TERM_FUTURE_0_7D", "predicted_abs_error_h")
    rho_age = _corr_value(numeric, "NEAR_TERM_FUTURE_0_7D", "position_age_min")
    rho_spread = _corr_value(numeric, "NEAR_TERM_FUTURE_0_7D", "expert_spread_h")

    m18c = json.loads((R / "M18C_TARGET_LABEL_AUDIT.json").read_text())
    non_near_share = float(m18c["findings"]["non_near_term_absolute_error_share"])

    matrix = pd.DataFrame([
        {
            "priority": "P0_BLOCKER",
            "hypothesis": "authoritative_target_ground_truth",
            "observed_evidence": f"107/386 non-near-term label-regime rows contribute {100*non_near_share:.1f}% of strict absolute error; official future ATA/PBP/berth truth unavailable.",
            "evidence_strength": "VERY_HIGH_FOR_LABEL_CONFOUNDING",
            "what_m18d_can_conclude": "Target semantics/quality is the dominant confounder in the strict aggregate.",
            "what_m18d_cannot_conclude": "It cannot determine the intended operational arrival event without authoritative port-call records.",
            "recommended_information": "Port-call/PCS/SafeSeaNet/pilotage/berth logs with event location and timestamp.",
            "m18e_action": "Do not treat strict 185.88 h as pure model error; preserve target contract and seek authoritative parallel labels before relabeling.",
        },
        {
            "priority": "P1_TEST_FIRST",
            "hypothesis": "destination_and_route_intent",
            "observed_evidence": f"Within FUTURE_0_7D, physics-eligible rows MAE={near_phys_true:.2f} h vs {near_phys_false:.2f} h when ineligible; non-specific destinations n={int(near_non_specific_n) if np.isfinite(near_non_specific_n) else 0} have MAE={near_non_specific_mae:.2f} h (small-n).",
            "evidence_strength": "MODERATE_ASSOCIATION_CONFOUNDED",
            "what_m18d_can_conclude": "Rows with resolvable destination/geometry are substantially easier in the cleanest label regime.",
            "what_m18d_cannot_conclude": "Great-circle geometry does not prove navigable-route causality and eligibility is confounded with destination quality.",
            "recommended_information": "Authoritative destination/port-call intent plus navigable TSS/fairway/channel/water-only route distance.",
            "m18e_action": "Run separate destination-resolution and navigable-routing ablations before weather or broad architecture changes.",
        },
        {
            "priority": "P1_DIAGNOSTIC_CONTROL",
            "hypothesis": "uncertainty_and_expert_selection",
            "observed_evidence": f"FUTURE_0_7D M16H predicted-error rank correlation with actual AE={rho_uncert:.3f}; LOW confidence MAE={near_conf_low:.2f} h vs HIGH={near_conf_high:.2f} h; existing-expert oracle gap={expert_summary['near_term']['oracle_gap_h']:.2f} h.",
            "evidence_strength": "HIGH_DIAGNOSTIC_NOT_ACTIONABLE_ORACLE",
            "what_m18d_can_conclude": "The frozen uncertainty sidecar identifies hard cases and the current expert set contains post-hoc headroom.",
            "what_m18d_cannot_conclude": "Oracle selection is target-dependent and cannot be deployed or used as a promotion result.",
            "recommended_information": "Target-free signals that discriminate expert applicability; use confidence tier to stratify M18E evaluation.",
            "m18e_action": "Require every external-data ablation to report gains by frozen M16H confidence tier; no oracle-trained routing on these labels.",
        },
        {
            "priority": "P2_CONDITIONAL",
            "hypothesis": "port_operations",
            "observed_evidence": "Near-term data contain almost no STOPPED/anchored/berthed supervised rows under the company-reference contract; operational delay state is not observed directly.",
            "evidence_strength": "INSUFFICIENT_CURRENT_OBSERVABILITY",
            "what_m18d_can_conclude": "Current AIS-only residual slices cannot isolate berth/pilot/tug/congestion effects.",
            "what_m18d_cannot_conclude": "Absence of evidence is not evidence that port operations are unimportant.",
            "recommended_information": "Anchorage queue, berth availability, pilot/tug/mooring readiness, terminal/PCS event timestamps.",
            "m18e_action": "Test only after target event semantics are resolved; prioritize if predicting berth/all-fast rather than company ETA reference.",
        },
        {
            "priority": "P3_LOW_CURRENT_SIGNAL",
            "hypothesis": "raw_ais_freshness_and_state",
            "observed_evidence": f"Within FUTURE_0_7D, position-age rank correlation with AE={rho_age:.3f}; MOVING MAE={near_moving:.2f} h vs KINEMATIC_CONFLICT={near_conflict:.2f} h; 275/279 rows have <=5 min feature freshness.",
            "evidence_strength": "LOW_TO_MODERATE",
            "what_m18d_can_conclude": "Snapshot freshness alone is not a dominant near-term residual separator in this sample.",
            "what_m18d_cannot_conclude": "The polling/latest-state representation still cannot recover missing message-level intent or ETA revision history.",
            "recommended_information": "Raw AIS message stream with source timestamps/message types and historical declared-ETA updates.",
            "m18e_action": "Lower priority than target/destination/routing, unless new raw AIS is readily available.",
        },
        {
            "priority": "P2_UNTESTABLE_WITH_CURRENT_DATA",
            "hypothesis": "weather_and_ocean",
            "observed_evidence": "No decision-time weather/wave/current features exist in the frozen residual ledger.",
            "evidence_strength": "UNTESTABLE",
            "what_m18d_can_conclude": "Nothing causal about weather/ocean contribution from current data.",
            "what_m18d_cannot_conclude": "It cannot rank weather importance from AIS proxies alone.",
            "recommended_information": "Decision-time-available archived forecast wind/wave/current fields; reanalysis only for exploratory attribution with leakage warning.",
            "m18e_action": "Run after destination/routing ablation, or in parallel only if archived forecast data are easy to obtain.",
        },
        {
            "priority": "P2_SUPPORTING_SIGNAL",
            "hypothesis": "navigable_route_complexity",
            "observed_evidence": f"Near-term expert-spread rank correlation with AE={rho_spread:.3f}; highest-disagreement quartile is materially harder, but route similarity/distance metrics have mostly weak univariate correlations.",
            "evidence_strength": "MIXED",
            "what_m18d_can_conclude": "Some route/physics disagreement marks difficult cases, but simple route-distance diagnostics do not explain most near-term residual variance.",
            "what_m18d_cannot_conclude": "It cannot infer TSS/fairway/channel effects without chart-constrained routing data.",
            "recommended_information": "S-57/S-101/S-102 where legally/technically available, historical AIS corridors and water-only route graph.",
            "m18e_action": "Evaluate as a controlled ablation targeted to physics-ineligible/low-confidence rows rather than globally assuming benefit.",
        },
    ])
    matrix.to_csv(R / "m18d_missing_information_matrix.csv", index=False)

    # Compact top slice and correlation tables for report/audit consumption.
    robust_near = slices[(slices.scope.eq("NEAR_TERM_FUTURE_0_7D")) & (slices.support_grade.eq("ROBUST_N_GE20"))].copy()
    robust_near["mae_delta_vs_scope_h"] = robust_near["mae_h"] - near_metrics["mae_h"]
    robust_near = robust_near.sort_values("mae_delta_vs_scope_h", ascending=False)
    robust_near.head(30).to_csv(R / "m18d_top_robust_near_term_slices.csv", index=False)

    near_numeric = numeric[numeric.scope.eq("NEAR_TERM_FUTURE_0_7D")].sort_values("abs_rank_corr_abs_error", ascending=False)
    near_numeric.head(30).to_csv(R / "m18d_top_near_term_numeric_signals.csv", index=False)

    findings = {
        "strict_rows": 386,
        "near_term_rows": int(len(near)),
        "strict_m16g_mae_h": float(strict_metrics["mae_h"]),
        "near_term_m16g_mae_h": float(near_metrics["mae_h"]),
        "strict_m16g_p90_ae_h": float(strict_metrics["p90_ae_h"]),
        "near_term_m16g_p90_ae_h": float(near_metrics["p90_ae_h"]),
        "non_near_term_absolute_error_share": non_near_share,
        "near_term_m16h_predicted_error_rank_corr": rho_uncert,
        "near_term_expert_spread_rank_corr": rho_spread,
        "near_term_position_age_rank_corr": rho_age,
        "near_term_physics_eligible_mae_h": near_phys_true,
        "near_term_physics_ineligible_mae_h": near_phys_false,
        "near_term_low_confidence_mae_h": near_conf_low,
        "near_term_high_confidence_mae_h": near_conf_high,
        "near_term_existing_expert_oracle_mae_h": float(expert_summary["near_term"]["oracle_existing_expert_mae_h"]),
        "near_term_existing_expert_oracle_gap_h": float(expert_summary["near_term"]["oracle_gap_h"]),
        "near_term_selected_expert_is_oracle_fraction": float(expert_summary["near_term"]["selected_expert_is_oracle_fraction"]),
    }

    audit = {
        "milestone": "M18D",
        "version": M18D_VERSION,
        "decision": "RESIDUAL_ERROR_ATTRIBUTION_FROZEN_NO_MODEL_CHANGE",
        "scope": {
            "model_change": False,
            "training": False,
            "tuning": False,
            "relabeling": False,
            "row_filtering": False,
            "fresh_holdout_opened": False,
            "old_final_used_for_selection": False,
        },
        "population": {
            "development_rows": 386,
            "near_term_diagnostic_rows": int(len(near)),
            "blocked_old_final_rows": 53,
            "primary_residual": "frozen M16G OOF signed/absolute error",
        },
        "findings": findings,
        "expert_diagnostics": expert_summary,
        "priority_order": matrix[["priority", "hypothesis"]].to_dict(orient="records"),
        "policy": {
            "strict_primary_score_changed": False,
            "near_term_slice_replaces_primary_benchmark": False,
            "attribution_is_causal_proof": False,
            "oracle_expert_is_deployable_result": False,
            "target_derived_diagnostics_are_runtime_features": False,
            "m18e_must_use_controlled_ablations": True,
        },
        "next_step": "M18E_EXTERNAL_INFORMATION_ABLATION_LAB",
    }
    validate_attribution_dict(audit)
    (R / "M18D_RESIDUAL_ATTRIBUTION.json").write_text(json.dumps(audit, indent=2) + "\n")

    report = f"""# M18D — Residual Error Attribution

## Decision

**RESIDUAL ATTRIBUTION FROZEN — NO MODEL CHANGE, NO RELABELING, NO ROW DELETION.**

M18D decomposes the frozen M16G OOF residuals. It does not create a new score, train a router, delete the 107 difficult label-regime rows, or open the fresh holdout. Associations are diagnostics, not causal effects.

## Two scopes are kept separate

- **Strict company-reference benchmark:** 386 rows, M16G MAE **{strict_metrics['mae_h']:.2f} h**, P90 **{strict_metrics['p90_ae_h']:.2f} h**.
- **M18C near-term diagnostic (`FUTURE_0_7D`):** 279 rows, M16G MAE **{near_metrics['mae_h']:.2f} h**, P90 **{near_metrics['p90_ae_h']:.2f} h**.

The near-term slice is never promoted to the primary benchmark. It exists to study model residual structure with less label-regime confounding.

## Main residual findings

1. **Target semantics remains the first blocker.** The 107 non-near-term reference rows still contribute **{100*non_near_share:.1f}%** of strict absolute error. M18D therefore refuses to interpret the 185.88 h aggregate as pure model failure.
2. **Destination/geometry applicability is the strongest external-information clue inside the cleaner slice.** Physics-eligible rows have **{near_phys_true:.2f} h MAE** versus **{near_phys_false:.2f} h** when physics is ineligible. This is association, not proof: eligibility is also linked to destination resolution.
3. **Frozen uncertainty works as a difficulty detector.** M16H predicted absolute error has near-term rank correlation **{rho_uncert:.3f}** with actual absolute error. LOW-confidence near-term rows have **{near_conf_low:.2f} h MAE** versus **{near_conf_high:.2f} h** for HIGH confidence.
4. **There is diagnostic expert-selection headroom, but it is not deployable evidence.** A hindsight oracle over the four already-frozen M16G experts would have **{expert_summary['near_term']['oracle_existing_expert_mae_h']:.2f} h MAE** versus M16G **{near_metrics['mae_h']:.2f} h**, an oracle gap of **{expert_summary['near_term']['oracle_gap_h']:.2f} h**. The selected expert equals the hindsight oracle on only **{100*expert_summary['near_term']['selected_expert_is_oracle_fraction']:.1f}%** of near-term rows. Because the oracle uses the target, it cannot be trained/promoted from this result.
5. **Snapshot freshness is not the dominant near-term separator in this dataset.** 275/279 near-term rows have <=5 min feature freshness and the position-age/error rank correlation is only **{rho_age:.3f}**. This does not prove raw AIS is unimportant; it only lowers priority relative to target/destination/routing evidence.
6. **Weather and port operations remain unidentifiable from the current feature set.** M18D does not manufacture importance for variables that do not exist in the ledger.

## Missing Information Matrix

{matrix.to_markdown(index=False)}

## M18E implication

The evidence does **not** support a broad "add everything" experiment. M18E should begin with controlled, one-source-at-a-time ablations, reporting strict and near-term diagnostics plus frozen M16H confidence strata. The first data hypothesis to test is destination/route intent and navigable routing, while authoritative operational ground truth remains the prerequisite for changing the target itself. Weather and port-operation integrations remain conditional because the present dataset cannot attribute their contribution.

## Research basis

Recent AIS ETA work emphasizes that preprocessing, voyage-safe validation and anomalous vessel behaviour can dominate apparent model quality. Recent port-arrival work also combines AIS with port-call records and analyzes vessel-reported ETA accuracy rather than assuming a reported ETA is authoritative ATA. Feature-importance results in recent maritime ETA studies repeatedly identify speed, distance/course and vessel characteristics as relevant, but those population-level findings are not substituted for project-specific evidence here.
"""
    (R / "M18D_REPORT.md").write_text(report)

    policy = f"""# M18D Residual Attribution Policy

M18D is diagnostic-only. It operates on frozen M16G OOF predictions and does not change M18A target semantics, M18B validation, M18C labels, M16G/M16H or the 53-row old-final block.

## Interpretation rules

- Always report the strict 386-row benchmark separately from the 279-row `FUTURE_0_7D` diagnostic slice.
- Do not call subgroup associations causal effects.
- Do not expose `target_tte_h`, `reference_eta_status`, M18C reliability tiers, residuals, hindsight oracle identity or coverage outcomes as runtime predictors.
- The existing-expert oracle is only a lower-bound/headroom diagnostic; it is target-dependent and non-deployable.
- Missing variables such as weather and port operations must be marked untestable rather than assigned invented importance.
- M18E experiments must add one information family at a time and preserve the M18A/M18B lockbox rules.

## Frozen headline diagnostics

- strict M16G MAE: {strict_metrics['mae_h']:.3f} h
- near-term M16G MAE: {near_metrics['mae_h']:.3f} h
- near-term M16H predicted-error rank correlation: {rho_uncert:.3f}
- near-term existing-expert oracle gap: {expert_summary['near_term']['oracle_gap_h']:.3f} h (diagnostic only)

## External research context

- Marreiros et al. (2026), *Forecasting*: reproducible AIS ETA workflow; leakage-free voyage grouping and anomalous/loitering behaviour as deployment constraints: https://www.mdpi.com/2571-9394/8/4/70
- Chu, Yan & Wang (2025), *Transportation Research Part C*: arrival-time prediction fusing AIS and port-call records and analyzing vessel-reported ETA accuracy: https://www.sciencedirect.com/science/article/pii/S0968090X25001329
- Maritime Transport Research (2025): ETA feature-importance analysis highlights speed, distance, course and vessel type: https://www.sciencedirect.com/science/article/pii/S2666822X2500005X
"""
    DOCS.mkdir(exist_ok=True)
    (DOCS / "M18D_RESIDUAL_ATTRIBUTION_POLICY.md").write_text(policy)

    artifacts = [
        R / "M18D_RESIDUAL_ATTRIBUTION.json",
        R / "M18D_REPORT.md",
        R / "m18d_residual_ledger.csv",
        R / "m18d_categorical_slice_metrics.csv",
        R / "m18d_numeric_attribution.csv",
        R / "m18d_quantile_slice_metrics.csv",
        R / "m18d_expert_diagnostics.json",
        R / "m18d_missing_information_matrix.csv",
        R / "m18d_top_robust_near_term_slices.csv",
        R / "m18d_top_near_term_numeric_signals.csv",
        DOCS / "M18D_RESIDUAL_ATTRIBUTION_POLICY.md",
    ]
    freeze = {
        "milestone": "M18D",
        "version": M18D_VERSION,
        "status": "FROZEN_RESIDUAL_ERROR_ATTRIBUTION",
        "model_change": False,
        "relabeling": False,
        "strict_rows_deleted": 0,
        "fresh_holdout_opened": False,
        "development_population": 386,
        "near_term_diagnostic_population": int(len(near)),
        "blocked_old_final_population": 53,
        "m18a_baseline_freeze_sha256": sha256_file(R / "M18A_BASELINE_FREEZE.json"),
        "m18b_validation_freeze_sha256": sha256_file(R / "M18B_VALIDATION_FREEZE.json"),
        "m18c_audit_freeze_sha256": sha256_file(R / "M18C_AUDIT_FREEZE.json"),
        "m16g_oof_sha256": sha256_file(R / "m16g_oof_predictions.csv"),
        "attribution_artifact_sha256": _sha_map(artifacts),
    }
    (R / "M18D_ATTRIBUTION_FREEZE.json").write_text(json.dumps(freeze, indent=2) + "\n")

    print("PASS M18D residual error attribution frozen; no model change")
    print(f"PASS strict=386 MAE={strict_metrics['mae_h']:.2f}h; near-term={len(near)} MAE={near_metrics['mae_h']:.2f}h")
    print(f"PASS M16H near-term error rank-corr={rho_uncert:.3f}; confidence LOW/HIGH MAE={near_conf_low:.2f}/{near_conf_high:.2f}h")
    print(f"PASS physics eligible/ineligible near-term MAE={near_phys_true:.2f}/{near_phys_false:.2f}h")
    print(f"PASS diagnostic existing-expert oracle gap={expert_summary['near_term']['oracle_gap_h']:.2f}h; not deployable")
    print("PASS Missing Information Matrix frozen; weather/port-ops marked untestable/conditional rather than assigned invented importance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
