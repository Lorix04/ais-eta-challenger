#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
DOCS = ROOT / "docs"
CONFIG = ROOT / "config"
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m18e import (  # noqa: E402
    M18E_VERSION,
    paired_gain_summary,
    sha256_file,
    simple_metrics,
    validate_ablation_audit,
    validate_source_registry,
)


def _source_registry() -> pd.DataFrame:
    rows = [
        {
            "source_id": "unlocode_frozen_m16b_catalog",
            "information_family": "destination_port_intent",
            "source_name": "UNECE UN/LOCODE-derived M16B frozen catalog",
            "source_url": "https://unlocode.unece.org/publications/",
            "source_authority": "UNECE / UN/CEFACT",
            "data_version": "M16B frozen catalog; aligned to UN/LOCODE provenance",
            "decision_time_safe": True,
            "artifact_pinned": True,
            "status": "RUN_FROZEN_RETROSPECTIVE",
            "m18e_action": "Re-express raw-vs-canonical destination OOF comparison as explicit external-information ablation; do not retrain M16G.",
            "notes": "UN/LOCODE is target-free reference data. Current UNECE publications page exposes official production 2025-1 and a continuously updated 2026 pre-release.",
        },
        {
            "source_id": "unlocode_2026_prerelease_refresh",
            "information_family": "destination_port_intent",
            "source_name": "UNECE UN/LOCODE pre-release",
            "source_url": "https://unlocode.unece.org/publications/",
            "source_authority": "UNECE / UN/CEFACT",
            "data_version": "pre-release last-updated 2026-08-31 per source page",
            "decision_time_safe": True,
            "artifact_pinned": False,
            "status": "BLOCKED_DATASET_NOT_PINNED",
            "m18e_action": "Do not partially scrape individual rows. Pin the complete official/pre-release artifact and checksum before any resolver refresh ablation.",
            "notes": "A moving pre-release cannot be a reproducible model input until a local snapshot is frozen.",
        },
        {
            "source_id": "iho_s100_navigable_route",
            "information_family": "navigable_routing",
            "source_name": "IHO S-100 Phase-1 products (S-101/S-102/S-111)",
            "source_url": "https://iho.int/en/the-s-100-framework-is-now-operational",
            "source_authority": "International Hydrographic Organization",
            "data_version": "operational editions entered into force 2026-01-01",
            "decision_time_safe": True,
            "artifact_pinned": False,
            "status": "BLOCKED_DATASET_NOT_PINNED",
            "m18e_action": "Require actual Sicily/Eastern-Mediterranean product cells, coverage/licence audit and frozen checksums before chart-constrained routing ablation.",
            "notes": "Operational product specifications do not imply that complete local production datasets are already available in this repository.",
        },
        {
            "source_id": "searoute_marnet_proxy",
            "information_family": "navigable_routing_proxy",
            "source_name": "searoute-py Marnet shortest-sea-route proxy",
            "source_url": "https://github.com/genthalili/searoute-py",
            "source_authority": "Open-source proxy network; not hydrographic authority",
            "data_version": "1.6.0 candidate",
            "decision_time_safe": True,
            "artifact_pinned": False,
            "status": "BLOCKED_DATASET_NOT_PINNED",
            "m18e_action": "May be evaluated only as a non-authoritative research proxy after pinning package/network artifacts and licence/source hashes.",
            "notes": "Upstream explicitly states the package is not for navigational routing; it can only be a research distance proxy.",
        },
        {
            "source_id": "archived_weather_ocean_forecast",
            "information_family": "weather_ocean",
            "source_name": "Archived decision-time weather/wave/current forecast fields",
            "source_url": "https://data.marine.copernicus.eu/",
            "source_authority": "Copernicus Marine / forecast provider",
            "data_version": "not supplied",
            "decision_time_safe": False,
            "artifact_pinned": False,
            "status": "BLOCKED_NO_HISTORICAL_DECISION_TIME_DATA",
            "m18e_action": "Do not substitute hindsight reanalysis for operational forecast availability. Obtain archived forecasts keyed to decision_time first.",
            "notes": "No aligned archived forecast artifact is present in the project.",
        },
        {
            "source_id": "historical_port_operations",
            "information_family": "port_operations",
            "source_name": "Historical berth/pilot/tug/anchorage/port-call operations",
            "source_url": "https://greenvoyage2050.imo.org/port-call-optimization-guide-launched-at-imo/",
            "source_authority": "Port/terminal/VTS/PCS data holder; IMO semantics reference",
            "data_version": "not supplied",
            "decision_time_safe": False,
            "artifact_pinned": False,
            "status": "BLOCKED_NO_HISTORICAL_DECISION_TIME_DATA",
            "m18e_action": "Acquire timestamped operational records and explicit availability timestamps before any berth/port-operations ablation.",
            "notes": "M18E refuses to infer berth/pilot/tug state from post-hoc or current web information.",
        },
    ]
    out = pd.DataFrame(rows)
    validate_source_registry(out)
    return out


def main() -> int:
    R.mkdir(exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    CONFIG.mkdir(exist_ok=True)

    # Frozen ledgers only. No old-final, no fresh holdout.
    m16b = pd.read_csv(R / "m16b_destination_resolutions.csv")
    m16e = pd.read_csv(R / "m16e_oof_predictions.csv")
    m18c = pd.read_csv(R / "m18c_reference_forensics.csv")
    m18d = pd.read_csv(R / "m18d_residual_ledger.csv")
    if len(m16b) != 386 or m16b.mmsi.nunique() != 386:
        raise AssertionError("M18E M16B development population drift")
    if len(m18d) != 386 or int(m18d.m18d_scope_near_term.sum()) != 279:
        raise AssertionError("M18E M18D scope drift")

    status = m18c[["mmsi", "reference_eta_status"]].copy()
    d = m16b.merge(status, on="mmsi", how="left", validate="one_to_one")
    rows = []

    def add_ablation(ablation_id: str, scope: str, frame: pd.DataFrame, baseline_col: str, candidate_col: str,
                     information_family: str, purity: str, interpretation: str) -> None:
        bm = simple_metrics(frame["target_tte_h"], frame[baseline_col])
        cm = simple_metrics(frame["target_tte_h"], frame[candidate_col])
        ps = paired_gain_summary(frame["target_tte_h"], frame[baseline_col], frame[candidate_col])
        rows.append({
            "ablation_id": ablation_id,
            "scope": scope,
            "information_family": information_family,
            "purity": purity,
            "rows": int(len(frame)),
            "baseline": baseline_col,
            "candidate": candidate_col,
            "baseline_mae_h": bm["mae_h"],
            "candidate_mae_h": cm["mae_h"],
            "mae_gain_h": bm["mae_h"] - cm["mae_h"],
            "baseline_medae_h": bm["medae_h"],
            "candidate_medae_h": cm["medae_h"],
            "medae_gain_h": bm["medae_h"] - cm["medae_h"],
            "baseline_p90_h": bm["p90_ae_h"],
            "candidate_p90_h": cm["p90_ae_h"],
            "p90_gain_h": bm["p90_ae_h"] - cm["p90_ae_h"],
            "paired_mean_gain_h": ps["mean_abs_error_gain_h"],
            "paired_win_fraction": ps["win_fraction"],
            "paired_ci95_low_h": ps["paired_bootstrap_ci95_h"][0],
            "paired_ci95_high_h": ps["paired_bootstrap_ci95_h"][1],
            "interpretation": interpretation,
        })

    add_ablation(
        "DEST_CANONICALIZATION", "STRICT_386", d,
        "pred_m16a_raw_destination_median_h", "pred_train_canonical_destination_median_h",
        "destination_port_intent", "RETROSPECTIVE_COMPONENT_ABLATION",
        "External/curated canonical destination semantics reduce cardinality but do not improve strict pooled MAE; retain as semantics/coverage infrastructure, not standalone accuracy promotion.",
    )
    near = d[d.reference_eta_status.eq("FUTURE_0_7D")].copy()
    add_ablation(
        "DEST_CANONICALIZATION", "NEAR_TERM_FUTURE_0_7D", near,
        "pred_m16a_raw_destination_median_h", "pred_train_canonical_destination_median_h",
        "destination_port_intent", "RETROSPECTIVE_COMPONENT_ABLATION",
        "Near-term slice is diagnostic only; target-derived status never participates in deployment gating.",
    )

    pe = m16e[m16e.physics_eligible.astype(bool)].copy()
    add_ablation(
        "DEST_GEOMETRY_PLUS_CAUSAL_PHYSICS", "PHYSICS_ELIGIBLE", pe,
        "pred_m16d_route_analogue_h", "pred_m16e_physics_h",
        "destination_geometry+routing_physics", "MIXED_COMPONENT_NOT_PURE_EXTERNAL",
        "This is not a pure external-data effect: verified external port geometry enables an interpretable physics expert that also uses internal causal motion.",
    )
    pe_near = pe[pe.reference_eta_status.eq("FUTURE_0_7D")].copy()
    add_ablation(
        "DEST_GEOMETRY_PLUS_CAUSAL_PHYSICS", "PHYSICS_ELIGIBLE_NEAR_TERM", pe_near,
        "pred_m16d_route_analogue_h", "pred_m16e_physics_h",
        "destination_geometry+routing_physics", "MIXED_COMPONENT_NOT_PURE_EXTERNAL",
        "Strong clean-label-regime diagnostic signal; cannot be reported as a new M18E model score because the component was selected/frozen before M18E.",
    )

    ablations = pd.DataFrame(rows)
    ablations.to_csv(R / "m18e_ablation_results.csv", index=False, float_format="%.12g")

    registry = _source_registry()
    registry.to_csv(R / "m18e_external_source_registry.csv", index=False)

    source_summary = {
        "source_count": int(len(registry)),
        "run_retrospective": int(registry.status.eq("RUN_FROZEN_RETROSPECTIVE").sum()),
        "ready_new_data_pinned": int(registry.status.eq("READY_NEW_DATA_PINNED").sum()),
        "blocked_dataset_not_pinned": int(registry.status.eq("BLOCKED_DATASET_NOT_PINNED").sum()),
        "blocked_no_historical_decision_time_data": int(registry.status.eq("BLOCKED_NO_HISTORICAL_DECISION_TIME_DATA").sum()),
    }

    strict_dest = ablations.query("ablation_id == 'DEST_CANONICALIZATION' and scope == 'STRICT_386'").iloc[0]
    near_phy = ablations.query("ablation_id == 'DEST_GEOMETRY_PLUS_CAUSAL_PHYSICS' and scope == 'PHYSICS_ELIGIBLE_NEAR_TERM'").iloc[0]

    audit = {
        "milestone": "M18E",
        "version": M18E_VERSION,
        "decision": "NO_NEW_EXTERNAL_PROMOTION__LAB_AND_BLOCKERS_FROZEN",
        "scope": {
            "development_population": 386,
            "blocked_old_final_population": 53,
            "fresh_holdout_opened": False,
            "old_final_used": False,
            "m16g_modified": False,
            "m16h_modified": False,
            "target_relabeling": False,
            "strict_rows_deleted": False,
            "new_external_model_promoted": False,
        },
        "source_summary": source_summary,
        "retrospective_findings": {
            "strict_destination_canonicalization_mae_gain_h": float(strict_dest.mae_gain_h),
            "strict_destination_canonicalization_p90_gain_h": float(strict_dest.p90_gain_h),
            "strict_destination_canonicalization_ci95_h": [float(strict_dest.paired_ci95_low_h), float(strict_dest.paired_ci95_high_h)],
            "near_term_physics_component_rows": int(near_phy.rows),
            "near_term_physics_component_mae_gain_h": float(near_phy.mae_gain_h),
            "near_term_physics_component_win_fraction": float(near_phy.paired_win_fraction),
            "near_term_physics_component_ci95_h": [float(near_phy.paired_ci95_low_h), float(near_phy.paired_ci95_high_h)],
        },
        "policy": {
            "external_source_requires_local_pinned_artifact": True,
            "external_source_requires_decision_time_availability": True,
            "blocked_sources_are_treated_as_zero_gain": False,
            "target_derived_slices_are_diagnostic_only": True,
            "historical_component_ablation_is_new_model_promotion": False,
            "moving_web_source_may_be_partially_scraped_into_model": False,
            "reanalysis_may_substitute_for_operational_forecast_without_warning": False,
        },
        "next_action": "Acquire and pin one genuinely new decision-time-safe source artifact before any M18E predictive promotion. Highest-priority candidates remain complete destination/port-intent reference and reproducible navigable-route data; do not bundle blocked weather/port-ops sources into a synthetic mega-ablation.",
    }
    validate_ablation_audit(audit)
    (R / "M18E_EXTERNAL_ABLATION.json").write_text(json.dumps(audit, indent=2) + "\n")

    policy_md = (
        "# M18E External Information Ablation Policy\n\n"
        "M18E is an evidence-first laboratory around the frozen M16G/M16H system. It does **not** open the fresh holdout.\n\n"
        "## Admission gate for a new external source\n\n"
        "A source may enter a predictive ablation only when: (1) the exact artifact is locally pinned and checksummed; "
        "(2) its licence/provenance are recorded; (3) each feature is available no later than `decision_time`; "
        "(4) missingness is explicit; (5) the ablation changes one information family at a time; and (6) evaluation uses the frozen M18A/M18B contract.\n\n"
        "Blocked sources are **not** assigned a zero gain. They are unmeasured. Target-derived slices such as `FUTURE_0_7D` remain diagnostic only.\n\n"
        "## Current result\n\n"
        f"The strict retrospective destination-canonicalization ablation changes MAE by **{strict_dest.mae_gain_h:.2f} h** (positive is better) and P90 by **{strict_dest.p90_gain_h:.2f} h**. This is mixed evidence, so canonicalization remains infrastructure rather than a standalone accuracy promotion.\n\n"
        f"On the already-frozen near-term physics-eligible component (n={int(near_phy.rows)}), external destination geometry + causal motion improves the route expert by **{near_phy.mae_gain_h:.2f} h MAE**. Because this is a mixed historical component and not a newly isolated M18E source, it is supporting evidence only.\n"
    )
    (DOCS / "M18E_EXTERNAL_ABLATION_POLICY.md").write_text(policy_md)

    report = (
        "# M18E — External Information Ablation Lab\n\n"
        "## Decision\n\n**NO_NEW_EXTERNAL_PROMOTION__LAB_AND_BLOCKERS_FROZEN**\n\n"
        "M16G/M16H and M18A-D remain unchanged. The fresh holdout and the 53-row old-final population remain sealed.\n\n"
        "## Why this milestone is conservative\n\n"
        "M18D ranked destination/route intent first, but M18E refuses to manufacture external features from moving web pages or present-day information. "
        "A new source is admissible only when its exact artifact is pinned and it could have been known at `decision_time`.\n\n"
        "## Retrospective component ablations\n\n" + ablations.to_markdown(index=False) + "\n\n"
        "## External-source registry\n\n" + registry.to_markdown(index=False) + "\n\n"
        "## Interpretation\n\n"
        "* The frozen M16B canonical destination layer reduces semantic/cardinality noise but **does not improve strict pooled MAE** versus the raw-destination median baseline; its benefit is therefore not promoted as a standalone accuracy gain.\n"
        "* The frozen M16E physics component shows a meaningful near-term gain when verified destination geometry is available, but this combines external coordinates with internal causal motion and is not a pure new-source effect.\n"
        "* Current S-100 product specifications are operational, but this project does not contain a pinned local Sicily/Eastern-Mediterranean S-101/S-102/S-111 dataset. Therefore chart-constrained routing is **blocked, not scored**.\n"
        "* Weather/ocean and port-operations ablations remain blocked until historical decision-time-aligned artifacts exist. Hindsight/current web data are prohibited.\n\n"
        "## Next gate\n\n"
        "Acquire **one** reproducible external artifact first, then run a single-family ablation. The first preferred target is complete destination/port-intent + navigable-route data, not an all-at-once feature bundle.\n"
    )
    (R / "M18E_REPORT.md").write_text(report)

    freeze_files = [
        "reports/M18E_EXTERNAL_ABLATION.json",
        "reports/m18e_ablation_results.csv",
        "reports/m18e_external_source_registry.csv",
        "docs/M18E_EXTERNAL_ABLATION_POLICY.md",
    ]
    freeze = {
        "milestone": "M18E",
        "status": "EXTERNAL_ABLATION_LAB_FROZEN_NO_NEW_PROMOTION",
        "version": M18E_VERSION,
        "development_population": 386,
        "blocked_old_final_population": 53,
        "fresh_holdout_opened": False,
        "model_change": False,
        "new_external_model_promoted": False,
        "m18a_baseline_freeze_sha256": sha256_file(R / "M18A_BASELINE_FREEZE.json"),
        "m18b_validation_freeze_sha256": sha256_file(R / "M18B_VALIDATION_FREEZE.json"),
        "m18c_audit_freeze_sha256": sha256_file(R / "M18C_AUDIT_FREEZE.json"),
        "m18d_attribution_freeze_sha256": sha256_file(R / "M18D_ATTRIBUTION_FREEZE.json"),
        "m16g_oof_sha256": sha256_file(R / "m16g_oof_predictions.csv"),
        "artifact_sha256": {rel: sha256_file(ROOT / rel) for rel in freeze_files},
    }
    (R / "M18E_ABLATION_FREEZE.json").write_text(json.dumps(freeze, indent=2) + "\n")

    # Update state last so frozen inputs above are not mutated.
    state_path = ROOT / "PROJECT_STATE.json"
    state = json.loads(state_path.read_text())
    if state.get("current_phase") != "M18E_DONE_EXTERNAL_INFORMATION_ABLATION_LAB":
        state["state_version"] = int(state.get("state_version", 0)) + 1
    state["current_phase"] = "M18E_DONE_EXTERNAL_INFORMATION_ABLATION_LAB"
    state["current_task"] = "M18E froze the external-information admission gate, source registry, retrospective destination/physics component ablations and explicit blockers. No new external predictive model was promoted because no genuinely new source artifact is both pinned and decision-time aligned; M16G/M16H and M18A-D remain frozen and the fresh holdout stays sealed."
    completed = list(state.get("completed", []))
    line = "M18E external-information ablation lab: decision-time/provenance admission gate, frozen retrospective destination-canonicalization and destination-geometry+physics component ablations, explicit S-100/searoute/weather/port-ops blockers, source registry and reproducibility freeze; no new model promotion, old-final use or fresh-holdout opening"
    if line not in completed:
        completed.append(line)
    state["completed"] = completed
    state["m18e_summary"] = {
        "decision": audit["decision"],
        "development_rows": 386,
        "blocked_old_final_rows": 53,
        "fresh_holdout_opened": False,
        "new_external_model_promoted": False,
        "source_count": source_summary["source_count"],
        "ready_new_data_pinned": source_summary["ready_new_data_pinned"],
        "strict_destination_canonicalization_mae_gain_h": audit["retrospective_findings"]["strict_destination_canonicalization_mae_gain_h"],
        "near_term_physics_component_mae_gain_h": audit["retrospective_findings"]["near_term_physics_component_mae_gain_h"],
    }
    state_path.write_text(json.dumps(state, indent=2) + "\n")

    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
