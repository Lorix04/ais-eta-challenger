#!/usr/bin/env python3
"""Run M5 port-specific/generalization stress tests without touching locked final calls."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m4 import M4Config, enrich_with_source_state  # noqa: E402
from ais_eta.m5 import (  # noqa: E402
    M5Config,
    build_call_stress_table,
    evaluate_m5_gate,
    grouped_route_metrics,
    port_interval_diagnostics,
    temporal_reliability_by_port,
    temporal_stability_by_port,
)

REPORTS = ROOT / "reports"
STATES_PATH = ROOT / "data" / "derived" / "m0c_ship_states.pkl.gz"


def _jsonify(v):
    if isinstance(v, dict):
        return {str(k): _jsonify(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonify(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    return v


def _ground_truth_confidence() -> pd.DataFrame:
    cat = pd.read_csv(REPORTS / "catania_m0e_eta_primary_cohort.csv")[["session_id", "ground_truth_confidence"]]
    aug = pd.read_csv(REPORTS / "augusta_m1x_eta_primary_cohort.csv")[["session_id", "ground_truth_confidence"]]
    return pd.concat([cat, aug], ignore_index=True).drop_duplicates("session_id")


def _entrance_support() -> pd.DataFrame:
    cat_all = pd.read_csv(REPORTS / "catania_m0e_audited_calls.csv")
    cat_eta = pd.read_csv(REPORTS / "catania_m0e_eta_primary_cohort.csv")
    aug_eta = pd.read_csv(REPORTS / "augusta_m1x_eta_primary_cohort.csv")
    aug_all = pd.read_csv(REPORTS / "augusta_m1x_audited_calls.csv")
    rows = [{
        "port": "CATANIA",
        "entrance": "CATANIA_RESEARCH_GATE",
        "audited_candidates": int(len(cat_all)),
        "validated_calls": int(cat_all["final_ground_truth"].astype(bool).sum()),
        "eta_eligible_calls": int(len(cat_eta)),
        "m5_status": "SINGLE_RESEARCH_GATE_ONLY",
    }]
    for entrance, g in aug_all.groupby("inbound_entrance", dropna=False):
        eligible = aug_eta[aug_eta["inbound_entrance"].astype(str).eq(str(entrance))]
        validated = int(g["final_ground_truth"].astype(bool).sum())
        rows.append({
            "port": "AUGUSTA",
            "entrance": str(entrance),
            "audited_candidates": int(len(g)),
            "validated_calls": validated,
            "eta_eligible_calls": int(len(eligible)),
            "m5_status": "SUPPORTED_FOR_ETA" if len(eligible) else "NOT_SUPPORTED_FOR_ETA_GENERALIZATION",
        })
    return pd.DataFrame(rows)


def _claim_scope(port_metrics: pd.DataFrame, interval_diag: pd.DataFrame, entrance: pd.DataFrame) -> pd.DataFrame:
    port_lookup = port_metrics.set_index("port")
    aug_interval = interval_diag[interval_diag.port.eq("AUGUSTA")].iloc[0]
    cat_interval = interval_diag[interval_diag.port.eq("CATANIA")].iloc[0]
    return pd.DataFrame([
        {
            "claim": "Retained route-kNN improves <=24h ETA versus geodesic within Catania OOF",
            "status": "SUPPORTED_DEVELOPMENT_OOF",
            "evidence": f"{int(port_lookup.loc['CATANIA','calls'])} calls; improvement {port_lookup.loc['CATANIA','route_improvement_fraction']*100:.1f}%",
        },
        {
            "claim": "Retained route-kNN improves <=24h ETA versus geodesic within Augusta OOF",
            "status": "SUPPORTED_DEVELOPMENT_OOF",
            "evidence": f"{int(port_lookup.loc['AUGUSTA','calls'])} calls; improvement {port_lookup.loc['AUGUSTA','route_improvement_fraction']*100:.1f}%",
        },
        {
            "claim": "M4 pooled HIGH-reliability interval transfers to later Augusta calls",
            "status": "SUPPORTED_SMALL_TEMPORAL_SAMPLE",
            "evidence": f"{int(aug_interval.temporal_high_calls)} calls; whole-call coverage {aug_interval.pooled_trajectory_coverage*100:.1f}%",
        },
        {
            "claim": "M4 pooled HIGH-reliability interval transfers to later Catania calls",
            "status": "UNDERPOWERED",
            "evidence": f"only {int(cat_interval.temporal_high_calls)} later HIGH-reliability calls",
        },
        {
            "claim": "Model generalizes to an unseen port",
            "status": "NOT_ESTABLISHED",
            "evidence": "retained route model is port/gate-specific; no scientifically valid leave-one-port-out route geometry",
        },
        {
            "claim": "Model generalizes across Augusta entrances",
            "status": "NOT_ESTABLISHED",
            "evidence": "primary ETA cohort contains Levante only; Scirocco has no ETA-eligible calls",
        },
        {
            "claim": "Model generalizes to Siracusa/Santa Panagia",
            "status": "NOT_ESTABLISHED",
            "evidence": "authoritative geometry/ground truth intentionally not promoted beyond exploratory status",
        },
        {
            "claim": "Locked final chronological holdout performance",
            "status": "NOT_YET_EVALUATED",
            "evidence": "16 final calls remain untouched until M6",
        },
    ])


def main() -> None:
    cfg = M5Config()
    m4cfg = M4Config()
    m2 = pd.read_csv(REPORTS / "m2_oof_route_predictions.csv")
    states = pd.read_pickle(STATES_PATH)
    splits = pd.read_csv(REPORTS / "m2_call_splits.csv")
    locked = set(splits.loc[splits.m2_split.eq("final_chronological_test"), "session_id"].astype(str))
    if set(m2.session_id.astype(str)) & locked:
        raise AssertionError("M5 M2 OOF source contains locked final calls")

    # Freeze the M4 reliability rules; M5 is diagnostic and does not tune them.
    enriched = enrich_with_source_state(m2, states, m4cfg)
    if set(enriched.session_id.astype(str)) & locked:
        raise AssertionError("M5 source-state enrichment leaked locked final calls")

    call_table = build_call_stress_table(enriched, _ground_truth_confidence(), cfg.max_eval_h)
    call_table.to_csv(REPORTS / "m5_call_stress_metrics.csv", index=False)

    port_metrics = grouped_route_metrics(call_table, ["port"], cfg)
    cold_metrics = grouped_route_metrics(call_table, ["port", "cold_vessel_in_fold"], cfg)
    route_family_metrics = grouped_route_metrics(call_table, ["port", "initial_route_family"], cfg)
    confidence_metrics = grouped_route_metrics(call_table, ["port", "ground_truth_confidence"], cfg)
    intent_metrics = grouped_route_metrics(call_table, ["port", "initial_intent_supported"], cfg)
    initial_reliability_metrics = grouped_route_metrics(call_table, ["port", "initial_reliability_tier"], cfg)

    port_metrics.to_csv(REPORTS / "m5_port_metrics.csv", index=False)
    cold_metrics.to_csv(REPORTS / "m5_cold_vessel_metrics.csv", index=False)
    route_family_metrics.to_csv(REPORTS / "m5_route_family_metrics.csv", index=False)
    confidence_metrics.to_csv(REPORTS / "m5_ground_truth_confidence_metrics.csv", index=False)
    intent_metrics.to_csv(REPORTS / "m5_initial_intent_metrics.csv", index=False)
    initial_reliability_metrics.to_csv(REPORTS / "m5_initial_reliability_metrics.csv", index=False)

    temporal = pd.read_csv(REPORTS / "m4_temporal_eval_predictions.csv")
    for c in ["decision_time", "ground_truth_time", "pred_raw_arrival", "pred_stabilized_arrival"]:
        temporal[c] = pd.to_datetime(temporal[c], errors="raise")
    if set(temporal.session_id.astype(str)) & locked:
        raise AssertionError("M4 temporal evaluation unexpectedly contains locked final calls")

    rel_port = temporal_reliability_by_port(temporal)
    stability_port = temporal_stability_by_port(temporal)
    rel_port.to_csv(REPORTS / "m5_temporal_reliability_by_port.csv", index=False)
    stability_port.to_csv(REPORTS / "m5_temporal_stability_by_port.csv", index=False)

    cal_scores = pd.read_csv(REPORTS / "m4_calibration_call_scores.csv")
    m4_summary = json.loads((REPORTS / "m4_summary.json").read_text())
    pooled_q = float(m4_summary["primary_interval"]["half_width_h"])
    interval_diag = port_interval_diagnostics(cal_scores, temporal, pooled_q, cfg.confidence_level)
    interval_diag.to_csv(REPORTS / "m5_port_interval_diagnostics.csv", index=False)

    entrance = _entrance_support()
    entrance.to_csv(REPORTS / "m5_entrance_support.csv", index=False)

    lopo = {
        "status": "NOT_SCIENTIFICALLY_MEANINGFUL_FOR_RETAINED_ROUTE_MODEL",
        "attempted": False,
        "reason": [
            "M2 historical routes terminate at a port-specific target gate; transferring those geometries to another port would not represent an unseen-port deployment.",
            "Catania OOF support is only 10 calls and its initial route family is effectively SSE-only, while Augusta is multimodal.",
            "Augusta ETA cohort is Levante-only, so even within-port entrance transfer is not supported.",
            "Siracusa/Santa Panagia remain exploratory because authoritative production-grade geometry was not established.",
        ],
        "allowed_claim": "within-seen-port temporal and cold-vessel stress tests only",
    }
    (REPORTS / "m5_leave_one_port_out_assessment.json").write_text(json.dumps(lopo, indent=2))

    claims = _claim_scope(port_metrics, interval_diag, entrance)
    claims.to_csv(REPORTS / "m5_claim_scope.csv", index=False)

    gate = evaluate_m5_gate(port_metrics, cold_metrics, interval_diag, route_family_metrics, cfg)
    # Add explicit stress warnings that do not change the pre-specified core gate.
    seen = cold_metrics[~cold_metrics.cold_vessel_in_fold.astype(bool)].copy()
    for r in seen.itertuples(index=False):
        if int(r.calls) >= cfg.min_seen_calls_for_warning and float(r.route_improvement_fraction) < -cfg.route_family_warning_degradation_fraction:
            gate["warnings"].append({
                "type": "SEEN_VESSEL_SUBGROUP_DEGRADATION",
                "port": str(r.port),
                "calls": int(r.calls),
                "route_improvement_fraction": float(r.route_improvement_fraction),
            })
    gate["locked_final_calls_scored"] = 0
    gate["leave_one_port_out"] = lopo["status"]
    (REPORTS / "m5_gate_result.json").write_text(json.dumps(_jsonify(gate), indent=2))

    port_dict = {r.port: r for r in port_metrics.itertuples(index=False)}
    cold_true = cold_metrics[cold_metrics.cold_vessel_in_fold.astype(bool)]
    cold_dict = {r.port: r for r in cold_true.itertuples(index=False)}
    family_warn = [w for w in gate["warnings"] if w["type"] == "ROUTE_FAMILY_DEGRADATION"]
    summary = {
        "status": gate["status"],
        "passed": gate["passed"],
        "retained_model": "M2 port-specific train-only route-kNN distance / robust trailing speed + M4 reliability/stability",
        "m5_design": "stress-test existing cross-fitted OOF and later M4 temporal-evaluation predictions; no refit and no locked-final scoring",
        "oof_calls": int(call_table.session_id.nunique()),
        "oof_unique_vessels": int(call_table.mmsi.nunique()),
        "locked_final_calls_scored": 0,
        "port_route_under24": {
            p: {
                "calls": int(r.calls),
                "geodesic_mae_min": float(r.geodesic_voyage_mae_h * 60),
                "route_mae_min": float(r.route_voyage_mae_h * 60),
                "route_improvement_fraction": float(r.route_improvement_fraction),
                "paired_gain_ci95_min": [float(r.bootstrap_gain_ci95_low_h * 60), float(r.bootstrap_gain_ci95_high_h * 60)],
                "route_better_call_fraction": float(r.route_better_call_fraction),
            }
            for p, r in port_dict.items()
        },
        "cold_vessel_under24": {
            p: {
                "calls": int(r.calls),
                "geodesic_mae_min": float(r.geodesic_voyage_mae_h * 60),
                "route_mae_min": float(r.route_voyage_mae_h * 60),
                "route_improvement_fraction": float(r.route_improvement_fraction),
            }
            for p, r in cold_dict.items()
        },
        "temporal_eval_port_counts": temporal.drop_duplicates("session_id").groupby("port").size().astype(int).to_dict(),
        "interval_diagnostics": interval_diag.to_dict(orient="records"),
        "route_family_warnings": family_warn,
        "entrance_generalization": entrance.to_dict(orient="records"),
        "unseen_port_generalization": "NOT_ESTABLISHED",
        "leave_one_port_out_attempted": False,
        "claim_scope_file": "reports/m5_claim_scope.csv",
        "next": "M6 final take-home deliverable and one-time locked chronological evaluation under frozen M2/M4 design",
    }
    (REPORTS / "m5_summary.json").write_text(json.dumps(_jsonify(summary), indent=2))

    # Human-readable report.
    aug_seen = seen[seen.port.eq("AUGUSTA")]
    aug_e = route_family_metrics[(route_family_metrics.port.eq("AUGUSTA")) & (route_family_metrics.initial_route_family.eq("E"))]
    report = "# M5 — Port generalization / stress tests\n\n"
    report += f"**Status:** `{gate['status']}`  \n"
    report += "**Locked final chronological calls scored:** 0\n\n"
    report += "M5 does not fit a new predictor. It stress-tests the retained M2 route-physics model and the fixed M4 reliability/stability layer using only already-cross-fitted OOF data and the later M4 temporal-evaluation block.\n\n"
    report += "## Main result\n\n"
    for port in ["CATANIA", "AUGUSTA"]:
        r = port_dict[port]
        report += f"- **{port}:** <=24 h voyage-balanced MAE {r.geodesic_voyage_mae_h*60:.1f} -> {r.route_voyage_mae_h*60:.1f} min ({r.route_improvement_fraction*100:.1f}% route gain) over {int(r.calls)} OOF calls.\n"
    report += "\nThe route signal therefore survives separately in both seen ports. This is a *within-port* claim, not an unseen-port claim.\n\n"
    report += "## Cold-vessel stress\n\n"
    for port in ["CATANIA", "AUGUSTA"]:
        r = cold_dict[port]
        report += f"- **{port}:** {int(r.calls)} cold-vessel calls, {r.geodesic_voyage_mae_h*60:.1f} -> {r.route_voyage_mae_h*60:.1f} min ({r.route_improvement_fraction*100:.1f}% gain).\n"
    report += "\nCold-vessel performance is therefore not the source of the aggregate route gain.\n\n"
    report += "## Stress failures / scope limits\n\n"
    if len(aug_e):
        r = aug_e.iloc[0]
        report += f"- Augusta's initial **E route-family** is a real stress failure: {int(r.calls)} calls and route-kNN degrades <=24 h MAE by {-r.route_improvement_fraction*100:.1f}% versus geodesic. Do not claim route-kNN improves every route family.\n"
    if len(aug_seen):
        r = aug_seen.iloc[0]
        report += f"- Augusta **seen-vessel** subgroup ({int(r.calls)} calls) degrades by {-r.route_improvement_fraction*100:.1f}%; MERSEY SPIRIT / multi-stage behavior is a major contributor. Repeated MMSI does not imply route stability.\n"
    report += "- Catania has only 2 later M4 temporal-evaluation calls, so port-specific temporal uncertainty claims for Catania are underpowered.\n"
    report += "- Augusta ETA ground truth is Levante-only; Scirocco entrance generalization is not established.\n"
    report += "- Leave-one-port-out was intentionally **not attempted**: the retained route representation is anchored to port-specific gates, so transferring Augusta route geometry to Catania (or vice versa) would test an invalid geometry rather than true unseen-port generalization.\n\n"
    report += "## Uncertainty by port\n\n"
    for r in interval_diag.itertuples(index=False):
        report += f"- **{r.port}:** pooled M4 interval whole-call coverage = {r.pooled_trajectory_coverage*100:.1f}% on {int(r.temporal_high_calls)} later HIGH-reliability calls. A port-only 90% calibration would use k={int(r.port_specific_conformal_order_k)}/{int(r.calibration_calls)} and q={r.port_specific_q_h*60:.1f} min; because k=n, this is a small-N diagnostic dominated by the maximum calibration error, not a replacement for M4's pooled interval.\n"
    report += "\n## Claim boundary\n\n"
    report += "Supported: seen-port OOF route value, cold-vessel stress within Catania/Augusta, scoped M4 reliability/coverage diagnostics.  \n"
    report += "Not established: unseen-port transfer, Augusta Scirocco, Siracusa/Santa Panagia, berth/all-fast ETA, reported-AIS-ETA superiority.\n\n"
    report += "## Gate\n\n"
    report += f"M5 gate = **{gate['status']}**. Core port and cold-vessel checks pass, but route-family heterogeneity and small port-specific uncertainty samples require scope-limited claims. Proceed to M6 without changing the predictor; the locked final chronological holdout is still untouched.\n\n"
    report += "## External context\n\n"
    report += "Recent reproducible AIS-ETA work explicitly notes that vessel-level and cross-port generalization require larger, dedicated validation rather than being inferred from row-level success. Recent port-trajectory research likewise models port phases with port-specific geospatial context. M5 follows that conservative interpretation instead of manufacturing a leave-one-port-out result from incompatible target geometries.\n"
    (REPORTS / "M5_REPORT.md").write_text(report)

    print(json.dumps(_jsonify(summary), indent=2))


if __name__ == "__main__":
    main()
