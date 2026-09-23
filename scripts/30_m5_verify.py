#!/usr/bin/env python3
"""Integration verification for M5 port/generalization stress tests."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m5 import M5Config  # noqa: E402

REPORTS = ROOT / "reports"


def ok(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS {message}")


def main() -> None:
    cfg = M5Config()
    summary = json.loads((REPORTS / "m5_summary.json").read_text())
    gate = json.loads((REPORTS / "m5_gate_result.json").read_text())
    calls = pd.read_csv(REPORTS / "m5_call_stress_metrics.csv")
    ports = pd.read_csv(REPORTS / "m5_port_metrics.csv")
    cold = pd.read_csv(REPORTS / "m5_cold_vessel_metrics.csv")
    fam = pd.read_csv(REPORTS / "m5_route_family_metrics.csv")
    intervals = pd.read_csv(REPORTS / "m5_port_interval_diagnostics.csv")
    entrance = pd.read_csv(REPORTS / "m5_entrance_support.csv")
    claims = pd.read_csv(REPORTS / "m5_claim_scope.csv")
    m2 = pd.read_csv(REPORTS / "m2_oof_route_predictions.csv")
    splits = pd.read_csv(REPORTS / "m2_call_splits.csv")
    locked = set(splits.loc[splits.m2_split.eq("final_chronological_test"), "session_id"].astype(str))

    ok(calls.session_id.nunique() == 40, "M5 stress table preserves the 40 independent M2 OOF calls")
    ok(calls.mmsi.nunique() == 33, "M5 preserves the 33 unique M2 OOF vessels")
    ok(not (set(calls.session_id.astype(str)) & locked), "locked final chronological calls remain untouched")
    ok(summary["locked_final_calls_scored"] == 0 and gate["locked_final_calls_scored"] == 0,
       "M5 explicitly records zero locked-final scoring")
    ok(set(ports.port) == {"CATANIA", "AUGUSTA"}, "port metrics separate Catania and Augusta")

    for port in ["CATANIA", "AUGUSTA"]:
        r = ports[ports.port.eq(port)].iloc[0]
        ok(int(r.calls) >= cfg.min_port_oof_calls, f"{port} has enough OOF calls for port stress test")
        ok(float(r.route_improvement_fraction) >= -cfg.max_route_degradation_fraction,
           f"{port} route model does not materially degrade <=24h port MAE")
        c = cold[(cold.port.eq(port)) & cold.cold_vessel_in_fold.astype(bool)].iloc[0]
        ok(int(c.calls) >= cfg.min_cold_calls_per_port, f"{port} has enough cold-vessel calls")
        ok(float(c.route_improvement_fraction) >= -cfg.max_route_degradation_fraction,
           f"{port} route model survives cold-vessel stress")

    aug = intervals[intervals.port.eq("AUGUSTA")].iloc[0]
    cat = intervals[intervals.port.eq("CATANIA")].iloc[0]
    ok(int(aug.temporal_high_calls) >= cfg.min_temporal_high_calls_per_port,
       "Augusta has enough later HIGH-reliability calls for port-level interval stress")
    ok(float(aug.pooled_trajectory_coverage) >= cfg.min_port_trajectory_coverage,
       "pooled M4 interval preserves required whole-call coverage in Augusta")
    ok(int(cat.temporal_high_calls) < cfg.min_temporal_high_calls_per_port,
       "Catania temporal interval evidence is explicitly recognized as underpowered")
    ok(bool(aug.uses_max_calibration_score) and bool(cat.uses_max_calibration_score),
       "port-specific 90% conformal diagnostics expose max-score small-N behavior")

    aug_e = fam[(fam.port.eq("AUGUSTA")) & fam.initial_route_family.eq("E")]
    ok(len(aug_e) == 1 and int(aug_e.iloc[0].calls) >= cfg.min_route_family_calls_for_warning,
       "Augusta E route-family stress subgroup is represented")
    ok(float(aug_e.iloc[0].route_improvement_fraction) < -cfg.route_family_warning_degradation_fraction,
       "Augusta E route-family degradation is preserved as a warning, not hidden")
    ok(any(w["type"] == "ROUTE_FAMILY_DEGRADATION" for w in gate["warnings"]),
       "M5 gate records route-family heterogeneity warning")

    sci = entrance[(entrance.port.eq("AUGUSTA")) & entrance.entrance.eq("SCIROCCO")]
    ok(len(sci) == 1 and int(sci.iloc[0].eta_eligible_calls) == 0,
       "Scirocco has no ETA-eligible support and is not promoted to generalization evidence")
    ok(gate["unseen_port_generalization"] == "NOT_ESTABLISHED",
       "unseen-port generalization is explicitly not claimed")
    ok(gate["leave_one_port_out"] == "NOT_SCIENTIFICALLY_MEANINGFUL_FOR_RETAINED_ROUTE_MODEL",
       "leave-one-port-out is not manufactured from incompatible port-specific route geometry")

    unseen = claims[claims.claim.eq("Model generalizes to an unseen port")].iloc[0]
    ok(unseen.status == "NOT_ESTABLISHED", "claim-scope table blocks unseen-port claim")
    final_claim = claims[claims.claim.eq("Locked final chronological holdout performance")].iloc[0]
    ok(final_claim.status == "NOT_YET_EVALUATED", "claim-scope table preserves final holdout")

    ok(len(m2) == 556, "M5 does not rewrite M2 OOF predictions")
    ok(gate["status"] == "M5_GO_M6_SCOPE_LIMITED" and gate["passed"] is True,
       "M5 gate passes with explicit scope limits")
    print("M5 verification complete")


if __name__ == "__main__":
    main()
