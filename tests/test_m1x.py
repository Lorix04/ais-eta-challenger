from __future__ import annotations

import pandas as pd

from ais_eta.m1x import AugustaGeometryV1, _merge_short_excursions
from ais_eta.m1x_audit import M1XGateCriteria, evaluate_m1x_evidence_gate, validate_augusta_manual_audit


def test_augusta_geometry_has_two_distinct_versioned_gates():
    g = AugustaGeometryV1()
    assert {x.name for x in g.gates} == {"LEVANTE", "SCIROCCO"}
    assert g.levante_a != g.scirocco_a
    assert "research" in g.version


def test_short_excursion_merge_does_not_duplicate_sessions():
    t = pd.Timestamp("2026-04-01 00:00:00")
    s = [
        {"mmsi": 1, "inbound_time": t, "outbound_time": t + pd.Timedelta(minutes=10), "hysteresis_merges": 0},
        {"mmsi": 1, "inbound_time": t + pd.Timedelta(hours=2), "outbound_time": t + pd.Timedelta(hours=3), "hysteresis_merges": 0},
    ]
    out = _merge_short_excursions(s, hysteresis_s=20*60)
    assert len(out) == 2


def test_augusta_audit_requires_exact_candidate_coverage():
    cand = pd.DataFrame({"session_id": ["AU-001", "AU-002"]})
    audit = pd.DataFrame({
        "session_id": ["AU-001"],
        "keep_for_port_call_truth": [True],
        "ground_truth_confidence": ["A"],
        "audited_event_type": ["VALIDATED_PORT_ENTRANCE"],
        "audit_reason_code": ["x"],
        "audit_note": ["x"],
    })
    try:
        validate_augusta_manual_audit(cand, audit)
    except ValueError as e:
        assert "coverage mismatch" in str(e)
    else:
        raise AssertionError("audit coverage mismatch should fail")


def test_m1x_gate_requires_all_prespecified_dimensions():
    calls = pd.DataFrame({"mmsi": range(25), "session_id": [f"S{i}" for i in range(25)]})
    hs = pd.DataFrame({"max_supported_horizon_h": [13.0]*5 + [7.0]*5 + [2.0]*15})
    result = evaluate_m1x_evidence_gate(calls, hs, M1XGateCriteria(min_combined_eta_calls=25))
    assert result["status"] == "GO_M2"
    result2 = evaluate_m1x_evidence_gate(calls.iloc[:20], hs.iloc[:20], M1XGateCriteria(min_combined_eta_calls=25))
    assert result2["status"] == "HOLD_M2"
