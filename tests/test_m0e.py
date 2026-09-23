from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m0e import assemble_audited_calls, build_session_diagnostics, validate_manual_audit, validate_vessel_roles


def load_candidates():
    return pd.read_csv(ROOT / "reports" / "catania_m0d_candidate_calls.csv")


def test_manual_audit_covers_every_candidate_exactly_once():
    c = load_candidates()
    a = pd.read_csv(ROOT / "config" / "catania_m0e_manual_audit.csv")
    out = validate_manual_audit(c, a)
    assert len(out) == len(c) == 61
    assert out["session_id"].nunique() == 61


def test_manual_audit_rejects_missing_session():
    c = load_candidates()
    a = pd.read_csv(ROOT / "config" / "catania_m0e_manual_audit.csv").iloc[:-1].copy()
    with pytest.raises(ValueError, match="coverage mismatch"):
        validate_manual_audit(c, a)


def test_vessel_role_table_covers_all_candidate_vessels():
    c = load_candidates()
    r = pd.read_csv(ROOT / "config" / "catania_m0e_vessel_roles.csv")
    out = validate_vessel_roles(c, r)
    assert set(out["mmsi"]) == set(c["mmsi"].astype(int))


def test_assembled_truth_never_promotes_excluded_rows_and_core_is_subset():
    c = load_candidates()
    a = pd.read_csv(ROOT / "config" / "catania_m0e_manual_audit.csv")
    r = pd.read_csv(ROOT / "config" / "catania_m0e_vessel_roles.csv")
    # Diagnostics are deterministic audit metadata; minimal synthetic frame is
    # enough here because detailed full-data diagnostics are tested by the
    # integration script.
    d = pd.DataFrame({"session_id": c["session_id"]})
    out = assemble_audited_calls(c, a, r, d)
    assert not out.loc[~out["keep_for_port_call_truth"], "final_ground_truth"].any()
    assert out.loc[out["eta_primary_eligible"], "final_ground_truth"].all()
    assert out.loc[out["eta_primary_eligible"], "eta_primary_scope"].all()
    assert set(out.loc[out["eta_primary_eligible"], "ground_truth_confidence"]) <= {"A", "B"}


def test_manual_decisions_expected_counts_are_stable():
    c = load_candidates()
    a = validate_manual_audit(c, pd.read_csv(ROOT / "config" / "catania_m0e_manual_audit.csv"))
    assert int(a["keep_for_port_call_truth"].sum()) == 51
    assert int((~a["keep_for_port_call_truth"]).sum()) == 10
    kept = a[a["keep_for_port_call_truth"]]
    assert kept["ground_truth_confidence"].value_counts().to_dict() == {"A": 37, "B": 14}
