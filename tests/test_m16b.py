from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ais_eta.m16b import DestinationResolver, M16B_FUZZY_THRESHOLD

CATALOG = ROOT / "data/static/m16b_destination_catalog.csv"


def resolver():
    return DestinationResolver.from_csv(CATALOG)


def test_augusta_exact_code_space_and_name_converge():
    r = resolver()
    vals = [r.resolve(x) for x in ["ITAUG", "IT AUG", "Augusta"]]
    assert {x.canonical_destination for x in vals} == {"ITAUG"}
    assert all(x.is_resolved_port for x in vals)


def test_catania_fixture_and_route_terminal_semantics():
    r = resolver()
    assert r.resolve("CATANIA SICILY").canonical_destination == "ITCTA"
    x = r.resolve("ITSAL>ITCTA")
    assert x.canonical_destination == "ITCTA"
    assert x.is_route_expression is True
    assert x.resolution_method == "route_terminal_unlocode"


def test_route_name_terminal_milazzo_is_resolved():
    x = resolver().resolve("MESSINA>MILAZZO")
    assert x.canonical_destination == "ITMLZ"
    assert x.is_route_expression


def test_gibraltar_common_wrong_country_prefix_is_curated_not_generic():
    x = resolver().resolve("GBGIB")
    assert x.canonical_destination == "GIGIB"
    assert x.resolution_method == "curated_alias"


def test_malta_is_not_silently_coerced_to_valletta():
    x = resolver().resolve("MALTA")
    assert x.canonical_destination == "AMBIGUOUS:MALTA"
    assert x.canonical_unlocode is None
    assert x.is_resolved_port is False
    assert x.resolution_method == "ambiguous_geography"


def test_for_orders_and_unknown_are_explicit_classes():
    r = resolver()
    assert r.resolve("FOR ORDER").canonical_destination == "FOR_ORDERS"
    assert r.resolve("BALTIC SEA FOR ORDERS").canonical_destination == "FOR_ORDERS"
    assert r.resolve("UNKNOWN").canonical_destination == "UNKNOWN"


def test_unverified_unlocode_like_value_is_not_claimed_as_verified_port():
    x = resolver().resolve("ITMNP")
    assert x.canonical_destination == "ITMNP"
    assert x.resolution_method == "unlocode_like_unverified"
    assert x.is_resolved_port is False


def test_conservative_fuzzy_matching_is_thresholded_and_deterministic():
    r = resolver()
    x1 = r.resolve("CATANIA SICIL")
    x2 = r.resolve("CATANIA SICIL")
    assert x1 == x2
    assert x1.canonical_destination == "ITCTA"
    assert x1.resolution_confidence >= M16B_FUZZY_THRESHOLD


def test_m16b_artifacts_are_dev_only_and_target_free_resolver_contract():
    p = ROOT / "reports/M16B_CARDINALITY_COVERAGE.json"
    ledger = ROOT / "reports/m16b_destination_resolutions.csv"
    if not p.exists() or not ledger.exists():
        return
    summary = json.loads(p.read_text())
    df = pd.read_csv(ledger)
    manifest = json.loads((ROOT / "reports/M16A_DEVELOPMENT_MANIFEST.json").read_text())
    assert summary["development_rows"] == 386
    assert summary["blocked_old_final_rows"] == 53
    assert summary["target_or_reference_eta_used_in_resolution"] is False
    assert summary["final_test_used_for_selection"] is False
    assert len(df) == 386 and df["mmsi"].nunique() == 386
    assert set(df["mmsi"].astype(int)).isdisjoint(manifest["blocked_old_final_mmsi"])


def test_resolver_source_does_not_accept_target_or_eta_inputs():
    import inspect
    from ais_eta.m16b import DestinationResolver
    sig = inspect.signature(DestinationResolver.resolve)
    assert list(sig.parameters) == ["self", "value"]


def test_project_state_preserves_m16b_after_later_m16_progress():
    state = json.loads((ROOT / "PROJECT_STATE.json").read_text())
    if "m16b_summary" not in state:
        return
    assert state["m16b_summary"]["status"] == "DONE_CANONICAL_DESTINATION_RESOLVER"
    assert state["m16b_summary"]["final_test_used_for_selection"] is False
    assert state["current_phase"].startswith(("M16B_", "M16C_", "M16D_", "M16E_", "M16F_", "M16G_", "M16H_", "M16I_", "M16J_", "M17A_", "M17B_", "M17C_", "M17D_", "M17E_", "M17F_", "M17G_", "M17H_", "M18A_", "M18B_", "M18C_", "M18D_", "M18E_", "M18F_", "M18G_", "M19_"))
