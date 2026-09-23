import pandas as pd
from ais_eta.m10 import (
    M10Config,
    deterministic_split,
    normalize_destination,
    parse_ais_eta_reference,
    reference_eta_status,
)


def test_eta_year_rollover_forward():
    ref = pd.Timestamp("2026-12-31 23:00:00")
    assert parse_ais_eta_reference("01/01 02:00", ref) == pd.Timestamp("2027-01-01 02:00:00")


def test_eta_year_rollover_backward():
    ref = pd.Timestamp("2026-01-02 03:00:00")
    assert parse_ais_eta_reference("12/31 20:00", ref) == pd.Timestamp("2025-12-31 20:00:00")


def test_eta_invalid_marker_is_unparseable():
    assert pd.isna(parse_ais_eta_reference("—", pd.Timestamp("2026-04-10")))


def test_reference_eta_status_boundaries():
    assert reference_eta_status(-25) == "PAST_GT24H"
    assert reference_eta_status(-2) == "PAST_1_24H"
    assert reference_eta_status(-0.5) == "PAST_LT1H"
    assert reference_eta_status(24) == "FUTURE_0_7D"
    assert reference_eta_status(24 * 8) == "FUTURE_7_14D"
    assert reference_eta_status(24 * 20) == "FUTURE_14_30D"
    assert reference_eta_status(24 * 40) == "FUTURE_GT30D"


def test_split_is_target_free_and_deterministic():
    cfg = M10Config(split_salt="unit-test")
    assert deterministic_split(247123456, cfg) == deterministic_split(247123456, cfg)
    assert deterministic_split("247123456", cfg) == deterministic_split(247123456, cfg)


def test_destination_normalization():
    assert normalize_destination("  IT AUG  ") == "IT AUG"
    assert normalize_destination("ITAUG>ITSPA") == "ITAUG>ITSPA"
    assert normalize_destination(None) == "UNKNOWN"
