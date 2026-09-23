import pandas as pd

from ais_eta.m11 import (
    destination_specificity,
    eta_granularity_bucket,
    eta_year_resolution,
    feature_freshness_bucket,
    top_error_concentration,
)


def test_year_resolution_forward_near_october_from_april():
    a = eta_year_resolution("10/03 20:25", pd.Timestamp("2026-04-16 15:39:13"))
    assert a.chosen_year == 2026
    assert 169 < a.chosen_delta_days < 171
    assert 24 < a.assignment_margin_days < 25


def test_year_resolution_backward_near_november_from_april():
    a = eta_year_resolution("11/04 19:00", pd.Timestamp("2026-04-09 17:01:56"))
    assert a.chosen_year == 2025
    assert -157 < a.chosen_delta_days < -155


def test_eta_granularity_buckets():
    assert eta_granularity_bucket(pd.Timestamp("2026-04-10 12:00")) == "EXACT_HOUR"
    assert eta_granularity_bucket(pd.Timestamp("2026-04-10 12:30")) == "HALF_HOUR"
    assert eta_granularity_bucket(pd.Timestamp("2026-04-10 12:25")) == "OTHER_5MIN_MULTIPLE"
    assert eta_granularity_bucket(pd.Timestamp("2026-04-10 12:18")) == "NON_5MIN"


def test_feature_freshness_buckets():
    assert feature_freshness_bucket(0.5) == "LE_5MIN"
    assert feature_freshness_bucket(30) == "GT5_LE60MIN"
    assert feature_freshness_bucket(120) == "GT60_LE180MIN"
    assert feature_freshness_bucket(181) == "GT180MIN"


def test_destination_specificity():
    assert destination_specificity("FOR ORDERS") == "NON_SPECIFIC"
    assert destination_specificity("UNKNOWN") == "NON_SPECIFIC"
    assert destination_specificity("ITAUG") == "SPECIFIC_OR_STRUCTURED"


def test_error_concentration():
    d = top_error_concentration([10, 5, 3, 2], ks=[1, 2])
    assert d.loc[d.top_k.eq(1), "share_of_total_absolute_error"].iloc[0] == 0.5
    assert d.loc[d.top_k.eq(2), "share_of_total_absolute_error"].iloc[0] == 0.75
