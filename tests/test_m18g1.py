from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m18g import sha256_file, validate_prediction_ledger
from ais_eta.m18g1 import (
    M18G1_FORBIDDEN_FRESH_COLUMNS,
    M18G1_VERSION,
    score_fresh_rows,
    validate_fresh_rows,
)

R = ROOT / "reports"
BUNDLE = ROOT / "models/m18g1_full_development_scoring_bundle.joblib"
REF = ROOT / "data/derived/m10_reference_rows.pkl.gz"
STATES = ROOT / "data/derived/m0c_ship_states.pkl.gz"


def test_m18g1_registration_and_hash():
    reg = json.loads((R / "M18G1_SCORING_BUNDLE_REGISTRATION.json").read_text())
    assert reg["version"] == M18G1_VERSION
    assert reg["status"] == "FULL_DEVELOPMENT_M16G_M16H_SCORING_BUNDLE_REGISTERED"
    assert reg["development_population"] == 386
    assert reg["blocked_old_final_population"] == 53
    assert reg["fresh_holdout_opened"] is False
    assert reg["holdout_labels_observed"] is False
    assert reg["historical_m16g_oof_score_changed"] is False
    assert sha256_file(BUNDLE) == reg["bundle_sha256"]


def test_m18g1_fresh_input_rejects_targets():
    raw = pd.read_pickle(REF)
    row = raw.loc[raw["split"].eq("train")].head(1).copy()
    with pytest.raises(ValueError, match="label/reference"):
        validate_fresh_rows(row)
    clean = row.drop(columns=[c for c in row.columns if c in M18G1_FORBIDDEN_FRESH_COLUMNS])
    validate_fresh_rows(clean)


def _synthetic_unseen_two():
    raw = pd.read_pickle(REF)
    dev = raw.loc[raw["split"].isin(["train", "calibration"])].sort_values("mmsi", kind="mergesort")
    old = dev.iloc[[5, 105]]["mmsi"].astype(int).tolist()
    rows = dev.set_index("mmsi").loc[old].reset_index().copy()
    new = [990100001, 990100002]
    rows["mmsi"] = new
    rows = rows.drop(columns=[c for c in rows.columns if c in M18G1_FORBIDDEN_FRESH_COLUMNS])
    states = pd.read_pickle(STATES)
    parts = []
    for a, b in zip(old, new):
        z = states.loc[states["mmsi"].eq(a)].copy()
        z["mmsi"] = b
        parts.append(z)
    return rows, pd.concat(parts, ignore_index=True)


def test_m18g1_scores_unseen_rows_deterministically_and_label_free():
    bundle = joblib.load(BUNDLE)
    rows, states = _synthetic_unseen_two()
    a, detail_a = score_fresh_rows(bundle, rows, states)
    b, detail_b = score_fresh_rows(bundle, rows, states)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    pd.testing.assert_frame_equal(detail_a, detail_b, check_exact=True)
    validate_prediction_ledger(a, require_candidate=False)
    assert len(a) == 2
    assert set(a["slice_vessel_seen"]) == {"UNSEEN"}
    assert not (set(a.columns) & M18G1_FORBIDDEN_FRESH_COLUMNS)
    assert a[["baseline_pred_h", "m16h_risk_h", "m16h_lower_h", "m16h_upper_h"]].notna().all().all()


def test_m18g1_readiness_advances_only_bundle_blocker():
    q = pd.read_csv(R / "m18g1_opening_readiness_after_bundle.csv").set_index("check")
    assert bool(q.loc["full_development_m16g_m16h_scoring_bundle_registered", "passed_now"]) is True
    assert bool(q.loc["fresh_holdout_received_unlabelled", "passed_now"]) is False
    assert bool(q.loc["blind_prediction_ledger_sealed_before_labels", "passed_now"]) is False
    assert bool(q.loc["labels_observed", "passed_now"]) is False
    smoke = pd.read_csv(R / "m18g1_blind_smoke_ledger.csv")
    validate_prediction_ledger(smoke, require_candidate=False)
