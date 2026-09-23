from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m18b import (
    M18B_EMBARGO_H,
    M18B_TEST_BLOCKS,
    assign_temporal_blocks,
    build_forward_membership,
    paired_mae_gain_bootstrap,
    temporal_block_boundaries,
    validate_protocol_dict,
)


def test_m18b_boundaries_keep_equal_timestamps_together():
    t = pd.Series(pd.to_datetime([
        "2026-04-01 00:00", "2026-04-01 00:00", "2026-04-02 00:00",
        "2026-04-03 00:00", "2026-04-04 00:00", "2026-04-05 00:00",
        "2026-04-06 00:00", "2026-04-07 00:00", "2026-04-08 00:00",
        "2026-04-09 00:00",
    ]))
    b = temporal_block_boundaries(t, n_blocks=5)
    block = assign_temporal_blocks(t, b)
    assert block[0] == block[1]
    assert list(np.unique(block)) == [0, 1, 2, 3, 4]


def test_m18b_forward_membership_purges_embargo_and_separates_owners():
    t = pd.date_range("2026-04-01", periods=10, freq="6h")
    manifest = pd.DataFrame({
        "mmsi": np.arange(100, 110),
        "decision_time": t,
        "m18b_temporal_block": np.repeat(np.arange(5), 2),
        "canonical_destination": ["A"] * 10,
    })
    membership, windows = build_forward_membership(manifest, embargo_h=6.0, test_blocks=(2, 3, 4))
    assert windows.owner_overlap_n.eq(0).all()
    for _, w in windows.iterrows():
        sub = membership[membership.m18b_forward_fold.eq(w.m18b_forward_fold)]
        train = sub[sub.m18b_role.eq("TRAIN")]
        test = sub[sub.m18b_role.eq("TEST")]
        assert set(train.mmsi).isdisjoint(set(test.mmsi))
        if len(train):
            assert pd.to_datetime(train.decision_time).max() < pd.Timestamp(w.train_end_exclusive)


def test_m18b_paired_bootstrap_detects_clear_gain():
    y = np.arange(30, dtype=float)
    baseline = y + 10.0
    candidate = y + 1.0
    out = paired_mae_gain_bootstrap(y, baseline, candidate, reps=1000, seed=7)
    assert out["observed_gain_h"] == 9.0
    assert out["gain_ci95_low_h"] > 0
    assert out["probability_gain_positive"] == 1.0


def test_m18b_frozen_artifacts_if_present():
    p = ROOT / "reports/M18B_VALIDATION_PROTOCOL.json"
    if not p.exists():
        return
    protocol = json.loads(p.read_text())
    validate_protocol_dict(protocol)
    freeze = json.loads((ROOT / "reports/M18B_VALIDATION_FREEZE.json").read_text())
    manifest = pd.read_csv(ROOT / "reports/m18b_validation_manifest.csv")
    windows = pd.read_csv(ROOT / "reports/m18b_forward_windows.csv")
    assert freeze["model_change"] is False
    assert freeze["fresh_holdout_opened"] is False
    assert len(manifest) == 386 and manifest.mmsi.nunique() == 386
    assert len(windows) == len(M18B_TEST_BLOCKS)
    assert windows.owner_overlap_n.eq(0).all()
    assert windows.embargo_h.eq(M18B_EMBARGO_H).all()
    assert protocol["cross_port_validation"]["status"] == "WITHHELD_NO_AUTHORITATIVE_PORT_EVENT_LABEL"
