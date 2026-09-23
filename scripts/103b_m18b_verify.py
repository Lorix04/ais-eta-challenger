#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m18a import sha256_file  # noqa: E402
from ais_eta.m18b import M18B_EMBARGO_H, M18B_TEST_BLOCKS, validate_protocol_dict  # noqa: E402


def main() -> int:
    protocol = json.loads((R / "M18B_VALIDATION_PROTOCOL.json").read_text())
    freeze = json.loads((R / "M18B_VALIDATION_FREEZE.json").read_text())
    state = json.loads((ROOT / "PROJECT_STATE.json").read_text())
    validate_protocol_dict(protocol)

    assert freeze["model_change"] is False
    assert freeze["m18a_contract_changed"] is False
    assert freeze["fresh_holdout_opened"] is False
    assert freeze["development_population"] == 386
    assert freeze["blocked_old_final_population"] == 53
    assert freeze["strict_forward_folds"] == len(M18B_TEST_BLOCKS)
    assert float(freeze["embargo_h"]) == M18B_EMBARGO_H
    assert sha256_file(R / "M18A_BASELINE_FREEZE.json") == freeze["m18a_baseline_freeze_sha256"]
    assert sha256_file(R / "m16g_oof_predictions.csv") == freeze["m16g_oof_sha256"]
    for rel, digest in freeze["validation_artifact_sha256"].items():
        assert sha256_file(ROOT / rel) == digest, rel

    manifest = pd.read_csv(R / "m18b_validation_manifest.csv", parse_dates=["decision_time"])
    membership = pd.read_csv(R / "m18b_forward_membership.csv", parse_dates=["decision_time"])
    windows = pd.read_csv(R / "m18b_forward_windows.csv", parse_dates=["test_start", "test_end", "train_end_exclusive"])
    assert len(manifest) == 386 and manifest.mmsi.nunique() == 386
    assert windows.owner_overlap_n.eq(0).all()
    assert windows.test_n.sum() == freeze["strict_forward_test_union_n"]
    tests = membership.loc[membership.m18b_role.eq("TEST"), ["m18b_forward_fold", "mmsi"]]
    assert not tests.duplicated("mmsi").any()
    assert tests.mmsi.nunique() == freeze["strict_forward_test_union_n"]
    for _, w in windows.iterrows():
        sub = membership[membership.m18b_forward_fold.eq(w.m18b_forward_fold)]
        train = sub[sub.m18b_role.eq("TRAIN")]
        test = sub[sub.m18b_role.eq("TEST")]
        assert set(train.mmsi.astype(int)).isdisjoint(set(test.mmsi.astype(int)))
        if len(train):
            assert train.decision_time.max() < w.train_end_exclusive
        assert test.decision_time.min() == w.test_start

    assert state["current_phase"].startswith(("M18B_", "M18C_", "M18D_", "M18E_", "M18F_", "M18G_", "M19_"))
    assert state["m18b_summary"]["fresh_holdout_opened"] is False
    assert state["m18b_summary"]["cross_port_claim"] == "WITHHELD"

    print("PASS M18B machine-readable validation protocol")
    print(f"PASS 386 unique owners; strict forward test union={freeze['strict_forward_test_union_n']}")
    print(f"PASS {len(windows)} expanding temporal folds with {M18B_EMBARGO_H:.0f}h embargo")
    print("PASS train/test owner overlap=0; old-final remains excluded")
    print("PASS cross-port claim withheld and fresh external lockbox unopened")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
