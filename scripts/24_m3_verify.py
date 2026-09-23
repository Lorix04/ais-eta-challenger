#!/usr/bin/env python3
"""Integration verification for M3 cross-fitted residual modelling."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m3 import (  # noqa: E402
    FORBIDDEN_MODEL_FEATURES,
    MODEL_CATEGORICAL_FEATURES,
    MODEL_NUMERIC_FEATURES,
    PORT_STATE_FEATURES,
    validate_feature_contract,
)

REPORTS = ROOT / "reports"


def ok(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS {message}")


def main() -> None:
    summary = json.loads((REPORTS / "m3_summary.json").read_text())
    gate = json.loads((REPORTS / "m3_gate_result.json").read_text())
    ps_gate = json.loads((REPORTS / "m3_port_state_gate.json").read_text())
    oof = pd.read_csv(REPORTS / "m3_oof_predictions.csv")
    splits = pd.read_csv(REPORTS / "m3_stack_splits.csv")
    manifest = pd.read_csv(REPORTS / "m3_fold_training_manifest.csv")
    m2 = pd.read_csv(REPORTS / "m2_oof_route_predictions.csv")
    locked = pd.read_csv(REPORTS / "m2_call_splits.csv")
    locked_ids = set(locked.loc[locked.m2_split.eq("final_chronological_test"), "session_id"].astype(str))

    ok(splits.session_id.nunique() == 40, "M3 starts from exactly 40 M2 cross-fitted calls")
    ok((splits.m3_role == "stacking_warmup").sum() == 10, "10 stacking warm-up calls frozen")
    ok((splits.m3_role == "oof_validation").sum() == 30, "30 calls assigned to M3 OOF validation")
    ok(oof.session_id.nunique() == 30, "M3 OOF contains 30 independent calls")
    ok(oof.m3_temporal_fold.nunique() == 3, "three expanding temporal M3 folds present")
    ok(not (set(oof.session_id.astype(str)) & locked_ids), "locked M2 final calls remain untouched")
    ok(summary["locked_final_calls_scored"] == 0, "summary records zero final-holdout scoring")

    ok((pd.to_datetime(manifest.training_max_arrival) < pd.to_datetime(manifest.validation_min_arrival)).all(),
       "every M3 training history ends before its validation block")
    ok((manifest.training_calls.to_numpy() == np.array([10, 20, 30])).all(),
       "expanding M3 train call counts are 10/20/30")

    pred_cols = [
        "pred_m3_route_physics_h", "pred_m3_direct_ridge_h", "pred_m3_residual_ridge_h",
        "pred_m3_residual_huber_h", "pred_m3_residual_lightgbm_h",
        "pred_m3_residual_lightgbm_portstate_h",
    ]
    ok(np.isfinite(oof[pred_cols].to_numpy(float)).all(), "all M3 predictions are finite")
    ok((oof[pred_cols].to_numpy(float) >= 0).all(), "all M3 predictions are non-negative")

    # M3 route baseline must be the existing M2 OOF route prediction, not a
    # newly in-sample fitted route feature.
    for c in ["decision_time"]:
        oof[c] = pd.to_datetime(oof[c], errors="raise")
        m2[c] = pd.to_datetime(m2[c], errors="raise")
    chk = oof.merge(
        m2[["session_id", "decision_time", "pred_m2_route_knn_h"]].rename(columns={"pred_m2_route_knn_h": "pred_m2_route_knn_h_ref"}),
        on=["session_id", "decision_time"], how="left", validate="one_to_one")
    ok(chk.pred_m2_route_knn_h_ref.notna().all(), "every M3 row maps back to an M2 OOF base prediction")
    ok(np.allclose(chk.pred_m3_route_physics_h, chk.pred_m2_route_knn_h_ref, atol=1e-12),
       "M3 physical baseline exactly preserves M2 cross-fitted route ETA")

    validate_feature_contract(MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES + PORT_STATE_FEATURES)
    ok(not (set(MODEL_NUMERIC_FEATURES + MODEL_CATEGORICAL_FEATURES + PORT_STATE_FEATURES) & FORBIDDEN_MODEL_FEATURES),
       "feature manifest excludes identity/truth columns")
    ok("mmsi" not in summary["feature_manifest"]["numeric"] + summary["feature_manifest"]["categorical"],
       "raw MMSI is not a model feature")
    ok("true_tta_h" not in summary["feature_manifest"]["numeric"], "true remaining time is not a model feature")

    ok(summary["selected_core_model"] == "pred_m3_residual_lightgbm_h", "fixed candidate selection picks residual LightGBM")
    ok(summary["status"] == "M3_COMPLEXITY_NOT_JUSTIFIED", "M3 stop condition fires when residual ML does not beat physics")
    ok(gate["status"] == "M3_COMPLEXITY_NOT_JUSTIFIED", "saved M3 gate agrees with summary")
    ok(ps_gate["status"] == "DROP_PORT_STATE_FOR_NOW", "port-state ablation is rejected by pre-specified gate")
    ok(summary["selected_under24_mae_h"] > summary["route_physics_under24_mae_h"],
       "selected residual model is worse than route physics at <=24h")
    ok(summary["cold_selected_under24_mae_h"] > summary["cold_route_physics_under24_mae_h"],
       "residual model also degrades cold-vessel <=24h performance")
    ok(summary["bootstrap"]["ci95_high_h"] < 0, "paired call bootstrap CI indicates negative residual-ML gain")

    print("M3 verification complete")


if __name__ == "__main__":
    main()
