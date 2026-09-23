#!/usr/bin/env python3
"""Verify M12 is explainability-only and preserves all frozen M10/M6 artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ais_eta.m12 import CURRENT_FEATURES, prepare_current_features

R = ROOT / "reports"
MODEL = ROOT / "models" / "m10_strict_reference_catboost.cbm"
D = ROOT / "data" / "derived" / "m10_reference_rows.pkl.gz"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    freeze = json.loads((R / "M10_FREEZE.json").read_text())
    summary = json.loads((R / "m12_summary.json").read_text())
    assert sha256(D) == freeze["reference_dataset_sha256"]
    assert sha256(MODEL) == summary["frozen_model_sha256_before"] == summary["frozen_model_sha256_after"]
    assert summary["interpretation"]["masking_used_for_selection"] is False
    assert summary["interpretation"]["final_test_used_for_model_change"] is False
    assert summary["interpretation"]["history_model_refit_in_m12"] is False

    pvc = pd.read_csv(R / "m12_prediction_values_change.csv")
    shap = pd.read_csv(R / "m12_mean_abs_shap.csv")
    grp = pd.read_csv(R / "m12_group_shap_importance.csv")
    mask = pd.read_csv(R / "m12_calibration_group_masking_stress.csv")
    abl = pd.read_csv(R / "m12_frozen_calibration_ablation.csv")
    local = pd.read_csv(R / "m12_local_explanation_examples.csv")
    assert set(pvc["feature"]) == set(CURRENT_FEATURES)
    assert set(shap["split"]) == {"train", "calibration", "final_test"}
    assert len(grp.loc[grp["split"].eq("calibration")]) == 6
    assert len(mask) == 6
    assert set(abl["task"]) == {"strict_all_parseable", "future_reference"}
    assert len(local) >= 6
    assert max(summary["shap_max_additivity_error_h"].values()) < 1e-8

    # Reproduce the frozen M10 strict final predictions byte-for-value within floating tolerance.
    df = prepare_current_features(pd.read_pickle(D))
    final = df.loc[df["split"].eq("final_test")].copy()
    model = CatBoostRegressor(); model.load_model(str(MODEL))
    pred = model.predict(final[CURRENT_FEATURES])
    stored = pd.read_csv(R / "m10_final_predictions.csv")
    stored = stored.loc[stored["task"].eq("strict_all_parseable"), ["mmsi", "prediction_tte_h"]]
    comp = final[["mmsi"]].copy(); comp["reproduced"] = pred
    comp = comp.merge(stored, on="mmsi", how="left", validate="one_to_one")
    assert len(comp) == 53
    assert np.max(np.abs(comp["reproduced"] - comp["prediction_tte_h"])) < 1e-9

    # M6 historical freeze remains byte-identical as inherited by M10/M11.
    m6 = json.loads((R / "M6_FREEZE.json").read_text())
    for rel, expected in m6["sha256"].items():
        p = ROOT / rel
        assert p.exists(), rel
        assert sha256(p) == expected, rel

    print("PASS M10 reference dataset and frozen CatBoost hashes unchanged")
    print("PASS 53/53 strict final predictions reproduced from frozen artifact")
    print("PASS native importance, SHAP, group masking and frozen calibration ablation tables")
    print("PASS SHAP additivity < 1e-8 h on train/calibration/final")
    print("PASS M12 contains no model selection or post-final tuning")
    print(f"PASS M6 frozen hashes unchanged: {len(m6['sha256'])}/{len(m6['sha256'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
