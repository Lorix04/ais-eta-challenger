from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m16f import candidate_configs, family_names, fit_predict, M16F_FEATURES


def _toy() -> pd.DataFrame:
    rows = []
    for i in range(20):
        r = {c: float(i + 1) for c in M16F_FEATURES if c not in {
            "destination_norm", "canonical_destination", "resolution_method", "ship_type_cat",
            "nav_status_cat", "flag_cat", "motion_state_cat", "stale_risk_cat",
        }}
        for c in ["destination_norm", "canonical_destination", "resolution_method", "ship_type_cat", "nav_status_cat", "flag_cat", "motion_state_cat", "stale_risk_cat"]:
            r[c] = "A" if i % 2 == 0 else "B"
        r["mmsi"] = 100000000 + i
        r["target_tte_h"] = float(i * 3 - 10)
        rows.append(r)
    return pd.DataFrame(rows)


def test_predeclared_panel_is_small_and_complete():
    cfg = candidate_configs()
    assert len(cfg) == 10
    assert family_names(cfg) == ["catboost", "extra_trees", "hist_gb", "random_forest", "svr"]
    assert len({c.config_id for c in cfg}) == len(cfg)


def test_m16f_features_do_not_include_target_or_expert_predictions():
    bad = [c for c in M16F_FEATURES if c.startswith("pred_") or "target" in c or "reference_eta_status" in c]
    assert bad == []


def test_each_family_can_fit_without_nonfinite_predictions():
    d = _toy()
    tr, va = d.iloc[:16].copy(), d.iloc[16:].copy()
    for family in family_names():
        cfg = next(c for c in candidate_configs() if c.family == family)
        p = fit_predict(tr, va, cfg)
        assert len(p) == len(va)
        assert np.isfinite(p).all()


def test_if_artifacts_exist_old_final_is_blocked_and_state_advances():
    p = ROOT / "reports/M16F_SUMMARY.json"
    if not p.exists():
        return
    s = json.loads(p.read_text())
    assert s["development_rows"] == 386
    assert s["blocked_old_final_rows"] == 53
    assert s["final_test_used_for_selection"] is False
    state = json.loads((ROOT / "PROJECT_STATE.json").read_text())
    assert state["current_phase"] in {
        "M16E_DONE_MARITIME_PHYSICS_EXPERT", "M16F_DONE_TABULAR_CHALLENGER_PANEL",
        "M16G_DONE_MIXTURE_OF_EXPERTS", "M16H_DONE_PROBABILISTIC_ETA_CONFIDENCE",
        "M16I_DONE_ABLATION_STRESS_RED_TEAM", "M16J_DONE_SMART_CHALLENGER_FREEZE",
        "M17A_DONE_SSL_LEARNED_TRAJECTORY_RETRIEVAL_BENCHMARK",
        "M17B_DONE_HISTORICAL_MEMORY_HNSW_BENCHMARK",
        "M17C_DONE_SICILY_HISTORICAL_CORRIDOR_KNOWLEDGE_GRAPH",
        "M17D_DONE_EXTENDED_MIXTURE_OF_EXPERTS",
        "M17E_DONE_CONSTRAINED_SELECTIVE_ROUTER",
        "M17F_DONE_LATENT_TARGET_NOISE_REGIME_MODEL",
        "M17G_DONE_STRONGER_SSL_ENCODER",
        "M17H_DONE_CONSERVATIVE_MOCO_INTEGRATION",
        "M18A_DONE_PREDICTION_CONTRACT_FROZEN_BASELINE",
        "M18B_DONE_VALIDATION_REDESIGN",
        "M18C_DONE_TARGET_LABEL_RELIABILITY_AUDIT",
        "M18D_DONE_RESIDUAL_ERROR_ATTRIBUTION",
        "M18E_DONE_EXTERNAL_INFORMATION_ABLATION_LAB",
        "M18F_DONE_UNCERTAINTY_SELECTIVE_ETA",
        "M18G_DONE_FROZEN_EXTERNAL_PROMOTION_GATE",
        "M18G_DONE_FULL_DEVELOPMENT_SCORING_BUNDLE_REGISTERED",
        "M19_IMPLEMENTED_AWAITING_EXTERNAL_MMDEC_BYTES",
        "M19_DONE_EXTERNAL_MMDEC_ONE_SHOT_NO_RETUNING",
    }
