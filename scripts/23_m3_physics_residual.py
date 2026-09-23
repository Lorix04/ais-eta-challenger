#!/usr/bin/env python3
"""M3 cross-fitted physics + residual ML.

M3 deliberately trains only on *M2 out-of-fold* route predictions.  This avoids
reconstructing in-sample route features for the residual learner: every base
route prediction used by M3 was itself produced without the current call in the
M2 route-training set.  A second expanding temporal split is then applied over
those 40 M2-OOF calls.  The 16 locked chronological final calls remain untouched.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lightgbm import LGBMRegressor  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import HuberRegressor, Ridge  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from ais_eta.m0d import _haversine_km  # noqa: E402
from ais_eta.m1 import KM_PER_NM  # noqa: E402
from ais_eta.m1x import AugustaGeometryV1  # noqa: E402
from ais_eta.m2 import eta_metrics, target_gate  # noqa: E402
from ais_eta.m3 import (  # noqa: E402
    MODEL_CATEGORICAL_FEATURES,
    MODEL_NUMERIC_FEATURES,
    PORT_STATE_FEATURES,
    M3Config,
    broad_horizon_band,
    build_same_snapshot_port_state,
    call_horizon_balanced_weights,
    circular_abs_diff_deg,
    destination_supports_port,
    evaluate_m3_gate,
    evaluate_port_state_gate,
    finalize_m3_features,
    subtract_target_from_port_state,
    validate_feature_contract,
    bearing_to_target_deg,
)

REPORTS = ROOT / "reports"
STATES_PATH = ROOT / "data" / "derived" / "m0c_ship_states.pkl.gz"

BASELINE = "pred_m3_route_physics_h"
CORE_MODELS = [
    "pred_m3_direct_ridge_h",
    "pred_m3_residual_ridge_h",
    "pred_m3_residual_huber_h",
    "pred_m3_residual_lightgbm_h",
]
PORTSTATE_MODEL = "pred_m3_residual_lightgbm_portstate_h"
ALL_MODELS = [BASELINE] + CORE_MODELS + [PORTSTATE_MODEL]


def _jsonify(value):
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def assign_m3_stack_folds(m2_oof: pd.DataFrame, cfg: M3Config) -> pd.DataFrame:
    calls = (
        m2_oof[["session_id", "port", "mmsi", "ground_truth_time"]]
        .drop_duplicates("session_id")
        .copy()
    )
    calls["ground_truth_time"] = pd.to_datetime(calls["ground_truth_time"], errors="raise")
    calls = calls.sort_values(["ground_truth_time", "session_id"], kind="mergesort").reset_index(drop=True)
    if len(calls) <= cfg.stacking_warmup_calls + cfg.stacking_temporal_folds:
        raise ValueError("Insufficient M2-OOF calls for M3 stacking split")
    calls["m3_role"] = "stacking_warmup"
    calls["m3_temporal_fold"] = pd.Series(pd.NA, index=calls.index, dtype="Int64")
    rest = calls.iloc[cfg.stacking_warmup_calls:].copy()
    for fold, positions in enumerate(np.array_split(np.arange(len(rest)), cfg.stacking_temporal_folds), start=1):
        idx = rest.iloc[positions].index
        calls.loc[idx, "m3_role"] = "oof_validation"
        calls.loc[idx, "m3_temporal_fold"] = fold
    return calls


def _decision_lookup() -> pd.DataFrame:
    cat = pd.read_csv(REPORTS / "catania_m1_decision_panel.csv")
    cat["port"] = "CATANIA"
    cat["inbound_entrance"] = "CATANIA"
    cat = cat[[
        "port", "session_id", "decision_time", "source_observation_time",
        "provider_observation_age_s", "progress_to_gate_30m_kn",
    ]]
    aug = pd.read_csv(REPORTS / "augusta_m1x_decision_panel.csv")
    aug = aug[[
        "port", "session_id", "inbound_entrance", "decision_time", "source_observation_time",
        "provider_observation_age_s", "progress_to_gate_30m_kn",
    ]]
    out = pd.concat([cat, aug], ignore_index=True)
    for c in ["decision_time", "source_observation_time"]:
        out[c] = pd.to_datetime(out[c], errors="raise")
    return out.drop_duplicates(["session_id", "decision_time"])


def build_m3_feature_panel(states: pd.DataFrame, m2_oof: pd.DataFrame) -> pd.DataFrame:
    x = m2_oof.copy()
    for c in ["ground_truth_time", "decision_time", "source_observation_time"]:
        x[c] = pd.to_datetime(x[c], errors="raise")
    x = x.merge(_decision_lookup(), on=["port", "session_id", "decision_time", "source_observation_time"], how="left", validate="one_to_one")
    if x["provider_observation_age_s"].isna().any():
        raise AssertionError("M3 decision lookup failed")

    needed_times = set(x["source_observation_time"].tolist())
    state_subset_all = states[states["recorded_at"].isin(needed_times)].copy()
    source_cols = [
        "mmsi", "recorded_at", "sog_clean", "cog_clean", "heading_clean", "draught_clean",
        "nav_status_name", "motion_state", "dynamic_state_age_s", "stop_duration_s",
        "stale_risk_level", "turn_candidate", "destination_clean",
    ]
    src = state_subset_all[source_cols].drop_duplicates(["mmsi", "recorded_at"])
    x = x.merge(
        src,
        left_on=["mmsi", "source_observation_time"],
        right_on=["mmsi", "recorded_at"],
        how="left", validate="many_to_one",
    ).drop(columns=["recorded_at"])
    x = x.rename(columns={"sog_clean": "sog_current_kn"})
    if x[["lat_clean", "lon_clean", "sog_current_kn"]].isna().all(axis=1).any():
        raise AssertionError("M3 exact source-state join failed")

    bearing = []
    port_ref = []
    dest = []
    aug_gates = [g.midpoint for g in AugustaGeometryV1().gates]
    for r in x.itertuples(index=False):
        entrance = "CATANIA" if r.port == "CATANIA" else r.inbound_entrance
        glon, glat = target_gate(r.port, entrance)
        bearing.append(bearing_to_target_deg(r.lon_clean, r.lat_clean, glon, glat))
        dest.append(destination_supports_port(r.destination_clean, r.port))
        if r.port == "CATANIA":
            port_ref.append(float(r.geodesic_distance_km))
        else:
            dd = [float(_haversine_km(
                np.asarray([float(r.lat_clean)]), np.asarray([float(r.lon_clean)]),
                float(g[1]), float(g[0])
            )[0]) for g in aug_gates]
            port_ref.append(min(dd))
    x["bearing_to_gate_deg"] = bearing
    x["cog_to_gate_error_deg"] = circular_abs_diff_deg(x["cog_clean"], x["bearing_to_gate_deg"])
    x["destination_support_port"] = dest
    x["port_reference_distance_km"] = port_ref

    # Same-snapshot port state only at the timestamps required by M3.
    ps = build_same_snapshot_port_state(
        state_subset_all,
        target_gate("CATANIA", "CATANIA"),
        aug_gates,
    )
    x = x.merge(
        ps,
        left_on=["port", "source_observation_time"],
        right_on=["port", "recorded_at"],
        how="left", validate="many_to_one",
    ).drop(columns=["recorded_at"])
    if x[PORT_STATE_FEATURES].isna().any().any():
        raise AssertionError("M3 port-state snapshot join failed")
    x = subtract_target_from_port_state(x)

    x["route_match_confidence"] = 1.0 / (1.0 + x["knn_mean_cross_track_km"].astype(float))
    x["pred_route_physics_h"] = x["pred_m2_route_knn_h"].astype(float)
    x["broad_horizon_band"] = broad_horizon_band(x["true_tta_h"])
    return finalize_m3_features(x)


def _linear_pipeline(estimator, numeric: list[str], categorical: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
        ]), categorical),
    ], remainder="drop", sparse_threshold=0.0)
    return Pipeline([("pre", pre), ("reg", estimator)])


def _lightgbm_pipeline(cfg: M3Config, numeric: list[str], categorical: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
        ]), categorical),
    ], remainder="drop", sparse_threshold=1.0)
    reg = LGBMRegressor(
        n_estimators=cfg.lightgbm_estimators,
        max_depth=cfg.lightgbm_max_depth,
        num_leaves=cfg.lightgbm_num_leaves,
        learning_rate=cfg.lightgbm_learning_rate,
        reg_lambda=cfg.lightgbm_reg_lambda,
        objective="regression_l1",
        min_child_samples=8,
        random_state=cfg.random_seed,
        n_jobs=2,
        verbosity=-1,
    )
    return Pipeline([("pre", pre), ("reg", reg)])


def _clean_frame(df: pd.DataFrame, numeric: list[str], cats: list[str]) -> pd.DataFrame:
    out = df[numeric + cats].copy()
    for c in numeric:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in cats:
        out[c] = out[c].astype("string").fillna("UNKNOWN").astype(str)
    return out


def fit_predict(train: pd.DataFrame, val: pd.DataFrame, cfg: M3Config) -> pd.DataFrame:
    numeric = list(MODEL_NUMERIC_FEATURES)
    cats = list(MODEL_CATEGORICAL_FEATURES)
    numeric_ps = numeric + list(PORT_STATE_FEATURES)
    validate_feature_contract(numeric + cats)
    validate_feature_contract(numeric_ps + cats)

    w = call_horizon_balanced_weights(train).to_numpy()
    y = train["true_tta_h"].to_numpy(float)
    btr = train["pred_route_physics_h"].to_numpy(float)
    bva = val["pred_route_physics_h"].to_numpy(float)
    residual = y - btr
    tr = _clean_frame(train, numeric, cats)
    va = _clean_frame(val, numeric, cats)

    out = val.copy()
    out[BASELINE] = bva
    out["pred_m3_geodesic_h"] = (out.geodesic_distance_km.astype(float) / KM_PER_NM) / out.sog_median_30m_kn.astype(float).clip(lower=1)

    dr = _linear_pipeline(Ridge(alpha=cfg.ridge_alpha), numeric, cats)
    dr.fit(tr, y, reg__sample_weight=w)
    out["pred_m3_direct_ridge_h"] = np.maximum(0, dr.predict(va))

    rr = _linear_pipeline(Ridge(alpha=cfg.ridge_alpha), numeric, cats)
    rr.fit(tr, residual, reg__sample_weight=w)
    out["pred_m3_residual_ridge_h"] = np.maximum(0, bva + rr.predict(va))

    hr = _linear_pipeline(HuberRegressor(epsilon=cfg.huber_epsilon, alpha=cfg.huber_alpha, max_iter=100), numeric, cats)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        hr.fit(tr, residual, reg__sample_weight=w)
    out["pred_m3_residual_huber_h"] = np.maximum(0, bva + hr.predict(va))

    lg = _lightgbm_pipeline(cfg, numeric, cats)
    lg.fit(tr, residual, reg__sample_weight=w)
    out["pred_m3_residual_lightgbm_h"] = np.maximum(0, bva + lg.predict(va))

    trp = _clean_frame(train, numeric_ps, cats)
    vap = _clean_frame(val, numeric_ps, cats)
    lgp = _lightgbm_pipeline(cfg, numeric_ps, cats)
    lgp.fit(trp, residual, reg__sample_weight=w)
    out[PORTSTATE_MODEL] = np.maximum(0, bva + lgp.predict(vap))
    return out


def metrics_by_scope(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, q in [
        ("all_oof", oof),
        ("under6h", oof[oof.true_tta_h.le(6)]),
        ("under12h", oof[oof.true_tta_h.le(12)]),
        ("under24h", oof[oof.true_tta_h.le(24)]),
    ]:
        m = eta_metrics(q, ALL_MODELS)
        m["scope"] = scope
        rows.append(m)
    return pd.concat(rows, ignore_index=True)


def fold_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, g in oof.groupby("m3_temporal_fold"):
        for scope, q in [("all_oof", g), ("under24h", g[g.true_tta_h.le(24)])]:
            m = eta_metrics(q, ALL_MODELS)
            m["m3_temporal_fold"] = int(fold)
            m["scope"] = scope
            rows.append(m)
    return pd.concat(rows, ignore_index=True)


def metric(metrics: pd.DataFrame, model: str, scope: str) -> float:
    x = metrics[(metrics.baseline.eq(model)) & metrics.scope.eq(scope)]
    return float(x.iloc[0].voyage_balanced_mae_h)


def cold_metric(oof: pd.DataFrame, model: str) -> float:
    q = oof[oof.cold_vessel_in_fold.astype(bool) & oof.true_tta_h.le(24)]
    return float(eta_metrics(q, [model]).iloc[0].voyage_balanced_mae_h)


def call_gain_diagnostics(oof: pd.DataFrame, model: str) -> tuple[pd.DataFrame, float]:
    q = oof[oof.true_tta_h.le(24)].copy()
    q["ae_base"] = (q[BASELINE] - q.true_tta_h).abs()
    q["ae_model"] = (q[model] - q.true_tta_h).abs()
    call = q.groupby(["session_id", "port", "mmsi"], as_index=False).agg(
        baseline_mae_h=("ae_base", "mean"), model_mae_h=("ae_model", "mean"), points=("true_tta_h", "size"))
    call["gain_h"] = call.baseline_mae_h - call.model_mae_h
    pos = call.loc[call.gain_h.gt(0), "gain_h"]
    share = float(pos.max() / pos.sum()) if len(pos) and pos.sum() > 0 else 1.0
    return call.sort_values("gain_h", ascending=False), share


def paired_call_bootstrap(oof: pd.DataFrame, model: str, seed: int = 42, n_boot: int = 2000) -> dict:
    q = oof[oof.true_tta_h.le(24)].copy()
    q["ae_base"] = (q[BASELINE] - q.true_tta_h).abs()
    q["ae_model"] = (q[model] - q.true_tta_h).abs()
    call = q.groupby("session_id", as_index=False).agg(base=("ae_base", "mean"), model=("ae_model", "mean"))
    delta = (call.base - call.model).to_numpy(float)
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(n_boot)])
    return {
        "calls": int(len(delta)),
        "mean_call_mae_gain_h": float(delta.mean()),
        "ci95_low_h": float(np.quantile(boot, 0.025)),
        "ci95_high_h": float(np.quantile(boot, 0.975)),
        "p_bootstrap_gain_gt_zero": float(np.mean(boot > 0)),
    }


def main() -> None:
    cfg = M3Config()
    m2 = pd.read_csv(REPORTS / "m2_oof_route_predictions.csv")
    for c in ["ground_truth_time", "decision_time", "source_observation_time"]:
        m2[c] = pd.to_datetime(m2[c], errors="raise")
    states = pd.read_pickle(STATES_PATH)
    panel = build_m3_feature_panel(states, m2)
    calls = assign_m3_stack_folds(m2, cfg)
    calls.to_csv(REPORTS / "m3_stack_splits.csv", index=False)

    rows = []
    manifest = []
    for fold in range(1, cfg.stacking_temporal_folds + 1):
        val_calls = calls[calls.m3_temporal_fold.eq(fold)]
        validation_start = pd.Timestamp(val_calls.ground_truth_time.min())
        train_calls = calls[calls.ground_truth_time < validation_start]
        if set(train_calls.session_id) & set(val_calls.session_id):
            raise AssertionError("M3 stack train/validation call overlap")
        train = panel[panel.session_id.isin(train_calls.session_id)].copy()
        val = panel[panel.session_id.isin(val_calls.session_id)].copy()
        train_vessels = set(train_calls.mmsi.astype(int))
        val["cold_vessel_in_fold"] = ~val.mmsi.astype(int).isin(train_vessels)
        print(f"M3 stack fold {fold}: train {train.session_id.nunique()} calls/{len(train)} rows -> val {val.session_id.nunique()} calls/{len(val)} rows", flush=True)
        pred = fit_predict(train, val, cfg)
        pred["m3_temporal_fold"] = fold
        rows.append(pred)
        manifest.append({
            "fold": fold,
            "validation_start": validation_start,
            "training_calls": int(train.session_id.nunique()),
            "training_unique_vessels": int(train.mmsi.nunique()),
            "training_rows": int(len(train)),
            "validation_calls": int(val.session_id.nunique()),
            "validation_unique_vessels": int(val.mmsi.nunique()),
            "validation_rows": int(len(val)),
            "cold_validation_calls": int(val.loc[val.cold_vessel_in_fold, "session_id"].nunique()),
            "training_max_arrival": pd.Timestamp(train_calls.ground_truth_time.max()),
            "validation_min_arrival": validation_start,
        })

    oof = pd.concat(rows, ignore_index=True)
    # 16 M2 final chronological calls never entered m2_oof, therefore cannot enter M3.
    locked = set(pd.read_csv(REPORTS / "m2_call_splits.csv").query("m2_split == 'final_chronological_test'").session_id.astype(str))
    if set(oof.session_id.astype(str)) & locked:
        raise AssertionError("Locked final calls entered M3")
    oof.to_csv(REPORTS / "m3_oof_predictions.csv", index=False)
    pd.DataFrame(manifest).to_csv(REPORTS / "m3_fold_training_manifest.csv", index=False)

    metrics = metrics_by_scope(oof)
    metrics.to_csv(REPORTS / "m3_eta_metrics.csv", index=False)
    fm = fold_metrics(oof)
    fm.to_csv(REPORTS / "m3_fold_metrics.csv", index=False)

    candidate_u24 = {m: metric(metrics, m, "under24h") for m in CORE_MODELS}
    selected = min(candidate_u24, key=candidate_u24.get)
    base_u24, sel_u24 = metric(metrics, BASELINE, "under24h"), metric(metrics, selected, "under24h")
    base_all, sel_all = metric(metrics, BASELINE, "all_oof"), metric(metrics, selected, "all_oof")
    cold_base, cold_sel = cold_metric(oof, BASELINE), cold_metric(oof, selected)

    pf = fm[fm.scope.eq("under24h")].pivot(index="m3_temporal_fold", columns="baseline", values="voyage_balanced_mae_h")
    fold_wins = int((pf[selected] < pf[BASELINE]).sum())
    call_gains, top_share = call_gain_diagnostics(oof, selected)
    call_gains.to_csv(REPORTS / "m3_selected_model_call_gains.csv", index=False)
    bootstrap = paired_call_bootstrap(oof, selected, seed=cfg.random_seed)

    gate = evaluate_m3_gate(
        base_all, sel_all, base_u24, sel_u24, fold_wins, len(pf), cold_base, cold_sel, top_share)

    ps_core = "pred_m3_residual_lightgbm_h"
    ps_wins = int((pf[PORTSTATE_MODEL] < pf[ps_core]).sum())
    ps_gate = evaluate_port_state_gate(
        metric(metrics, ps_core, "all_oof"), metric(metrics, PORTSTATE_MODEL, "all_oof"),
        metric(metrics, ps_core, "under24h"), metric(metrics, PORTSTATE_MODEL, "under24h"),
        ps_wins, len(pf))

    summary = {
        "status": gate["status"],
        "design": "second-stage expanding temporal stack over M2 cross-fitted route predictions",
        "m2_oof_calls_available": int(calls.session_id.nunique()),
        "stacking_warmup_calls": int((calls.m3_role == "stacking_warmup").sum()),
        "m3_oof_calls": int(oof.session_id.nunique()),
        "m3_oof_unique_vessels": int(oof.mmsi.nunique()),
        "m3_oof_rows": int(len(oof)),
        "cold_vessel_oof_calls": int(oof.loc[oof.cold_vessel_in_fold, "session_id"].nunique()),
        "locked_final_calls_scored": 0,
        "selected_core_model": selected,
        "candidate_under24_mae_h": candidate_u24,
        "route_physics_under24_mae_h": base_u24,
        "selected_under24_mae_h": sel_u24,
        "selected_under24_improvement_fraction": (base_u24 - sel_u24) / base_u24,
        "route_physics_all_oof_mae_h": base_all,
        "selected_all_oof_mae_h": sel_all,
        "cold_route_physics_under24_mae_h": cold_base,
        "cold_selected_under24_mae_h": cold_sel,
        "fold_wins": fold_wins,
        "fold_total": int(len(pf)),
        "top_positive_call_gain_share": top_share,
        "bootstrap": bootstrap,
        "port_state_gate": ps_gate,
        "feature_manifest": {"numeric": MODEL_NUMERIC_FEATURES, "categorical": MODEL_CATEGORICAL_FEATURES, "port_state": PORT_STATE_FEATURES},
        "gate": gate,
    }
    (REPORTS / "m3_summary.json").write_text(json.dumps(_jsonify(summary), indent=2), encoding="utf-8")
    (REPORTS / "m3_gate_result.json").write_text(json.dumps(_jsonify(gate), indent=2), encoding="utf-8")
    (REPORTS / "m3_port_state_gate.json").write_text(json.dumps(_jsonify(ps_gate), indent=2), encoding="utf-8")
    print(json.dumps(_jsonify(summary), indent=2))


if __name__ == "__main__":
    main()
