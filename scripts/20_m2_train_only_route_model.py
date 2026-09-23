#!/usr/bin/env python3
"""Run M2 train-only route modelling and OOF route-distance ablation."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m1 import HORIZON_LABELS  # noqa: E402
from ais_eta.m2 import (  # noqa: E402
    M2Config,
    add_horizon_band,
    assign_m2_splits,
    eta_metrics,
    evaluate_m2_gate,
    fit_train_only_route_model,
    predict_route_distances,
)

REPORTS = ROOT / "reports"
STATES_PATH = ROOT / "data" / "derived" / "m0c_ship_states.pkl.gz"

PRED_COLS = [
    "pred_m2_geodesic_h",
    "pred_m2_sector_prototype_h",
    "pred_m2_route_knn_h",
]


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
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    return value


def _voyage_start_lookup() -> dict[str, pd.Timestamp]:
    cat = pd.read_csv(REPORTS / "catania_m1_decision_panel.csv")
    aug = pd.read_csv(REPORTS / "augusta_m1x_decision_panel.csv")
    out = {}
    for frame in (cat, aug):
        frame["voyage_start_time"] = pd.to_datetime(frame["voyage_start_time"], errors="raise")
        out.update(frame.groupby("session_id")["voyage_start_time"].first().to_dict())
    return {str(k): pd.Timestamp(v) for k, v in out.items()}


def _normalized_decision_panel(states: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    cat = pd.read_csv(REPORTS / "catania_m1_decision_panel.csv")
    cat = cat.rename(columns={"distance_to_gate_mid_km": "geodesic_distance_km"})
    cat["port"] = "CATANIA"
    cat["inbound_entrance"] = "CATANIA"

    aug = pd.read_csv(REPORTS / "augusta_m1x_decision_panel.csv")
    aug = aug.rename(columns={"distance_to_target_gate_km": "geodesic_distance_km"})
    aug["port"] = "AUGUSTA"

    cols = [
        "port", "session_id", "mmsi", "name", "inbound_entrance",
        "ground_truth_time", "ground_truth_confidence", "decision_time",
        "source_observation_time", "true_tta_h", "geodesic_distance_km",
        "sog_median_30m_kn", "scope_inbound_approach",
    ]
    panel = pd.concat([cat[cols], aug[cols]], ignore_index=True)
    for c in ["ground_truth_time", "decision_time", "source_observation_time"]:
        panel[c] = pd.to_datetime(panel[c], errors="raise")

    state_pos = states[["mmsi", "recorded_at", "lon_clean", "lat_clean"]].copy()
    state_pos = state_pos.drop_duplicates(["mmsi", "recorded_at"])
    panel = panel.merge(
        state_pos,
        left_on=["mmsi", "source_observation_time"],
        right_on=["mmsi", "recorded_at"],
        how="left",
        validate="many_to_one",
    )
    panel = panel.drop(columns=["recorded_at"])
    if panel[["lon_clean", "lat_clean"]].isna().any().any():
        raise AssertionError("M2 decision panel lost source positions")

    attach = splits[["session_id", "m2_split", "m2_role", "m2_temporal_fold"]].copy()
    panel = panel.merge(attach, on="session_id", how="left", validate="many_to_one")
    if panel["m2_split"].isna().any():
        raise AssertionError("M2 split coverage mismatch")
    return add_horizon_band(panel)


def _metric_row(df: pd.DataFrame, pred: str) -> pd.Series:
    m = eta_metrics(df, [pred])
    if m.empty:
        raise ValueError(f"No metric rows for {pred}")
    return m.iloc[0]


def _group_metrics(oof: pd.DataFrame, group_cols: list[str], scope: str) -> pd.DataFrame:
    rows = []
    for keys, group in oof.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        if scope == "under24h":
            group = group[group["true_tta_h"].le(24.0)]
        elif scope == "under12h":
            group = group[group["true_tta_h"].le(12.0)]
        elif scope == "under6h":
            group = group[group["true_tta_h"].le(6.0)]
        if group.empty:
            continue
        for pred in PRED_COLS:
            m = eta_metrics(group, [pred])
            if m.empty:
                continue
            row = {c: k for c, k in zip(group_cols, keys)}
            row["scope"] = scope
            row.update(m.iloc[0].to_dict())
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    cfg = M2Config()
    states = pd.read_pickle(STATES_PATH)
    cat_cohort = pd.read_csv(REPORTS / "catania_m0e_eta_primary_cohort.csv")
    aug_cohort = pd.read_csv(REPORTS / "augusta_m1x_eta_primary_cohort.csv")
    cat_m1 = pd.read_csv(REPORTS / "catania_m1_call_splits.csv")
    splits = assign_m2_splits(cat_cohort, aug_cohort, cat_m1, cfg)
    splits.to_csv(REPORTS / "m2_call_splits.csv", index=False)

    voyage_starts = _voyage_start_lookup()
    missing_starts = sorted(set(splits["session_id"].astype(str)) - set(voyage_starts))
    if missing_starts:
        raise AssertionError(f"Missing voyage starts for {missing_starts}")

    panel = _normalized_decision_panel(states, splits)
    approach = panel[panel["scope_inbound_approach"].astype(bool)].copy()
    final_ids = set(splits.loc[splits["m2_split"].eq("final_chronological_test"), "session_id"].astype(str))

    models: dict[str, dict] = {}
    rows: list[dict] = []
    fold_training_manifest = []

    for port in ["CATANIA", "AUGUSTA"]:
        port_calls = splits[(splits["port"].eq(port)) & splits["m2_split"].eq("development")].copy()
        for fold in range(1, cfg.temporal_folds + 1):
            val_calls = port_calls[port_calls["m2_temporal_fold"].eq(fold)].copy()
            if val_calls.empty:
                continue
            validation_start = pd.Timestamp(val_calls["ground_truth_time"].min())
            train_calls = port_calls[pd.to_datetime(port_calls["ground_truth_time"]) < validation_start].copy()
            if set(train_calls["session_id"]) & set(val_calls["session_id"]):
                raise AssertionError("M2 train/validation session overlap")
            if set(train_calls["session_id"]) & final_ids:
                raise AssertionError("M2 final holdout leaked into route training")

            model = fit_train_only_route_model(states, train_calls, voyage_starts, cfg)
            model_key = f"{port}_fold{fold}"
            models[model_key] = model
            train_vessels = set(train_calls["mmsi"].astype(int))
            fold_training_manifest.append({
                "port": port,
                "m2_temporal_fold": fold,
                "validation_start": validation_start,
                "validation_calls": int(len(val_calls)),
                "validation_unique_vessels": int(val_calls["mmsi"].astype(int).nunique()),
                "training_calls": int(len(train_calls)),
                "training_unique_vessels": int(len(train_vessels)),
                "training_max_arrival": pd.Timestamp(train_calls["ground_truth_time"].max()),
                "family_counts": json.dumps(model["family_counts"], sort_keys=True),
            })

            val_panel = approach[
                approach["port"].eq(port)
                & approach["session_id"].isin(val_calls["session_id"])
            ].copy()
            for obs in val_panel.itertuples(index=False):
                route = predict_route_distances(model, float(obs.lon_clean), float(obs.lat_clean), cfg)
                speed = max(float(obs.sog_median_30m_kn), cfg.speed_floor_kn)
                geo_km = float(obs.geodesic_distance_km)
                # Independent geodesic calculation inside the route model is a
                # useful contract check against panel geometry.
                if abs(route["geodesic_km"] - geo_km) > 0.25:
                    raise AssertionError(
                        f"Geodesic mismatch {obs.session_id}: {route['geodesic_km']} vs {geo_km}"
                    )
                rows.append({
                    "port": port,
                    "session_id": str(obs.session_id),
                    "mmsi": int(obs.mmsi),
                    "name": obs.name,
                    "m2_temporal_fold": fold,
                    "ground_truth_time": pd.Timestamp(obs.ground_truth_time),
                    "decision_time": pd.Timestamp(obs.decision_time),
                    "source_observation_time": pd.Timestamp(obs.source_observation_time),
                    "true_tta_h": float(obs.true_tta_h),
                    "horizon_band": str(obs.horizon_band),
                    "sog_median_30m_kn": float(obs.sog_median_30m_kn),
                    "lon_clean": float(obs.lon_clean),
                    "lat_clean": float(obs.lat_clean),
                    "geodesic_distance_km": geo_km,
                    "prototype_route_km": route["prototype_route_km"],
                    "prototype_cross_track_km": route["prototype_cross_track_km"],
                    "route_knn_distance_km": route["knn_route_km"],
                    "knn_mean_cross_track_km": route["knn_mean_cross_track_km"],
                    "route_family": route["current_route_family"],
                    "prototype_kind": route["prototype_kind"],
                    "knn_neighbor_sessions": route["knn_neighbor_sessions"],
                    "knn_neighbor_mmsi": route["knn_neighbor_mmsi"],
                    "cold_vessel_in_fold": int(obs.mmsi) not in train_vessels,
                    "pred_m2_geodesic_h": (geo_km / 1.852) / speed,
                    "pred_m2_sector_prototype_h": (route["prototype_route_km"] / 1.852) / speed,
                    "pred_m2_route_knn_h": (route["knn_route_km"] / 1.852) / speed,
                })

    oof = pd.DataFrame(rows)
    if oof.empty:
        raise AssertionError("M2 produced no OOF predictions")
    if set(oof["session_id"]) & final_ids:
        raise AssertionError("Final chronological holdout was scored in M2")
    oof = add_horizon_band(oof)
    oof.to_csv(REPORTS / "m2_oof_route_predictions.csv", index=False)
    pd.DataFrame(fold_training_manifest).to_csv(REPORTS / "m2_fold_training_manifest.csv", index=False)
    (REPORTS / "m2_route_models.json").write_text(json.dumps(_jsonify(models), indent=2), encoding="utf-8")

    overall = eta_metrics(oof, PRED_COLS)
    overall["scope"] = "all_oof"
    under6 = eta_metrics(oof[oof["true_tta_h"].le(6.0)], PRED_COLS); under6["scope"] = "under6h"
    under12 = eta_metrics(oof[oof["true_tta_h"].le(12.0)], PRED_COLS); under12["scope"] = "under12h"
    under24 = eta_metrics(oof[oof["true_tta_h"].le(24.0)], PRED_COLS); under24["scope"] = "under24h"
    metrics = pd.concat([overall, under6, under12, under24], ignore_index=True)
    metrics.to_csv(REPORTS / "m2_eta_metrics.csv", index=False)

    fold_metrics = pd.concat(
        [_group_metrics(oof, ["port", "m2_temporal_fold"], s) for s in ["all", "under6h", "under12h", "under24h"]],
        ignore_index=True,
    )
    fold_metrics.to_csv(REPORTS / "m2_fold_metrics.csv", index=False)
    port_metrics = pd.concat(
        [_group_metrics(oof, ["port"], s) for s in ["all", "under6h", "under12h", "under24h"]],
        ignore_index=True,
    )
    port_metrics.to_csv(REPORTS / "m2_port_metrics.csv", index=False)

    horizon_rows = []
    for band in HORIZON_LABELS:
        q = oof[oof["horizon_band"].astype(str).eq(band)]
        if q.empty:
            continue
        mm = eta_metrics(q, PRED_COLS)
        mm["horizon_band"] = band
        horizon_rows.append(mm)
    horizon_metrics = pd.concat(horizon_rows, ignore_index=True) if horizon_rows else pd.DataFrame()
    horizon_metrics.to_csv(REPORTS / "m2_horizon_metrics.csv", index=False)

    cold = oof[oof["cold_vessel_in_fold"].astype(bool)].copy()
    cold_all = eta_metrics(cold, PRED_COLS); cold_all["scope"] = "cold_all"
    cold24 = eta_metrics(cold[cold["true_tta_h"].le(24.0)], PRED_COLS); cold24["scope"] = "cold_under24h"
    cold_metrics = pd.concat([cold_all, cold24], ignore_index=True)
    cold_metrics.to_csv(REPORTS / "m2_cold_vessel_metrics.csv", index=False)

    # Per-call diagnostics and route-distance inflation relative to the straight line.
    diag = (
        oof.groupby(["port", "session_id", "mmsi", "name"], as_index=False)
        .agg(
            points=("decision_time", "size"),
            max_horizon_h=("true_tta_h", "max"),
            cold_vessel=("cold_vessel_in_fold", "max"),
            median_route_ratio=("route_knn_distance_km", lambda s: float(np.median(s / oof.loc[s.index, "geodesic_distance_km"]))),
            median_knn_cross_track_km=("knn_mean_cross_track_km", "median"),
        )
    )
    for pred, short in [
        ("pred_m2_geodesic_h", "geo"),
        ("pred_m2_sector_prototype_h", "prototype"),
        ("pred_m2_route_knn_h", "knn"),
    ]:
        x = oof.copy(); x["ae"] = (x[pred] - x["true_tta_h"]).abs()
        cm = x.groupby("session_id")["ae"].mean()
        diag[f"mae_{short}_h"] = diag["session_id"].map(cm)
    diag.to_csv(REPORTS / "m2_call_diagnostics.csv", index=False)

    def get_metric(frame: pd.DataFrame, scope: str, pred: str) -> float:
        return float(frame[(frame["scope"].eq(scope)) & frame["baseline"].eq(pred)]["voyage_balanced_mae_h"].iloc[0])

    geo_overall = get_metric(metrics, "all_oof", "pred_m2_geodesic_h")
    knn_overall = get_metric(metrics, "all_oof", "pred_m2_route_knn_h")
    geo24 = get_metric(metrics, "under24h", "pred_m2_geodesic_h")
    knn24 = get_metric(metrics, "under24h", "pred_m2_route_knn_h")
    geo_cold24 = get_metric(cold_metrics, "cold_under24h", "pred_m2_geodesic_h")
    knn_cold24 = get_metric(cold_metrics, "cold_under24h", "pred_m2_route_knn_h")

    all_fold = fold_metrics[fold_metrics["scope"].eq("all")]
    pivot = all_fold.pivot_table(
        index=["port", "m2_temporal_fold"], columns="baseline", values="voyage_balanced_mae_h"
    )
    pivot = pivot.dropna(subset=["pred_m2_geodesic_h", "pred_m2_route_knn_h"])
    fold_wins = int((pivot["pred_m2_route_knn_h"] < pivot["pred_m2_geodesic_h"]).sum())
    fold_total = int(len(pivot))

    gate = evaluate_m2_gate(
        geo_overall,
        knn_overall,
        geo24,
        knn24,
        fold_wins,
        fold_total,
        geo_cold24,
        knn_cold24,
    )
    (REPORTS / "m2_gate_result.json").write_text(json.dumps(_jsonify(gate), indent=2), encoding="utf-8")

    summary = {
        "design": {
            "primary_evaluation": "strict expanding temporal OOF by port",
            "catania_final_holdout_preserved": int((splits["port"].eq("CATANIA") & splits["m2_split"].eq("final_chronological_test")).sum()),
            "augusta_final_holdout_locked": int((splits["port"].eq("AUGUSTA") & splits["m2_split"].eq("final_chronological_test")).sum()),
            "temporal_folds_per_port": cfg.temporal_folds,
            "knn_k": cfg.knn_k,
            "route_family_sectors": "NNE[0,60), E[60,120), SSE[120,180), OTHER",
        },
        "oof": {
            "prediction_points": int(len(oof)),
            "calls": int(oof["session_id"].nunique()),
            "unique_vessels": int(oof["mmsi"].nunique()),
            "cold_vessel_calls": int(oof.loc[oof["cold_vessel_in_fold"], "session_id"].nunique()),
            "final_holdout_rows_scored": 0,
        },
        "overall_mae_h": {
            "geodesic": geo_overall,
            "sector_prototype": get_metric(metrics, "all_oof", "pred_m2_sector_prototype_h"),
            "route_knn": knn_overall,
        },
        "under24_mae_h": {
            "geodesic": geo24,
            "sector_prototype": get_metric(metrics, "under24h", "pred_m2_sector_prototype_h"),
            "route_knn": knn24,
        },
        "cold_under24_mae_h": {
            "geodesic": geo_cold24,
            "route_knn": knn_cold24,
        },
        "route_distance": {
            "median_knn_to_geodesic_ratio": float(np.median(oof["route_knn_distance_km"] / oof["geodesic_distance_km"])),
            "median_knn_cross_track_km": float(oof["knn_mean_cross_track_km"].median()),
        },
        "fold_consistency": {
            "knn_wins": fold_wins,
            "folds": fold_total,
        },
        "gate": gate,
        "interpretation": (
            "Route geometry adds reproducible train-only signal in the navigation-dominated <=24 h regime, "
            "including cold-vessel calls, while very-long-horizon errors remain dominated by waiting/behaviour "
            "that route distance alone cannot explain. Retain geodesic as a baseline and route-kNN as an M3 "
            "feature/alternative physics distance; do not claim a nautical ground-truth route."
        ),
    }
    (REPORTS / "m2_summary.json").write_text(json.dumps(_jsonify(summary), indent=2), encoding="utf-8")
    print(json.dumps(_jsonify(summary), indent=2))


if __name__ == "__main__":
    main()
