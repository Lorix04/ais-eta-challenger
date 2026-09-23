#!/usr/bin/env python3
"""Full-data M2 verification and train-only leakage audit."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def check(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)
    print(f"PASS {message}")


def main() -> None:
    splits = pd.read_csv(REPORTS / "m2_call_splits.csv")
    oof = pd.read_csv(REPORTS / "m2_oof_route_predictions.csv")
    manifest = pd.read_csv(REPORTS / "m2_fold_training_manifest.csv")
    models = json.loads((REPORTS / "m2_route_models.json").read_text())
    gate = json.loads((REPORTS / "m2_gate_result.json").read_text())
    summary = json.loads((REPORTS / "m2_summary.json").read_text())

    splits["ground_truth_time"] = pd.to_datetime(splits["ground_truth_time"], errors="raise")
    oof["ground_truth_time"] = pd.to_datetime(oof["ground_truth_time"], errors="raise")
    manifest["validation_start"] = pd.to_datetime(manifest["validation_start"], errors="raise")
    manifest["training_max_arrival"] = pd.to_datetime(manifest["training_max_arrival"], errors="raise")

    check(len(splits) == 83, "83 combined ETA calls frozen in M2 split table")
    check(int((splits.m2_split == "final_chronological_test").sum()) == 16, "16 final chronological calls locked")
    check(int((splits.m2_role == "oof_validation").sum()) == 44, "44 calls assigned to expanding OOF validation blocks")
    check(int((splits.m2_role == "warmup_train").sum()) == 23, "23 calls reserved as route warm-up history")

    final_ids = set(splits.loc[splits.m2_split.eq("final_chronological_test"), "session_id"].astype(str))
    check(not (set(oof.session_id.astype(str)) & final_ids), "final chronological holdout never scored in M2")
    check(len(oof) == 556, "556 causal OOF approach prediction points produced")
    check(oof.session_id.nunique() == 40, "40 independent calls contribute M2 OOF route evaluation")
    check(oof.mmsi.nunique() == 33, "33 unique vessels contribute M2 OOF route evaluation")
    check(oof.loc[oof.cold_vessel_in_fold.astype(bool), "session_id"].nunique() == 28, "28 OOF calls are cold-vessel relative to their fold")

    check((manifest.training_max_arrival < manifest.validation_start).all(), "every fold route model uses only earlier-arriving calls")
    check(len(manifest) == 6, "six port x expanding-fold route models fitted")

    split_lookup = splits.set_index("session_id")
    for key, model in models.items():
        port, fold_text = key.rsplit("_fold", 1)
        fold = int(fold_text)
        train_ids = set(model["training_call_ids"])
        val_ids = set(
            splits.loc[
                splits.port.eq(port)
                & splits.m2_role.eq("oof_validation")
                & splits.m2_temporal_fold.eq(fold),
                "session_id",
            ].astype(str)
        )
        check(not (train_ids & val_ids), f"{key} has no train/validation call overlap")
        check(not (train_ids & final_ids), f"{key} excludes all locked final calls")
        val_start = split_lookup.loc[list(val_ids), "ground_truth_time"].min()
        train_max = split_lookup.loc[list(train_ids), "ground_truth_time"].max()
        check(pd.Timestamp(train_max) < pd.Timestamp(val_start), f"{key} training history is strictly earlier than validation")

    check((oof.route_knn_distance_km + 1e-9 >= oof.geodesic_distance_km).all(), "route-kNN distance never claims shorter-than-geodesic travel")
    check((oof.prototype_route_km + 1e-9 >= oof.geodesic_distance_km).all(), "prototype distance never claims shorter-than-geodesic travel")
    check(np.isfinite(oof[["pred_m2_geodesic_h", "pred_m2_sector_prototype_h", "pred_m2_route_knn_h"]].to_numpy()).all(), "all M2 OOF ETA predictions are finite")

    # Neighbor IDs must come from the corresponding train-only route model.
    for row in oof.itertuples(index=False):
        model = models[f"{row.port}_fold{int(row.m2_temporal_fold)}"]
        train_ids = set(model["training_call_ids"])
        neighbors = set(str(row.knn_neighbor_sessions).split("|"))
        if not neighbors <= train_ids:
            raise AssertionError(f"route-kNN neighbor leakage for {row.session_id}: {neighbors-train_ids}")
        if str(row.session_id) in neighbors:
            raise AssertionError(f"self-neighbor leakage for {row.session_id}")
    print("PASS every route-kNN neighbor belongs to the fold training history and never to self")

    check(gate["status"] == "GO_M3_ROUTE_SIGNAL_CONFIRMED", "pre-specified M2 route-value gate passed")
    check(gate["observed"]["under24_mae_improvement_fraction"] >= 0.10, "route-kNN improves <=24h voyage-balanced MAE by at least 10%")
    check(gate["observed"]["cold_under24_mae_improvement_fraction"] >= 0.05, "route-kNN improves cold-vessel <=24h MAE")
    check(gate["observed"]["fold_wins"] == gate["observed"]["fold_total"] == 6, "route-kNN wins all six port x temporal folds on all-horizon MAE")
    check(summary["oof"]["final_holdout_rows_scored"] == 0, "summary records zero final-holdout rows scored")

    print("M2 verification complete")


if __name__ == "__main__":
    main()
