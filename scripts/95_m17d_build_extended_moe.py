#!/usr/bin/env python3
"""Build M17D: extend frozen M16G MoE with M17A learned retrieval + M17C graph signal.

The outer-validation rows are never used to fit/select the meta learner.  Inside
an outer-train partition all six base experts are regenerated inner-OOF before
meta selection.  M17A representation learning is also restricted to the
current inner-train MMSIs.  M17C graph construction excludes both outer-valid
and inner-valid MMSI owners while retaining target-free unlabeled AIS owners.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m16a import assert_development_only, balanced_hash_folds, extended_metrics
from ais_eta.m16c import add_m16c_features, candidate_configs as c_configs, predict_hierarchical_prior
from ais_eta.m16d import candidate_configs as d_configs, predict_route_analogue
from ais_eta.m16e import candidate_configs as e_configs, predict_physics_expert
from ais_eta.m16f import candidate_configs as f_configs, fit_predict
from ais_eta.m17a import (
    M17A_RETRIEVAL_CONFIG, M17A_SEED, build_causal_pretraining_windows,
    cosine_distance_matrix, train_ssl_encoder,
)
from ais_eta.m17c import prepare_transition_events, predict_corridor_eta
from ais_eta.m17d import (
    M17D_EXPERT_NAMES, M17D_INNER_FOLDS, M17D_INNER_SALT, M17D_META_CANDIDATES,
    M17D_VERSION, build_gating_features, fit_predict_meta, promotion_gate,
    score_candidates_inner_cv,
)

REF = ROOT / "data/derived/m10_reference_rows.pkl.gz"
STATES = ROOT / "data/derived/m0c_ship_states.pkl.gz"
R = ROOT / "reports"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_abs(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, dtype=float) - np.asarray(p, dtype=float))))


def prepare_development() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_pickle(REF)
    final = raw.loc[raw["split"].eq("final_test")].copy()
    dev = raw.loc[raw["split"].isin(["train", "calibration"])].copy()
    if len(dev) != 386 or len(final) != 53:
        raise AssertionError("unexpected M17D development/final cardinality")
    assert_development_only(dev, final["mmsi"])

    b = pd.read_csv(R / "m16b_destination_resolutions.csv")
    e = pd.read_csv(R / "m16e_physics_feature_audit.csv")
    a = pd.read_csv(R / "m16a_oof_predictions.csv")
    d = pd.read_csv(R / "m16d_oof_predictions.csv")
    bcols = [
        "mmsi", "canonical_destination", "canonical_port_name", "canonical_lat", "canonical_lon",
        "resolution_method", "resolution_confidence", "is_resolved_port", "is_non_specific", "is_route_expression",
    ]
    ecols = [
        "mmsi", "physics_distance_gc_nm", "physics_bearing_to_port_deg", "physics_course_alignment_deg",
        "physics_recent_speed_max_kn", "physics_local_sinuosity_180m", "physics_local_sinuosity_360m",
        "physics_eligible",
    ]
    dev = dev.merge(b[bcols], on="mmsi", how="left")
    dev = dev.merge(e[ecols], on="mmsi", how="left")
    dev = dev.merge(
        a[["mmsi", "m16a_outer_fold", "pred_train_destination_median_h", "pred_catboost_current_snapshot_h"]],
        on="mmsi", how="left",
    )
    dev = dev.merge(d[["mmsi", "pred_m16d_route_analogue_h"]], on="mmsi", how="left")
    dev = dev.sort_values("mmsi", kind="mergesort").reset_index(drop=True)
    dev = add_m16c_features(dev)
    if dev["m16a_outer_fold"].isna().any():
        raise AssertionError("M17D missing frozen M16A outer fold")
    return dev, final


def _config_maps():
    return (
        {c.config_id: c for c in c_configs()},
        {c.config_id: c for c in d_configs()},
        {c.config_id: c for c in e_configs()},
        {c.config_id: c for c in f_configs()},
    )


def _selected_configs(outer: int):
    cm, dm, em, fm = _config_maps()
    sc = pd.read_csv(R / "m16c_outer_selected_configs.csv")
    sd = pd.read_csv(R / "m16d_outer_selected_configs.csv")
    se = pd.read_csv(R / "m16e_outer_selected_configs.csv")
    sf = pd.read_csv(R / "m16f_outer_selected_configs.csv")
    cid = str(sc.loc[sc["outer_fold"].eq(outer), "selected_config_id"].iloc[0])
    did = str(sd.loc[sd["outer_fold"].eq(outer), "selected_config_id"].iloc[0])
    eid = str(se.loc[se["outer_fold"].eq(outer), "selected_config_id"].iloc[0])
    fid = str(sf.loc[sf["outer_fold"].eq(outer) & sf["family"].eq("hist_gb"), "selected_config_id"].iloc[0])
    return cm[cid], dm[did], em[eid], fm[fid]


def _quality_m16_inner(valid: pd.DataFrame, prior: pd.DataFrame, route: pd.DataFrame, physics: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "m16c_deepest_support": prior["deepest_support"].to_numpy(float),
        "m16c_deepest_weight": prior["deepest_weight"].to_numpy(float),
        "m16c_fallback_count": prior["fallback_count"].to_numpy(float),
        "m16d_neighbour_count": route["neighbour_count"].to_numpy(float),
        "m16d_nearest_distance": route["nearest_distance"].to_numpy(float),
        "m16d_median_neighbour_distance": route["median_neighbour_distance"].to_numpy(float),
        "m16d_similarity_gap": route["similarity_gap"].to_numpy(float),
        "m16d_gate_used": route["gate_used"].astype(str).to_numpy(),
        "physics_eligible": valid["physics_eligible"].astype(bool).to_numpy(),
        "resolution_confidence": pd.to_numeric(valid["resolution_confidence"], errors="coerce").to_numpy(float),
        "physics_distance_gc_nm": pd.to_numeric(valid["physics_distance_gc_nm"], errors="coerce").to_numpy(float),
        "physics_course_alignment_deg": pd.to_numeric(valid["physics_course_alignment_deg"], errors="coerce").to_numpy(float),
        "physics_recent_speed_max_kn": pd.to_numeric(valid["physics_recent_speed_max_kn"], errors="coerce").to_numpy(float),
        "m16e_effective_speed_kn": physics["effective_speed_kn"].to_numpy(float),
        "m16e_distance_factor": physics["distance_factor"].to_numpy(float),
        "m16e_route_distance_proxy_nm": physics["route_distance_proxy_nm"].to_numpy(float),
    })


def _add_learned_quality(q: pd.DataFrame, learned: pd.DataFrame) -> pd.DataFrame:
    out = q.copy()
    out["m17a_neighbour_count"] = learned["neighbour_count"].to_numpy(float)
    out["m17a_nearest_distance"] = learned["nearest_distance"].to_numpy(float)
    out["m17a_median_neighbour_distance"] = learned["median_neighbour_distance"].to_numpy(float)
    out["m17a_similarity_gap"] = learned["similarity_gap"].to_numpy(float)
    out["m17a_gate_used"] = learned["gate_used"].astype(str).to_numpy()
    return out


def _empty_graph_quality(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        "m17c_graph_supported": np.zeros(n, dtype=float),
        "m17c_path_edges": np.zeros(n, dtype=float),
        "m17c_min_path_support": np.zeros(n, dtype=float),
        "m17c_median_path_support": np.zeros(n, dtype=float),
        "m17c_contextual_edge_share": np.zeros(n, dtype=float),
        "m17c_source_snap_nm": np.zeros(n, dtype=float),
        "m17c_destination_snap_nm": np.zeros(n, dtype=float),
    })


def _graph_gate_inner(
    valid: pd.DataFrame,
    physics_gate: np.ndarray,
    fold_events: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame, int]:
    pred = np.asarray(physics_gate, dtype=float).copy()
    q = _empty_graph_quality(len(valid))
    supported = 0
    for j, row in enumerate(valid.itertuples(index=False)):
        if not bool(row.physics_eligible):
            continue
        g = predict_corridor_eta(
            fold_events,
            query_mmsi=int(row.mmsi), query_at=pd.Timestamp(row.last_update),
            query_lat=float(row.track_lat), query_lon=float(row.track_lon),
            destination_lat=float(row.canonical_lat), destination_lon=float(row.canonical_lon),
            query_ship_type=str(row.ship_type_cat), recent_speed_kn=float(row.physics_recent_speed_max_kn),
        )
        if not g.supported:
            continue
        supported += 1
        pred[j] = float(g.prediction_h)
        q.loc[j, "m17c_graph_supported"] = 1.0
        q.loc[j, "m17c_path_edges"] = float(g.path_edges)
        q.loc[j, "m17c_min_path_support"] = float(g.min_path_support)
        q.loc[j, "m17c_median_path_support"] = float(g.median_path_support)
        q.loc[j, "m17c_contextual_edge_share"] = float(g.contextual_edge_share)
        q.loc[j, "m17c_source_snap_nm"] = float(g.source_snap_nm)
        q.loc[j, "m17c_destination_snap_nm"] = float(g.destination_snap_nm)
    return pred, q, supported


def _frozen_outer_tables(dev: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    g = pd.read_csv(R / "m16g_oof_predictions.csv")
    c = pd.read_csv(R / "m16c_oof_predictions.csv")
    d = pd.read_csv(R / "m16d_oof_predictions.csv")
    e = pd.read_csv(R / "m16e_oof_predictions.csv")
    a = pd.read_csv(R / "m17a_oof_predictions.csv")
    ac = pd.read_csv(R / "m17a_neighbour_details.csv")
    ac = ac.loc[ac["method"].eq("ssl_masked_contrastive_cosine")].copy()
    z = pd.read_csv(R / "m17c_oof_predictions.csv")
    base = (
        g[["mmsi", "pred_m16c_prior_h", "pred_m16d_route_h", "pred_m16e_physics_gate_h", "pred_m16f_tabular_h", "pred_m16g_selected_h"]]
        .merge(a[["mmsi", "pred_m17a_ssl_h"]], on="mmsi")
        .merge(z[["mmsi", "pred_m17c_graph_gate_m16e_fallback_h"]], on="mmsi")
    )
    quality = (
        c[["mmsi", "m16c_deepest_support", "m16c_deepest_weight", "m16c_fallback_count"]]
        .merge(d[["mmsi", "m16d_neighbour_count", "m16d_nearest_distance", "m16d_median_neighbour_distance", "m16d_similarity_gap", "m16d_gate_used"]], on="mmsi")
        .merge(e[["mmsi", "physics_eligible", "resolution_confidence", "physics_distance_gc_nm", "physics_course_alignment_deg", "physics_recent_speed_max_kn", "m16e_effective_speed_kn", "m16e_distance_factor", "m16e_route_distance_proxy_nm"]], on="mmsi")
        .merge(ac[["query_mmsi", "neighbour_count", "nearest_distance", "median_neighbour_distance", "similarity_gap", "gate_used"]].rename(columns={
            "query_mmsi":"mmsi", "neighbour_count":"m17a_neighbour_count", "nearest_distance":"m17a_nearest_distance",
            "median_neighbour_distance":"m17a_median_neighbour_distance", "similarity_gap":"m17a_similarity_gap", "gate_used":"m17a_gate_used",
        }), on="mmsi", how="left")
        .merge(z[["mmsi", "m17c_graph_supported", "m17c_path_edges", "m17c_min_path_support", "m17c_median_path_support", "m17c_contextual_edge_share", "m17c_source_snap_nm", "m17c_destination_snap_nm"]], on="mmsi", how="left")
    )
    base = dev[["mmsi"]].merge(base, on="mmsi", how="left")
    quality = dev[["mmsi"]].merge(quality, on="mmsi", how="left")
    mat = np.column_stack([
        base["pred_m16c_prior_h"].to_numpy(float),
        base["pred_m16d_route_h"].to_numpy(float),
        base["pred_m16e_physics_gate_h"].to_numpy(float),
        base["pred_m16f_tabular_h"].to_numpy(float),
        base["pred_m17a_ssl_h"].to_numpy(float),
        base["pred_m17c_graph_gate_m16e_fallback_h"].to_numpy(float),
    ])
    return mat, quality.drop(columns=["mmsi"]), base["pred_m16g_selected_h"].to_numpy(float)


def _prepare_ssl_and_graph(dev: pd.DataFrame, final: pd.DataFrame):
    states = pd.read_pickle(STATES)
    states["recorded_at"] = pd.to_datetime(states["recorded_at"])
    dev_states = states.loc[states["mmsi"].isin(dev["mmsi"])].copy()
    windows, owners, _ = build_causal_pretraining_windows(dev_states, dev)
    seq = np.asarray(np.load(R / "m16d_route_sequences.npz")["geo_kin_6h"], dtype=np.float32)
    owner_ship_type = dict(zip(dev["mmsi"].astype(int), dev["ship_type_cat"].astype(str)))
    transitions = prepare_transition_events(states, owner_ship_type=owner_ship_type)
    final_set = set(final["mmsi"].astype(int))
    transitions = transitions.loc[~transitions["mmsi"].astype(int).isin(final_set)].copy()
    if set(transitions["mmsi"].astype(int)) & final_set:
        raise AssertionError("old-final MMSI leaked into M17D graph transitions")
    return windows, owners, seq, transitions


def run_worker(outer: int, worker_dir: Path) -> int:
    worker_dir.mkdir(parents=True, exist_ok=True)
    dev, final = prepare_development()
    assert_development_only(dev, final["mmsi"])
    y = dev["target_tte_h"].to_numpy(float)
    mmsis = dev["mmsi"].to_numpy(int)
    folds = dev["m16a_outer_fold"].to_numpy(int)
    destinations = dev["canonical_destination"].fillna("UNKNOWN").astype(str).to_numpy()
    matrices = dict(np.load(R / "m16d_route_distance_matrices.npz"))
    cc, dc, ec, fc = _selected_configs(outer)
    windows, owners, query_sequences, transitions = _prepare_ssl_and_graph(dev, final)

    outer_train = np.flatnonzero(folds != outer)
    outer_valid = np.flatnonzero(folds == outer)
    outer_valid_mmsi = set(int(x) for x in mmsis[outer_valid])
    if set(mmsis[outer_train]) & outer_valid_mmsi:
        raise AssertionError("M17D outer train/valid overlap")

    inner_map = balanced_hash_folds(
        mmsis[outer_train], n_splits=M17D_INNER_FOLDS,
        salt=f"{M17D_INNER_SALT}:outer={outer}",
    )
    inner_fold = np.asarray([inner_map[int(x)] for x in mmsis[outer_train]], dtype=int)
    train_base = np.full((len(outer_train), len(M17D_EXPERT_NAMES)), np.nan, dtype=float)
    quality_parts: list[pd.DataFrame] = []
    quality_local_positions: list[np.ndarray] = []
    ssl_audit: list[dict] = []
    graph_audit: list[dict] = []

    for inner in range(M17D_INNER_FOLDS):
        va_local = np.flatnonzero(inner_fold == inner)
        tr_local = np.flatnonzero(inner_fold != inner)
        va_idx = outer_train[va_local]
        tr_idx = outer_train[tr_local]
        tr = dev.iloc[tr_idx].copy()
        va = dev.iloc[va_idx].copy()
        tr_mmsi = set(int(x) for x in mmsis[tr_idx])
        va_mmsi = set(int(x) for x in mmsis[va_idx])

        prior = predict_hierarchical_prior(tr, va, cc)
        route = predict_route_analogue(
            train_indices=tr_idx, valid_indices=va_idx,
            distance_matrix=matrices[dc.representation], targets=y,
            destinations=destinations, config=dc,
        )
        physics = predict_physics_expert(tr, va, ec)
        tabular = fit_predict(tr, va, fc)
        route_pred = route["prediction_h"].to_numpy(float)
        physics_pred = physics["prediction_h"].to_numpy(float)
        eligible = va["physics_eligible"].astype(bool).to_numpy()
        physics_gate = np.where(eligible & np.isfinite(physics_pred), physics_pred, route_pred)

        corpus_mask = np.asarray([int(x) in tr_mmsi for x in owners], dtype=bool)
        if any(int(x) in va_mmsi or int(x) in outer_valid_mmsi for x in owners[corpus_mask]):
            raise AssertionError("M17D validation owner leaked into SSL pretraining")
        tr_windows = windows[corpus_mask]
        seed = M17A_SEED + 100 * int(outer) + int(inner)
        ssl = train_ssl_encoder(tr_windows, query_sequences, seed=seed)
        ssl_dist = cosine_distance_matrix(ssl.embeddings)
        learned = predict_route_analogue(
            train_indices=tr_idx, valid_indices=va_idx, distance_matrix=ssl_dist,
            targets=y, destinations=destinations, config=M17A_RETRIEVAL_CONFIG,
        )
        learned_pred = learned["prediction_h"].to_numpy(float)
        ssl_audit.append({
            "outer_fold": outer, "inner_fold": inner,
            "pretraining_windows": int(len(tr_windows)), "pretraining_owners": int(len(tr_mmsi)),
            "inner_valid_owners": int(len(va_mmsi)), "inner_valid_used": False,
            "train_loss_start": float(ssl.train_loss_start), "train_loss_end": float(ssl.train_loss_end),
        })

        blocked = outer_valid_mmsi | va_mmsi
        fold_events = transitions.loc[~transitions["mmsi"].astype(int).isin(blocked)].copy()
        overlap = len(set(fold_events["mmsi"].astype(int)) & blocked)
        if overlap:
            raise AssertionError("M17D graph contains validation owner")
        graph_gate, graph_q, graph_supported = _graph_gate_inner(va, physics_gate, fold_events)
        graph_audit.append({
            "outer_fold": outer, "inner_fold": inner,
            "events_available": int(len(fold_events)), "event_owners": int(fold_events["mmsi"].nunique()),
            "inner_valid_owner_overlap": int(overlap), "supported_rows": int(graph_supported),
        })

        train_base[va_local, :] = np.column_stack([
            prior["prediction_h"].to_numpy(float), route_pred, physics_gate,
            tabular, learned_pred, graph_gate,
        ])
        q = _quality_m16_inner(va, prior, route, physics)
        q = _add_learned_quality(q, learned)
        for col in graph_q.columns:
            q[col] = graph_q[col].to_numpy()
        quality_parts.append(q)
        quality_local_positions.append(va_local)

    if not np.isfinite(train_base).all():
        raise AssertionError("M17D inner-OOF expert matrix contains non-finite values")
    train_quality = pd.DataFrame(index=np.arange(len(outer_train)))
    for pos, part in zip(quality_local_positions, quality_parts):
        part = part.reset_index(drop=True)
        for col in part.columns:
            train_quality.loc[pos, col] = part[col].to_numpy()
    train_features = build_gating_features(train_base, train_quality)

    frozen_base, frozen_quality, frozen_m16g = _frozen_outer_tables(dev)
    valid_base = frozen_base[outer_valid, :]
    valid_quality = frozen_quality.iloc[outer_valid].reset_index(drop=True)
    valid_features = build_gating_features(valid_base, valid_quality)
    train_features, valid_features = train_features.align(valid_features, join="outer", axis=1, fill_value=0.0)

    search = score_candidates_inner_cv(train_base, y[outer_train], train_features, inner_fold)
    selected_candidate = str(search.iloc[0]["candidate_id"])
    selected = fit_predict_meta(
        selected_candidate, train_base, y[outer_train], train_features,
        valid_base, valid_features,
    )

    pred = dev.iloc[outer_valid][["mmsi", "target_tte_h", "m16a_outer_fold"]].copy().reset_index(drop=True)
    for j, name in enumerate(M17D_EXPERT_NAMES):
        pred[f"pred_m17d_expert_{name}_h"] = valid_base[:, j]
    pred["pred_m17d_selected_h"] = selected.prediction
    pred["m17d_selected_meta_id"] = selected_candidate
    pred["m17d_selected_expert"] = selected.selected_expert
    pred["pred_m16g_frozen_h"] = frozen_m16g[outer_valid]

    for candidate in M17D_META_CANDIDATES:
        r = fit_predict_meta(candidate, train_base, y[outer_train], train_features, valid_base, valid_features)
        pred[f"pred_m17d_{candidate}_h"] = r.prediction

    search.insert(0, "outer_fold", outer)
    search["selected"] = search["candidate_id"].eq(selected_candidate)
    selected_row = {
        "outer_fold": outer, "selected_meta_id": selected_candidate,
        "inner_best_mae_h": float(search.iloc[0]["mae_h"]),
        "inner_best_p90_ae_h": float(search.iloc[0]["p90_ae_h"]),
        "outer_train_n": int(len(outer_train)), "outer_valid_n": int(len(outer_valid)),
        "inner_folds": M17D_INNER_FOLDS,
        "m17a_regenerated_inner_oof": True, "m17c_regenerated_inner_oof": True,
        "selected_on_inner_only": True,
    }

    pred.to_csv(worker_dir / f"outer_{outer}_pred.csv", index=False, float_format="%.12g")
    search.to_csv(worker_dir / f"outer_{outer}_search.csv", index=False, float_format="%.12g")
    pd.DataFrame([selected_row]).to_csv(worker_dir / f"outer_{outer}_selected.csv", index=False, float_format="%.12g")
    pd.DataFrame(ssl_audit).to_csv(worker_dir / f"outer_{outer}_ssl_audit.csv", index=False, float_format="%.12g")
    pd.DataFrame(graph_audit).to_csv(worker_dir / f"outer_{outer}_graph_audit.csv", index=False, float_format="%.12g")
    print(
        f"M17D worker outer={outer} selected={selected_candidate} "
        f"outer_mae={mean_abs(y[outer_valid], selected.prediction):.6f} "
        f"m16g={mean_abs(y[outer_valid], frozen_m16g[outer_valid]):.6f}", flush=True,
    )
    return 0


def _run_workers(tmp: Path) -> None:
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS":"1", "OPENBLAS_NUM_THREADS":"1", "MKL_NUM_THREADS":"1", "NUMEXPR_NUM_THREADS":"1"})
    for outer in range(5):
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker-fold", str(outer), "--worker-dir", str(tmp)],
            cwd=ROOT, env=env, check=True,
        )


def run(output_dir: Path, *, reuse_workers: Path | None = None) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    dev, final = prepare_development()
    if set(dev["mmsi"].astype(int)) & set(final["mmsi"].astype(int)):
        raise AssertionError("M17D development contains old-final MMSI")
    tmp = reuse_workers or (output_dir / ".m17d_workers")
    if reuse_workers is None:
        _run_workers(tmp)

    required = []
    for o in range(5):
        for suffix in ["pred", "search", "selected", "ssl_audit", "graph_audit"]:
            p = tmp / f"outer_{o}_{suffix}.csv"
            if not p.exists():
                required.append(str(p))
    if required:
        raise FileNotFoundError("missing M17D worker artifacts: " + ", ".join(required))

    pred = pd.concat([pd.read_csv(tmp / f"outer_{o}_pred.csv") for o in range(5)], ignore_index=True)
    search = pd.concat([pd.read_csv(tmp / f"outer_{o}_search.csv") for o in range(5)], ignore_index=True)
    selected = pd.concat([pd.read_csv(tmp / f"outer_{o}_selected.csv") for o in range(5)], ignore_index=True)
    ssl_audit = pd.concat([pd.read_csv(tmp / f"outer_{o}_ssl_audit.csv") for o in range(5)], ignore_index=True)
    graph_audit = pd.concat([pd.read_csv(tmp / f"outer_{o}_graph_audit.csv") for o in range(5)], ignore_index=True)
    if reuse_workers is None:
        shutil.rmtree(tmp)

    pred = pred.sort_values("mmsi", kind="mergesort").reset_index(drop=True)
    if len(pred) != 386 or pred["mmsi"].nunique() != 386:
        raise AssertionError("M17D OOF ledger must contain exactly 386 unique development MMSIs")
    if set(pred["mmsi"].astype(int)) & set(final["mmsi"].astype(int)):
        raise AssertionError("M17D ledger contains old-final MMSI")

    y = pred["target_tte_h"].to_numpy(float)
    mix = pred["pred_m17d_selected_h"].to_numpy(float)
    base = pred["pred_m16g_frozen_h"].to_numpy(float)
    mix_m = extended_metrics(y, mix)
    base_m = extended_metrics(y, base)
    ae_mix = np.abs(y - mix); ae_base = np.abs(y - base)
    delta = ae_base - ae_mix
    changed = np.abs(mix - base) > 1e-12
    changed_row_win_share = float(np.mean(delta[changed] > 0)) if np.any(changed) else 0.0
    all_row_win_share = float(np.mean(delta > 1e-12))
    tie_share = float(np.mean(~changed))

    fold_rows = []
    fold_wins = 0
    for outer in range(5):
        m = pred["m16a_outer_fold"].astype(int).eq(outer).to_numpy()
        mm = extended_metrics(y[m], mix[m]); bm = extended_metrics(y[m], base[m])
        win = float(mm["mae_h"]) < float(bm["mae_h"])
        fold_wins += int(win)
        fold_rows.append({
            "outer_fold": outer, "m17d_mae_h": float(mm["mae_h"]), "m16g_mae_h": float(bm["mae_h"]),
            "gain_h": float(bm["mae_h"] - mm["mae_h"]), "m17d_wins": bool(win),
            "m17d_p90_ae_h": float(mm["p90_ae_h"]), "m16g_p90_ae_h": float(bm["p90_ae_h"]),
        })
    p90_ratio = float(mix_m["p90_ae_h"] / base_m["p90_ae_h"])
    gate = promotion_gate(base_m, mix_m, fold_wins=fold_wins, changed_row_win_share=changed_row_win_share)
    gain = float(base_m["mae_h"] - mix_m["mae_h"])

    metric_rows = [
        {"model":"m17d_extended_nested_mixture", "source":"M17D nested OOF", **mix_m},
        {"model":"m16g_frozen_nested_mixture", "source":"frozen M16G OOF", **base_m},
    ]
    for name in M17D_EXPERT_NAMES:
        metric_rows.append({
            "model": f"expert_{name}", "source":"frozen outer expert OOF",
            **extended_metrics(y, pred[f"pred_m17d_expert_{name}_h"]),
        })
    for candidate in M17D_META_CANDIDATES:
        metric_rows.append({
            "model": f"m17d_{candidate}", "source":"M17D candidate outer OOF",
            **extended_metrics(y, pred[f"pred_m17d_{candidate}_h"]),
        })
    metrics = pd.DataFrame(metric_rows).sort_values(["mae_h", "p90_ae_h", "model"], kind="mergesort").reset_index(drop=True)

    usage = pred.groupby(["m17d_selected_meta_id", "m17d_selected_expert"], dropna=False).size().reset_index(name="rows")
    usage["share"] = usage["rows"] / len(pred)
    paired = pred[["mmsi", "m16a_outer_fold", "target_tte_h", "m17d_selected_meta_id", "m17d_selected_expert"]].copy()
    paired["ae_m17d_h"] = ae_mix; paired["ae_m16g_h"] = ae_base; paired["gain_h"] = delta
    paired["prediction_changed"] = changed; paired["m17d_wins"] = delta > 1e-12

    search = search.sort_values(["outer_fold", "mae_h", "p90_ae_h", "candidate_id"], kind="mergesort").reset_index(drop=True)
    selected = selected.sort_values("outer_fold", kind="mergesort").reset_index(drop=True)
    fold_compare = pd.DataFrame(fold_rows)
    ssl_audit = ssl_audit.sort_values(["outer_fold", "inner_fold"], kind="mergesort").reset_index(drop=True)
    graph_audit = graph_audit.sort_values(["outer_fold", "inner_fold"], kind="mergesort").reset_index(drop=True)

    pred.to_csv(output_dir / "m17d_oof_predictions.csv", index=False, float_format="%.12g")
    search.to_csv(output_dir / "m17d_inner_meta_search.csv", index=False, float_format="%.12g")
    selected.to_csv(output_dir / "m17d_outer_selected_meta.csv", index=False, float_format="%.12g")
    metrics.to_csv(output_dir / "m17d_model_metrics.csv", index=False, float_format="%.12g")
    fold_compare.to_csv(output_dir / "m17d_fold_comparison.csv", index=False, float_format="%.12g")
    usage.sort_values(["m17d_selected_meta_id", "rows"], ascending=[True, False]).to_csv(output_dir / "m17d_expert_usage.csv", index=False, float_format="%.12g")
    paired.to_csv(output_dir / "m17d_paired_error_deltas.csv", index=False, float_format="%.12g")
    ssl_audit.to_csv(output_dir / "m17d_ssl_inner_audit.csv", index=False, float_format="%.12g")
    graph_audit.to_csv(output_dir / "m17d_graph_inner_audit.csv", index=False, float_format="%.12g")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    names = ["M16G frozen", "M17D extended MoE", "M17A learned", "M17C graph gate"]
    vals = [
        float(base_m["mae_h"]), float(mix_m["mae_h"]),
        float(metrics.loc[metrics["model"].eq("expert_learned_retrieval"), "mae_h"].iloc[0]),
        float(metrics.loc[metrics["model"].eq("expert_corridor_graph_gate"), "mae_h"].iloc[0]),
    ]
    ax.bar(names, vals); ax.set_ylabel("OOF MAE (hours)"); ax.set_title("M17D extended Mixture of Experts vs frozen M16G")
    for i, v in enumerate(vals): ax.text(i, v + 1, f"{v:.2f}", ha="center")
    fig.tight_layout(); fig.savefig(output_dir / "m17d_mixture_comparison.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(fold_compare["outer_fold"].astype(str), fold_compare["gain_h"])
    ax.axhline(0, linewidth=1); ax.set_xlabel("Outer fold"); ax.set_ylabel("MAE gain vs M16G (hours)")
    ax.set_title("M17D fold stability (positive = better)")
    fig.tight_layout(); fig.savefig(output_dir / "m17d_fold_gain.png", dpi=170); plt.close(fig)

    summary = {
        "milestone":"M17D", "version":M17D_VERSION, "gate":gate,
        "development_rows":386, "blocked_old_final_rows":53, "final_test_used_for_selection":False,
        "outer_folds":5, "inner_folds":M17D_INNER_FOLDS,
        "base_experts":list(M17D_EXPERT_NAMES), "meta_candidates":list(M17D_META_CANDIDATES),
        "m17a_regenerated_inner_oof":True, "m17c_regenerated_inner_oof":True,
        "target_derived_gating_features_used":False,
        "selected_meta_by_outer_fold":{str(int(r.outer_fold)):str(r.selected_meta_id) for r in selected.itertuples()},
        "m16g_frozen_mae_h":float(base_m["mae_h"]), "m17d_mae_h":float(mix_m["mae_h"]),
        "m17d_medae_h":float(mix_m["medae_h"]), "m17d_p90_ae_h":float(mix_m["p90_ae_h"]),
        "gain_h_vs_m16g":gain, "fold_wins_vs_m16g":int(fold_wins),
        "changed_row_win_share_vs_m16g":changed_row_win_share, "all_row_win_share_vs_m16g":all_row_win_share,
        "tie_share_vs_m16g":tie_share, "p90_ratio_vs_m16g":p90_ratio,
        "promotion_gate":{
            "min_mae_gain_h":1.0, "min_fold_wins":3, "min_changed_row_win_share":0.50, "max_p90_ratio":1.05,
        },
        "claim":"Development-only nested OOF extension; M16G/M17A/M17C frozen and old final remains blocked.",
    }
    (output_dir / "M17D_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    report = f"""# M17D — Extend Frozen Mixture of Experts with M17A + M17C

## Decision

**{gate}**

M17D extends the frozen M16G expert panel with two post-freeze signals: M17A learned trajectory retrieval and M17C causal corridor/knowledge-graph gating. The 53 old-final MMSIs remain hard-blocked.

## Leakage-safe stacking protocol

The final/meta learner is trained only on cross-fitted base predictions, matching the standard stacking principle that the meta-model must learn from out-of-fold base predictions rather than in-sample fits. For every M16A outer fold, all six experts are regenerated inner-OOF inside the outer-train. The M17A encoder is trained only on inner-train MMSI trajectories, while the M17C graph excludes both outer-valid and inner-valid owners. Outer-validation uses only frozen OOF predictions from M16C/D/E/F, M17A and M17C.

## Result

- Frozen M16G MAE: **{float(base_m['mae_h']):.3f} h**
- M17D extended MoE MAE: **{float(mix_m['mae_h']):.3f} h**
- MAE gain vs M16G: **{gain:.3f} h** (positive = better)
- M17D MedAE: **{float(mix_m['medae_h']):.3f} h**
- M17D P90 absolute error: **{float(mix_m['p90_ae_h']):.3f} h**
- P90 ratio vs M16G: **{p90_ratio:.4f}**
- Fold wins vs M16G: **{fold_wins}/5**
- Changed-row win share: **{changed_row_win_share*100:.2f}%**

Predeclared promotion gate: >=1 h pooled MAE gain, >=3/5 fold wins, >=50% wins among rows whose prediction changes, and P90 <=1.05x frozen M16G.

## Interpretation

A pass means the two new post-freeze signals add stable information beyond M16G under the same development-only nested protocol. A fail is retained as a valid negative result: M17A and M17C remain independently frozen components but are not allowed to replace the frozen M16G mixture without meeting the predeclared gate. No result here reopens the official M14 final submission or the already-observed 53-row M10 final set.
"""
    (output_dir / "M17D_REPORT.md").write_text(report)

    artifacts = [
        "M17D_REPORT.md", "M17D_SUMMARY.json", "m17d_oof_predictions.csv", "m17d_inner_meta_search.csv",
        "m17d_outer_selected_meta.csv", "m17d_model_metrics.csv", "m17d_fold_comparison.csv",
        "m17d_expert_usage.csv", "m17d_paired_error_deltas.csv", "m17d_ssl_inner_audit.csv",
        "m17d_graph_inner_audit.csv", "m17d_mixture_comparison.png", "m17d_fold_gain.png",
    ]
    freeze = {
        "milestone":"M17D", "version":M17D_VERSION, "gate":gate,
        "development_population":386, "blocked_old_final_population":53,
        "selection_population_rule":"M10 train+calibration only; old M10 final hard-blocked",
        "inner_selection_only":True, "final_test_used_for_selection":False,
        "m17a_regenerated_inner_oof":True, "m17c_regenerated_inner_oof":True,
        "immutable_inputs":{
            "m16g_freeze":sha(R / "M16G_MIXTURE_FREEZE.json"),
            "m17a_freeze":sha(R / "M17A_SSL_RETRIEVAL_FREEZE.json"),
            "m17c_freeze":sha(R / "M17C_CORRIDOR_GRAPH_FREEZE.json"),
            "m16j_freeze":sha(R / "M16_FINAL_FREEZE.json"),
            "m14_submission":sha(ROOT / "dist/ais_eta_takehome_submission_final.zip"),
        },
        "artifact_sha256":{name:sha(output_dir / name) for name in artifacts},
    }
    (output_dir / "M17D_EXTENDED_MOE_FREEZE.json").write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=R)
    ap.add_argument("--worker-fold", type=int)
    ap.add_argument("--worker-dir", type=Path)
    ap.add_argument("--reuse-workers", type=Path)
    args = ap.parse_args()
    if args.worker_fold is not None:
        if args.worker_dir is None:
            raise SystemExit("--worker-dir required with --worker-fold")
        return run_worker(int(args.worker_fold), args.worker_dir)
    s = run(args.output_dir, reuse_workers=args.reuse_workers)
    print(json.dumps(s, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
