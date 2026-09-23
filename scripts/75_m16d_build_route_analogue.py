#!/usr/bin/env python3
"""Build M16D historical route-analogue expert with nested development-only OOF selection."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ais_eta.m16a import assert_development_only, balanced_hash_folds, extended_metrics
from ais_eta.m16d import (
    M16D_DTW_BAND, M16D_INNER_FOLDS, M16D_INNER_SALT, M16D_POINTS, M16D_VERSION,
    RouteConfig, candidate_configs, dtw_distance, predict_route_analogue,
    representation_spec, resample_causal_trajectory,
)

REF = ROOT / "data/derived/m10_reference_rows.pkl.gz"
STATES = ROOT / "data/derived/m0c_ship_states.pkl.gz"
R = ROOT / "reports"

ID_COLS = {
    "mmsi", "name", "split", "last_update", "eta_reference_raw", "eta_reference_dt",
    "target_tte_h", "reference_eta_status", "history_last_at",
}
CAT_COLS = [
    "destination_norm", "ship_type_cat", "nav_status_cat", "flag_cat",
    "motion_state_cat", "stale_risk_cat", "hist_destination_last",
]
CATBOOST_PARAMS = {
    "loss_function": "MAE", "depth": 5, "learning_rate": 0.03, "l2_leaf_reg": 10.0,
    "random_seed": 42, "verbose": False, "allow_writing_files": False, "thread_count": 1,
}



def save_npz_deterministic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write compressed NPZ with fixed ZIP metadata for byte-reproducibility."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name in sorted(arrays):
            buf = io.BytesIO()
            np.save(buf, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, buf.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


try:
    from numba import njit

    @njit(cache=False)
    def _DTW_NUMBA(a, b, band):
        n = a.shape[0]; m = b.shape[0]
        inf = 1e300
        prev = np.empty(m + 1); prev[:] = inf; prev[0] = 0.0
        for i in range(1, n + 1):
            cur = np.empty(m + 1); cur[:] = inf
            j0 = max(1, i - band); j1 = min(m, i + band)
            for j in range(j0, j1 + 1):
                ss = 0.0
                for q in range(a.shape[1]):
                    d = a[i - 1, q] - b[j - 1, q]
                    ss += d * d
                cost = np.sqrt(ss)
                best = prev[j]
                if cur[j - 1] < best: best = cur[j - 1]
                if prev[j - 1] < best: best = prev[j - 1]
                cur[j] = cost + best
            prev = cur
        return prev[m] / (n + m)
except Exception:  # optional acceleration only
    _DTW_NUMBA = None


def _accelerated_pairwise(sequences: list[np.ndarray]) -> tuple[np.ndarray, str]:
    """Use one shared numba kernel when available; pure-Python remains the fallback."""
    n = len(sequences)
    out = np.zeros((n, n), dtype=np.float64)
    if _DTW_NUMBA is not None:
        _DTW_NUMBA(sequences[0], sequences[min(1, len(sequences)-1)], M16D_DTW_BAND)
        for i in range(n):
            for j in range(i):
                d = float(_DTW_NUMBA(sequences[i], sequences[j], M16D_DTW_BAND))
                out[i, j] = d; out[j, i] = d
        return out, "numba_optional_acceleration_same_formula"
    for i in range(n):
        for j in range(i):
            d = dtw_distance(sequences[i], sequences[j], band=M16D_DTW_BAND)
            out[i, j] = d; out[j, i] = d
    return out, "pure_python"


def prepare_development() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_pickle(REF)
    final = raw.loc[raw["split"].eq("final_test")].copy()
    dev = raw.loc[raw["split"].isin(["train", "calibration"])].copy()
    if len(dev) != 386 or len(final) != 53:
        raise AssertionError("unexpected M16D dev/final cardinality")
    assert_development_only(dev, final["mmsi"])

    a = pd.read_csv(R / "m16a_oof_predictions.csv")
    b = pd.read_csv(R / "m16b_destination_resolutions.csv")
    c = pd.read_csv(R / "m16c_oof_predictions.csv")
    dev = dev.merge(a[["mmsi", "m16a_outer_fold", "pred_train_destination_median_h", "pred_catboost_current_snapshot_h"]], on="mmsi", how="left")
    dev = dev.merge(b[["mmsi", "canonical_destination", "resolution_method", "resolution_confidence"]], on="mmsi", how="left")
    dev = dev.merge(c[["mmsi", "pred_m16c_hierarchical_prior_h"]], on="mmsi", how="left")
    dev = dev.sort_values("mmsi", kind="mergesort").reset_index(drop=True)
    if dev["m16a_outer_fold"].isna().any():
        raise AssertionError("missing frozen M16A outer fold")
    return dev, final, raw


def build_representations(dev: pd.DataFrame, states: pd.DataFrame, output_dir: Path):
    reps: dict[str, list[np.ndarray]] = {}
    support_rows = []
    matrices: dict[str, np.ndarray] = {}
    engines: dict[str, str] = {}

    grouped = {int(k): g.copy() for k, g in states.groupby("mmsi", sort=False)}
    for rep in ["geo_3h", "geo_kin_3h", "geo_kin_6h"]:
        hours, kin = representation_spec(rep)
        seqs = []
        for idx, row in dev.iterrows():
            g = grouped.get(int(row["mmsi"]))
            if g is None:
                raise AssertionError(f"M16D missing history for MMSI {row['mmsi']}")
            seq, meta = resample_causal_trajectory(
                g, end_at=pd.Timestamp(row["history_last_at"]), hours=hours,
                n_points=M16D_POINTS, include_kinematics=kin,
            )
            seqs.append(seq)
            support_rows.append({"mmsi": int(row["mmsi"]), "representation": rep, **meta})
        reps[rep] = seqs
        matrices[rep], engines[rep] = _accelerated_pairwise(seqs)

    save_npz_deterministic(output_dir / "m16d_route_sequences.npz", {k: np.stack(v) for k, v in reps.items()})
    save_npz_deterministic(output_dir / "m16d_route_distance_matrices.npz", matrices)
    pd.DataFrame(support_rows).sort_values(["representation", "mmsi"]).to_csv(output_dir / "m16d_route_support.csv", index=False)
    return reps, matrices, engines


def fit_history_catboost_oof(dev: pd.DataFrame, raw_reference: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    work = raw_reference.loc[raw_reference["split"].isin(["train", "calibration"])].copy()
    fold_map = dict(zip(dev["mmsi"].astype(int), dev["m16a_outer_fold"].astype(int)))
    work["m16a_outer_fold"] = work["mmsi"].map(fold_map)
    work = work.sort_values("mmsi", kind="mergesort").reset_index(drop=True)
    features = [c for c in raw_reference.columns if c not in ID_COLS]
    cats = [c for c in CAT_COLS if c in features]
    for col in cats:
        work[col] = work[col].fillna("UNKNOWN").astype(str)

    pred = np.zeros(len(work), dtype=float)
    tree_rows = []
    for outer in range(5):
        tr = work.loc[work["m16a_outer_fold"].ne(outer)].copy()
        va = work.loc[work["m16a_outer_fold"].eq(outer)].copy()
        inner_map = balanced_hash_folds(tr["mmsi"], n_splits=6, salt=f"M16D_HISTORY_ES:outer={outer}")
        inner = tr["mmsi"].map(inner_map)
        fit = tr.loc[inner.ne(0)].copy(); es = tr.loc[inner.eq(0)].copy()
        selector = CatBoostRegressor(iterations=700, **CATBOOST_PARAMS)
        selector.fit(fit[features], fit["target_tte_h"], cat_features=cats,
                     eval_set=(es[features], es["target_tte_h"]), early_stopping_rounds=80, verbose=False)
        trees = max(1, int(selector.tree_count_))
        model = CatBoostRegressor(iterations=trees, **CATBOOST_PARAMS)
        model.fit(tr[features], tr["target_tte_h"], cat_features=cats, verbose=False)
        pred[work["m16a_outer_fold"].eq(outer).to_numpy()] = model.predict(va[features])
        tree_rows.append({"outer_fold": outer, "chosen_trees": trees, "train_n": len(tr), "valid_n": len(va)})

    pred_map = dict(zip(work["mmsi"].astype(int), pred))
    aligned = dev["mmsi"].astype(int).map(pred_map).to_numpy(float)
    return aligned, pd.DataFrame(tree_rows)


def metric_triplet(y, p):
    m = extended_metrics(y, p)
    return float(m["mae_h"]), float(m["p90_ae_h"]), float(m["medae_h"])


def build_route_oof(dev: pd.DataFrame, matrices: dict[str, np.ndarray]):
    y = dev["target_tte_h"].to_numpy(float)
    mmsis = dev["mmsi"].to_numpy(int)
    folds = dev["m16a_outer_fold"].to_numpy(int)
    destinations = dev["canonical_destination"].fillna("UNKNOWN").astype(str).to_numpy()
    configs = candidate_configs()

    route_pred = np.zeros(len(dev), dtype=float)
    detail_frames = []
    search_rows = []
    selected_rows = []

    for outer in range(5):
        outer_train = np.flatnonzero(folds != outer)
        outer_valid = np.flatnonzero(folds == outer)
        inner_map = balanced_hash_folds(mmsis[outer_train], M16D_INNER_FOLDS, salt=f"{M16D_INNER_SALT}:outer={outer}")
        inner_fold = np.asarray([inner_map[int(x)] for x in mmsis[outer_train]], dtype=int)
        scored = []
        for cfg in configs:
            pp = []; yy = []
            for inner in range(M16D_INNER_FOLDS):
                tr = outer_train[inner_fold != inner]
                va = outer_train[inner_fold == inner]
                o = predict_route_analogue(
                    train_indices=tr, valid_indices=va, distance_matrix=matrices[cfg.representation],
                    targets=y, destinations=destinations, config=cfg,
                )
                pp.extend(o["prediction_h"].to_numpy(float)); yy.extend(y[va])
            mae, p90, medae = metric_triplet(np.asarray(yy), np.asarray(pp))
            row = {"outer_fold": outer, "config_id": cfg.config_id, "representation": cfg.representation,
                   "gate": cfg.gate, "k": cfg.k, "inner_mae_h": mae, "inner_p90_ae_h": p90, "inner_medae_h": medae}
            search_rows.append(row); scored.append((mae, p90, medae, cfg.config_id, cfg))
        best = min(scored, key=lambda x: (x[0], x[1], x[2], x[3]))
        cfg = best[-1]
        selected_rows.append({"outer_fold": outer, "selected_config_id": cfg.config_id, "inner_mae_h": best[0],
                              "inner_p90_ae_h": best[1], "inner_medae_h": best[2]})
        out = predict_route_analogue(
            train_indices=outer_train, valid_indices=outer_valid, distance_matrix=matrices[cfg.representation],
            targets=y, destinations=destinations, config=cfg,
        )
        route_pred[outer_valid] = out["prediction_h"].to_numpy(float)
        out["outer_fold"] = outer
        out["selected_config_id"] = cfg.config_id
        out["query_mmsi"] = mmsis[out["row_index"].to_numpy(int)]
        out["nearest_mmsi"] = mmsis[out["nearest_index"].to_numpy(int)]
        detail_frames.append(out)

    details = pd.concat(detail_frames, ignore_index=True).sort_values("row_index", kind="mergesort").reset_index(drop=True)
    return route_pred, pd.DataFrame(search_rows), pd.DataFrame(selected_rows), details


def manual_audit_plot(dev, states, details, selected, output_dir):
    # One low-distance validation example per outer fold.  Neighbours are guaranteed
    # by construction to be outer-train rows; this is a geometric audit only.
    picks = details.sort_values(["outer_fold", "nearest_distance"], kind="mergesort").groupby("outer_fold", as_index=False).head(1)
    audit_rows = []
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    state_groups = {int(k): g.sort_values("recorded_at") for k, g in states.groupby("mmsi", sort=False)}
    selected_map = dict(zip(selected["outer_fold"], selected["selected_config_id"]))
    for ax, rec in zip(axes, picks.itertuples()):
        qidx = int(rec.row_index); q = dev.iloc[qidx]
        rep = str(selected_map[int(rec.outer_fold)]).split("__")[0]
        hours, _ = representation_spec(rep)
        end = pd.Timestamp(q["history_last_at"]); start = end - pd.Timedelta(hours=hours)
        neighbour_indices = [int(x) for x in str(rec.neighbour_indices).split(";")[:3]]
        qg = state_groups[int(q["mmsi"])]
        qg = qg.loc[qg["recorded_at"].between(start, end) & qg["position_valid"].fillna(False)]
        ax.plot(qg["lon_clean"], qg["lat_clean"], linewidth=2.5, label=f"query {int(q['mmsi'])}")
        for rank, ni in enumerate(neighbour_indices, 1):
            nr = dev.iloc[ni]; ne = pd.Timestamp(nr["history_last_at"]); ns = ne - pd.Timedelta(hours=hours)
            ng = state_groups[int(nr["mmsi"])]
            ng = ng.loc[ng["recorded_at"].between(ns, ne) & ng["position_valid"].fillna(False)]
            ax.plot(ng["lon_clean"], ng["lat_clean"], alpha=0.75, label=f"N{rank} {int(nr['mmsi'])}")
            audit_rows.append({"outer_fold": int(rec.outer_fold), "query_mmsi": int(q["mmsi"]), "neighbor_rank": rank,
                               "neighbor_mmsi": int(nr["mmsi"]), "representation": rep,
                               "query_target_h": float(q["target_tte_h"]), "neighbor_target_h": float(nr["target_tte_h"]),
                               "selected_config_id": selected_map[int(rec.outer_fold)]})
        ax.set_title(f"Fold {int(rec.outer_fold)} | {rep} | d={float(rec.nearest_distance):.3f}")
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.legend(fontsize=7)
    axes[-1].axis("off")
    fig.suptitle("M16D manual route-neighbour geometry audit (query + Top-3 outer-train neighbours)")
    fig.tight_layout()
    fig.savefig(output_dir / "m16d_neighbour_geometry_audit.png", dpi=160)
    plt.close(fig)
    pd.DataFrame(audit_rows).to_csv(output_dir / "m16d_neighbour_audit.csv", index=False)


def run(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    dev, final, raw = prepare_development()
    states = pd.read_pickle(STATES)
    states = states.loc[states["mmsi"].isin(dev["mmsi"])].copy()
    states["recorded_at"] = pd.to_datetime(states["recorded_at"])
    if len(states) < 500_000:
        raise AssertionError("unexpectedly sparse M16D trajectory source")

    reps, matrices, engines = build_representations(dev, states, output_dir)
    pred_hist, trees = fit_history_catboost_oof(dev, raw)
    pred_route, inner_search, selected, details = build_route_oof(dev, matrices)

    ledger = dev[["mmsi", "split", "reference_eta_status", "destination_norm", "canonical_destination", "target_tte_h", "m16a_outer_fold"]].copy()
    ledger["pred_m16d_route_analogue_h"] = pred_route
    ledger["pred_m16d_history_aggregate_catboost_h"] = pred_hist
    ledger["pred_m16a_raw_destination_median_h"] = dev["pred_train_destination_median_h"].to_numpy(float)
    ledger["pred_m16a_catboost_current_snapshot_h"] = dev["pred_catboost_current_snapshot_h"].to_numpy(float)
    ledger["pred_m16c_hierarchical_prior_h"] = dev["pred_m16c_hierarchical_prior_h"].to_numpy(float)
    for c in ["selected_config_id", "neighbour_count", "nearest_distance", "median_neighbour_distance", "similarity_gap", "gate_used", "nearest_mmsi"]:
        mp = dict(zip(details["query_mmsi"].astype(int), details[c]))
        ledger[f"m16d_{c}"] = ledger["mmsi"].astype(int).map(mp)

    models = {
        "m16d_route_analogue": "pred_m16d_route_analogue_h",
        "m16d_history_aggregate_catboost": "pred_m16d_history_aggregate_catboost_h",
        "m16a_raw_destination_median": "pred_m16a_raw_destination_median_h",
        "m16a_catboost_current_snapshot": "pred_m16a_catboost_current_snapshot_h",
        "m16c_hierarchical_prior": "pred_m16c_hierarchical_prior_h",
    }
    metrics = pd.DataFrame([{"model": name, **extended_metrics(ledger["target_tte_h"], ledger[col])} for name, col in models.items()])
    fold_rows = []
    for fold, g in ledger.groupby("m16a_outer_fold"):
        for name, col in models.items():
            fold_rows.append({"outer_fold": int(fold), "model": name, **extended_metrics(g["target_tte_h"], g[col])})
    fold_metrics = pd.DataFrame(fold_rows)

    route_m = metrics.set_index("model").loc["m16d_route_analogue"]
    hist_m = metrics.set_index("model").loc["m16d_history_aggregate_catboost"]
    raw_m = metrics.set_index("model").loc["m16a_raw_destination_median"]
    pvt = fold_metrics.pivot(index="outer_fold", columns="model", values="mae_h")
    route_wins_hist = int((pvt["m16d_route_analogue"] < pvt["m16d_history_aggregate_catboost"]).sum())
    route_wins_raw = int((pvt["m16d_route_analogue"] < pvt["m16a_raw_destination_median"]).sum())
    gain_hist = float(hist_m["mae_h"] - route_m["mae_h"])
    gain_raw = float(raw_m["mae_h"] - route_m["mae_h"])
    history_gate_pass = bool(gain_hist > 0 and route_wins_hist >= 3)
    # M16D's predeclared gate is contribution beyond simple history aggregates,
    # not standalone model selection against every previous expert.  Even though
    # the diagnostic pooled/fold comparison is encouraging, M16G owns any final
    # expert combination/promotion.  Keep this field false to avoid post-hoc
    # promotion after seeing development OOF results.
    standalone_promoted = False
    gate = "PASS_ROUTE_SIGNAL_COMPONENT" if history_gate_pass else "NO_PROMOTION"

    # Untuned diagnostic only; M16G owns ensemble fitting/selection.
    blend = 0.5 * ledger["pred_m16d_route_analogue_h"].to_numpy(float) + 0.5 * ledger["pred_m16a_raw_destination_median_h"].to_numpy(float)
    blend_m = extended_metrics(ledger["target_tte_h"], blend)

    ledger.to_csv(output_dir / "m16d_oof_predictions.csv", index=False, float_format="%.12g")
    inner_search.sort_values(["outer_fold", "inner_mae_h", "config_id"]).to_csv(output_dir / "m16d_inner_search.csv", index=False, float_format="%.12g")
    selected.to_csv(output_dir / "m16d_outer_selected_configs.csv", index=False, float_format="%.12g")
    details.to_csv(output_dir / "m16d_neighbour_details.csv", index=False, float_format="%.12g")
    metrics.sort_values(["mae_h", "p90_ae_h", "model"]).to_csv(output_dir / "m16d_model_metrics.csv", index=False, float_format="%.12g")
    fold_metrics.sort_values(["outer_fold", "model"]).to_csv(output_dir / "m16d_fold_metrics.csv", index=False, float_format="%.12g")
    trees.to_csv(output_dir / "m16d_history_catboost_tree_counts.csv", index=False)
    manual_audit_plot(dev, states, details, selected, output_dir)

    summary = {
        "milestone": "M16D", "status": "HISTORICAL_ROUTE_ANALOGUE_BUILT", "version": M16D_VERSION,
        "gate": gate, "standalone_promoted": standalone_promoted,
        "development_rows": 386, "blocked_old_final_rows": 53, "final_test_used_for_selection": False,
        "trajectory_source_rows": int(len(states)), "trajectory_source": "data/derived/m0c_ship_states.pkl.gz",
        "causal_cutoff": "per-row history_last_at, always <= Tracks.last_update",
        "route_points": M16D_POINTS, "dtw_band": M16D_DTW_BAND,
        "pairwise_distance_engine": engines,
        "candidate_config_count": len(candidate_configs()), "outer_folds": 5, "inner_folds": M16D_INNER_FOLDS,
        "selection_rule": "minimum pooled inner-CV MAE; P90/MedAE/config_id deterministic tie-break",
        "route_analogue_oof": {k: (int(v) if k == "n" or k.endswith("count") else float(v)) for k, v in route_m.to_dict().items()},
        "history_aggregate_catboost_oof": {k: (int(v) if k == "n" or k.endswith("count") else float(v)) for k, v in hist_m.to_dict().items()},
        "raw_destination_median_oof": {k: (int(v) if k == "n" or k.endswith("count") else float(v)) for k, v in raw_m.to_dict().items()},
        "route_mae_gain_h_vs_history_aggregate_catboost": gain_hist,
        "route_mae_gain_h_vs_raw_destination": gain_raw,
        "route_fold_wins_vs_history_aggregate_catboost": route_wins_hist,
        "route_fold_wins_vs_raw_destination": route_wins_raw,
        "history_gate_pass": history_gate_pass,
        "standalone_promotion_rule": "not evaluated/promoted in M16D; M16G owns expert combination and promotion",
        "equal_blend_50_50_route_raw_destination_diagnostic": {**blend_m, "weight_selected_or_tuned": False},
        "selected_configs_by_outer_fold": {str(int(r.outer_fold)): str(r.selected_config_id) for r in selected.itertuples()},
        "manual_geometry_audit_rows": int(pd.read_csv(output_dir / "m16d_neighbour_audit.csv").shape[0]),
        "claim": "Nested development-only OOF route-analogue result; no use of the old M10 final test for selection or scoring.",
    }
    (output_dir / "M16D_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Comparison plot.
    p = metrics.set_index("model").loc[["m16a_raw_destination_median", "m16a_catboost_current_snapshot", "m16d_history_aggregate_catboost", "m16d_route_analogue"]]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(p)); w = 0.25
    ax.bar(x-w, p["mae_h"], width=w, label="MAE")
    ax.bar(x, p["medae_h"], width=w, label="MedAE")
    ax.bar(x+w, p["p90_ae_h"], width=w, label="P90 AE")
    ax.set_xticks(x); ax.set_xticklabels(["raw dest", "current CatBoost", "history CatBoost", "route analogue"], rotation=12)
    ax.set_ylabel("Hours"); ax.set_title("M16D development-only OOF comparison"); ax.legend()
    fig.tight_layout(); fig.savefig(output_dir / "m16d_route_analogue_comparison.png", dpi=160); plt.close(fig)

    freeze_files = [
        "m16d_oof_predictions.csv", "m16d_inner_search.csv", "m16d_outer_selected_configs.csv",
        "m16d_neighbour_details.csv", "m16d_model_metrics.csv", "m16d_fold_metrics.csv",
        "m16d_history_catboost_tree_counts.csv", "m16d_route_support.csv", "m16d_route_sequences.npz",
        "m16d_route_distance_matrices.npz", "m16d_neighbour_audit.csv", "m16d_neighbour_geometry_audit.png",
        "m16d_route_analogue_comparison.png", "M16D_SUMMARY.json",
    ]
    freeze = {
        "milestone": "M16D", "status": "ROUTE_ANALOGUE_FROZEN", "version": M16D_VERSION,
        "development_population": 386, "blocked_old_final_population": 53,
        "selection_population_rule": "M10 train+calibration only; 53-row old final hard-blocked",
        "outer_fold_source": "M16A frozen outer folds", "inner_selection_only": True,
        "final_test_used_for_selection": False, "candidate_config_count": len(candidate_configs()),
        "primary_metric": "pooled OOF MAE hours", "gate": gate,
        "artifact_sha256": {name: sha(output_dir / name) for name in freeze_files},
    }
    (output_dir / "M16D_ROUTE_ANALOGUE_FREEZE.json").write_text(json.dumps(freeze, indent=2) + "\n")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--output-dir", type=Path, default=R); args = ap.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
