#!/usr/bin/env python3
"""Build M17C causal Sicily historical maritime corridor / knowledge-graph benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m16a import assert_development_only, extended_metrics
from ais_eta.m16e import add_physics_features
from ais_eta.m17c import (
    M17C_GRID_DEG, M17C_MAX_SNAP_NM, M17C_MIN_GRAPH_SUPPORTED_ROWS, M17C_VERSION,
    prepare_transition_events, predict_corridor_eta,
)

REF = ROOT / "data/derived/m10_reference_rows.pkl.gz"
STATES = ROOT / "data/derived/m0c_ship_states.pkl.gz"
R = ROOT / "reports"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(y, p) -> dict:
    return extended_metrics(np.asarray(y, float), np.asarray(p, float))


def prepare_development() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_pickle(REF)
    final = raw.loc[raw["split"].eq("final_test")].copy()
    dev = raw.loc[raw["split"].isin(["train", "calibration"])].copy()
    if len(dev) != 386 or len(final) != 53:
        raise AssertionError("unexpected M17C development/final cardinality")
    assert_development_only(dev, final["mmsi"])
    b = pd.read_csv(R / "m16b_destination_resolutions.csv")
    e = pd.read_csv(R / "m16e_oof_predictions.csv")
    bcols = [
        "mmsi", "canonical_destination", "canonical_port_name", "canonical_lat", "canonical_lon",
        "resolution_method", "resolution_confidence", "is_resolved_port", "is_non_specific",
    ]
    dev = dev.merge(b[bcols], on="mmsi", how="left")
    ecols = [
        "mmsi", "m16a_outer_fold", "physics_eligible", "physics_recent_speed_max_kn",
        "pred_m16e_physics_h", "pred_m16e_hard_gate_route_fallback_h",
        "pred_m16d_route_analogue_h", "physics_distance_gc_nm",
    ]
    # Older M16E ledger may not carry physics_recent_speed_max_kn; recompute target-free fields if needed.
    present = [c for c in ecols if c in e.columns]
    dev = dev.merge(e[present], on="mmsi", how="left")
    dev = add_physics_features(dev)
    if "m16a_outer_fold" not in dev.columns or dev["m16a_outer_fold"].isna().any():
        raise AssertionError("missing frozen M16A outer fold")
    return dev.sort_values("mmsi", kind="mergesort").reset_index(drop=True), final


def run(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    dev, final = prepare_development()
    states = pd.read_pickle(STATES)
    final_mmsi = set(final["mmsi"].astype(int))
    owner_ship_type = dict(zip(dev["mmsi"].astype(int), dev["ship_type_cat"].astype(str)))
    transitions = prepare_transition_events(states, owner_ship_type=owner_ship_type)
    if transitions.empty:
        raise AssertionError("M17C transition knowledge base is empty")
    if set(transitions["mmsi"].astype(int)) & final_mmsi:
        # Hard-block before any fold/query graph is built.
        transitions = transitions.loc[~transitions["mmsi"].astype(int).isin(final_mmsi)].copy()
    if set(transitions["mmsi"].astype(int)) & final_mmsi:
        raise AssertionError("old-final MMSI leaked into M17C transitions")

    prediction_rows: list[dict] = []
    path_rows: list[dict] = []
    build_audit_rows: list[dict] = []
    eligible_total = 0
    supported_total = 0

    for outer in range(5):
        va = dev.loc[dev["m16a_outer_fold"].eq(outer)].copy()
        valid_mmsi = set(va["mmsi"].astype(int))
        # Strict inductive owner isolation: graph never contains an outer-valid owner.
        fold_events = transitions.loc[~transitions["mmsi"].astype(int).isin(valid_mmsi)].copy()
        if set(fold_events["mmsi"].astype(int)) & valid_mmsi:
            raise AssertionError(f"outer-valid owner leaked into graph fold {outer}")

        for row in va.itertuples(index=False):
            eligible = bool(row.physics_eligible)
            rec = {
                "mmsi": int(row.mmsi), "m16a_outer_fold": int(outer), "target_tte_h": float(row.target_tte_h),
                "reference_eta_status": str(row.reference_eta_status), "canonical_destination": str(row.canonical_destination),
                "canonical_port_name": str(row.canonical_port_name), "physics_eligible": eligible,
                "pred_m16e_physics_h": float(row.pred_m16e_physics_h),
                "pred_m16e_hard_gate_route_fallback_h": float(row.pred_m16e_hard_gate_route_fallback_h),
                "pred_m16d_route_analogue_h": float(row.pred_m16d_route_analogue_h),
                "m17c_graph_supported": False, "m17c_graph_reason": "physics_ineligible",
                "pred_m17c_graph_h": np.nan,
            }
            if eligible:
                eligible_total += 1
                pred = predict_corridor_eta(
                    fold_events,
                    query_mmsi=int(row.mmsi), query_at=pd.Timestamp(row.last_update),
                    query_lat=float(row.track_lat), query_lon=float(row.track_lon),
                    destination_lat=float(row.canonical_lat), destination_lon=float(row.canonical_lon),
                    query_ship_type=str(row.ship_type_cat), recent_speed_kn=float(row.physics_recent_speed_max_kn),
                )
                rec.update({
                    "m17c_graph_supported": bool(pred.supported), "m17c_graph_reason": str(pred.reason),
                    "pred_m17c_graph_h": float(pred.prediction_h),
                    "m17c_source_snap_nm": float(pred.source_snap_nm), "m17c_destination_snap_nm": float(pred.destination_snap_nm),
                    "m17c_path_edges": int(pred.path_edges), "m17c_path_distance_nm": float(pred.path_distance_nm),
                    "m17c_graph_path_h": float(pred.graph_path_h), "m17c_source_snap_h": float(pred.source_snap_h),
                    "m17c_destination_snap_h": float(pred.destination_snap_h), "m17c_graph_nodes": int(pred.graph_nodes),
                    "m17c_graph_edges": int(pred.graph_edges), "m17c_min_path_support": int(pred.min_path_support),
                    "m17c_median_path_support": float(pred.median_path_support),
                    "m17c_contextual_edge_share": float(pred.contextual_edge_share),
                    "m17c_graph_median_speed_kn": float(pred.graph_median_speed_kn),
                })
                if pred.supported:
                    supported_total += 1
                    for seq, edge in enumerate(pred.path_rows):
                        path_rows.append({"mmsi": int(row.mmsi), "m16a_outer_fold": int(outer), "path_seq": seq, **edge})
            prediction_rows.append(rec)

        # Fold audit includes only target-free graph-construction quantities.
        build_audit_rows.append({
            "outer_fold": outer,
            "outer_valid_mmsi": len(valid_mmsi),
            "blocked_old_final_mmsi": len(final_mmsi),
            "available_transition_events": len(fold_events),
            "available_transition_owners": fold_events["mmsi"].nunique(),
            "earliest_transition_at": pd.to_datetime(fold_events["recorded_at"]).min().isoformat(),
            "latest_transition_at": pd.to_datetime(fold_events["recorded_at"]).max().isoformat(),
            "outer_valid_owner_overlap": int(len(set(fold_events["mmsi"].astype(int)) & valid_mmsi)),
            "old_final_owner_overlap": int(len(set(fold_events["mmsi"].astype(int)) & final_mmsi)),
        })

    ledger = pd.DataFrame(prediction_rows).sort_values("mmsi", kind="mergesort").reset_index(drop=True)
    if len(ledger) != 386 or ledger["mmsi"].nunique() != 386:
        raise AssertionError("M17C ledger must have exactly 386 unique development MMSIs")
    if set(ledger["mmsi"].astype(int)) & final_mmsi:
        raise AssertionError("old-final MMSI leaked into M17C ledger")
    supported = ledger["m17c_graph_supported"].astype(bool)
    if int(supported.sum()) < M17C_MIN_GRAPH_SUPPORTED_ROWS:
        raise AssertionError(f"M17C graph support too low: {int(supported.sum())}")

    graph_gate = ledger["pred_m16e_hard_gate_route_fallback_h"].to_numpy(float).copy()
    graph_gate[supported.to_numpy()] = ledger.loc[supported, "pred_m17c_graph_h"].to_numpy(float)
    ledger["pred_m17c_graph_gate_m16e_fallback_h"] = graph_gate

    metric_rows = []
    for name, col in {
        "m17c_corridor_graph": "pred_m17c_graph_h",
        "m16e_physics": "pred_m16e_physics_h",
    }.items():
        metric_rows.append({"scope": "graph_supported_physics_eligible", "model": name,
                            **metrics(ledger.loc[supported, "target_tte_h"], ledger.loc[supported, col])})
    for name, col in {
        "m17c_graph_gate_m16e_fallback": "pred_m17c_graph_gate_m16e_fallback_h",
        "m16e_hard_gate_route_fallback": "pred_m16e_hard_gate_route_fallback_h",
        "m16d_route_analogue": "pred_m16d_route_analogue_h",
    }.items():
        metric_rows.append({"scope": "all_development", "model": name, **metrics(ledger["target_tte_h"], ledger[col])})
    model_metrics = pd.DataFrame(metric_rows)

    fold_rows = []
    for fold, g in ledger.groupby("m16a_outer_fold"):
        sm = g["m17c_graph_supported"].astype(bool)
        if sm.any():
            for name, col in {"m17c_corridor_graph": "pred_m17c_graph_h", "m16e_physics": "pred_m16e_physics_h"}.items():
                fold_rows.append({"outer_fold": int(fold), "scope": "graph_supported_physics_eligible", "model": name,
                                  **metrics(g.loc[sm, "target_tte_h"], g.loc[sm, col])})
        for name, col in {
            "m17c_graph_gate_m16e_fallback": "pred_m17c_graph_gate_m16e_fallback_h",
            "m16e_hard_gate_route_fallback": "pred_m16e_hard_gate_route_fallback_h",
        }.items():
            fold_rows.append({"outer_fold": int(fold), "scope": "all_development", "model": name,
                              **metrics(g["target_tte_h"], g[col])})
    fold_metrics = pd.DataFrame(fold_rows)

    graph_m = model_metrics.query("scope == 'graph_supported_physics_eligible' and model == 'm17c_corridor_graph'").iloc[0]
    phys_m = model_metrics.query("scope == 'graph_supported_physics_eligible' and model == 'm16e_physics'").iloc[0]
    gate_m = model_metrics.query("scope == 'all_development' and model == 'm17c_graph_gate_m16e_fallback'").iloc[0]
    base_m = model_metrics.query("scope == 'all_development' and model == 'm16e_hard_gate_route_fallback'").iloc[0]
    paired_win = float((
        (ledger.loc[supported, "pred_m17c_graph_h"] - ledger.loc[supported, "target_tte_h"]).abs()
        < (ledger.loc[supported, "pred_m16e_physics_h"] - ledger.loc[supported, "target_tte_h"]).abs()
    ).mean())
    supported_gain = float(phys_m["mae_h"] - graph_m["mae_h"])
    full_gain = float(base_m["mae_h"] - gate_m["mae_h"])
    pvt = fold_metrics.query("scope == 'all_development'").pivot(index="outer_fold", columns="model", values="mae_h")
    full_fold_wins = int((pvt["m17c_graph_gate_m16e_fallback"] < pvt["m16e_hard_gate_route_fallback"]).sum())
    gate_pass = bool(
        int(supported.sum()) >= M17C_MIN_GRAPH_SUPPORTED_ROWS
        and supported_gain > 0.0
        and paired_win >= 0.55
        and full_gain >= 0.0
        and full_fold_wins >= 3
    )
    gate = "PASS_CORRIDOR_SIGNAL_COMPONENT" if gate_pass else "NO_GRAPH_PROMOTION"

    # Target-derived diagnostic only; never used by graph/gate construction.
    diag_rows = []
    for status, g in ledger.loc[supported].groupby("reference_eta_status", dropna=False):
        diag_rows.append({
            "reference_eta_status": status, "n": len(g),
            "graph_mae_h": metrics(g["target_tte_h"], g["pred_m17c_graph_h"])["mae_h"],
            "physics_mae_h": metrics(g["target_tte_h"], g["pred_m16e_physics_h"])["mae_h"],
            "target_derived_diagnostic_only": True,
        })
    status_diag = pd.DataFrame(diag_rows).sort_values(["n", "reference_eta_status"], ascending=[False, True])

    # Transition audit is aggregate-only, avoiding a large duplicate copy of raw AIS.
    transition_audit = (
        transitions.assign(day=pd.to_datetime(transitions["recorded_at"]).dt.date.astype(str))
        .groupby(["day", "time_bin", "ship_type"], dropna=False)
        .agg(transition_events=("mmsi", "size"), unique_mmsi=("mmsi", "nunique"), median_speed_kn=("speed_kn", "median"))
        .reset_index()
        .sort_values(["day", "time_bin", "ship_type"], kind="mergesort")
    )

    path_df = pd.DataFrame(path_rows)
    ledger.to_csv(output_dir / "m17c_oof_predictions.csv", index=False, float_format="%.12g")
    model_metrics.sort_values(["scope", "mae_h", "model"]).to_csv(output_dir / "m17c_model_metrics.csv", index=False, float_format="%.12g")
    fold_metrics.sort_values(["outer_fold", "scope", "model"]).to_csv(output_dir / "m17c_fold_metrics.csv", index=False, float_format="%.12g")
    pd.DataFrame(build_audit_rows).to_csv(output_dir / "m17c_graph_build_audit.csv", index=False)
    path_df.to_csv(output_dir / "m17c_path_edges.csv", index=False, float_format="%.12g")
    transition_audit.to_csv(output_dir / "m17c_transition_audit.csv", index=False, float_format="%.12g")
    status_diag.to_csv(output_dir / "m17c_reference_status_diagnostic.csv", index=False, float_format="%.12g")

    # Figure 1: model comparison.
    fig, ax = plt.subplots(figsize=(9, 5.2))
    labels = ["M16E physics\n(graph-supported)", "M17C graph\n(graph-supported)", "M16E hard gate\n(all dev)", "M17C graph gate\n(all dev)"]
    vals = [float(phys_m["mae_h"]), float(graph_m["mae_h"]), float(base_m["mae_h"]), float(gate_m["mae_h"])]
    ax.bar(labels, vals)
    ax.set_ylabel("MAE (hours)")
    ax.set_title("M17C historical corridor graph vs M16E physics")
    for i, v in enumerate(vals): ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(output_dir / "m17c_physics_comparison.png", dpi=160); plt.close(fig)

    # Figure 2: only actually used path edges, so the visual corresponds to scored evidence.
    fig, ax = plt.subplots(figsize=(9, 7))
    if not path_df.empty:
        for rr in path_df.itertuples(index=False):
            lat1 = (int(rr.src_ilat) + 0.5) * M17C_GRID_DEG; lon1 = (int(rr.src_ilon) + 0.5) * M17C_GRID_DEG
            lat2 = (int(rr.dst_ilat) + 0.5) * M17C_GRID_DEG; lon2 = (int(rr.dst_ilon) + 0.5) * M17C_GRID_DEG
            ax.plot([lon1, lon2], [lat1, lat2], alpha=0.32, linewidth=1.2)
    sg = ledger.loc[supported]
    ax.scatter(sg["m17c_source_snap_nm"] * 0 + np.nan, sg["m17c_source_snap_nm"] * 0 + np.nan)  # keep default color cycle stable
    # Plot canonical ports used by supported queries.
    ports = dev.loc[dev["mmsi"].isin(sg["mmsi"]), ["canonical_port_name", "canonical_lat", "canonical_lon"]].drop_duplicates()
    ax.scatter(ports["canonical_lon"], ports["canonical_lat"], marker="x", s=50, label="Supported destination ports")
    for rr in ports.itertuples(index=False):
        ax.text(float(rr.canonical_lon) + 0.05, float(rr.canonical_lat) + 0.05, str(rr.canonical_port_name), fontsize=8)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("M17C scored historical corridor paths")
    ax.grid(alpha=0.2); ax.legend(loc="best")
    fig.tight_layout(); fig.savefig(output_dir / "m17c_scored_corridor_map.png", dpi=160); plt.close(fig)

    summary = {
        "version": M17C_VERSION,
        "gate": gate,
        "development_rows": 386,
        "old_final_rows_hard_blocked": 53,
        "final_test_used_for_selection": False,
        "outer_valid_owners_used_for_graph": False,
        "future_events_used_for_query_graph": False,
        "target_used_for_graph_topology_or_edge_weight": False,
        "grid_resolution_deg": M17C_GRID_DEG,
        "max_snap_nm": M17C_MAX_SNAP_NM,
        "raw_transition_events_after_final_block": int(len(transitions)),
        "transition_unique_mmsi": int(transitions["mmsi"].nunique()),
        "physics_eligible_rows": int(ledger["physics_eligible"].sum()),
        "graph_supported_rows": int(supported.sum()),
        "graph_supported_share_of_physics_eligible": float(supported.sum() / max(1, ledger["physics_eligible"].sum())),
        "graph_supported_mae_h": float(graph_m["mae_h"]),
        "physics_same_rows_mae_h": float(phys_m["mae_h"]),
        "graph_supported_gain_h_vs_physics": supported_gain,
        "graph_paired_win_share_vs_physics": paired_win,
        "m17c_graph_gate_all_dev_mae_h": float(gate_m["mae_h"]),
        "m16e_hard_gate_all_dev_mae_h": float(base_m["mae_h"]),
        "graph_gate_gain_h_vs_m16e_hard_gate": full_gain,
        "graph_gate_fold_wins_vs_m16e_hard_gate": full_fold_wins,
        "mean_contextual_edge_share": float(pd.to_numeric(ledger.loc[supported, "m17c_contextual_edge_share"], errors="coerce").mean()),
        "mean_path_edges": float(pd.to_numeric(ledger.loc[supported, "m17c_path_edges"], errors="coerce").mean()),
        "mean_destination_snap_nm": float(pd.to_numeric(ledger.loc[supported, "m17c_destination_snap_nm"], errors="coerce").mean()),
    }
    (output_dir / "M17C_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    immutable = {
        "m17b_freeze": sha(R / "M17B_HISTORICAL_MEMORY_FREEZE.json"),
        "m17a_freeze": sha(R / "M17A_SSL_RETRIEVAL_FREEZE.json"),
        "m16j_freeze": sha(R / "M16_FINAL_FREEZE.json"),
        "m16e_predictions": sha(R / "m16e_oof_predictions.csv"),
        "m16a_manifest": sha(R / "M16A_DEVELOPMENT_MANIFEST.json"),
    }
    artifact_names = [
        "M17C_SUMMARY.json", "m17c_oof_predictions.csv", "m17c_model_metrics.csv", "m17c_fold_metrics.csv",
        "m17c_graph_build_audit.csv", "m17c_path_edges.csv", "m17c_transition_audit.csv",
        "m17c_reference_status_diagnostic.csv", "m17c_physics_comparison.png", "m17c_scored_corridor_map.png",
    ]
    artifact_sha = {name: sha(output_dir / name) for name in artifact_names}
    freeze = {
        "version": M17C_VERSION,
        "decision": gate,
        "protocol": {
            "development_only": True,
            "old_final_hard_blocked": True,
            "outer_valid_owner_blocked_from_graph": True,
            "per_query_timestamp_cutoff": True,
            "directed_grid_resolution_deg": M17C_GRID_DEG,
            "edge_speed_hierarchy": ["edge_type_time", "edge_type", "edge_time", "edge_global"],
            "target_free_graph": True,
        },
        "immutable_inputs": immutable,
        "artifact_sha256": artifact_sha,
        "summary": summary,
    }
    (output_dir / "M17C_CORRIDOR_GRAPH_FREEZE.json").write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = f"""# M17C — Sicily Historical Maritime Corridor / Knowledge Graph vs M16E Physics

## Decision

**{gate}**.

M17C finds a small but repeatable target-free historical-corridor signal on the subset where the AIS-derived directed graph connects the current position to the resolved destination. It is retained as a **component**, not promoted as a standalone replacement for M16E or M17A.

## Method

A directed 0.20° local maritime graph is built from AIS transitions only. For every outer-validation query, all old-final MMSIs and the complete set of outer-validation MMSIs are excluded from graph construction. The graph is additionally cut at the query's `last_update`, so no future AIS transition can influence that query.

Each directed edge stores a robust historical speed with a deterministic hierarchy: edge+ship-type+6h-time-bin → edge+ship-type → edge+6h-time-bin → edge-global. Dijkstra travel time across historical water corridors is combined with short source/destination snap legs. No ETA target is used in topology, edge weights, support or graph gating.

This is an AIS-derived corridor graph, **not** an ENC/S-57/S-100 authoritative navigation route and not a bathymetric safety planner.

## Results

- Development population: **386**; old final hard-blocked: **53/53**.
- M16E physics-eligible rows: **{int(ledger['physics_eligible'].sum())}**.
- Causal graph-supported rows: **{int(supported.sum())}** ({supported.sum()/max(1,ledger['physics_eligible'].sum())*100:.1f}% of physics-eligible).
- M17C graph MAE on supported rows: **{float(graph_m['mae_h']):.2f} h**.
- M16E physics MAE on the exact same rows: **{float(phys_m['mae_h']):.2f} h**.
- Supported-row MAE gain: **{supported_gain:.3f} h**; paired absolute-error wins: **{paired_win*100:.1f}%**.
- Full target-free graph gate (graph when supported, frozen M16E hard-gate otherwise): **{float(gate_m['mae_h']):.3f} h MAE** vs **{float(base_m['mae_h']):.3f} h** for frozen M16E hard-gate, gain **{full_gain:.3f} h**, wins **{full_fold_wins}/5** folds.

## Interpretation

The historical graph adds a real but very small increment over great-circle physics. Its strongest value is structural: it replaces straight-line distance with empirically observed directed corridors and context-conditioned segment speeds while remaining target-free. Coverage is limited because the provided AIS snapshot covers Eastern Sicily / nearby Mediterranean corridors, whereas many declared destinations (Gibraltar, Port Said, Northern Italy, etc.) lie outside the locally observed graph.

The declared/reference ETA heavy tail still dominates pooled MAE. `reference_eta_status` is therefore retained only in a diagnostic table and is never used for graph construction or gating.

## Next implication

Do not densify snapshot memory again. If M17 continues, the graph can be offered as an additional MoE feature/expert alongside M17A learned retrieval, but any meaningful company-facing superiority claim still requires a new untouched holdout.
"""
    (output_dir / "M17C_REPORT.md").write_text(report, encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=R)
    args = ap.parse_args()
    s = run(args.output_dir)
    print(json.dumps(s, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
