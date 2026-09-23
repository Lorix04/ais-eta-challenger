#!/usr/bin/env python3
"""Build M16E leakage-safe maritime / physics expert with nested development-only selection."""
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
from ais_eta.m16e import (
    M16E_INNER_FOLDS, M16E_VERSION, add_physics_features, candidate_configs,
    predict_physics_expert, select_config_inner_cv,
)

REF = ROOT / "data/derived/m10_reference_rows.pkl.gz"
R = ROOT / "reports"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_development() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_pickle(REF)
    final = raw.loc[raw["split"].eq("final_test")].copy()
    dev = raw.loc[raw["split"].isin(["train", "calibration"])].copy()
    if len(dev) != 386 or len(final) != 53:
        raise AssertionError("unexpected M16E dev/final cardinality")
    assert_development_only(dev, final["mmsi"])

    b = pd.read_csv(R / "m16b_destination_resolutions.csv")
    c = pd.read_csv(R / "m16c_oof_predictions.csv")
    d = pd.read_csv(R / "m16d_oof_predictions.csv")
    bcols = [
        "mmsi", "canonical_destination", "canonical_unlocode", "canonical_port_name",
        "canonical_lat", "canonical_lon", "resolution_method", "resolution_confidence",
        "is_resolved_port", "is_non_specific",
    ]
    dev = dev.merge(b[bcols], on="mmsi", how="left")
    dev = dev.merge(c[["mmsi", "pred_m16c_hierarchical_prior_h"]], on="mmsi", how="left")
    dev = dev.merge(d[[
        "mmsi", "m16a_outer_fold", "pred_m16d_route_analogue_h",
        "pred_m16a_raw_destination_median_h", "pred_m16a_catboost_current_snapshot_h",
        "m16d_nearest_distance", "m16d_median_neighbour_distance",
    ]], on="mmsi", how="left")
    dev = dev.sort_values("mmsi", kind="mergesort").reset_index(drop=True)
    if dev["m16a_outer_fold"].isna().any():
        raise AssertionError("missing frozen M16A fold")
    return add_physics_features(dev), final


def nested_oof(dev: pd.DataFrame):
    pred = np.full(len(dev), np.nan, dtype=float)
    base = np.full(len(dev), np.nan, dtype=float)
    eff_speed = np.full(len(dev), np.nan, dtype=float)
    factor = np.full(len(dev), np.nan, dtype=float)
    route_distance = np.full(len(dev), np.nan, dtype=float)
    correction = np.full(len(dev), np.nan, dtype=float)
    search_frames = []
    selected_rows = []

    for outer in range(5):
        tr = dev.loc[dev["m16a_outer_fold"].ne(outer)].copy()
        va = dev.loc[dev["m16a_outer_fold"].eq(outer)].copy()
        best, search = select_config_inner_cv(tr, outer)
        search_frames.append(search)
        out = predict_physics_expert(tr, va, best)
        idx = va.index.to_numpy(int)
        pred[idx] = out["prediction_h"].to_numpy(float)
        base[idx] = out["base_prediction_h"].to_numpy(float)
        eff_speed[idx] = out["effective_speed_kn"].to_numpy(float)
        factor[idx] = out["distance_factor"].to_numpy(float)
        route_distance[idx] = out["route_distance_proxy_nm"].to_numpy(float)
        correction[idx] = out["residual_correction_h"].to_numpy(float)
        sel = search.iloc[0]
        selected_rows.append({
            "outer_fold": outer,
            "selected_config_id": best.config_id,
            "speed_window_min": best.speed_window_min,
            "distance_mode": best.distance_mode,
            "correction": best.correction,
            "inner_best_mae_h": float(sel["mae_h"]),
            "inner_best_p90_ae_h": float(sel["p90_ae_h"]),
            "inner_best_medae_h": float(sel["medae_h"]),
            "inner_eligible_rows": int(sel["eligible_inner_rows"]),
            "outer_eligible_rows": int(va["physics_eligible"].sum()),
            "selected_on_inner_only": True,
        })
    return pred, base, eff_speed, factor, route_distance, correction, pd.concat(search_frames, ignore_index=True), pd.DataFrame(selected_rows)


def _metrics(y, p):
    return extended_metrics(np.asarray(y, dtype=float), np.asarray(p, dtype=float))


def run(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    dev, final = prepare_development()
    pred, base, eff_speed, factor, route_distance, correction, inner_search, selected = nested_oof(dev)
    eligible = dev["physics_eligible"].to_numpy(bool)
    if int(eligible.sum()) < 100:
        raise AssertionError("M16E physics coverage unexpectedly low")

    ledger_cols = [
        "mmsi", "split", "reference_eta_status", "destination_norm", "canonical_destination",
        "canonical_port_name", "resolution_method", "resolution_confidence", "target_tte_h",
        "m16a_outer_fold", "physics_distance_gc_nm", "physics_bearing_to_port_deg",
        "physics_course_alignment_deg", "physics_recent_speed_max_kn", "physics_eligible",
        "physics_local_sinuosity_180m", "physics_local_sinuosity_360m",
    ]
    ledger = dev[ledger_cols].copy()
    ledger["pred_m16e_physics_h"] = pred
    ledger["pred_m16e_physics_base_h"] = base
    ledger["m16e_effective_speed_kn"] = eff_speed
    ledger["m16e_distance_factor"] = factor
    ledger["m16e_route_distance_proxy_nm"] = route_distance
    ledger["m16e_residual_correction_h"] = correction
    ledger["pred_m16d_route_analogue_h"] = dev["pred_m16d_route_analogue_h"].to_numpy(float)
    ledger["pred_m16c_hierarchical_prior_h"] = dev["pred_m16c_hierarchical_prior_h"].to_numpy(float)
    ledger["pred_m16a_raw_destination_median_h"] = dev["pred_m16a_raw_destination_median_h"].to_numpy(float)
    ledger["pred_m16a_catboost_current_snapshot_h"] = dev["pred_m16a_catboost_current_snapshot_h"].to_numpy(float)

    sel_map = dict(zip(selected["outer_fold"].astype(int), selected["selected_config_id"]))
    ledger["m16e_selected_config_id"] = ledger["m16a_outer_fold"].astype(int).map(sel_map)

    # Hard gate is deliberately predeclared and target-free: when physics has a
    # verified destination + recent motion, use it; otherwise keep M16D route.
    hard = ledger["pred_m16d_route_analogue_h"].to_numpy(float).copy()
    hard[eligible] = ledger.loc[eligible, "pred_m16e_physics_h"].to_numpy(float)
    ledger["pred_m16e_hard_gate_route_fallback_h"] = hard

    eligible_models = {
        "m16e_physics": "pred_m16e_physics_h",
        "m16d_route_analogue": "pred_m16d_route_analogue_h",
        "m16c_hierarchical_prior": "pred_m16c_hierarchical_prior_h",
        "m16a_raw_destination_median": "pred_m16a_raw_destination_median_h",
    }
    metric_rows = []
    for name, col in eligible_models.items():
        metric_rows.append({"scope": "physics_eligible", "model": name, **_metrics(ledger.loc[eligible, "target_tte_h"], ledger.loc[eligible, col])})
    for name, col in {
        "m16e_hard_gate_route_fallback": "pred_m16e_hard_gate_route_fallback_h",
        "m16d_route_analogue": "pred_m16d_route_analogue_h",
        "m16c_hierarchical_prior": "pred_m16c_hierarchical_prior_h",
        "m16a_raw_destination_median": "pred_m16a_raw_destination_median_h",
        "m16a_catboost_current_snapshot": "pred_m16a_catboost_current_snapshot_h",
    }.items():
        metric_rows.append({"scope": "all_development", "model": name, **_metrics(ledger["target_tte_h"], ledger[col])})
    metrics = pd.DataFrame(metric_rows)

    fold_rows = []
    for fold, g in ledger.groupby("m16a_outer_fold"):
        em = g["physics_eligible"].astype(bool)
        for name, col in eligible_models.items():
            fold_rows.append({"outer_fold": int(fold), "scope": "physics_eligible", "model": name,
                              **_metrics(g.loc[em, "target_tte_h"], g.loc[em, col])})
        for name, col in {
            "m16e_hard_gate_route_fallback": "pred_m16e_hard_gate_route_fallback_h",
            "m16d_route_analogue": "pred_m16d_route_analogue_h",
        }.items():
            fold_rows.append({"outer_fold": int(fold), "scope": "all_development", "model": name,
                              **_metrics(g["target_tte_h"], g[col])})
    fold_metrics = pd.DataFrame(fold_rows)

    # Status is target-derived; this table is forensic/diagnostic only and never
    # participates in eligibility, inner selection or the hard gate.
    status_rows = []
    eg = ledger.loc[eligible].copy()
    for status, g in eg.groupby("reference_eta_status", dropna=False):
        status_rows.append({
            "reference_eta_status": status, "n": len(g),
            "physics_mae_h": _metrics(g["target_tte_h"], g["pred_m16e_physics_h"])["mae_h"],
            "physics_medae_h": _metrics(g["target_tte_h"], g["pred_m16e_physics_h"])["medae_h"],
            "route_mae_h": _metrics(g["target_tte_h"], g["pred_m16d_route_analogue_h"])["mae_h"],
            "target_derived_diagnostic_only": True,
        })
    status_diag = pd.DataFrame(status_rows).sort_values(["n", "reference_eta_status"], ascending=[False, True])

    phy_e = metrics.query("scope == 'physics_eligible' and model == 'm16e_physics'").iloc[0]
    route_e = metrics.query("scope == 'physics_eligible' and model == 'm16d_route_analogue'").iloc[0]
    hard_m = metrics.query("scope == 'all_development' and model == 'm16e_hard_gate_route_fallback'").iloc[0]
    route_m = metrics.query("scope == 'all_development' and model == 'm16d_route_analogue'").iloc[0]
    physics_win_share = float((
        (ledger.loc[eligible, "pred_m16e_physics_h"] - ledger.loc[eligible, "target_tte_h"]).abs()
        < (ledger.loc[eligible, "pred_m16d_route_analogue_h"] - ledger.loc[eligible, "target_tte_h"]).abs()
    ).mean())
    pvt = fold_metrics.query("scope == 'all_development'").pivot(index="outer_fold", columns="model", values="mae_h")
    hard_fold_wins = int((pvt["m16e_hard_gate_route_fallback"] < pvt["m16d_route_analogue"]).sum())
    hard_gain = float(route_m["mae_h"] - hard_m["mae_h"])
    physics_gain_eligible = float(route_e["mae_h"] - phy_e["mae_h"])
    gate_pass = bool(physics_win_share > 0.5 and hard_gain > 0 and hard_fold_wins >= 3)
    gate = "PASS_PHYSICS_GATING_COMPONENT" if gate_pass else "NO_PROMOTION"

    # Target-free feature audit; no reference ETA or prediction target included.
    feature_audit = dev[[
        "mmsi", "canonical_destination", "canonical_port_name", "resolution_confidence",
        "physics_distance_gc_nm", "physics_bearing_to_port_deg", "physics_course_alignment_deg",
        "sog_median_180m", "sog_median_360m", "physics_recent_speed_max_kn",
        "physics_local_sinuosity_180m", "physics_local_sinuosity_360m", "physics_eligible",
    ]].copy()

    ledger.to_csv(output_dir / "m16e_oof_predictions.csv", index=False, float_format="%.12g")
    feature_audit.to_csv(output_dir / "m16e_physics_feature_audit.csv", index=False, float_format="%.12g")
    inner_search.sort_values(["outer_fold", "mae_h", "config_id"]).to_csv(output_dir / "m16e_inner_search.csv", index=False, float_format="%.12g")
    selected.to_csv(output_dir / "m16e_outer_selected_configs.csv", index=False, float_format="%.12g")
    metrics.sort_values(["scope", "mae_h", "model"]).to_csv(output_dir / "m16e_model_metrics.csv", index=False, float_format="%.12g")
    fold_metrics.sort_values(["outer_fold", "scope", "model"]).to_csv(output_dir / "m16e_fold_metrics.csv", index=False, float_format="%.12g")
    status_diag.to_csv(output_dir / "m16e_reference_status_diagnostic.csv", index=False, float_format="%.12g")

    # Comparison plot: full hard gate and eligible physics signal.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fullp = metrics.query("scope == 'all_development' and model in ['m16d_route_analogue','m16e_hard_gate_route_fallback']").set_index("model")
    fullp[["mae_h", "medae_h", "p90_ae_h"]].plot(kind="bar", ax=axes[0])
    axes[0].set_title("All development: route vs fixed physics hard-gate")
    axes[0].set_ylabel("Hours"); axes[0].tick_params(axis="x", rotation=15)
    eligp = metrics.query("scope == 'physics_eligible' and model in ['m16e_physics','m16d_route_analogue','m16c_hierarchical_prior']").set_index("model")
    eligp[["mae_h", "medae_h", "p90_ae_h"]].plot(kind="bar", ax=axes[1])
    axes[1].set_title(f"Physics-eligible subset (n={int(eligible.sum())})")
    axes[1].set_ylabel("Hours"); axes[1].tick_params(axis="x", rotation=15)
    fig.tight_layout(); fig.savefig(output_dir / "m16e_physics_comparison.png", dpi=160); plt.close(fig)

    # Interpretability plot: physical estimate vs target, with stale/future labels
    # shown only as a forensic overlay (not used in selection/gating).
    fig, ax = plt.subplots(figsize=(8, 6))
    pe = ledger.loc[eligible].copy()
    ax.scatter(pe["pred_m16e_physics_h"], pe["target_tte_h"], s=22, alpha=0.65)
    lo = float(min(pe["pred_m16e_physics_h"].min(), pe["target_tte_h"].quantile(0.05)))
    hi = float(max(pe["pred_m16e_physics_h"].max(), pe["target_tte_h"].quantile(0.95)))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    ax.set_xlabel("M16E physics prediction (h)"); ax.set_ylabel("Reference TTE target (h)")
    ax.set_title("M16E physics-eligible OOF predictions (heavy stale-label tail visible)")
    fig.tight_layout(); fig.savefig(output_dir / "m16e_physics_scatter.png", dpi=160); plt.close(fig)

    selected_counts = selected["selected_config_id"].value_counts().to_dict()
    future_diag = status_diag.loc[status_diag["reference_eta_status"].eq("FUTURE_0_7D")]
    future_diag_rec = future_diag.iloc[0].to_dict() if len(future_diag) else {}
    summary = {
        "milestone": "M16E", "status": "MARITIME_PHYSICS_EXPERT_BUILT", "version": M16E_VERSION,
        "gate": gate, "standalone_promoted": False,
        "development_rows": 386, "blocked_old_final_rows": 53, "final_test_used_for_selection": False,
        "eligible_rows": int(eligible.sum()), "eligible_share": float(eligible.mean()),
        "eligibility_rule": "catalog-resolved destination confidence>=0.99; geodesic distance>=1nm; max causal 180/360m median SOG>=2kn",
        "distance_semantics": "great-circle lower bound; optional causal local-sinuosity multiplier tested as a route-detour proxy; no chart/water-mask routing dataset used",
        "speed_semantics": "causal recent median SOG with alternate-window fallback; clipped to 2..25 kn",
        "candidate_config_count": len(candidate_configs()), "outer_folds": 5, "inner_folds": M16E_INNER_FOLDS,
        "selection_rule": "eligible-only pooled inner-CV MAE; P90/MedAE/config_id deterministic tie-break",
        "residual_correction_rule": "none/global/destination-shrunk residual fit only on eligible inner/outer training rows",
        "physics_eligible_oof": {k: (int(v) if k == "n" else float(v)) for k, v in phy_e.to_dict().items() if k not in {"scope", "model"}},
        "route_on_physics_eligible_oof": {k: (int(v) if k == "n" else float(v)) for k, v in route_e.to_dict().items() if k not in {"scope", "model"}},
        "physics_mae_gain_h_vs_route_on_eligible": physics_gain_eligible,
        "physics_wins_vs_route_share_on_eligible": physics_win_share,
        "hard_gate_oof": {k: (int(v) if k == "n" else float(v)) for k, v in hard_m.to_dict().items() if k not in {"scope", "model"}},
        "route_oof": {k: (int(v) if k == "n" else float(v)) for k, v in route_m.to_dict().items() if k not in {"scope", "model"}},
        "hard_gate_mae_gain_h_vs_route": hard_gain, "hard_gate_fold_wins_vs_route": hard_fold_wins,
        "hard_gate_is_target_free": True,
        "future_0_7d_status_diagnostic_only": future_diag_rec,
        "reference_eta_status_used_for_selection_or_gate": False,
        "selected_configs_by_outer_fold": {str(int(r.outer_fold)): str(r.selected_config_id) for r in selected.itertuples()},
        "selected_config_counts": {str(k): int(v) for k, v in selected_counts.items()},
        "claim": "Nested development-only physics signal. The 53-row old final is hard-blocked; target-derived ETA status is diagnostic only.",
    }
    (output_dir / "M16E_SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")

    report = f"""# M16E — Maritime / Physics Expert\n\n## Decision\n\n**{gate}**. M16E is retained as an interpretable gating expert, not promoted as a new final model.\n\n## Method\n\nThe expert uses M16B catalog-resolved destination coordinates, great-circle remaining distance, causal recent median SOG, and an optional local-sinuosity route-detour proxy. It tests 12 predeclared configurations (180/360 minute speed window × geodesic/local-sinuosity distance × none/global/destination-shrunk residual correction). Each outer fold selects its configuration only with 4-fold inner CV. The 53 old-final MMSIs are never used.\n\nNo S-57/S-100 chart, bathymetric water mask, commercial routing API, or future trajectory is used; therefore the distance term is explicitly a lower-bound/proxy rather than a navigational route distance.\n\n## OOF result\n\n- Physics-eligible coverage: **{int(eligible.sum())}/386 ({eligible.mean()*100:.1f}%)**.\n- Physics MAE on eligible rows: **{float(phy_e['mae_h']):.2f} h** vs M16D route **{float(route_e['mae_h']):.2f} h**.\n- Physics MedAE: **{float(phy_e['medae_h']):.2f} h** vs route **{float(route_e['medae_h']):.2f} h**.\n- Physics wins paired absolute error on **{physics_win_share*100:.1f}%** of eligible rows.\n- Fixed target-free hard gate (physics when eligible, M16D otherwise): **{float(hard_m['mae_h']):.2f} h MAE** vs route **{float(route_m['mae_h']):.2f} h**, gain **{hard_gain:.2f} h**, with **{hard_fold_wins}/5** outer-fold wins.\n\n## Interpretation\n\nThe physics signal is highly informative when a canonical port and meaningful recent motion exist. The remaining tail is dominated by the fact that the company target is a reported/reference AIS ETA and may be stale or already in the past; a physical travel-time model cannot logically reproduce negative/stale TTE values. `reference_eta_status` is therefore used only after scoring for diagnostics and never for model selection or gating.\n\nM16G should use physics eligibility, distance, route support, alignment and recent-motion quality as target-free gating features rather than forcing physics onto every row.\n"""
    (output_dir / "M16E_REPORT.md").write_text(report)

    freeze_files = [
        "m16e_oof_predictions.csv", "m16e_physics_feature_audit.csv", "m16e_inner_search.csv",
        "m16e_outer_selected_configs.csv", "m16e_model_metrics.csv", "m16e_fold_metrics.csv",
        "m16e_reference_status_diagnostic.csv", "m16e_physics_comparison.png", "m16e_physics_scatter.png",
        "M16E_SUMMARY.json", "M16E_REPORT.md",
    ]
    freeze = {
        "milestone": "M16E", "status": "MARITIME_PHYSICS_EXPERT_FROZEN", "version": M16E_VERSION,
        "development_population": 386, "blocked_old_final_population": 53,
        "selection_population_rule": "M10 train+calibration only; 53-row old final hard-blocked",
        "outer_fold_source": "M16A frozen outer folds", "inner_selection_only": True,
        "final_test_used_for_selection": False, "candidate_config_count": len(candidate_configs()),
        "primary_metric": "eligible-only pooled OOF MAE hours", "gate": gate,
        "artifact_sha256": {name: sha(output_dir / name) for name in freeze_files},
    }
    (output_dir / "M16E_MARITIME_PHYSICS_FREEZE.json").write_text(json.dumps(freeze, indent=2) + "\n")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--output-dir", type=Path, default=R); args = ap.parse_args()
    print(json.dumps(run(args.output_dir), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
