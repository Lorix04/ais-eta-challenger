#!/usr/bin/env python3
"""Build and freeze the full-development M16G+M16H fresh-row scoring bundle.

This is the operational prerequisite discovered by M18G.  It does *not* open an
external holdout and does not change any historical OOF score.  Choices are
aggregated deterministically from previously frozen outer-fold selections, then
learnable components are refit on all 386 development rows.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m16a import assert_development_only
from ais_eta.m16b import DestinationResolver
from ais_eta.m16c import add_m16c_features, candidate_configs as c_configs
from ais_eta.m16d import candidate_configs as d_configs
from ais_eta.m16e import add_physics_features, candidate_configs as e_configs
from ais_eta.m16f import M16F_FEATURES, _build_sklearn_model, candidate_configs as f_configs, prepare_features
from ais_eta.m16g import M16G_EXPERT_NAMES, _classifier, build_gating_features
from ais_eta.m16h import M16H_NOMINAL_COVERAGE, build_error_features, candidate_floor, conformal_upper_quantile, fit_error_model
from ais_eta.m18g import M18G_SELECTIVE_RISK_THRESHOLD_H, sha256_file, validate_prediction_ledger
from ais_eta.m18g1 import (
    M18G1_FORBIDDEN_FRESH_COLUMNS, M18G1_REQUIRED_ROW_COLUMNS, M18G1_REQUIRED_STATE_COLUMNS,
    M18G1_VERSION, plurality_choice, score_fresh_rows,
)

R = ROOT / "reports"
MODELS = ROOT / "models"
DATA = ROOT / "data"
BUNDLE = MODELS / "m18g1_full_development_scoring_bundle.joblib"
REF = DATA / "derived/m10_reference_rows.pkl.gz"
STATES = DATA / "derived/m0c_ship_states.pkl.gz"
CATALOG = DATA / "static/m16b_destination_catalog.csv"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append_once(path: Path, marker: str, block: str) -> None:
    text = path.read_text() if path.exists() else ""
    if marker not in text:
        if text and not text.endswith("\n"):
            text += "\n"
        text += "\n" + block.strip() + "\n"
        path.write_text(text)


def selected_plurality(path: Path, column: str = "selected_config_id", *, where=None):
    df = pd.read_csv(path)
    if where is not None:
        df = where(df)
    selected, counts = plurality_choice(df[column].astype(str))
    return selected, counts


def config_by_id(configs, cid: str):
    found = [c for c in configs if c.config_id == cid]
    if len(found) != 1:
        raise AssertionError(f"config id not uniquely found: {cid}")
    return found[0]


def prepare_development() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_pickle(REF)
    final = raw.loc[raw["split"].eq("final_test")].copy()
    dev = raw.loc[raw["split"].isin(["train", "calibration"])].copy()
    if len(dev) != 386 or len(final) != 53:
        raise AssertionError("unexpected M18G1 development/final cardinality")
    assert_development_only(dev, final["mmsi"])
    dev = dev.sort_values("mmsi", kind="mergesort").reset_index(drop=True)

    catalog = pd.read_csv(CATALOG)
    resolver = DestinationResolver(catalog)
    resolved = resolver.transform(dev["destination_norm"])
    dev = pd.concat([dev.reset_index(drop=True), resolved.reset_index(drop=True)], axis=1)
    dev = add_physics_features(dev)
    dev = add_m16c_features(dev)

    # Refit input parity: recomputed target-free semantics must agree with the
    # historical frozen ledgers used by M16G/M16H evaluation.
    b = pd.read_csv(R / "m16b_destination_resolutions.csv").sort_values("mmsi", kind="mergesort").reset_index(drop=True)
    if list(dev.mmsi.astype(int)) != list(b.mmsi.astype(int)):
        raise AssertionError("M18G1 M16B row alignment drift")
    for col in ["canonical_destination", "resolution_method", "is_resolved_port", "is_non_specific", "is_route_expression"]:
        if not dev[col].astype(str).equals(b[col].astype(str)):
            raise AssertionError(f"M18G1 recomputed M16B drift: {col}")
    if not np.allclose(pd.to_numeric(dev["resolution_confidence"]), pd.to_numeric(b["resolution_confidence"]), equal_nan=True):
        raise AssertionError("M18G1 resolution confidence drift")

    e = pd.read_csv(R / "m16e_physics_feature_audit.csv").sort_values("mmsi", kind="mergesort").reset_index(drop=True)
    for col in [
        "physics_distance_gc_nm", "physics_bearing_to_port_deg", "physics_course_alignment_deg",
        "physics_recent_speed_max_kn", "physics_local_sinuosity_180m", "physics_local_sinuosity_360m",
    ]:
        if not np.allclose(pd.to_numeric(dev[col]), pd.to_numeric(e[col]), equal_nan=True, rtol=1e-10, atol=1e-10):
            raise AssertionError(f"M18G1 recomputed physics drift: {col}")
    if not dev["physics_eligible"].astype(bool).equals(e["physics_eligible"].astype(bool)):
        raise AssertionError("M18G1 physics eligibility drift")
    return dev, final, catalog


def oof_quality_and_base(dev: pd.DataFrame):
    g = pd.read_csv(R / "m16g_oof_predictions.csv").sort_values("mmsi", kind="mergesort").reset_index(drop=True)
    c = pd.read_csv(R / "m16c_oof_predictions.csv")
    d = pd.read_csv(R / "m16d_oof_predictions.csv")
    e = pd.read_csv(R / "m16e_oof_predictions.csv")
    q = (
        c[["mmsi", "m16c_deepest_support", "m16c_deepest_weight", "m16c_fallback_count"]]
        .merge(d[["mmsi", "m16d_neighbour_count", "m16d_nearest_distance", "m16d_median_neighbour_distance", "m16d_similarity_gap", "m16d_gate_used"]], on="mmsi")
        .merge(e[["mmsi", "physics_eligible", "resolution_confidence", "physics_distance_gc_nm", "physics_course_alignment_deg", "physics_recent_speed_max_kn", "m16e_effective_speed_kn", "m16e_distance_factor", "m16e_route_distance_proxy_nm"]], on="mmsi")
    )
    q = dev[["mmsi"]].merge(q, on="mmsi", how="left")
    if list(g.mmsi.astype(int)) != list(dev.mmsi.astype(int)):
        raise AssertionError("M18G1 M16G OOF row alignment drift")
    base = g[["pred_m16c_prior_h", "pred_m16d_route_h", "pred_m16e_physics_gate_h", "pred_m16f_tabular_h"]].to_numpy(float)
    quality = q.drop(columns=["mmsi"])
    return g, base, quality


def main() -> int:
    MODELS.mkdir(exist_ok=True)
    dev, final, catalog = prepare_development()

    c_id, c_votes = selected_plurality(R / "m16c_outer_selected_configs.csv")
    d_id, d_votes = selected_plurality(R / "m16d_outer_selected_configs.csv")
    e_id, e_votes = selected_plurality(R / "m16e_outer_selected_configs.csv")
    f_id, f_votes = selected_plurality(
        R / "m16f_outer_selected_configs.csv",
        where=lambda x: x.loc[x["family"].eq("hist_gb")],
    )
    meta_id, meta_votes = selected_plurality(R / "m16g_outer_selected_meta.csv", column="selected_meta_id")
    interval_id, interval_votes = selected_plurality(R / "m16h_outer_selected_interval.csv", column="selected_candidate")

    cc = config_by_id(c_configs(), c_id)
    dc = config_by_id(d_configs(), d_id)
    ec = config_by_id(e_configs(), e_id)
    fc = config_by_id(f_configs(), f_id)
    if fc.family != "hist_gb":
        raise AssertionError("M18G1 operational tabular expert must remain M16F hist_gb family")
    floor_h = candidate_floor(interval_id)
    if floor_h is None:
        raise AssertionError("M18G1 plurality selected non-adaptive M16H interval unexpectedly")

    # Full-development tabular refit.
    tab_model = _build_sklearn_model(fc)
    tab_train = prepare_features(dev)[M16F_FEATURES]
    tab_model.fit(tab_train, dev["target_tte_h"].to_numpy(float))

    # Full-development M16G meta refit.  Importantly, training inputs are the
    # official cross-fitted base-expert predictions, not in-sample refit preds.
    g, base_oof, quality_oof = oof_quality_and_base(dev)
    gating = build_gating_features(base_oof, quality_oof)
    oracle = np.argmin(np.abs(base_oof - g["target_tte_h"].to_numpy(float)[:, None]), axis=1).astype(int)
    meta_model = clone(_classifier(meta_id)).fit(gating, oracle)

    # Full-development M16H risk refit.  Error targets remain official nested-OOF
    # M16G errors.  Interval scale and confidence thresholds use cross-fitted M16H
    # risks, preserving out-of-fold calibration semantics before external labels.
    hframe = pd.DataFrame({
        "pred_m16c_prior_h": base_oof[:, 0],
        "pred_m16d_route_h": base_oof[:, 1],
        "pred_m16e_physics_gate_h": base_oof[:, 2],
        "pred_m16f_tabular_h": base_oof[:, 3],
        "pred_m16g_selected_h": g["pred_m16g_selected_h"].to_numpy(float),
        "m16g_selected_meta_id": g["m16g_selected_meta_id"].astype(str),
        "m16c_deepest_support": quality_oof["m16c_deepest_support"],
        "m16c_deepest_weight": quality_oof["m16c_deepest_weight"],
        "m16c_fallback_count": quality_oof["m16c_fallback_count"],
        "m16d_neighbour_count": quality_oof["m16d_neighbour_count"],
        "m16d_nearest_distance": quality_oof["m16d_nearest_distance"],
        "m16d_median_neighbour_distance": quality_oof["m16d_median_neighbour_distance"],
        "m16d_similarity_gap": quality_oof["m16d_similarity_gap"],
        "m16d_gate_used": quality_oof["m16d_gate_used"],
        "physics_eligible": quality_oof["physics_eligible"],
        "resolution_confidence": quality_oof["resolution_confidence"],
        "physics_distance_gc_nm": quality_oof["physics_distance_gc_nm"],
        "physics_course_alignment_deg": quality_oof["physics_course_alignment_deg"],
        "physics_recent_speed_max_kn": quality_oof["physics_recent_speed_max_kn"],
    })
    error_features = build_error_features(hframe)
    abs_error = np.abs(g["target_tte_h"].to_numpy(float) - g["pred_m16g_selected_h"].to_numpy(float))
    risk_model = fit_error_model(error_features, abs_error)
    h_oof = pd.read_csv(R / "m16h_oof_intervals.csv").sort_values("mmsi", kind="mergesort").reset_index(drop=True)
    if list(h_oof.mmsi.astype(int)) != list(dev.mmsi.astype(int)):
        raise AssertionError("M18G1 M16H OOF row alignment drift")
    risk_cv = h_oof["predicted_abs_error_h"].to_numpy(float)
    q1, q2 = np.quantile(risk_cv, [1/3, 2/3], method="linear")
    scale_scores = abs_error / (float(floor_h) + np.maximum(risk_cv, 0.0))
    interval_q = conformal_upper_quantile(scale_scores, M16H_NOMINAL_COVERAGE)
    half_cv = interval_q * (float(floor_h) + np.maximum(risk_cv, 0.0))
    empirical_cv_coverage = float(np.mean(abs_error <= half_cv))

    # Frozen M16D route memory aligned to the same sorted 386 development rows.
    seq_npz = np.load(R / "m16d_route_sequences.npz")
    seq = np.asarray(seq_npz[dc.representation], dtype=float)
    support = pd.read_csv(R / "m16d_route_support.csv")
    sup = support.loc[support["representation"].eq(dc.representation)].sort_values("mmsi", kind="mergesort")
    if list(sup.mmsi.astype(int)) != list(dev.mmsi.astype(int)) or len(seq) != len(dev):
        raise AssertionError("M18G1 route sequence alignment drift")

    bundle = {
        "version": M18G1_VERSION,
        "status": "FROZEN_FULL_DEVELOPMENT_SCORER",
        "development_rows": 386,
        "blocked_old_final_rows": 53,
        "historical_oof_score_changed": False,
        "external_holdout_opened": False,
        "labels_required_for_scoring": False,
        "configs": {
            "m16c_id": c_id,
            "m16d_id": d_id,
            "m16d": {"representation": dc.representation, "gate": dc.gate, "k": dc.k},
            "m16e_id": e_id,
            "m16f_id": f_id,
            "m16g_meta_id": meta_id,
            "m16h_interval_id": interval_id,
            "selection_rule": "plurality of already-frozen outer-fold selections; lexical tie-break; no new external labels",
        },
        "vote_counts": {
            "m16c": c_votes, "m16d": d_votes, "m16e": e_votes, "m16f_hist_gb": f_votes,
            "m16g_meta": meta_votes, "m16h_interval": interval_votes,
        },
        "training_frame": dev,
        "training_mmsis": dev["mmsi"].astype(int).tolist(),
        "catalog": catalog,
        "route_memory": {
            "representation": dc.representation,
            "sequences": seq,
            "targets_h": dev["target_tte_h"].to_numpy(float),
            "destinations": dev["canonical_destination"].fillna("UNKNOWN").astype(str).to_numpy(),
            "mmsis": dev["mmsi"].to_numpy(np.int64),
        },
        "objects": {
            "m16c_config": cc,
            "m16e_config": ec,
            "m16f_model": tab_model,
            "m16g_meta_model": meta_model,
            "m16h_risk_model": risk_model,
        },
        "feature_columns": {
            "gating": list(gating.columns),
            "risk": list(error_features.columns),
        },
        "uncertainty": {
            "nominal_marginal_coverage": M16H_NOMINAL_COVERAGE,
            "interval_candidate": interval_id,
            "interval_floor_h": float(floor_h),
            "interval_scale_q": float(interval_q),
            "calibration_source": "official M16G nested-OOF absolute errors normalized by official M16H cross-fitted predicted risk",
            "development_empirical_coverage_using_cross_fitted_risk": empirical_cv_coverage,
            "confidence_q1_h": float(q1),
            "confidence_q2_h": float(q2),
            "m18f_frozen_selective_risk_threshold_h": float(M18G_SELECTIVE_RISK_THRESHOLD_H),
            "selection_conditional_guarantee_claimed": False,
        },
        "input_contract": {
            "required_row_columns": list(M18G1_REQUIRED_ROW_COLUMNS),
            "required_state_columns": list(M18G1_REQUIRED_STATE_COLUMNS),
            "forbidden_fresh_columns": sorted(M18G1_FORBIDDEN_FRESH_COLUMNS),
            "fresh_states_must_be_causal": "recorded_at <= row.history_last_at; selected M16D window is applied by scorer",
        },
    }

    # Uncompressed joblib avoids timestamp-bearing compression headers and is
    # deterministic under the pinned environment/random seeds used here.
    joblib.dump(bundle, BUNDLE, compress=0, protocol=5)
    bundle_sha = sha(BUNDLE)
    (MODELS / "m18g1_full_development_scoring_bundle.joblib.sha256").write_text(f"{bundle_sha}  {BUNDLE.name}\n")

    # Consensus/refit audit table.
    vote_rows = []
    for component, counts in bundle["vote_counts"].items():
        selected = bundle["configs"].get({
            "m16c": "m16c_id", "m16d": "m16d_id", "m16e": "m16e_id", "m16f_hist_gb": "m16f_id",
            "m16g_meta": "m16g_meta_id", "m16h_interval": "m16h_interval_id",
        }[component])
        for choice, n in counts.items():
            vote_rows.append({"component": component, "choice": choice, "outer_fold_votes": int(n), "selected_for_full_refit": choice == selected})
    pd.DataFrame(vote_rows).sort_values(["component", "choice"]).to_csv(R / "m18g1_refit_consensus.csv", index=False)

    # Synthetic *unlabelled* smoke test: clone five prediction-time rows/history
    # under synthetic MMSIs, proving unseen-row scoring and M18G ledger validity.
    pick = dev.iloc[[0, 80, 160, 240, 320]].copy()
    original_mmsi = pick["mmsi"].astype(int).tolist()
    synthetic_mmsi = [990000001 + i for i in range(len(pick))]
    raw = pd.read_pickle(REF).sort_values("mmsi", kind="mergesort")
    smoke = raw.loc[raw["mmsi"].isin(original_mmsi)].copy().sort_values("mmsi", kind="mergesort").reset_index(drop=True)
    # Order exactly as selected development rows.
    smoke = smoke.set_index("mmsi").loc[original_mmsi].reset_index()
    smoke["mmsi"] = synthetic_mmsi
    drop = [c for c in smoke.columns if c in M18G1_FORBIDDEN_FRESH_COLUMNS]
    smoke = smoke.drop(columns=drop)
    states = pd.read_pickle(STATES)
    parts = []
    for old, new in zip(original_mmsi, synthetic_mmsi):
        z = states.loc[states["mmsi"].eq(old)].copy()
        z["mmsi"] = new
        parts.append(z)
    smoke_states = pd.concat(parts, ignore_index=True)
    smoke_ledger, smoke_detail = score_fresh_rows(bundle, smoke, smoke_states)
    validate_prediction_ledger(smoke_ledger, require_candidate=False)
    if any(c in smoke_ledger.columns for c in M18G1_FORBIDDEN_FRESH_COLUMNS):
        raise AssertionError("M18G1 smoke ledger leaked label columns")
    smoke_ledger2, _ = score_fresh_rows(bundle, smoke, smoke_states)
    pd.testing.assert_frame_equal(smoke_ledger, smoke_ledger2, check_exact=True)
    smoke_ledger.to_csv(R / "m18g1_blind_smoke_ledger.csv", index=False, float_format="%.12g")
    smoke_detail.to_csv(R / "m18g1_blind_smoke_detail.csv", index=False, float_format="%.12g")

    source_freezes = [
        "M16G_MIXTURE_FREEZE.json", "M16H_PROBABILISTIC_FREEZE.json", "M18A_BASELINE_FREEZE.json",
        "M18B_VALIDATION_FREEZE.json", "M18C_AUDIT_FREEZE.json", "M18D_ATTRIBUTION_FREEZE.json",
        "M18E_ABLATION_FREEZE.json", "M18F_SELECTIVE_FREEZE.json", "M18G_PROMOTION_FREEZE.json",
    ]
    source_hash = {name: sha(R / name) for name in source_freezes}
    schema = {
        "version": M18G1_VERSION,
        "required_row_columns": list(M18G1_REQUIRED_ROW_COLUMNS),
        "required_state_columns": list(M18G1_REQUIRED_STATE_COLUMNS),
        "forbidden_fresh_columns": sorted(M18G1_FORBIDDEN_FRESH_COLUMNS),
        "ledger_columns": list(smoke_ledger.columns),
        "detail_only_columns": [c for c in smoke_detail.columns if c not in smoke_ledger.columns],
    }
    (R / "M18G1_SCORING_INPUT_SCHEMA.json").write_text(json.dumps(schema, indent=2) + "\n")

    registration = {
        "milestone": "M18G1",
        "version": M18G1_VERSION,
        "status": "FULL_DEVELOPMENT_M16G_M16H_SCORING_BUNDLE_REGISTERED",
        "bundle_path": str(BUNDLE.relative_to(ROOT)),
        "bundle_sha256": bundle_sha,
        "development_population": 386,
        "blocked_old_final_population": 53,
        "fresh_holdout_opened": False,
        "holdout_labels_observed": False,
        "historical_m16g_oof_score_changed": False,
        "point_candidate_registered": False,
        "baseline_operational_scorer_registered": True,
        "selective_operational_scorer_registered": True,
        "scoring_requires_labels": False,
        "consensus_configs": bundle["configs"],
        "uncertainty": bundle["uncertainty"],
        "source_freeze_sha256": source_hash,
        "smoke_ledger_sha256": sha(R / "m18g1_blind_smoke_ledger.csv"),
        "input_schema_sha256": sha(R / "M18G1_SCORING_INPUT_SCHEMA.json"),
        "next_gate_step": "receive a truly fresh unlabeled cohort, hash its provenance, score it with scripts/111_m18g1_score_blind.py, then seal prediction ledger + opening manifest before label reveal",
    }
    (R / "M18G1_SCORING_BUNDLE_REGISTRATION.json").write_text(json.dumps(registration, indent=2) + "\n")

    readiness = pd.DataFrame([
        ("gate_freeze", True, "Historical M18G gate remains unchanged"),
        ("full_development_m16g_m16h_scoring_bundle_registered", True, bundle_sha),
        ("fresh_holdout_received_unlabelled", False, "No fresh external cohort supplied"),
        ("fresh_holdout_provenance_manifest_sealed", False, "Must be done before scoring/opening"),
        ("blind_prediction_ledger_sealed_before_labels", False, "Now executable via scripts/111_m18g1_score_blind.py once unlabeled cohort exists"),
        ("point_candidate_registered_pre_label", False, "M18E promoted no point challenger; optional for baseline/selective external validation"),
        ("labels_observed", False, "Must remain false until prediction ledger/opening manifest are sealed"),
        ("one_time_evaluation_executed", False, "Not executed"),
    ], columns=["check", "passed_now", "note"])
    readiness.to_csv(R / "m18g1_opening_readiness_after_bundle.csv", index=False)

    report = f"""# M18G1 — Full-Development M16G + M16H Scoring Bundle\n\n## Decision\n\n**FULL-DEVELOPMENT BLIND SCORER REGISTERED. HOLDOUT STILL SEALED.**\n\nM18G identified that the repository contained only nested-OOF M16G/M16H evaluation artifacts. M18G1 converts the frozen recipe into an operational scorer for previously unseen rows without changing the historical OOF benchmark.\n\n## Refit rule\n\nNo new external label or fresh-holdout information is used. For each tunable M16C/M16D/M16E/M16F/M16G/M16H component, the operational configuration is the deterministic plurality of the already-frozen outer-fold selections (lexical tie-break only). The selected recipe is then refit on all **386** development owners.\n\nSelected operational recipe:\n\n- M16C: `{c_id}`\n- M16D: `{d_id}`\n- M16E: `{e_id}`\n- M16F HistGB: `{f_id}`\n- M16G meta: `{meta_id}`\n- M16H interval: `{interval_id}`\n\nM16H risk is fit on official M16G nested-OOF absolute errors. Its full-development interval uses the already-selected adaptive interval family and a conformal upper quantile computed from **cross-fitted M16H risk**; the resulting development empirical marginal coverage is **{empirical_cv_coverage:.2%}**. This is not a guarantee under external distribution shift or selective filtering.\n\n## Blind scoring contract\n\n`scripts/111_m18g1_score_blind.py` consumes a target-free current-row table plus causal AIS state history, emits the exact M18G baseline ledger fields, refuses target/reference-ETA columns, validates unique sample keys and writes SHA-256 seals.\n\nA five-row synthetic-unlabelled smoke cohort was scored twice with byte/value-identical ledgers. This is a software test only and is not model performance evidence.\n\n## Freeze\n\nBundle SHA-256: `{bundle_sha}`\n\nThe 53 old-final owners remain blocked from development. No fresh holdout has been received or opened and no M18G one-shot evaluation has run.\n"""
    (R / "M18G1_REPORT.md").write_text(report)

    doc = f"""# M18G1 Full-Development Scoring Bundle\n\nThe trusted serialized scorer is `models/{BUNDLE.name}` (SHA-256 `{bundle_sha}`). Only load this repository-generated joblib file; joblib/pickle files are executable serialization formats.\n\n## Required sequence before any fresh labels\n\n1. Receive a **fresh unlabeled** row table and matching causal state history.\n2. Freeze/hash a provenance manifest for that cohort.\n3. Run `python scripts/111_m18g1_score_blind.py --rows <rows> --states <states> --output <prediction.csv> --holdout-id <id> --provenance-sha256 <sha> --opening-manifest <manifest.json>`.\n4. Independently verify bundle, ledger and manifest hashes.\n5. Only then may the label ledger be revealed and `scripts/109_m18g_evaluate_external_holdout.py` be run once.\n\nThe blind scorer rejects `{', '.join(sorted(M18G1_FORBIDDEN_FRESH_COLUMNS))}` if present in fresh rows. The full schema is frozen in `reports/M18G1_SCORING_INPUT_SCHEMA.json`.\n\nThe operational refit is not a new development score and must not replace M16G's frozen nested-OOF **185.882697 h MAE** claim. It exists solely to produce pre-label predictions for unseen rows.\n"""
    (ROOT / "docs/M18G1_FULL_DEVELOPMENT_SCORING_BUNDLE.md").write_text(doc)

    append_once(
        ROOT / "README.md",
        "## M18G — Full-Development Scoring Bundle — REGISTERED",
        f"""## M18G — Full-Development Scoring Bundle — REGISTERED

The external gate discovered one remaining deployment blocker: M16G/M16H existed only as nested-OOF development evidence. The repository now contains a frozen full-development operational refit at `models/{BUNDLE.name}` (SHA-256 `{bundle_sha}`) plus `scripts/111_m18g1_score_blind.py` for generating a target-free, hash-sealed M18G prediction ledger on a truly fresh cohort before labels are revealed. Historical M16G OOF metrics are unchanged and the fresh holdout remains unopened. See `reports/M18G1_REPORT.md` and `docs/M18G1_FULL_DEVELOPMENT_SCORING_BUNDLE.md`.""",
    )
    append_once(
        ROOT / "WORKFLOW.md",
        "## M18G — Full-development blind scorer registration",
        """## M18G — Full-development blind scorer registration

1. Keep the historical M18G gate and all M16/M17/M18A-F freezes immutable.
2. Aggregate each operational hyperparameter/model-family choice only from already-frozen outer-fold selections using deterministic plurality voting; do not use fresh-holdout labels.
3. Refit M16C/M16D/M16E/M16F, M16G meta gating and M16H risk on all 386 development owners. Train M16G meta on official cross-fitted base-expert predictions and train M16H risk on official nested-OOF M16G errors.
4. Freeze the serialized scorer, input schema, source-freeze hashes and label-free smoke ledger.
5. On receipt of a fresh **unlabelled** cohort, first seal provenance, then run `scripts/111_m18g1_score_blind.py`; the resulting ledger and opening manifest must be SHA-256 sealed before label reveal.
6. Only after those hashes are fixed may the one-shot `scripts/109_m18g_evaluate_external_holdout.py` evaluation be run. Historical M16G 185.882697 h OOF MAE is not replaced by the operational refit.
""",
    )
    append_once(
        ROOT / "REFERENCES.md",
        "## M18G full-development scoring-bundle methodology",
        """## M18G full-development scoring-bundle methodology

- scikit-learn, *Common pitfalls and recommended practices — Data leakage*: https://scikit-learn.org/dev/common_pitfalls.html — preprocessing and fitting must be learned from training data only; the external test must not influence model choices.
- Oliveira, Orenstein, Ramos & Romano (2024), *Split Conformal Prediction and Non-Exchangeable Data*, JMLR 25(225):1–38. https://jmlr.org/papers/v25/23-1553.html — used as context for why external temporal/spatiotemporal shift may weaken nominal conformal claims even when the development calibration is valid.
- Bao et al. (2025), *CAP: A General Algorithm for Online Selective Conformal Prediction with FCR Control*, JMLR 26(287):1–74. https://jmlr.org/papers/v26/24-0452.html — methodological context for post-selection coverage; M18G/M18F therefore keep retained-set coverage empirical rather than claiming automatic selection-conditional validity.
""",
    )
    append_once(
        ROOT / "docs/MODEL_CARD.md",
        "## Operational external-scoring form (M18G)",
        f"""## Operational external-scoring form (M18G)

A full-development operational refit of the frozen M16G+M16H recipe is registered at `models/{BUNDLE.name}`. It is **not** a new benchmark result: the official development evidence remains the nested-OOF M16G/M16H artifacts. The operational bundle exists solely to score previously unseen, target-free rows before external labels are exposed. Its fresh-row scorer requires causal AIS history for M16D and rejects target/reference-ETA columns. External promotion remains subject to the frozen one-shot M18G gate.
""",
    )

    # Freeze all new M18G1 artifacts; source M18G freeze remains untouched.
    new_files = [
        "models/m18g1_full_development_scoring_bundle.joblib",
        "models/m18g1_full_development_scoring_bundle.joblib.sha256",
        "reports/M18G1_SCORING_BUNDLE_REGISTRATION.json",
        "reports/M18G1_SCORING_INPUT_SCHEMA.json",
        "reports/M18G1_REPORT.md",
        "reports/m18g1_refit_consensus.csv",
        "reports/m18g1_blind_smoke_ledger.csv",
        "reports/m18g1_blind_smoke_detail.csv",
        "reports/m18g1_opening_readiness_after_bundle.csv",
        "docs/M18G1_FULL_DEVELOPMENT_SCORING_BUNDLE.md",
    ]
    freeze = {
        "milestone": "M18G1",
        "version": M18G1_VERSION,
        "status": "FULL_DEVELOPMENT_SCORING_BUNDLE_FROZEN",
        "m18g_gate_modified": False,
        "fresh_holdout_opened": False,
        "holdout_labels_observed": False,
        "artifact_sha256": {rel: sha(ROOT / rel) for rel in new_files},
        "source_freeze_sha256": source_hash,
    }
    (R / "M18G1_SCORING_BUNDLE_FREEZE.json").write_text(json.dumps(freeze, indent=2) + "\n")

    # Update project state only after all assertions/smoke tests pass.
    sp = ROOT / "PROJECT_STATE.json"
    state = json.loads(sp.read_text())
    if state.get("current_phase") != "M18G_DONE_FULL_DEVELOPMENT_SCORING_BUNDLE_REGISTERED":
        state["state_version"] = int(state.get("state_version", 0)) + 1
    state["updated_at"] = "2026-09-23T15:12:00+02:00"
    state["current_phase"] = "M18G_DONE_FULL_DEVELOPMENT_SCORING_BUNDLE_REGISTERED"
    state["current_task"] = "M18G1 registered and froze the full-development M16G+M16H label-free scorer. Fresh holdout remains sealed; next step is to score a truly fresh unlabeled cohort and seal its ledger before label reveal."
    state["m18g1_summary"] = {
        "status": registration["status"],
        "bundle_sha256": bundle_sha,
        "fresh_holdout_opened": False,
        "holdout_labels_observed": False,
        "blind_scoring_ready": True,
        "point_candidate_registered": False,
        "historical_m16g_oof_score_changed": False,
        "next_action": registration["next_gate_step"],
    }
    line = "M18G1 full-development scorer: M16C/D/E/F + M16G meta + M16H risk/interval refit on all 386 development owners using only previously frozen selection decisions; label-free blind scorer and SHA-256 sealing path ready; external holdout still unopened"
    completed = list(state.get("completed", []))
    if line not in completed:
        completed.append(line)
    state["completed"] = completed
    sp.write_text(json.dumps(state, indent=2) + "\n")

    print(f"PASS M18G1 bundle: {bundle_sha}")
    print(f"PASS config consensus: C={c_id} D={d_id} E={e_id} F={f_id} G={meta_id} H={interval_id}")
    print(f"PASS full-dev interval cross-fitted calibration coverage: {empirical_cv_coverage:.6f}")
    print(f"PASS synthetic blind smoke rows: {len(smoke_ledger)}")
    print("PASS fresh holdout remains sealed; labels observed: false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
