#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
DOCS = ROOT / "docs"
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m18g import (  # noqa: E402
    M18G_BOOTSTRAP_REPS,
    M18G_BOOTSTRAP_SEED,
    M18G_CRITICAL_SLICE_COLUMNS,
    M18G_EXPECTED_DEV_ROWS,
    M18G_EXPECTED_OLD_FINAL_ROWS,
    M18G_POINT_MAX_P90_RATIO,
    M18G_POINT_MAX_SLICE_REGRESSION_RATIO,
    M18G_POINT_MIN_CRITICAL_SLICE_N,
    M18G_REQUIRED_LABEL_COLUMNS,
    M18G_REQUIRED_PREDICTION_COLUMNS,
    M18G_SELECTIVE_MAX_COVERAGE,
    M18G_SELECTIVE_MIN_COVERAGE,
    M18G_SELECTIVE_MIN_INTERVAL_COVERAGE,
    M18G_SELECTIVE_RISK_THRESHOLD_H,
    M18G_VERSION,
    sample_key,
    sha256_file,
    validate_gate_contract,
)


def _write_templates() -> None:
    pred_cols = list(M18G_REQUIRED_PREDICTION_COLUMNS) + ["candidate_pred_h"]
    pd.DataFrame(columns=pred_cols).to_csv(R / "m18g_external_prediction_ledger_template.csv", index=False)
    pd.DataFrame(columns=list(M18G_REQUIRED_LABEL_COLUMNS)).to_csv(R / "m18g_external_label_ledger_template.csv", index=False)


def _forbidden_sample_keys() -> pd.DataFrame:
    raw = pd.read_pickle(ROOT / "data/derived/m10_reference_rows.pkl.gz")
    out = raw[["mmsi", "last_update", "split"]].copy()
    out["decision_time"] = pd.to_datetime(out["last_update"]).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["sample_key"] = [sample_key(m, t) for m, t in zip(out.mmsi, out.decision_time)]
    out = out[["sample_key", "mmsi", "decision_time", "split"]].sort_values(["sample_key"], kind="mergesort").reset_index(drop=True)
    if len(out) != 439 or out.sample_key.nunique() != 439:
        raise AssertionError("M18G original supervised-row key cardinality drift")
    if int(out["split"].eq("final_test").sum()) != M18G_EXPECTED_OLD_FINAL_ROWS:
        raise AssertionError("M18G old-final key cardinality drift")
    return out


def main() -> int:
    R.mkdir(exist_ok=True)
    DOCS.mkdir(exist_ok=True)

    m18b = json.loads((R / "M18B_VALIDATION_PROTOCOL.json").read_text())
    m18f = json.loads((R / "M18F_SELECTIVE_ETA.json").read_text())
    if m18b["fresh_external_lockbox"]["opened"] is not False:
        raise AssertionError("M18B fresh holdout must remain sealed")
    if m18f["scope"]["fresh_holdout_opened"] is not False:
        raise AssertionError("M18F fresh holdout must remain sealed")
    if abs(float(m18f["balanced_policy"]["frozen_pre_holdout_risk_threshold_h"]) - M18G_SELECTIVE_RISK_THRESHOLD_H) > 1e-12:
        raise AssertionError("M18F frozen threshold drift")

    forbidden = _forbidden_sample_keys()
    forbidden.to_csv(R / "m18g_forbidden_sample_keys.csv", index=False)
    _write_templates()

    candidate_registry = pd.DataFrame([
        {
            "track": "POINT_MODEL",
            "candidate_id": "UNREGISTERED",
            "policy_status": "BLOCKED_PRE_LABEL_REGISTRATION_REQUIRED",
            "external_opening_allowed_now": False,
            "reason": "M18E promoted no new external point model; any future candidate must be registered and hashed before holdout labels are observed.",
        },
        {
            "track": "SELECTIVE_ETA_LAYER",
            "candidate_id": "M18F_BALANCED_80",
            "policy_status": "POLICY_FROZEN_SCORING_BUNDLE_STILL_REQUIRED",
            "external_opening_allowed_now": False,
            "reason": "Risk threshold and policy are frozen, but a fresh-row full-development M16G/M16H scoring bundle has not been registered in this repository.",
        },
    ])
    candidate_registry.to_csv(R / "m18g_candidate_registry.csv", index=False)

    checklist = pd.DataFrame([
        ("gate_freeze", True, "M18G thresholds/procedure frozen before labels"),
        ("fresh_holdout_received", False, "No fresh external holdout supplied"),
        ("fresh_holdout_provenance_manifest", False, "Must be supplied and hashed before opening"),
        ("prediction_ledger_sealed_before_labels", False, "Must be generated without targets then SHA-256 sealed"),
        ("point_candidate_registered_pre_label", False, "No M18E point candidate promoted/registered"),
        ("full_development_m16g_m16h_scoring_bundle_registered", False, "Current artifacts are development OOF evaluation artifacts, not a registered fresh-row scoring bundle"),
        ("labels_observed", False, "Must remain false until every prior item required for the chosen track passes"),
        ("one_time_evaluation_executed", False, "Not executed in M18G build"),
    ], columns=["check", "passed_now", "note"])
    checklist.to_csv(R / "m18g_opening_readiness_checklist.csv", index=False)

    gate = {
        "milestone": "M18G",
        "version": M18G_VERSION,
        "status": "EXTERNAL_PROMOTION_GATE_FROZEN__HOLDOUT_REMAINS_SEALED",
        "scope": {
            "development_population": M18G_EXPECTED_DEV_ROWS,
            "blocked_old_final_population": M18G_EXPECTED_OLD_FINAL_ROWS,
            "fresh_holdout_opened": False,
            "holdout_labels_observed": False,
            "development_reused_for_new_model_selection": False,
            "post_holdout_retuning_allowed": False,
            "one_time_evaluation_executed": False,
        },
        "opening_readiness": {
            "ready_now": False,
            "blockers": [
                "fresh external holdout not supplied",
                "blind prediction ledger not yet generated/sealed",
                "no point candidate registered before labels",
                "no full-development M16G/M16H fresh-row scoring bundle registered",
            ],
            "important_note": "The evaluation gate is executable now, but holdout opening is deliberately blocked until a pre-label scoring bundle and sealed prediction ledger exist.",
        },
        "blind_opening_protocol": [
            "1_freeze_holdout_schema_provenance_and_source_hashes_without_labels",
            "2_register_baseline_and_optional_candidate_scoring_bundle_hashes_before_labels",
            "3_generate_prediction_risk_interval_and_prelabel_slice_ledger_without_target_columns",
            "4_assert_zero_exact_sample_key_overlap_with_439_original_supervised_rows",
            "5_sha256_seal_prediction_ledger_and_opening_manifest",
            "6_only_then_reveal_target_tte_h_labels",
            "7_run_scripts/109_m18g_evaluate_external_holdout.py_once",
            "8_freeze_result_no_retuning_on_same_holdout",
        ],
        "point_model_gate": {
            "baseline": "registered full-development operational form of frozen M16G",
            "candidate": "must be registered and hashed before label reveal; currently UNREGISTERED",
            "primary_metric": "MAE_h",
            "secondary_metrics": ["MedAE_h", "RMSE_h", "P90_absolute_error_h", "mean_signed_error_h"],
            "paired_bootstrap_unit": "sample/owner row",
            "paired_bootstrap_reps": M18G_BOOTSTRAP_REPS,
            "paired_bootstrap_seed": M18G_BOOTSTRAP_SEED,
            "primary_mae_gain_must_be_positive": True,
            "paired_bootstrap_ci95_lower_bound_must_exceed_h": 0.0,
            "max_p90_ratio": M18G_POINT_MAX_P90_RATIO,
            "critical_slice_columns": list(M18G_CRITICAL_SLICE_COLUMNS),
            "critical_slice_min_n": M18G_POINT_MIN_CRITICAL_SLICE_N,
            "max_critical_slice_regression_ratio": M18G_POINT_MAX_SLICE_REGRESSION_RATIO,
            "slice_rule": "Only prediction-time/pre-label slices are gating slices; target-derived slices remain diagnostics only after the decision.",
        },
        "selective_eta_gate": {
            "candidate": "M18F_BALANCED_80",
            "point_prediction": "same registered M16G prediction",
            "uncertainty": "same registered M16H risk/interval",
            "frozen_risk_threshold_h": M18G_SELECTIVE_RISK_THRESHOLD_H,
            "realized_coverage_min": M18G_SELECTIVE_MIN_COVERAGE,
            "realized_coverage_max": M18G_SELECTIVE_MAX_COVERAGE,
            "retained_mae_must_be_below_full_mae": True,
            "deferred_mae_must_exceed_retained_mae": True,
            "risk_abs_error_spearman_must_be_positive": True,
            "retained_empirical_interval_coverage_min": M18G_SELECTIVE_MIN_INTERVAL_COVERAGE,
            "selection_conditional_conformal_guarantee": False,
        },
        "failure_policy": {
            "point_gate_failure": "KEEP_FROZEN_M16G; do not tune using this holdout",
            "selective_gate_failure": "M18F remains development-only/advisory; do not retune threshold on this holdout",
            "next_attempt_after_failure": "Any revised model/policy requires a different future external cohort.",
        },
        "original_supervised_row_exclusion": {
            "forbidden_key_file": "reports/m18g_forbidden_sample_keys.csv",
            "rows": int(len(forbidden)),
            "old_final_rows_in_forbidden_set": int(forbidden["split"].eq("final_test").sum()),
            "key_definition": "SHA256(MMSI|decision_time_UTC)",
        },
    }
    validate_gate_contract(gate)
    (R / "M18G_EXTERNAL_PROMOTION_GATE.json").write_text(json.dumps(gate, indent=2) + "\n")

    opening_template = {
        "status": "TEMPLATE_NOT_AN_OPENING_RECORD",
        "gate_freeze_sha256": "FILL_AFTER_M18G_FREEZE",
        "holdout_id": "FILL_BEFORE_LABEL_REVEAL",
        "holdout_provenance_sha256": "FILL_BEFORE_LABEL_REVEAL",
        "prediction_ledger_sha256": "FILL_BEFORE_LABEL_REVEAL",
        "registered_scoring_bundle_sha256": "FILL_BEFORE_LABEL_REVEAL",
        "point_candidate_registered": False,
        "point_candidate_bundle_sha256": None,
        "labels_observed_when_prediction_ledger_sealed": False,
        "opening_count_before_this_evaluation": 0,
    }
    (R / "M18G_OPENING_MANIFEST_TEMPLATE.json").write_text(json.dumps(opening_template, indent=2) + "\n")

    policy = f"""# M18G Frozen External Promotion Gate

## Status

**Gate frozen; fresh holdout still sealed.** M18G is a governance/evaluation milestone, not another model-development iteration.

## One-shot order

1. Freeze the external cohort provenance/schema while labels remain unavailable.
2. Register SHA-256 hashes of the full-development baseline scorer and, if applicable, point-candidate scorer.
3. Generate a **blind prediction ledger** containing predictions, M16H risk/interval and the four predeclared prediction-time slices. It must contain no target/error columns.
4. Reject any exact `MMSI + decision_time` row already present among the original 439 supervised rows.
5. SHA-256 seal the prediction ledger/opening manifest.
6. Reveal labels only after steps 1–5.
7. Execute the external evaluator once.
8. Freeze the result. Failure cannot be repaired by tuning on the same holdout.

## Point-model gate

The frozen M18B requirements are executable: positive MAE gain; paired owner-row bootstrap ({M18G_BOOTSTRAP_REPS:,} reps, seed {M18G_BOOTSTRAP_SEED}) with 95% lower bound > 0 h; candidate P90 / baseline P90 <= {M18G_POINT_MAX_P90_RATIO:.2f}; and no more than {(M18G_POINT_MAX_SLICE_REGRESSION_RATIO-1):.0%} MAE regression in any predeclared critical slice with n >= {M18G_POINT_MIN_CRITICAL_SLICE_N}.

Critical slices are frozen **before labels**: `{', '.join(M18G_CRITICAL_SLICE_COLUMNS)}`. Target-derived horizon/reference-status slices may be reported after evaluation but cannot determine promotion.

## Selective-ETA gate

M18F `BALANCED_80` uses the unchanged pre-holdout risk threshold **{M18G_SELECTIVE_RISK_THRESHOLD_H:.9f} h**. On the fresh holdout it must realize {M18G_SELECTIVE_MIN_COVERAGE:.0%}–{M18G_SELECTIVE_MAX_COVERAGE:.0%} automatic coverage, retained MAE below full-population M16G MAE, deferred MAE above retained MAE, positive risk-vs-absolute-error Spearman association, and empirical retained interval coverage >= {M18G_SELECTIVE_MIN_INTERVAL_COVERAGE:.0%}. This remains an empirical selected-set audit; no selection-conditional conformal guarantee is claimed.

## Current blocker

The gate code is ready, but **opening is not ready**. The repository currently contains development OOF evaluation artifacts, not a registered full-development M16G/M16H fresh-row scoring bundle, and M18E did not promote a new point candidate. M18G therefore refuses to pretend that an external holdout can be opened safely today.
"""
    (DOCS / "M18G_EXTERNAL_PROMOTION_GATE.md").write_text(policy)

    report = f"""# M18G — Frozen External Promotion Gate

## Decision

**EXTERNAL PROMOTION GATE FROZEN; HOLDOUT NOT OPENED.**

M18G converts the M18B/M18F intentions into executable gate code and immutable pre-label thresholds. It deliberately produces no external accuracy number because no fresh holdout was supplied and no labels were opened.

## What is frozen

- point-model primary/secondary metrics and paired bootstrap uncertainty;
- 10,000 paired bootstrap repetitions with seed 18,023;
- P90 non-regression ratio <= 1.02;
- prediction-time critical-slice regression ratio <= 1.10 for n >= 20;
- the four critical slice dimensions;
- M18F BALANCED_80 risk threshold {M18G_SELECTIVE_RISK_THRESHOLD_H:.9f} h and its external validation criteria;
- zero reuse of the original 439 supervised `(MMSI, decision_time)` rows;
- one opening / no retuning on the same holdout.

## Why opening remains blocked

M18E promoted no new point model, and the current M16G/M16H artifacts are nested OOF development-evaluation artifacts rather than a registered full-development scorer for unseen rows. A blind prediction ledger therefore cannot yet be honestly produced for a new cohort. This is a **readiness blocker**, not a failed model result.

## Deliverables

See `docs/M18G_EXTERNAL_PROMOTION_GATE.md`, `reports/M18G_EXTERNAL_PROMOTION_GATE.json`, `reports/m18g_candidate_registry.csv`, `reports/m18g_opening_readiness_checklist.csv`, the blind prediction/label templates and `scripts/109_m18g_evaluate_external_holdout.py`.
"""
    (R / "M18G_REPORT.md").write_text(report)

    freeze_files = [
        "reports/M18G_EXTERNAL_PROMOTION_GATE.json",
        "reports/M18G_OPENING_MANIFEST_TEMPLATE.json",
        "reports/m18g_candidate_registry.csv",
        "reports/m18g_opening_readiness_checklist.csv",
        "reports/m18g_forbidden_sample_keys.csv",
        "reports/m18g_external_prediction_ledger_template.csv",
        "reports/m18g_external_label_ledger_template.csv",
        "docs/M18G_EXTERNAL_PROMOTION_GATE.md",
        "reports/M18G_REPORT.md",
        "scripts/109_m18g_evaluate_external_holdout.py",
    ]
    freeze = {
        "milestone": "M18G",
        "version": M18G_VERSION,
        "status": "FROZEN_EXTERNAL_GATE_HOLDOUT_SEALED",
        "fresh_holdout_opened": False,
        "holdout_labels_observed": False,
        "one_time_evaluation_executed": False,
        "opening_ready_now": False,
        "development_population": M18G_EXPECTED_DEV_ROWS,
        "blocked_old_final_population": M18G_EXPECTED_OLD_FINAL_ROWS,
        "m18a_freeze_sha256": sha256_file(R / "M18A_BASELINE_FREEZE.json"),
        "m18b_freeze_sha256": sha256_file(R / "M18B_VALIDATION_FREEZE.json"),
        "m18c_freeze_sha256": sha256_file(R / "M18C_AUDIT_FREEZE.json"),
        "m18d_freeze_sha256": sha256_file(R / "M18D_ATTRIBUTION_FREEZE.json"),
        "m18e_freeze_sha256": sha256_file(R / "M18E_ABLATION_FREEZE.json"),
        "m18f_freeze_sha256": sha256_file(R / "M18F_SELECTIVE_FREEZE.json"),
        "m16g_oof_sha256": sha256_file(R / "m16g_oof_predictions.csv"),
        "m16h_oof_intervals_sha256": sha256_file(R / "m16h_oof_intervals.csv"),
        "artifact_sha256": {rel: sha256_file(ROOT / rel) for rel in freeze_files},
    }
    (R / "M18G_PROMOTION_FREEZE.json").write_text(json.dumps(freeze, indent=2) + "\n")

    state_path = ROOT / "PROJECT_STATE.json"
    state = json.loads(state_path.read_text())
    if state.get("current_phase") != "M18G_DONE_FROZEN_EXTERNAL_PROMOTION_GATE":
        state["state_version"] = int(state.get("state_version", 0)) + 1
    state["updated_at"] = "2026-09-23T14:16:00+02:00"
    state["current_phase"] = "M18G_DONE_FROZEN_EXTERNAL_PROMOTION_GATE"
    state["current_task"] = "M18G froze the one-shot blind external promotion gate. No holdout labels were opened; opening remains blocked until a full-development scoring bundle and blind prediction ledger are registered before labels."
    line = "M18G frozen external promotion gate: blind prediction-ledger sealing before labels, paired-bootstrap/P90/critical-slice point gate, frozen M18F selective external gate, original-row exclusion keys, one-shot no-retuning rule; holdout remains sealed because fresh-row scoring bundle/candidate registration is incomplete"
    completed = list(state.get("completed", []))
    if line not in completed:
        completed.append(line)
    state["completed"] = completed
    state["m18g_summary"] = {
        "status": "EXTERNAL_PROMOTION_GATE_FROZEN_HOLDOUT_SEALED",
        "fresh_holdout_opened": False,
        "holdout_labels_observed": False,
        "one_time_evaluation_executed": False,
        "opening_ready_now": False,
        "point_candidate_registered": False,
        "selective_policy_registered": "M18F_BALANCED_80",
        "frozen_selective_risk_threshold_h": M18G_SELECTIVE_RISK_THRESHOLD_H,
        "point_bootstrap_reps": M18G_BOOTSTRAP_REPS,
        "point_bootstrap_seed": M18G_BOOTSTRAP_SEED,
        "point_p90_ratio_max": M18G_POINT_MAX_P90_RATIO,
        "critical_slice_regression_ratio_max": M18G_POINT_MAX_SLICE_REGRESSION_RATIO,
        "critical_slice_min_n": M18G_POINT_MIN_CRITICAL_SLICE_N,
        "forbidden_original_supervised_rows": int(len(forbidden)),
        "blocked_old_final_rows": M18G_EXPECTED_OLD_FINAL_ROWS,
        "next_action": "register a full-development M16G/M16H fresh-row scoring bundle and, if relevant, a point candidate before receiving/revealing fresh-holdout labels",
    }
    state_path.write_text(json.dumps(state, indent=2) + "\n")

    print(json.dumps(gate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
