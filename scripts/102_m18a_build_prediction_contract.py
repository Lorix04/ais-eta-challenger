#!/usr/bin/env python3
"""M18A: formalize the prediction contract and freeze the M16G baseline.

No model is trained and no prediction artifact is rewritten.  The script audits
existing frozen artifacts, writes the machine/human-readable contract, records
baseline metrics, and cryptographically anchors the pre-M18 champion.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m18a import (  # noqa: E402
    M18A_DECISION_TIME_FIELD,
    M18A_PRIMARY_TASK_ID,
    M18A_TARGET_FIELD,
    M18A_VERSION,
    assert_feature_contract,
    sha256_file,
    validate_contract_dict,
)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    e = np.asarray(y, float) - np.asarray(p, float)
    ae = np.abs(e)
    return {
        "n": int(len(ae)),
        "mae_h": float(np.mean(ae)),
        "medae_h": float(np.median(ae)),
        "rmse_h": float(np.sqrt(np.mean(e ** 2))),
        "p90_ae_h": float(np.quantile(ae, 0.90)),
        "p95_ae_h": float(np.quantile(ae, 0.95)),
        "bias_h": float(np.mean(e)),
    }


def _sha_map(paths: list[Path]) -> dict[str, str]:
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}


def main() -> int:
    ref_path = ROOT / "data/derived/m10_reference_rows.pkl.gz"
    dev_manifest_path = R / "M16A_DEVELOPMENT_MANIFEST.json"
    g_summary_path = R / "M16G_SUMMARY.json"
    g_freeze_path = R / "M16G_MIXTURE_FREEZE.json"
    h_summary_path = R / "M16H_SUMMARY.json"
    h_freeze_path = R / "M16H_PROBABILISTIC_FREEZE.json"
    final_freeze_path = R / "M16_FINAL_FREEZE.json"
    m14_path = ROOT / "dist/ais_eta_takehome_submission_final.zip"

    ref = pd.read_pickle(ref_path)
    manifest = json.loads(dev_manifest_path.read_text())
    gs = json.loads(g_summary_path.read_text())
    gf = json.loads(g_freeze_path.read_text())
    hs = json.loads(h_summary_path.read_text())
    hf = json.loads(h_freeze_path.read_text())
    final_freeze = json.loads(final_freeze_path.read_text())
    ledger = pd.read_csv(R / "m16g_oof_predictions.csv")

    # Baseline integrity before a new contract is allowed to exist.
    assert len(ref) == 439 and ref.mmsi.nunique() == 439
    assert int(ref.split.isin(["train", "calibration"]).sum()) == 386
    assert int(ref.split.eq("final_test").sum()) == 53
    assert len(ledger) == 386 and ledger.mmsi.nunique() == 386
    blocked = set(int(x) for x in manifest["blocked_old_final_mmsi"])
    assert set(ledger.mmsi.astype(int)).isdisjoint(blocked)
    assert len(blocked) == 53
    assert gf["gate"] == "PASS_MOE_STABLE_GAIN"
    assert gf["final_test_used_for_selection"] is False
    assert gs["target_derived_gating_features_used"] is False
    assert hf["p50_is_frozen_m16g"] is True
    assert hf["final_test_used_for_calibration"] is False
    assert final_freeze["fresh_untouched_holdout_required"] is True

    # M16G feature families are target-free by construction.  Explicitly audit
    # the strict M10 current-snapshot features again under the new contract.
    assert_feature_contract(manifest["current_features"])

    m = _metrics(ledger.target_tte_h.to_numpy(float), ledger.pred_m16g_selected_h.to_numpy(float))
    for key, summary_key in [("mae_h", "mixture_mae_h"), ("medae_h", "mixture_medae_h"), ("p90_ae_h", "mixture_p90_ae_h")]:
        if not np.isclose(m[key], float(gs[summary_key]), rtol=0, atol=1e-9):
            raise AssertionError(f"M16G metric drift: {key}")

    contract = {
        "milestone": "M18A",
        "version": M18A_VERSION,
        "status": "PREDICTION_CONTRACT_FROZEN",
        "purpose": "Freeze problem semantics and M16G baseline before any new data source, feature family, validation redesign, or external holdout is examined.",
        "primary_prediction_contract": {
            "task_id": M18A_PRIMARY_TASK_ID,
            "decision_time": M18A_DECISION_TIME_FIELD,
            "label_source": M18A_TARGET_FIELD,
            "target_formula": "target_tte_h = parsed Tracks.eta - Tracks.last_update",
            "eta_year_resolution": "choose the temporally nearest valid previous/current/next calendar year because AIS Message 5 ETA omits year",
            "unit": "hours",
            "supervision_unit": "one Tracks row / one MMSI",
            "population": "ship rows with parseable company-provided ETA reference",
            "semantic_status": "company reference ETA; not observed ATA and not berth/all-fast arrival",
            "location_semantics": "unspecified by the provided label; must not be silently relabelled as Pilot Boarding Place, port entrance, berth or all-fast",
            "strict_reference_policy": "retain every parseable company reference, including past/stale/far-future values; quality regimes are diagnostics unless a future target policy is predeclared before evaluation",
        },
        "leakage_boundary": {
            "positions_rule": "Positions.recorded_at <= Tracks.last_update",
            "current_snapshot_rule": "only fields available in the Tracks snapshot at Tracks.last_update; Tracks.eta is target-only",
            "training_aggregate_rule": "target-derived priors/aggregates may be fitted only on the training partition visible to the current nested fold",
            "representation_learning_rule": "outer/inner validation owners remain excluded whenever trajectory representation training would otherwise expose their history",
            "prohibited_prediction_inputs": [
                "Tracks.eta / eta_reference_raw / eta_reference_dt / target_tte_h",
                "reference_eta_status or any target-derived quality bucket",
                "future Positions rows or any post-decision event",
                "absolute error, residual, coverage or hindsight diagnostic",
                "old 53-row M10 final set for selection, fitting or threshold tuning",
                "fresh external holdout labels before the final promotion gate",
            ],
        },
        "frozen_baseline": {
            "point_champion": "M16G nested development-only mixture-of-experts",
            "point_version": gs["version"],
            "point_gate": gs["gate"],
            "development_rows": 386,
            "blocked_old_final_rows": 53,
            "outer_folds": int(gs["outer_folds"]),
            "point_metrics": m,
            "best_single_baseline": gs["best_single_baseline"],
            "gain_h_vs_best_single": float(gs["gain_h_vs_best_single"]),
            "uncertainty_sidecar": "M16H development-only empirical 80% residual/conformal-style interval around frozen M16G P50",
            "m16h_nominal_coverage": float(hs["nominal_coverage"]),
            "m16h_observed_pooled_coverage": float(hs["pooled_coverage"]),
            "official_company_submission_remains": "M14/M10 strict reference benchmark; M16G is a post-submission development challenger for a new untouched holdout only",
        },
        "selection_and_validation_policy": {
            "reused_development_set": "386 M10 train+calibration MMSIs only",
            "old_final_policy": "53 M10 final MMSIs permanently blocked for all post-final selection/tuning",
            "model_selection_after_m18a": "no further claim may use repeated improvement on the same 386 rows as external validation",
            "future_validation_requirement": "validation redesign must preserve complete owner/voyage groups and temporal/owner separation appropriate to the evaluated claim",
            "metrics_required": ["MAE", "MedAE", "RMSE", "P90 absolute error", "P95 absolute error", "bias", "fold/regime breakdown", "paired uncertainty/CI where applicable"],
        },
        "fresh_holdout_policy": {
            "status": "NOT_AVAILABLE_IN_REPOSITORY",
            "selection_use_allowed": False,
            "feature_engineering_use_allowed": False,
            "threshold_tuning_use_allowed": False,
            "label_peeking_allowed": False,
            "required_before_opening": [
                "record immutable file/content hashes",
                "verify schema and decision-time semantics without reading target-dependent performance",
                "predeclare candidate and promotion criteria",
                "score the frozen baseline and candidate under the same contract",
            ],
            "promotion_principle": "a fresh holdout is an evaluation lockbox, not another development set",
        },
        "target_change_policy": {
            "new_operational_targets": ["ATA at specified port location", "Pilot Boarding Place", "port entrance", "berth arrival", "all-fast"],
            "rule": "any operational-event target becomes a new versioned task with its own event definition, location, timestamp provenance and leakage boundary; it cannot replace this contract retrospectively",
        },
    }
    validate_contract_dict(contract)

    contract_json = R / "M18A_PREDICTION_CONTRACT.json"
    contract_json.write_text(json.dumps(contract, indent=2) + "\n")

    metrics = pd.DataFrame([
        {"artifact": "M16G", "scope": "386-row development nested OOF", **m},
        {
            "artifact": "M10/M14 official submission",
            "scope": "53-row once-opened old final; context only, permanently blocked for M18 selection",
            "n": 53,
            "mae_h": 312.01882325980154,
            "medae_h": 25.330220729072487,
            "rmse_h": np.nan,
            "p90_ae_h": 560.4,
            "p95_ae_h": np.nan,
            "bias_h": np.nan,
        },
    ])
    metrics.to_csv(R / "m18a_frozen_baseline_metrics.csv", index=False, float_format="%.12g")

    report = f"""# M18A — Prediction Contract & Frozen Baseline\n\n## Decision\n\n**PREDICTION_CONTRACT_FROZEN — NO MODEL CHANGE.**\n\nM18A freezes the semantics of the primary company-reference ETA task and cryptographically anchors the development-only M16G champion before M18 introduces any new validation design or external information source. No model is trained, no M16/M17 prediction ledger is rewritten, and the already-observed 53-row M10 final set remains permanently blocked from post-final selection.\n\n## Primary contract\n\n- **Decision time:** `Tracks.last_update`.\n- **Label:** company-provided `Tracks.eta`, treated as a reference ETA rather than observed ATA.\n- **Target:** `target_tte_h = parsed Tracks.eta - Tracks.last_update`.\n- **AIS year policy:** nearest valid previous/current/next calendar year around the decision time because Message 5 ETA carries month/day/hour/minute but not year.\n- **Supervision:** one Tracks row / one MMSI.\n- **History boundary:** `Positions.recorded_at <= Tracks.last_update`.\n- **Location/event caveat:** the supplied label does not establish Pilot Boarding Place, port-entry, berth or all-fast semantics. Any such target must be a separately versioned task.\n\n## Frozen baseline\n\n- M16G nested development-only MoE: **{m['mae_h']:.3f} h MAE**, **{m['medae_h']:.3f} h MedAE**, **{m['p90_ae_h']:.3f} h P90**, N=386.\n- M16G gain vs frozen best single expert: **{float(gs['gain_h_vs_best_single']):.3f} h**.\n- M16H remains the uncertainty sidecar around the unchanged M16G P50: nominal 80% interval, observed pooled development coverage **{float(hs['pooled_coverage'])*100:.2f}%**.\n- M14/M10 remains the official company-submission benchmark; M16G is not retroactively substituted into that already-opened final result.\n\n## Fresh-holdout rule\n\nAny newly issued holdout is a lockbox. It cannot be used for feature selection, threshold tuning, target-policy cleanup or model choice. Hashes, schema checks and promotion criteria must be fixed before target-dependent scoring.\n\n## What M18A deliberately does not do\n\nM18A does not add weather, routing, port-operations features, new target cleaning, a new split, or a new learner. Those are later hypotheses and must be evaluated under this frozen contract or explicitly declare a new contract version.\n"""
    (R / "M18A_REPORT.md").write_text(report)

    human = f"""# M18A Prediction Contract — Company `Tracks.eta` Reference\n\nThis document is the human-readable counterpart of `reports/M18A_PREDICTION_CONTRACT.json`. It is frozen before any M18 feature/data experiment.\n\n## 1. Prediction question\n\nAt **`Tracks.last_update`**, predict the number of hours to the company-provided **`Tracks.eta` reference**. The benchmark target is:\n\n`target_tte_h = parsed Tracks.eta - Tracks.last_update`\n\nThis is a company-reference ETA task. It is **not** automatically an observed ATA, Pilot Boarding Place ETA, port-entry ETA, berth ETA or all-fast timestamp. IMO port-call data models distinguish ETA/ATA and the location to which a timestamp refers, so future operational-event work must create a new target contract instead of silently changing this one.\n\n## 2. ETA parsing\n\nAIS Message 5 ETA is month/day/hour/minute in UTC and contains no year. The existing frozen M10 parser evaluates the previous, current and next calendar year around `Tracks.last_update` and chooses the nearest valid timestamp. M18A preserves this rule byte-for-byte; changing it is a target-contract change.\n\n## 3. Decision-time boundary\n\nAllowed information must exist at or before `Tracks.last_update`. Historical AIS rows satisfy:\n\n`Positions.recorded_at <= Tracks.last_update`\n\nPrediction-time inputs must not include `Tracks.eta`, parsed ETA fields, target/status derivatives, residuals, future positions, post-arrival events, or any hindsight diagnostic. Train-derived aggregates are allowed only when fitted inside the training partition visible to the current nested fold.\n\n## 4. Frozen baseline\n\nThe frozen point baseline is **M16G** (`{gs['version']}`), evaluated development-only on 386 M10 train+calibration MMSIs with the 53 old-final MMSIs hard-blocked. Its locked OOF metrics are **{m['mae_h']:.3f} h MAE**, **{m['medae_h']:.3f} h MedAE**, **{m['p90_ae_h']:.3f} h P90**. M16H is the frozen uncertainty sidecar and does not alter the M16G P50.\n\nThe official company submission remains M14/M10. M18A does not rewrite the already-opened 53-row final result.\n\n## 5. Fresh holdout\n\nA future company holdout is an evaluation lockbox, not a new development set. Before target-dependent scoring: hash the inputs, verify schema/decision-time semantics, freeze the candidate and promotion criteria. Holdout labels cannot be used to clean the target, choose features or tune thresholds.\n\n## 6. Contract changes\n\nAny change to prediction event, location, label source, decision time, ETA year resolution, allowed information boundary or holdout-use policy requires a new versioned contract. New port-operational targets such as PBP, port entrance, berth or all-fast are separate tasks.\n\n## External standards / evidence\n\n- IMO Compendium — JIT port-call data: ETA/ATA are timestamps at specified locations; Pilot Boarding Place and berth are separately defined locations.\n- USCG NAVCEN — AIS Message 5 ETA is encoded as `MMDDHHMM` UTC.\n- Recent reproducible AIS ETA work uses voyage-level grouping to keep complete voyages out of both train and evaluation partitions, reducing leakage.\n"""
    (ROOT / "docs/M18A_PREDICTION_CONTRACT.md").write_text(human)

    # The M18A freeze hashes the baseline inputs plus the newly generated contract
    # outputs.  It deliberately excludes PROJECT_STATE to avoid self-referential
    # hashes while still anchoring every prediction artifact that matters.
    immutable_baseline = [
        ref_path,
        dev_manifest_path,
        g_summary_path,
        g_freeze_path,
        R / "m16g_oof_predictions.csv",
        h_summary_path,
        h_freeze_path,
        final_freeze_path,
        R / "M17G_STRONG_SSL_FREEZE.json",
        R / "M17H_CONSERVATIVE_INTEGRATION_FREEZE.json",
        m14_path,
    ]
    contract_artifacts = [
        R / "M18A_PREDICTION_CONTRACT.json",
        R / "M18A_REPORT.md",
        R / "m18a_frozen_baseline_metrics.csv",
        ROOT / "docs/M18A_PREDICTION_CONTRACT.md",
    ]
    freeze = {
        "milestone": "M18A",
        "version": M18A_VERSION,
        "status": "FROZEN_BASELINE_AND_CONTRACT",
        "model_change": False,
        "prediction_artifact_rewrite": False,
        "point_champion": "M16G",
        "uncertainty_sidecar": "M16H",
        "development_population": 386,
        "blocked_old_final_population": 53,
        "fresh_holdout_opened": False,
        "immutable_baseline_sha256": _sha_map(immutable_baseline),
        "contract_artifact_sha256": _sha_map(contract_artifacts),
    }
    (R / "M18A_BASELINE_FREEZE.json").write_text(json.dumps(freeze, indent=2) + "\n")

    print("PASS M18A prediction contract frozen; no model change")
    print(f"PASS M16G baseline: n={m['n']} MAE={m['mae_h']:.6f} h MedAE={m['medae_h']:.6f} h P90={m['p90_ae_h']:.6f} h")
    print("PASS old M10 final blocked: 53/53; fresh holdout remains unopened")
    print(f"PASS baseline hashes anchored: {len(freeze['immutable_baseline_sha256'])}")
    print(f"PASS contract artifacts hashed: {len(freeze['contract_artifact_sha256'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
