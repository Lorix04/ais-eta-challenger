#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m18a import sha256_file, validate_contract_dict  # noqa: E402
from ais_eta.m18b import (  # noqa: E402
    M18B_BOOTSTRAP_REPS,
    M18B_BOOTSTRAP_SEED,
    M18B_EMBARGO_H,
    M18B_TEMPORAL_BLOCKS,
    M18B_TEST_BLOCKS,
    M18B_VERSION,
    build_forward_membership,
    build_validation_manifest,
    regression_metrics,
    validate_protocol_dict,
)

REF = ROOT / "data/derived/m10_reference_rows.pkl.gz"


def _sha_map(paths: list[Path]) -> dict[str, str]:
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha256_file(p) for p in paths}


def _metrics_by_block(manifest: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    x = manifest.merge(pred[["mmsi", "pred_m16g_selected_h"]], on="mmsi", how="left", validate="one_to_one")
    if x["pred_m16g_selected_h"].isna().any():
        raise AssertionError("M18B missing frozen M16G OOF prediction")
    rows = []
    rows.append({"slice": "ALL_386_LEGACY_OOF_DIAGNOSTIC", **regression_metrics(x.target_tte_h, x.pred_m16g_selected_h)})
    for block, g in x.groupby("m18b_temporal_block", sort=True):
        rows.append({"slice": f"TEMPORAL_BLOCK_{int(block)}_LEGACY_OOF_DIAGNOSTIC", **regression_metrics(g.target_tte_h, g.pred_m16g_selected_h)})
    strict = x[x["m18b_strict_forward_test"]]
    rows.append({"slice": "STRICT_TEST_UNION_LEGACY_OOF_DIAGNOSTIC_ONLY", **regression_metrics(strict.target_tte_h, strict.pred_m16g_selected_h)})
    return pd.DataFrame(rows)


def main() -> int:
    contract = json.loads((R / "M18A_PREDICTION_CONTRACT.json").read_text())
    validate_contract_dict(contract)
    m18a_freeze = json.loads((R / "M18A_BASELINE_FREEZE.json").read_text())
    if m18a_freeze["fresh_holdout_opened"] is not False:
        raise AssertionError("M18B refuses to run after fresh holdout opening")

    raw = pd.read_pickle(REF)
    dev = raw.loc[raw["split"].isin(["train", "calibration"])].copy()
    final = raw.loc[raw["split"].eq("final_test")].copy()
    if len(dev) != 386 or len(final) != 53:
        raise AssertionError("unexpected M18B dev/final cardinality")

    a = pd.read_csv(R / "m16a_oof_predictions.csv")
    b = pd.read_csv(R / "m16b_destination_resolutions.csv")
    g = pd.read_csv(R / "m16g_oof_predictions.csv")
    outer_map = dict(zip(a.mmsi.astype(int), a.m16a_outer_fold.astype(int)))
    destination_map = dict(zip(b.mmsi.astype(int), b.canonical_destination.fillna("UNKNOWN").astype(str)))
    manifest, boundaries = build_validation_manifest(
        dev,
        blocked_old_final_mmsi=final.mmsi.astype(int).tolist(),
        outer_fold_by_mmsi=outer_map,
        canonical_destination_by_mmsi=destination_map,
    )
    membership, windows = build_forward_membership(manifest)

    # Add destination-cold status for TEST rows without turning destination text
    # into an authoritative port label.
    mdest = manifest[["mmsi", "canonical_destination"]]
    membership = membership.merge(mdest, on="mmsi", how="left", validate="many_to_one")
    for fold in sorted(membership.m18b_forward_fold.unique()):
        f = membership.m18b_forward_fold.eq(fold)
        train_dest = set(membership.loc[f & membership.m18b_role.eq("TRAIN"), "canonical_destination"].astype(str))
        test = f & membership.m18b_role.eq("TEST")
        membership.loc[test, "destination_seen_in_train"] = membership.loc[test, "canonical_destination"].astype(str).isin(train_dest)
    membership["destination_seen_in_train"] = membership["destination_seen_in_train"].fillna(False).astype(bool)

    # Strict invariants.
    assert manifest.mmsi.nunique() == 386
    assert set(manifest.mmsi.astype(int)).isdisjoint(set(final.mmsi.astype(int)))
    strict_test = membership.loc[membership.m18b_role.eq("TEST"), ["m18b_forward_fold", "mmsi"]]
    if strict_test.duplicated("mmsi").any():
        raise AssertionError("M18B strict test owner appears in more than one forward fold")
    if len(strict_test) != int(manifest.m18b_strict_forward_test.sum()):
        raise AssertionError("M18B strict test union mismatch")
    if not windows.owner_overlap_n.eq(0).all():
        raise AssertionError("M18B owner overlap detected")

    manifest.to_csv(R / "m18b_validation_manifest.csv", index=False, float_format="%.12g", date_format="%Y-%m-%dT%H:%M:%S")
    membership.to_csv(R / "m18b_forward_membership.csv", index=False, float_format="%.12g", date_format="%Y-%m-%dT%H:%M:%S")
    windows.to_csv(R / "m18b_forward_windows.csv", index=False, float_format="%.12g", date_format="%Y-%m-%dT%H:%M:%S")
    diagnostic = _metrics_by_block(manifest, g)
    diagnostic.to_csv(R / "m18b_legacy_oof_temporal_diagnostics.csv", index=False, float_format="%.12g")

    block_counts = manifest.groupby("m18b_temporal_block").size().astype(int).to_dict()
    test_union_n = int(manifest.m18b_strict_forward_test.sum())
    protocol = {
        "milestone": "M18B",
        "version": M18B_VERSION,
        "status": "VALIDATION_PROTOCOL_FROZEN_NO_MODEL_CHANGE",
        "m18a_contract_version": contract["version"],
        "model_change": False,
        "prediction_artifact_rewrite": False,
        "development_population": 386,
        "blocked_old_final_population": 53,
        "owner_unit": "MMSI (one supervised Tracks row per MMSI under M18A)",
        "owner_grouping_note": "Because the primary benchmark has one target row per MMSI, owner-grouped and row-grouped outer evaluation coincide; repeated AIS history rows are inputs, not independent supervised labels.",
        "strict_forward_temporal_design": {
            "temporal_blocks": M18B_TEMPORAL_BLOCKS,
            "block_assignment": "near-equal-count contiguous decision-time blocks; identical timestamps never split",
            "block_counts": {str(int(k)): int(v) for k, v in block_counts.items()},
            "test_blocks": list(M18B_TEST_BLOCKS),
            "forward_folds": len(M18B_TEST_BLOCKS),
            "warmup_blocks": [0, 1],
            "strict_test_union_n": test_union_n,
            "strict_test_union_fraction": float(test_union_n / 386),
            "embargo_h": M18B_EMBARGO_H,
            "embargo_rationale": "matches the longest 6 h causal AIS history used by the frozen route representation and prevents overlapping decision-history windows at train/test boundaries",
            "fit_rule": "all preprocessing, train-derived aggregates, expert fitting, meta-selection and calibration must be refit using TRAIN only inside each forward fold",
            "purged_rows_are_training_forbidden": True,
            "future_rows_are_training_forbidden": True,
            "legacy_oof_metrics_are_forward_generalization": False,
        },
        "reporting_metrics": {
            "primary": "MAE_h",
            "secondary": ["MedAE_h", "RMSE_h", "P90_absolute_error_h", "mean_signed_error_h"],
            "fold_reporting": ["pooled strict-test union", "macro mean across forward folds", "each forward fold separately"],
            "selection_forbidden_slices": "slice metrics are diagnostics/robustness checks and cannot be used to tune after test labels are observed",
        },
        "predeclared_diagnostic_slices": [
            "forward fold / decision-time block",
            "M18A reference ETA status",
            "destination seen-vs-unseen in forward-train (diagnostic proxy only)",
            "target horizon regime; to be audited in M18D, not used as a prediction-time feature",
        ],
        "cross_port_validation": {
            "status": "WITHHELD_NO_AUTHORITATIVE_PORT_EVENT_LABEL",
            "reason": "M18A target is company Tracks.eta reference and noisy destination text does not establish an authoritative arrival port/event.",
            "allowed_now": "destination-family support diagnostics only; no cross-port generalization claim",
        },
        "fresh_external_lockbox": {
            "opened": False,
            "selection_use_allowed": False,
            "required_before_open": [
                "hash/schema/provenance freeze",
                "M18A target semantics confirmation or explicit new contract version",
                "executable full-development baseline/candidate recipes frozen",
                "promotion metrics and critical slices frozen",
            ],
            "paired_uncertainty": {
                "unit": "owner/MMSI row",
                "method": "paired bootstrap of baseline MAE minus candidate MAE",
                "reps": M18B_BOOTSTRAP_REPS,
                "seed": M18B_BOOTSTRAP_SEED,
            },
            "promotion_gate": {
                "primary_mae_gain_must_be_positive": True,
                "paired_bootstrap_ci95_lower_bound_must_exceed_h": 0.0,
                "candidate_p90_ratio_vs_baseline_max": 1.02,
                "critical_slice_regression_ratio_max": 1.10,
                "critical_slice_min_n": 20,
                "single_opening_no_retuning_after_labels": True,
            },
        },
        "legacy_m16g_oof_diagnostic_only": {
            "ledger": "reports/m16g_oof_predictions.csv",
            "purpose": "sanity-check chronology heterogeneity only",
            "not_a_claim": "These predictions came from legacy balanced hash folds; temporal slicing them does not create forward-temporal generalization evidence.",
        },
        "temporal_boundaries": [pd.Timestamp(x).isoformat() for x in boundaries],
    }
    validate_protocol_dict(protocol)
    (R / "M18B_VALIDATION_PROTOCOL.json").write_text(json.dumps(protocol, indent=2) + "\n")

    report = f"""# M18B — Validation Redesign\n\n## Decision\n\n**VALIDATION_PROTOCOL_FROZEN — NO MODEL CHANGE.**\n\nM18B keeps the M18A target contract and M16G/M16H baseline untouched. It replaces vague validation language with an executable, predeclared protocol for future M18 experiments.\n\n## What changed\n\n- The 386 development rows are partitioned into **5 contiguous decision-time blocks** with counts {block_counts}. Identical timestamps are never split.\n- Strict forward evaluation uses **3 expanding windows**: blocks 2, 3 and 4 are test windows; blocks 0–1 form warm-up history.\n- A **{M18B_EMBARGO_H:.0f} h embargo** is enforced before every test window, matching the longest causal AIS history used by the current route representation.\n- Strict forward test union: **{test_union_n}/386 rows ({100*test_union_n/386:.1f}%)**, each appearing as TEST exactly once.\n- Every strict test MMSI is absent from its fold training owners. With one supervised row per MMSI, owner-grouped validation is the relevant grouping unit for the M18A benchmark.\n- Cross-port claims are explicitly withheld because `Tracks.eta` plus destination text does not provide an authoritative arrival port/event label.\n- The fresh external holdout remains sealed and now has a predeclared paired-bootstrap promotion gate.\n\n## Critical distinction\n\n`m18b_legacy_oof_temporal_diagnostics.csv` slices the already-frozen M16G OOF ledger by chronology, but **it is not forward-temporal validation**: those predictions were produced under the legacy balanced M16A folds. Genuine M18B forward evidence requires refitting preprocessing, train-derived aggregates, experts, meta-strategy and calibration inside each forward TRAIN partition.\n\n## Strict forward windows\n\n{windows.to_markdown(index=False)}\n\n## Why 6 h embargo\n\nThe frozen route representation uses causal 3 h/6 h histories. A six-hour gap prevents a training decision window immediately adjacent to a test decision from overlapping the test row's longest historical context.\n\n## External lockbox promotion rule\n\nOn a genuinely fresh company holdout, promotion requires a positive MAE gain whose paired owner-level 95% bootstrap interval is entirely above zero, P90 no worse than 1.02× the baseline, and no >10% MAE regression in predeclared critical slices with at least 20 rows. No tuning is allowed after lockbox labels are exposed.\n\n## Research basis\n\nRecent reproducible AIS ETA work explicitly groups complete voyages across train/validation/test and reports a separate chronological robustness split, emphasizing that preprocessing and leakage control can dominate apparent model gains. General time-series validation guidance likewise recommends future-facing splits rather than IID K-fold when adjacent observations are correlated. M18B adapts those principles to this project's one-row-per-MMSI primary contract rather than pretending the primary benchmark contains repeated supervised voyage rows.\n"""
    (R / "M18B_REPORT.md").write_text(report)

    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    human = f"""# M18B Validation Protocol\n\nM18B freezes how future M18 candidates must be evaluated under the M18A prediction contract. It does **not** retrain or replace M16G.\n\n## Primary supervised unit\n\nOne `Tracks` target row per MMSI. Therefore the primary benchmark has 386 unique supervised owners, not many target rows per voyage. AIS position histories remain causal input sequences and must not be treated as independent labels.\n\n## Forward-temporal design\n\nFive contiguous near-equal-count blocks are formed from `Tracks.last_update`; equal timestamps remain in the same block. Test blocks are 2, 3 and 4. Each fold trains only on prior blocks, removes the six hours immediately preceding the test start, and never trains on future blocks.\n\nStrict forward membership is frozen in `reports/m18b_forward_membership.csv`. Any candidate claiming M18B forward performance must rebuild every train-derived component inside those TRAIN rows. Re-slicing old OOF predictions is diagnostic only.\n\n## Cross-port limitation\n\nNo leave-one-port-out claim is authorized yet. `Tracks.eta` is a company reference ETA and destination text is not an authoritative port-call event label. Destination seen/unseen is retained only as a diagnostic proxy until M18C or an external authoritative target establishes the event/port.\n\n## Fresh holdout\n\nThe future company lockbox remains unopened. Before opening, freeze input hashes/schema, executable full-development recipes, target semantics, metrics, critical slices and the promotion gate. After labels are exposed there is no retuning.\n\n## External methodological references\n\n- Guerreiro et al. (2026), *A Reproducible Workflow for AIS-Based ETA Forecasting* — voyage-grouped partitions, independent held-out test and chronological robustness analysis: https://www.mdpi.com/2571-9394/8/4/70\n- scikit-learn cross-validation guidance — time-series data should be evaluated on future observations rather than IID folds; `TimeSeriesSplit` supports a gap/embargo: https://scikit-learn.org/dev/modules/cross_validation.html#time-series-split\n- MAPEX (2026) — disjoint monthly test periods and encounter-level grouping motivated by temporal correlation in AIS trajectories: https://www.mdpi.com/2079-8954/14/5/536\n"""
    (docs / "M18B_VALIDATION_PROTOCOL.md").write_text(human)

    artifacts = [
        R / "M18B_VALIDATION_PROTOCOL.json",
        R / "M18B_REPORT.md",
        R / "m18b_validation_manifest.csv",
        R / "m18b_forward_membership.csv",
        R / "m18b_forward_windows.csv",
        R / "m18b_legacy_oof_temporal_diagnostics.csv",
        docs / "M18B_VALIDATION_PROTOCOL.md",
    ]
    freeze = {
        "milestone": "M18B",
        "version": M18B_VERSION,
        "status": "FROZEN_VALIDATION_REDESIGN",
        "model_change": False,
        "m18a_contract_changed": False,
        "fresh_holdout_opened": False,
        "development_population": 386,
        "blocked_old_final_population": 53,
        "strict_forward_test_union_n": test_union_n,
        "strict_forward_folds": len(M18B_TEST_BLOCKS),
        "embargo_h": M18B_EMBARGO_H,
        "m18a_baseline_freeze_sha256": sha256_file(R / "M18A_BASELINE_FREEZE.json"),
        "m16g_oof_sha256": sha256_file(R / "m16g_oof_predictions.csv"),
        "validation_artifact_sha256": _sha_map(artifacts),
    }
    (R / "M18B_VALIDATION_FREEZE.json").write_text(json.dumps(freeze, indent=2) + "\n")

    print("PASS M18B validation redesign frozen; no model change")
    print(f"PASS temporal blocks={M18B_TEMPORAL_BLOCKS} counts={block_counts}")
    print(f"PASS strict forward folds={len(M18B_TEST_BLOCKS)} test_union={test_union_n}/386 embargo={M18B_EMBARGO_H:.0f}h")
    print("PASS owner overlap=0 in every forward fold; old-final 53/53 blocked")
    print("PASS legacy M16G temporal slices labelled DIAGNOSTIC ONLY, not forward evidence")
    print("PASS cross-port claim withheld; fresh external lockbox remains unopened")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
