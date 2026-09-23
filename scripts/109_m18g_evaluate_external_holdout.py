#!/usr/bin/env python3
"""One-shot M18G external holdout evaluator.

DO NOT run this on a real holdout until the blind prediction ledger has been
sealed before labels and the opening manifest references the frozen M18G gate.
This script refuses to overwrite an existing output directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m18g import (  # noqa: E402
    evaluate_point_candidate,
    evaluate_selective_layer,
    sha256_file,
    validate_gate_contract,
    validate_label_ledger,
    validate_prediction_ledger,
)

ACK = "I_ACKNOWLEDGE_M18G_ONE_TIME_HOLDOUT_OPENING"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--opening-manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--acknowledge", required=True)
    args = ap.parse_args()
    if args.acknowledge != ACK:
        raise SystemExit(f"refusing holdout opening: --acknowledge must equal {ACK}")

    out = Path(args.output_dir)
    if out.exists():
        raise SystemExit("refusing to overwrite an existing M18G evaluation directory")

    gate = json.loads((R / "M18G_EXTERNAL_PROMOTION_GATE.json").read_text())
    freeze = json.loads((R / "M18G_PROMOTION_FREEZE.json").read_text())
    opening = json.loads(Path(args.opening_manifest).read_text())
    validate_gate_contract(gate)
    if opening.get("status") == "TEMPLATE_NOT_AN_OPENING_RECORD":
        raise SystemExit("opening manifest is still the template")
    if opening.get("labels_observed_when_prediction_ledger_sealed") is not False:
        raise SystemExit("blind prediction ledger must have been sealed before labels")
    if int(opening.get("opening_count_before_this_evaluation", -1)) != 0:
        raise SystemExit("M18G permits one opening only")
    if opening.get("gate_freeze_sha256") != sha256_file(R / "M18G_PROMOTION_FREEZE.json"):
        raise SystemExit("opening manifest references wrong gate freeze")
    if opening.get("prediction_ledger_sha256") != sha256_file(args.predictions):
        raise SystemExit("prediction ledger hash differs from pre-label opening manifest")

    pred = pd.read_csv(args.predictions)
    labels = pd.read_csv(args.labels)
    require_candidate = bool(opening.get("point_candidate_registered"))
    validate_prediction_ledger(pred, require_candidate=require_candidate)
    validate_label_ledger(labels)

    forbidden = set(pd.read_csv(R / "m18g_forbidden_sample_keys.csv").sample_key.astype(str))
    overlap = set(pred.sample_key.astype(str)) & forbidden
    if overlap:
        raise SystemExit(f"fresh holdout reuses {len(overlap)} original supervised sample keys")
    if set(pred.sample_key.astype(str)) != set(labels.sample_key.astype(str)):
        raise SystemExit("prediction and label ledgers must contain the exact same sample keys")
    merged = pred.merge(labels[["sample_key", "target_tte_h"]], on="sample_key", how="inner", validate="one_to_one")

    result = {
        "milestone": "M18G_EXTERNAL_EVALUATION",
        "gate_freeze_sha256": sha256_file(R / "M18G_PROMOTION_FREEZE.json"),
        "prediction_ledger_sha256": sha256_file(args.predictions),
        "label_ledger_sha256": sha256_file(args.labels),
        "rows": int(len(merged)),
        "point_model": evaluate_point_candidate(merged) if require_candidate else {
            "decision": "NOT_EVALUATED_NO_PRELABEL_POINT_CANDIDATE_REGISTERED"
        },
        "selective_eta": evaluate_selective_layer(merged),
        "no_retuning_on_same_holdout": True,
    }
    out.mkdir(parents=True)
    (out / "M18G_EXTERNAL_EVALUATION.json").write_text(json.dumps(result, indent=2) + "\n")
    if require_candidate:
        pd.DataFrame(result["point_model"]["critical_slice_rows"]).to_csv(out / "m18g_external_critical_slices.csv", index=False)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
