#!/usr/bin/env python3
"""Write M10 report and reconcile top-level documentation with the company-defined ETA reference task."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
D = ROOT / "docs"


def main() -> int:
    ds = json.loads((R / "m10_reference_dataset_summary.json").read_text())
    bench = json.loads((R / "m10_benchmark_summary.json").read_text())
    cal = pd.read_csv(R / "m10_calibration_model_comparison.csv")
    by = pd.read_csv(R / "m10_future_test_by_reference_status.csv")
    strict_pred = pd.read_csv(R / "m10_final_predictions.csv")
    strict_pred = strict_pred[strict_pred["task"].eq("strict_all_parseable")]
    zero7_strict = strict_pred[strict_pred["reference_eta_status"].eq("FUTURE_0_7D")]["abs_error_h"]
    zero7_future = by[by["reference_eta_status"].eq("FUTURE_0_7D")].iloc[0]
    s = bench["strict"]
    f = bench["future_reference"]

    cal_md = cal[["task", "model", "mae_h", "medae_h", "p90_ae_h", "within_24h", "within_48h"]].copy()
    for c in ["mae_h", "medae_h", "p90_ae_h"]:
        cal_md[c] = cal_md[c].map(lambda x: f"{x:.1f}")
    for c in ["within_24h", "within_48h"]:
        cal_md[c] = cal_md[c].map(lambda x: f"{100*x:.1f}%")

    report = f"""# M10 — Company Reference-ETA Benchmark

Date: 2026-09-21

## Decision

The company clarified that, **for this technical exercise, the ETA stored in `Tracks` may be used as the ground-truth/reference value**. M10 therefore creates a new benchmark that is separate from the earlier AIS-derived physical-arrival research branch.

This is a target change, not a retrospective rewrite of M6. The M6 predictor and holdout remain frozen.

## Target semantics

The supervised target is:

`target_tte_h = parsed(Tracks.eta) - Tracks.last_update`

with one supervised row per MMSI/Tracks row. `Tracks.eta` is never a feature. AIS Message 5 represents ETA as `MMDDHHMM UTC` without a year, so M10 resolves the nearest valid previous/current/next year relative to `last_update` deterministically.

**Important:** the company asked us to use this ETA as the exercise reference. It is still an *estimated* arrival value, not an observed Actual Time of Arrival (ATA). M10 therefore calls it `reference_eta`, not `actual_arrival`.

Official AIS semantics: USCG NAVCEN, *AIS Class A Ship Static and Voyage Related Data (Message 5)*: https://www.navcen.uscg.gov/ais-class-a-static-voyage-message-5

## Reference-label audit

- Tracks rows total: **{ds['tracks_rows_total']}**
- Ship Tracks rows: **{ds['ship_tracks']}**
- Ship rows with non-null ETA field: **{ds['ship_eta_nonnull']}**
- Parseable supervised ship references: **{ds['supervised_parseable_ship_eta_rows']}**
- Independent MMSI rows: **{ds['supervised_unique_mmsi']}**

Reference ETA status relative to `last_update`:

| Status | Rows |
|---|---:|
| PAST >24 h | {ds['status_counts']['PAST_GT24H']} |
| PAST 1–24 h | {ds['status_counts']['PAST_1_24H']} |
| PAST <1 h | {ds['status_counts']['PAST_LT1H']} |
| FUTURE 0–7 d | {ds['status_counts']['FUTURE_0_7D']} |
| FUTURE 7–14 d | {ds['status_counts']['FUTURE_7_14D']} |
| FUTURE 14–30 d | {ds['status_counts']['FUTURE_14_30D']} |
| FUTURE >30 d | {ds['status_counts']['FUTURE_GT30D']} |

So **{ds['past_reference_rows']}/{ds['supervised_parseable_ship_eta_rows']} ({100*ds['past_reference_rows']/ds['supervised_parseable_ship_eta_rows']:.1f}%)** of parseable references are already in the past at the latest-state timestamp. These rows are not silently deleted from the strict benchmark: they are part of the supplied reference quality and are reported explicitly.

## Causal feature construction

For each MMSI:

1. take the latest state from `Tracks`, excluding ETA from features;
2. attach historical `Positions` only when `recorded_at <= Tracks.last_update`;
3. derive trailing 15/30/60/180/360-minute movement summaries;
4. preserve destination, ship type, navigation state, current position/SOG/COG/heading/draught and data-freshness diagnostics;
5. create **one** supervised example per MMSI rather than thousands of minute-level pseudo-labels.

Median age of the latest matching historical Position relative to `Tracks.last_update` is **{ds['median_position_age_min']:.2f} min**; P95 is **{ds['p95_position_age_min']:.1f} min**.

## Split and freeze

M10 uses a deterministic SHA-256 split based on MMSI only:

- train: **{ds['split_counts']['train']}**
- calibration: **{ds['split_counts']['calibration']}**
- final test: **{ds['split_counts']['final_test']}**

The split is target-free. Candidate models and the selection metric were frozen in `reports/M10_FREEZE.json` before opening the M10 final test.

## Candidate models

The fixed suite was:

- global training median;
- destination-specific training median with global fallback;
- Ridge on current + causal-history features;
- CatBoost current snapshot;
- CatBoost current snapshot + causal-history aggregates.

No large search or post-final-test tuning was performed.

### Calibration results

{cal_md.to_markdown(index=False)}

## Final test — strict company-reference benchmark

The strict task includes **every parseable company reference ETA**, including past/stale-looking and very far-future values.

Selected on calibration: **`{s['selected_model']}`**.

Final test (`n={s['final_test_n']}`):

- MAE: **{s['selected_final_metrics']['mae_h']:.1f} h**
- median absolute error: **{s['selected_final_metrics']['medae_h']:.1f} h**
- P90 absolute error: **{s['selected_final_metrics']['p90_ae_h']:.1f} h**
- within ±24 h: **{100*s['selected_final_metrics']['within_24h']:.1f}%**
- within ±48 h: **{100*s['selected_final_metrics']['within_48h']:.1f}%**

The large gap between MAE and median error is caused by extreme reference values. On the strict final-test subset whose reference ETA lies **0–7 days in the future** (`n={len(zero7_strict)}`), the same frozen model has MAE **{zero7_strict.mean():.1f} h** and median absolute error **{zero7_strict.median():.1f} h**.

## Final test — future-reference diagnostic

A second diagnostic uses only references that are not already in the past at `last_update`; it does **not** impose an upper horizon cutoff.

Selected on calibration: **`{f['selected_model']}`**.

Final test (`n={f['final_test_n']}`):

- MAE: **{f['selected_final_metrics']['mae_h']:.1f} h**
- median absolute error: **{f['selected_final_metrics']['medae_h']:.1f} h**
- within ±24 h: **{100*f['selected_final_metrics']['within_24h']:.1f}%**
- within ±48 h: **{100*f['selected_final_metrics']['within_48h']:.1f}%**

Within the 0–7 day band (`n={int(zero7_future['n'])}`), the selected future-reference model obtains:

- MAE **{zero7_future['mae_h']:.1f} h**
- median absolute error **{zero7_future['medae_h']:.1f} h**
- within ±24 h **{100*zero7_future['within_24h']:.1f}%**
- within ±48 h **{100*zero7_future['within_48h']:.1f}%**

## Main finding

The most important M10 result is not that a complex model wins. It is almost the opposite.

For the future-reference task, a **train-only destination median** was the calibration winner. CatBoost on current state was close, while adding causal historical aggregates did not consistently improve calibration error. This indicates that, for the latest-state ETA reference supplied in `Tracks`, the strongest reproducible signal in this small sample is largely destination prior/current snapshot information rather than the full 13-day minute history.

This does **not** prove historical AIS is useless for physical ETA. The earlier M0–M9 branch showed that route/history matters when the target is a reconstructed physical port-entry event. It means the company-defined reference target is a different statistical problem.

## What M10 supports

- a benchmark that exactly follows the company's instruction to use `Tracks.eta` as reference;
- one independent supervised row per MMSI;
- causal use of historical Positions;
- explicit reference-quality diagnostics rather than silently deleting stale-looking values;
- calibration-only model selection and a separate M10 final holdout.

## What M10 does not support

- interpreting `Tracks.eta` as observed ATA;
- claiming superiority over the pre-existing external ETA model (its inputs and outputs are unavailable);
- claiming that 312 h / 137 h aggregate MAEs measure physical-arrival accuracy;
- backfilling the latest ETA into historical Positions rows;
- claiming that history-heavy ML is superior when calibration did not show it.

## Recommended delivery framing

The company-facing project should now lead with **M10 as the requested exercise benchmark**. The previous port-call/route-physics branch should remain as an additional domain-aware analysis showing what changes when the target is actual physical port entry rather than the latest declared ETA reference.
"""
    (R / "M10_REPORT.md").write_text(report)

    company = f"""# Company alignment — final clarification applied in M10

Date: 2026-09-21

The company provided a final clarification after M9:

> For the exercise, use the ETA present in the dataset as the ground-truth/reference value; do not reconstruct actual arrival from Positions and no port-area polygon is required.

## Consequence

The primary exercise target is now **`Tracks.eta`**, treated as `reference_eta` throughout M10. This supersedes the M9 assumption that the physical destination-port-area event had to be reconstructed for the main exercise.

The earlier M0–M9 physical-arrival work remains valid as a separate research/challenger branch, but it is no longer the main ground truth used for the requested benchmark.

## File semantics retained from the previous clarification

- `Positions`: historical AIS observations for each vessel.
- `Tracks`: latest available vessel state.
- `recorded_at`: timestamp associated with each Positions observation; upstream generation/reception semantics remain unknown.
- historical destination: available in Positions.
- historical ETA: not available; ETA exists only in Tracks latest state.
- no longer AIS history is available for this exercise.

## Target wording

Because AIS Message 5 defines ETA as an **Estimated Time of Arrival** (`MMDDHHMM UTC`) and does not include a year, M10 uses the wording **company reference ETA** rather than Actual Time of Arrival. The nearest valid calendar year around `Tracks.last_update` is inferred deterministically for timestamp arithmetic.

## Benchmark policy

- one supervised row per MMSI/Tracks record;
- ETA never enters the feature matrix;
- Positions history is restricted to `recorded_at <= Tracks.last_update`;
- stale/past/far-future references are diagnosed explicitly;
- final model selection is calibration-only;
- M6 frozen physical-arrival results are preserved and not rewritten.
"""
    (D / "COMPANY_ALIGNMENT.md").write_text(company)

    print("PASS wrote M10_REPORT.md and reconciled COMPANY_ALIGNMENT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
