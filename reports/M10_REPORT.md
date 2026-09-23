# M10 — Company Reference-ETA Benchmark

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

- Tracks rows total: **1196**
- Ship Tracks rows: **1146**
- Ship rows with non-null ETA field: **449**
- Parseable supervised ship references: **439**
- Independent MMSI rows: **439**

Reference ETA status relative to `last_update`:

| Status | Rows |
|---|---:|
| PAST >24 h | 51 |
| PAST 1–24 h | 10 |
| PAST <1 h | 3 |
| FUTURE 0–7 d | 319 |
| FUTURE 7–14 d | 31 |
| FUTURE 14–30 d | 17 |
| FUTURE >30 d | 8 |

So **64/439 (14.6%)** of parseable references are already in the past at the latest-state timestamp. These rows are not silently deleted from the strict benchmark: they are part of the supplied reference quality and are reported explicitly.

## Causal feature construction

For each MMSI:

1. take the latest state from `Tracks`, excluding ETA from features;
2. attach historical `Positions` only when `recorded_at <= Tracks.last_update`;
3. derive trailing 15/30/60/180/360-minute movement summaries;
4. preserve destination, ship type, navigation state, current position/SOG/COG/heading/draught and data-freshness diagnostics;
5. create **one** supervised example per MMSI rather than thousands of minute-level pseudo-labels.

Median age of the latest matching historical Position relative to `Tracks.last_update` is **0.55 min**; P95 is **174.7 min**.

## Split and freeze

M10 uses a deterministic SHA-256 split based on MMSI only:

- train: **321**
- calibration: **65**
- final test: **53**

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

| task                 | model                                         |   mae_h |   medae_h |   p90_ae_h | within_24h   | within_48h   |
|:---------------------|:----------------------------------------------|--------:|----------:|-----------:|:-------------|:-------------|
| strict_all_parseable | catboost_current_snapshot                     |   339.8 |      26.6 |      728.2 | 38.5%        | 60.0%        |
| strict_all_parseable | train_destination_median_with_global_fallback |   343.4 |      28.2 |      711.2 | 44.6%        | 70.8%        |
| strict_all_parseable | catboost_current_plus_causal_history          |   346.2 |      34.6 |      704.9 | 41.5%        | 55.4%        |
| strict_all_parseable | global_train_median                           |   353.4 |      35.5 |      700   | 29.2%        | 63.1%        |
| strict_all_parseable | ridge_current_plus_history                    |   448.9 |     196.8 |      646.2 | 9.2%         | 20.0%        |
| future_reference     | train_destination_median_with_global_fallback |   191.5 |      21.3 |      179.7 | 55.6%        | 79.6%        |
| future_reference     | catboost_current_snapshot                     |   200.7 |      23.3 |      185.9 | 50.0%        | 77.8%        |
| future_reference     | catboost_current_plus_causal_history          |   201   |      21.2 |      194   | 51.9%        | 68.5%        |
| future_reference     | global_train_median                           |   207.2 |      28.2 |      216.1 | 42.6%        | 72.2%        |
| future_reference     | ridge_current_plus_history                    |   307.5 |      89.7 |      333.3 | 14.8%        | 31.5%        |

## Final test — strict company-reference benchmark

The strict task includes **every parseable company reference ETA**, including past/stale-looking and very far-future values.

Selected on calibration: **`catboost_current_snapshot`**.

Final test (`n=53`):

- MAE: **312.0 h**
- median absolute error: **25.3 h**
- P90 absolute error: **560.4 h**
- within ±24 h: **43.4%**
- within ±48 h: **60.4%**

The large gap between MAE and median error is caused by extreme reference values. On the strict final-test subset whose reference ETA lies **0–7 days in the future** (`n=40`), the same frozen model has MAE **28.6 h** and median absolute error **18.1 h**.

## Final test — future-reference diagnostic

A second diagnostic uses only references that are not already in the past at `last_update`; it does **not** impose an upper horizon cutoff.

Selected on calibration: **`train_destination_median_with_global_fallback`**.

Final test (`n=46`):

- MAE: **136.8 h**
- median absolute error: **22.4 h**
- within ±24 h: **52.2%**
- within ±48 h: **71.7%**

Within the 0–7 day band (`n=40`), the selected future-reference model obtains:

- MAE **32.8 h**
- median absolute error **18.3 h**
- within ±24 h **57.5%**
- within ±48 h **80.0%**

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
