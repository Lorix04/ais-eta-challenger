# M11 — Reference ETA Quality & Error Forensics

Date: 2026-09-21

## Decision

M11 is **diagnostic only**. It does not retrain, retune, reselect, or alter the M10 benchmark. The company instruction remains authoritative for the exercise: `Tracks.eta` is retained as the reference label for every parseable row in the strict benchmark.

The goal of M11 is narrower: explain why the strict M10 final-test MAE (**312.0 h**) is much larger than the median absolute error (**25.3 h**) and quantify which properties of the supplied ETA reference generate that heavy tail.

## Frozen invariants

- M10 reference dataset hash still matches the pre-final-test freeze.
- The M10 train/calibration/final split is unchanged.
- The selected strict model remains `catboost_current_snapshot`.
- The M10 final test is not reopened for selection or tuning.
- **0** parseable company-reference rows are removed from the strict benchmark.
- All M6 frozen artifacts remain byte-identical.

## Why reference-ETA forensics are justified

AIS Message 5 carries ETA as `MMDDHHMM UTC`; it **does not encode a year**. USCG NAVCEN documents the format and unavailable sentinels. IMO Resolution A.1106(29) further states that departure, destination and ETA are manually entered at the start of the voyage and whenever changes occur. These facts do not invalidate the company reference; they explain why a quality audit is part of a defensible benchmark.

Sources:
- USCG NAVCEN, *AIS Class A Ship Static and Voyage Related Data (Message 5)*: https://www.navcen.uscg.gov/ais-class-a-static-voyage-message-5
- IMO Resolution A.1106(29), *Revised guidelines for the onboard operational use of shipborne AIS*: https://www.navcen.uscg.gov/sites/default/files/pdf/ais/references/IMO_A1106_29_Revised_guidelines.pdf

## 1. Temporal reference quality

Across all **439** parseable ship ETA references:

| Diagnostic | Rows | Share |
|---|---:|---:|
| ETA already in the past at `last_update` | 64 | 14.6% |
| ETA 0–7 days in the future | 319 | 72.7% |
| ETA >7 days in the future | 56 | 12.8% |
| ETA >30 days in the future | 8 | 1.8% |
| Absolute reference horizon >90 days | 19 | 4.3% |

These are **descriptive properties**, not automatic exclusion rules.

The strict benchmark preserves all of them exactly because the company asked to use the ETA in `Tracks` as the reference value.

## 2. AIS ETA has no year: the extreme tail includes year-resolution uncertainty

M10 resolves the missing ETA year by choosing the closest valid previous/current/next year relative to `Tracks.last_update`. M11 does not change that policy; it measures how separated that choice is from the second-best year assignment.

Five references have a nearest-vs-second-nearest year-assignment margin below **90 days**; two are below **30 days**:

| Vessel | Raw ETA | Chosen interpretation | Margin to alternative year |
|---|---|---|---:|
| TAORMINA | `10/03 20:25` | 2026-10-03 | 24.6 d |
| GABBIANO | `10/02 11:30` | 2026-10-02 | 27.3 d |
| CAPO PASSERO | `09/21 07:00` | 2026-09-21 | 49.5 d |
| LIPARI M | `11/04 19:00` | 2025-11-04 | 53.2 d |
| MIDEA | `09/04 11:00` | 2026-09-04 | 68.9 d |

This is not evidence that a different year is correct. It demonstrates that for dates approximately half a year away from the snapshot, **the AIS payload alone cannot encode the year**, so extreme-horizon interpretation contains unavoidable calendar ambiguity.

## 3. Seven placeholder-like January patterns

Seven rows use either `01/01 00:00` or `01/01 01:01`. In an April latest-state snapshot, the nearest-year resolver maps them to roughly **100–105 days in the past**.

M11 labels these rows `PLACEHOLDER_LIKE_PATTERN` for audit visibility only. It does **not** claim they are invalid and does not remove them from the company benchmark.

## 4. The ETA values are highly coarse in time

The minute component of the 439 ETA references is strongly concentrated:

- **86.1%** are exactly on `:00`;
- **93.8%** are on `:00` or `:30`;
- **96.1%** lie on a five-minute grid.

There are only **300 unique ETA timestamps** among 439 MMSIs; 83 timestamp values occur more than once, with a maximum multiplicity of 7.

This is consistent with ETA being manually maintained voyage-related information rather than sensor-derived continuous telemetry. It should not be interpreted as an error by itself.

## 5. Why strict M10 MAE is 312 h

On the frozen strict final test (`n=53`), the selected CatBoost model has:

- MAE: **312.0 h**;
- MedAE: **25.3 h**;
- P90: **560.4 h**.

The absolute-error mass is extremely concentrated:

| Largest rows | Share of total absolute error |
|---:|---:|
| Top 1 | 21.3% |
| Top 3 | 55.0% |
| Top 5 | 79.1% |
| Top 10 | 91.4% |

Therefore the 312 h mean is not representative of the typical row. A handful of references dominate it.

The largest final-test errors include references such as:

- `MIDEA`: reference about **+3553 h**, model about +35 h;
- `DURO`: reference about **−3055 h**, model about +8 h;
- `GIOVANNI E TOMMASO`: reference about **−2515 h**, model about +6 h;
- `GIUSEPPE PADRE I`: reference about **−2387 h**, model about −88 h;
- `CAPO MOLINI`: reference about **−1791 h**, model about −110 h.

These rows remain in the strict score.

## 6. Error by reference temporal status

Frozen strict-model diagnostics:

| Reference status | n | MAE | MedAE |
|---|---:|---:|---:|
| FUTURE 0–7 d | 40 | **28.6 h** | 18.1 h |
| FUTURE 7–14 d | 2 | 171.4 h | 171.4 h |
| FUTURE 14–30 d | 3 | 404.5 h | 427.2 h |
| FUTURE >30 d | 1 | 3518.4 h | 3518.4 h |
| PAST >24 h | 7 | 1474.2 h | 1681.8 h |

No final-test rows happened to fall in `PAST <1 h` or `PAST 1–24 h`.

This explains the gap between mean and median: the model is being scored not only on near-term future ETA references, but also on a small number of very distant or already-past reference timestamps.

## 7. Post-hoc sensitivity — diagnostic only

The **same frozen strict model** is evaluated under descriptive subsets; these are not alternative benchmarks used for model selection:

| Diagnostic subset | n | MAE | MedAE |
|---|---:|---:|---:|
| All parseable | 53 | 312.0 h | 25.3 h |
| Future only | 46 | 135.2 h | 22.5 h |
| Future 0–7 d | 40 | 28.6 h | 18.1 h |
| Future 0–14 d | 42 | 35.4 h | 18.8 h |
| Future 0–30 d | 45 | 60.0 h | 20.6 h |
| |TTE| ≤7 d | 42 | 31.4 h | 18.8 h |
| |TTE| ≤30 d | 48 | 71.9 h | 24.5 h |

The purpose is explanatory: it shows how the strict aggregate changes as the temporal tail enters the metric. It does **not** justify retroactively filtering the company target.

## 8. Feature freshness is a separate issue

The latest causal `Positions` record is usually close to the Tracks snapshot, but not always. In the strict final set, the only row with feature age >180 min is one of the extreme-error rows; four more lie between 60 and 180 min.

This is worth reporting as an input-quality limitation, but M11 does not use it to change the model or score.

## Interpretation

M11 supports four conclusions:

1. **The strict company benchmark remains the headline benchmark.** Nothing is deleted or relabeled.
2. **The mean error is tail-dominated.** Five final rows explain 79.1% of total absolute error.
3. **The reference itself has observable structure.** It contains past values, very long horizons, coarse minute granularity and a few calendar-year-sensitive cases.
4. **This does not mean the company target is “wrong”.** It means `Tracks.eta` is a manually maintained AIS estimate and therefore a different target from observed physical arrival.

## Consequence for the final presentation

The correct wording is:

> “Following the company instruction, every parseable `Tracks.eta` is retained in the strict benchmark. The resulting MAE is 312 h, while the median absolute error is 25.3 h. M11 shows that five extreme reference rows explain 79.1% of the total absolute error; this is reported as reference-target forensics rather than removed by cleaning.”

That statement is reproducible and does not move the goalposts after observing the test set.

## Next milestone

**M12 — Frozen Model Explainability & Ablation** should explain what the already-selected M10 model uses, compare the already-frozen candidate models/features, and inspect why causal history did not beat the current snapshot. It must not refit or retune against the final test.
