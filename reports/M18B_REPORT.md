# M18B — Validation Redesign

## Decision

**VALIDATION_PROTOCOL_FROZEN — NO MODEL CHANGE.**

M18B keeps the M18A target contract and M16G/M16H baseline untouched. It replaces vague validation language with an executable, predeclared protocol for future M18 experiments.

## What changed

- The 386 development rows are partitioned into **5 contiguous decision-time blocks** with counts {0: 78, 1: 77, 2: 77, 3: 77, 4: 77}. Identical timestamps are never split.
- Strict forward evaluation uses **3 expanding windows**: blocks 2, 3 and 4 are test windows; blocks 0–1 form warm-up history.
- A **6 h embargo** is enforced before every test window, matching the longest causal AIS history used by the current route representation.
- Strict forward test union: **231/386 rows (59.8%)**, each appearing as TEST exactly once.
- Every strict test MMSI is absent from its fold training owners. With one supervised row per MMSI, owner-grouped validation is the relevant grouping unit for the M18A benchmark.
- Cross-port claims are explicitly withheld because `Tracks.eta` plus destination text does not provide an authoritative arrival port/event label.
- The fresh external holdout remains sealed and now has a predeclared paired-bootstrap promotion gate.

## Critical distinction

`m18b_legacy_oof_temporal_diagnostics.csv` slices the already-frozen M16G OOF ledger by chronology, but **it is not forward-temporal validation**: those predictions were produced under the legacy balanced M16A folds. Genuine M18B forward evidence requires refitting preprocessing, train-derived aggregates, experts, meta-strategy and calibration inside each forward TRAIN partition.

## Strict forward windows

|   m18b_forward_fold |   test_block |   train_n |   purged_n |   test_n |   future_unused_n |   train_owner_n |   test_owner_n |   owner_overlap_n | test_start          | test_end            | train_end_exclusive   |   embargo_h |
|--------------------:|-------------:|----------:|-----------:|---------:|------------------:|----------------:|---------------:|------------------:|:--------------------|:--------------------|:----------------------|------------:|
|                   0 |            2 |       145 |         10 |       77 |               154 |             145 |             77 |                 0 | 2026-04-10 06:45:55 | 2026-04-12 02:24:52 | 2026-04-10 00:45:55   |           6 |
|                   1 |            3 |       220 |         12 |       77 |                77 |             220 |             77 |                 0 | 2026-04-12 02:26:39 | 2026-04-13 16:52:29 | 2026-04-11 20:26:39   |           6 |
|                   2 |            4 |       305 |          4 |       77 |                 0 |             305 |             77 |                 0 | 2026-04-13 19:13:57 | 2026-04-16 15:39:13 | 2026-04-13 13:13:57   |           6 |

## Why 6 h embargo

The frozen route representation uses causal 3 h/6 h histories. A six-hour gap prevents a training decision window immediately adjacent to a test decision from overlapping the test row's longest historical context.

## External lockbox promotion rule

On a genuinely fresh company holdout, promotion requires a positive MAE gain whose paired owner-level 95% bootstrap interval is entirely above zero, P90 no worse than 1.02× the baseline, and no >10% MAE regression in predeclared critical slices with at least 20 rows. No tuning is allowed after lockbox labels are exposed.

## Research basis

Recent reproducible AIS ETA work explicitly groups complete voyages across train/validation/test and reports a separate chronological robustness split, emphasizing that preprocessing and leakage control can dominate apparent model gains. General time-series validation guidance likewise recommends future-facing splits rather than IID K-fold when adjacent observations are correlated. M18B adapts those principles to this project's one-row-per-MMSI primary contract rather than pretending the primary benchmark contains repeated supervised voyage rows.
