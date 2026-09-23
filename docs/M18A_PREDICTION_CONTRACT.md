# M18A Prediction Contract — Company `Tracks.eta` Reference

This document is the human-readable counterpart of `reports/M18A_PREDICTION_CONTRACT.json`. It is frozen before any M18 feature/data experiment.

## 1. Prediction question

At **`Tracks.last_update`**, predict the number of hours to the company-provided **`Tracks.eta` reference**. The benchmark target is:

`target_tte_h = parsed Tracks.eta - Tracks.last_update`

This is a company-reference ETA task. It is **not** automatically an observed ATA, Pilot Boarding Place ETA, port-entry ETA, berth ETA or all-fast timestamp. IMO port-call data models distinguish ETA/ATA and the location to which a timestamp refers, so future operational-event work must create a new target contract instead of silently changing this one.

## 2. ETA parsing

AIS Message 5 ETA is month/day/hour/minute in UTC and contains no year. The existing frozen M10 parser evaluates the previous, current and next calendar year around `Tracks.last_update` and chooses the nearest valid timestamp. M18A preserves this rule byte-for-byte; changing it is a target-contract change.

## 3. Decision-time boundary

Allowed information must exist at or before `Tracks.last_update`. Historical AIS rows satisfy:

`Positions.recorded_at <= Tracks.last_update`

Prediction-time inputs must not include `Tracks.eta`, parsed ETA fields, target/status derivatives, residuals, future positions, post-arrival events, or any hindsight diagnostic. Train-derived aggregates are allowed only when fitted inside the training partition visible to the current nested fold.

## 4. Frozen baseline

The frozen point baseline is **M16G** (`m16g-mixture-of-experts-v1-20260921`), evaluated development-only on 386 M10 train+calibration MMSIs with the 53 old-final MMSIs hard-blocked. Its locked OOF metrics are **185.883 h MAE**, **19.342 h MedAE**, **290.680 h P90**. M16H is the frozen uncertainty sidecar and does not alter the M16G P50.

The official company submission remains M14/M10. M18A does not rewrite the already-opened 53-row final result.

## 5. Fresh holdout

A future company holdout is an evaluation lockbox, not a new development set. Before target-dependent scoring: hash the inputs, verify schema/decision-time semantics, freeze the candidate and promotion criteria. Holdout labels cannot be used to clean the target, choose features or tune thresholds.

## 6. Contract changes

Any change to prediction event, location, label source, decision time, ETA year resolution, allowed information boundary or holdout-use policy requires a new versioned contract. New port-operational targets such as PBP, port entrance, berth or all-fast are separate tasks.

## External standards / evidence

- IMO Compendium — JIT port-call data: ETA/ATA are timestamps at specified locations; Pilot Boarding Place and berth are separately defined locations.
- USCG NAVCEN — AIS Message 5 ETA is encoded as `MMDDHHMM` UTC.
- Recent reproducible AIS ETA work uses voyage-level grouping to keep complete voyages out of both train and evaluation partitions, reducing leakage.
