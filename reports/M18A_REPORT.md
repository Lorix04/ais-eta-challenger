# M18A — Prediction Contract & Frozen Baseline

## Decision

**PREDICTION_CONTRACT_FROZEN — NO MODEL CHANGE.**

M18A freezes the semantics of the primary company-reference ETA task and cryptographically anchors the development-only M16G champion before M18 introduces any new validation design or external information source. No model is trained, no M16/M17 prediction ledger is rewritten, and the already-observed 53-row M10 final set remains permanently blocked from post-final selection.

## Primary contract

- **Decision time:** `Tracks.last_update`.
- **Label:** company-provided `Tracks.eta`, treated as a reference ETA rather than observed ATA.
- **Target:** `target_tte_h = parsed Tracks.eta - Tracks.last_update`.
- **AIS year policy:** nearest valid previous/current/next calendar year around the decision time because Message 5 ETA carries month/day/hour/minute but not year.
- **Supervision:** one Tracks row / one MMSI.
- **History boundary:** `Positions.recorded_at <= Tracks.last_update`.
- **Location/event caveat:** the supplied label does not establish Pilot Boarding Place, port-entry, berth or all-fast semantics. Any such target must be a separately versioned task.

## Frozen baseline

- M16G nested development-only MoE: **185.883 h MAE**, **19.342 h MedAE**, **290.680 h P90**, N=386.
- M16G gain vs frozen best single expert: **7.043 h**.
- M16H remains the uncertainty sidecar around the unchanged M16G P50: nominal 80% interval, observed pooled development coverage **80.05%**.
- M14/M10 remains the official company-submission benchmark; M16G is not retroactively substituted into that already-opened final result.

## Fresh-holdout rule

Any newly issued holdout is a lockbox. It cannot be used for feature selection, threshold tuning, target-policy cleanup or model choice. Hashes, schema checks and promotion criteria must be fixed before target-dependent scoring.

## What M18A deliberately does not do

M18A does not add weather, routing, port-operations features, new target cleaning, a new split, or a new learner. Those are later hypotheses and must be evaluated under this frozen contract or explicitly declare a new contract version.
