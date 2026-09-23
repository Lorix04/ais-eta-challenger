# M18G Frozen External Promotion Gate

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

The frozen M18B requirements are executable: positive MAE gain; paired owner-row bootstrap (10,000 reps, seed 18023) with 95% lower bound > 0 h; candidate P90 / baseline P90 <= 1.02; and no more than 10% MAE regression in any predeclared critical slice with n >= 20.

Critical slices are frozen **before labels**: `slice_vessel_seen, slice_confidence_tier, slice_physics_eligible, slice_destination_resolved`. Target-derived horizon/reference-status slices may be reported after evaluation but cannot determine promotion.

## Selective-ETA gate

M18F `BALANCED_80` uses the unchanged pre-holdout risk threshold **61.116040125 h**. On the fresh holdout it must realize 70%–90% automatic coverage, retained MAE below full-population M16G MAE, deferred MAE above retained MAE, positive risk-vs-absolute-error Spearman association, and empirical retained interval coverage >= 78%. This remains an empirical selected-set audit; no selection-conditional conformal guarantee is claimed.

## Current blocker

The gate code is ready, but **opening is not ready**. The repository currently contains development OOF evaluation artifacts, not a registered full-development M16G/M16H fresh-row scoring bundle, and M18E did not promote a new point candidate. M18G therefore refuses to pretend that an external holdout can be opened safely today.
