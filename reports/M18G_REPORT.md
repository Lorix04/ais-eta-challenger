# M18G — Frozen External Promotion Gate

## Decision

**EXTERNAL PROMOTION GATE FROZEN; HOLDOUT NOT OPENED.**

M18G converts the M18B/M18F intentions into executable gate code and immutable pre-label thresholds. It deliberately produces no external accuracy number because no fresh holdout was supplied and no labels were opened.

## What is frozen

- point-model primary/secondary metrics and paired bootstrap uncertainty;
- 10,000 paired bootstrap repetitions with seed 18,023;
- P90 non-regression ratio <= 1.02;
- prediction-time critical-slice regression ratio <= 1.10 for n >= 20;
- the four critical slice dimensions;
- M18F BALANCED_80 risk threshold 61.116040125 h and its external validation criteria;
- zero reuse of the original 439 supervised `(MMSI, decision_time)` rows;
- one opening / no retuning on the same holdout.

## Why opening remains blocked

M18E promoted no new point model, and the current M16G/M16H artifacts are nested OOF development-evaluation artifacts rather than a registered full-development scorer for unseen rows. A blind prediction ledger therefore cannot yet be honestly produced for a new cohort. This is a **readiness blocker**, not a failed model result.

## Deliverables

See `docs/M18G_EXTERNAL_PROMOTION_GATE.md`, `reports/M18G_EXTERNAL_PROMOTION_GATE.json`, `reports/m18g_candidate_registry.csv`, `reports/m18g_opening_readiness_checklist.csv`, the blind prediction/label templates and `scripts/109_m18g_evaluate_external_holdout.py`.
