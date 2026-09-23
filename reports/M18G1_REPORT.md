# M18G1 — Full-Development M16G + M16H Scoring Bundle

## Decision

**FULL-DEVELOPMENT BLIND SCORER REGISTERED. HOLDOUT STILL SEALED.**

M18G identified that the repository contained only nested-OOF M16G/M16H evaluation artifacts. M18G1 converts the frozen recipe into an operational scorer for previously unseen rows without changing the historical OOF benchmark.

## Refit rule

No new external label or fresh-holdout information is used. For each tunable M16C/M16D/M16E/M16F/M16G/M16H component, the operational configuration is the deterministic plurality of the already-frozen outer-fold selections (lexical tie-break only). The selected recipe is then refit on all **386** development owners.

Selected operational recipe:

- M16C: `full__n2__a10__clip_none`
- M16D: `geo_kin_6h__destination__k9`
- M16E: `sog180__geodesic__corr_destination_residual`
- M16F HistGB: `hgb_quant50_leaf20`
- M16G meta: `extra_trees_hard_gate`
- M16H interval: `adaptive_scale_24h`

M16H risk is fit on official M16G nested-OOF absolute errors. Its full-development interval uses the already-selected adaptive interval family and a conformal upper quantile computed from **cross-fitted M16H risk**; the resulting development empirical marginal coverage is **80.31%**. This is not a guarantee under external distribution shift or selective filtering.

## Blind scoring contract

`scripts/111_m18g1_score_blind.py` consumes a target-free current-row table plus causal AIS state history, emits the exact M18G baseline ledger fields, refuses target/reference-ETA columns, validates unique sample keys and writes SHA-256 seals.

A five-row synthetic-unlabelled smoke cohort was scored twice with byte/value-identical ledgers. This is a software test only and is not model performance evidence.

## Freeze

Bundle SHA-256: `682619085b60beed4ea25084e492fdd1516e30f230bcfd1c5e5bc12c73348078`

The 53 old-final owners remain blocked from development. No fresh holdout has been received or opened and no M18G one-shot evaluation has run.
