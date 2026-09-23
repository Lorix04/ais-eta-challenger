# M19 — Fresh External Holdout: real MMDEC one-shot result

## Status

**ONE-SHOT EXTERNAL EVALUATION COMPLETED. MMDEC IS NOW SPENT. NO RETUNING IS PERMITTED ON THIS HOLDOUT.**

The 500-row target-free cohort was built from the canonical MMDEC v1 Parquet files only after their published MD5 checks matched. M16G+M16H predictions were generated and SHA-256 sealed before ETA label reveal. The sealed prediction ledger SHA-256 is `41466369f9e30ef5618d85cb69b106f05633bce3b78f5b553acaea3c526adf44`.

## Frozen external result

- rows: **500 unique MMSI**
- M16G operational baseline full MAE: **550.409 h**
- M18F BALANCED_80 retained rows: **76/500 (15.2%)**
- retained MAE: **342.545 h**
- deferred MAE: **587.668 h**
- retained interval empirical coverage: **69.7%**
- risk vs absolute-error Spearman: **0.326**
- selective external gate: **DO_NOT_EXTERNALLY_VALIDATE_M18F_LAYER**
- point challenger gate: **not evaluated**, because no separate point candidate was registered before label reveal.

The external result is therefore a **negative generalization result**. It does not replace the historical M16G 185.883 h nested-OOF development score; it measures the frozen operational refit under a severe geographic/temporal/provider domain shift.

## Label-semantic stress visible after opening

The MMDEC Message-5 declared-ETA labels are highly heterogeneous: {'FUTURE_0_7D': 197, 'FUTURE_14_30D': 14, 'FUTURE_7_14D': 27, 'FUTURE_GT30D': 38, 'PAST_1_24H': 44, 'PAST_GT24H': 178, 'PAST_LT1H': 2}. In particular, 224/500 labels are already in the past at decision time, while 38 are >30 days in the future. These rows remain in the official external score because removing them after seeing the holdout would be post-hoc target selection.

For interpretation only, the pre-declared M18C-style `FUTURE_0_7D` regime contains **197 rows** and has **38.360 h MAE**. This is **diagnostic only**, not a replacement external score and not a new tuning target.

## Integrity incidents

1. An early experimental fallback Parquet reader was found to truncate RLE-dictionary indices above 8 bits. That run was invalidated and never used for model/gate tuning. The corrected decoder implements the Parquet RLE/bit-packed hybrid rule for arbitrary supported bit widths and has dedicated 13-bit/20-bit regression tests.
2. After the valid prediction ledger had been sealed, the first label-reveal command failed before reading ETA values because pandas renames leading-underscore columns in `itertuples()`. Only that accessor was fixed; the sealed ledger SHA, scoring bundle, thresholds and gate remained unchanged.

## Consequence

MMDEC cannot be reused for iterative improvement. Any M20 adaptation/domain-generalization work must be performed without using MMDEC labels for hyperparameter or feature selection, and its success must be measured on a **new untouched external cohort**.
