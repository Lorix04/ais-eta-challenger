# M18F Uncertainty & Selective ETA Policy

## Frozen inputs

- point ETA: **M16G**, unchanged;
- uncertainty/risk and interval: **M16H**, unchanged;
- development population: **386** owners;
- old-final owners: **53/53 blocked**;
- fresh holdout: **sealed**.

## Selection rule

M18F uses **predicted absolute error risk** from M16H. Lower score means more reliable. During development evaluation, each M16A outer-validation fold receives a threshold derived only from cross-fitted uncertainty scores inside that fold's outer-train. Validation targets are never used to decide whether a row is retained.

The frozen service levels are 100%, 90%, 80%, 70%, 60%, 50% target coverage. The advisory balanced profile is **BALANCED_80**. It automatically emits the frozen M16G ETA + frozen M16H interval when risk is accepted and otherwise returns **DEFER_LOW_CONFIDENCE**.

For a future one-time fresh holdout, the development-derived M16H OOF 80th-percentile risk threshold is frozen at **61.116 h predicted absolute error risk** before the holdout is opened. It must not be tuned after seeing holdout labels.

## Interval validity statement

M16H's central interval is an empirical, marginal development calibration. Selecting low-risk rows changes the evaluated population. M18F therefore reports retained-set interval coverage **empirically only** and makes **no selection-conditional conformal guarantee**.

## Balanced development result

- realized automatic-ETA coverage: **78.76%** (304/386);
- retained MAE: **108.70 h** vs full-population **185.88 h**;
- retained MedAE: **12.99 h**;
- retained P90 absolute error: **133.18 h**;
- retained rows contain **46.1%** of strict total absolute error;
- empirical M16H interval coverage among retained rows: **84.21%**;
- near-term diagnostic retention: **91.40%** with MAE **14.21 h**;
- retained MAE improves relative to full-fold MAE in **5/5** outer folds.

Selective ETA is a **reliability policy**, not a claim that the underlying point model improved on all 386 rows.
