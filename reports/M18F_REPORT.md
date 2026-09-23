# M18F — Uncertainty & Selective ETA

## Decision

**PASS_SELECTIVE_ETA_ADVISORY_LAYER**

M18F promotes an **advisory selective-output layer**, not a new ETA model. M16G remains the point champion and M16H remains the frozen uncertainty sidecar.

## Why selective ETA

M18D showed that M16H uncertainty meaningfully ranks residual difficulty. M18F converts that signal into an auditable action: automatically issue ETA when expected error risk is sufficiently low; otherwise defer rather than pretending all predictions are equally trustworthy.

## Fold-safe evaluation

Thresholds are computed from cross-fitted risk scores inside each outer-train only. The corresponding outer-validation targets are not used in selection. Six target service levels are frozen: 100%, 90%, 80%, 70%, 60%, 50%.

### Strict risk–coverage curve

| scope      |   target_coverage |   rows |   retained_rows |   deferred_rows |   realized_coverage |   retained_mae_h |   retained_medae_h |   retained_p90_ae_h |   retained_interval_empirical_coverage |   retained_mean_interval_width_h |   retained_abs_error_share |   deferred_mae_h |
|:-----------|------------------:|-------:|----------------:|----------------:|--------------------:|-----------------:|-------------------:|--------------------:|---------------------------------------:|---------------------------------:|---------------------------:|-----------------:|
| STRICT_386 |               1   |    386 |             386 |               0 |            1        |         185.883  |           19.3416  |            290.68   |                               0.800518 |                         159.424  |                   1        |          nan     |
| STRICT_386 |               0.9 |    386 |             351 |              35 |            0.909326 |         165.229  |           15.9791  |            276.922  |                               0.797721 |                         125.061  |                   0.808291 |          393.008 |
| STRICT_386 |               0.8 |    386 |             304 |              82 |            0.787565 |         108.696  |           12.9925  |            133.182  |                               0.842105 |                         100.325  |                   0.460534 |          472.038 |
| STRICT_386 |               0.7 |    386 |             271 |             115 |            0.702073 |          98.6528 |           10.3536  |             61.5021 |                               0.878229 |                          89.1141 |                   0.372608 |          391.442 |
| STRICT_386 |               0.6 |    386 |             231 |             155 |            0.598446 |         101.772  |            8.13753 |             53.4514 |                               0.883117 |                          82.7519 |                   0.327654 |          311.234 |
| STRICT_386 |               0.5 |    386 |             194 |             192 |            0.502591 |          86.0645 |            6.78464 |             44.8052 |                               0.886598 |                          77.1625 |                   0.232702 |          286.741 |

### Near-term diagnostic curve

`FUTURE_0_7D` is target-derived and appears **only after** selection flags are frozen.

| scope                            |   target_coverage |   rows |   retained_rows |   deferred_rows |   realized_coverage |   retained_mae_h |   retained_medae_h |   retained_p90_ae_h |   retained_interval_empirical_coverage |   retained_mean_interval_width_h |   retained_abs_error_share |   deferred_mae_h |
|:---------------------------------|------------------:|-------:|----------------:|----------------:|--------------------:|-----------------:|-------------------:|--------------------:|---------------------------------------:|---------------------------------:|---------------------------:|-----------------:|
| NEAR_TERM_FUTURE_0_7D_DIAGNOSTIC |               1   |    279 |             279 |               0 |            1        |          20.2607 |           11.0147  |             42.6044 |                               0.956989 |                         114.48   |                   1        |         nan      |
| NEAR_TERM_FUTURE_0_7D_DIAGNOSTIC |               0.9 |    279 |             272 |               7 |            0.97491  |          17.3495 |            9.51502 |             38.7637 |                               0.955882 |                         104.425  |                   0.834828 |         133.382  |
| NEAR_TERM_FUTURE_0_7D_DIAGNOSTIC |               0.8 |    279 |             255 |              24 |            0.913978 |          14.213  |            8.72889 |             35.3143 |                               0.960784 |                          92.8438 |                   0.641161 |          84.5175 |
| NEAR_TERM_FUTURE_0_7D_DIAGNOSTIC |               0.7 |    279 |             245 |              34 |            0.878136 |          13.7768 |            7.55532 |             33.8972 |                               0.959184 |                          89.17   |                   0.597112 |          66.9829 |
| NEAR_TERM_FUTURE_0_7D_DIAGNOSTIC |               0.6 |    279 |             209 |              70 |            0.749104 |          11.8673 |            6.56194 |             30.4304 |                               0.966507 |                          82.602  |                   0.438773 |          45.3209 |
| NEAR_TERM_FUTURE_0_7D_DIAGNOSTIC |               0.5 |    279 |             176 |             103 |            0.630824 |          11.1306 |            5.33644 |             29.4615 |                               0.965909 |                          77.0037 |                   0.346555 |          35.8617 |

## BALANCED_80

At the predeclared 80% target service level, realized coverage is **78.76%** (304/386). Retained MAE falls from **185.88 h** on all rows to **108.70 h**, a **41.5%** reduction among automatically served rows. Those retained rows account for only **46.1%** of total strict absolute error, so deferring roughly one fifth of rows removes a disproportionate share of error. All **5/5** outer folds show lower retained MAE than their full-fold MAE.

Within the M18C near-term diagnostic regime, BALANCED_80 retains **91.40%** of rows and reduces MAE from **20.26 h** to **14.21 h**.

## Uncertainty interval caution

Among BALANCED_80 retained rows the existing M16H central interval has empirical coverage **84.21%** with mean width **100.33 h**. This is a descriptive audit only. Selection can alter conformal coverage properties, so M18F does not claim a selection-conditional guarantee.

## Fresh-holdout rule

The fresh holdout remains sealed. The development-derived 80th-percentile risk threshold **61.116 h** is frozen before any future holdout scoring. It cannot be retuned after holdout labels are observed. A true external promotion still requires M18G/the final external gate.
