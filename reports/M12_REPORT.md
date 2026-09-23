# M12 — Frozen Model Explainability & Ablation

## Scope and non-negotiable rule

M12 is **diagnostic only**. The M10 company-reference benchmark has already opened its final test, so this milestone does **not** fit, tune, select, prune, or replace a model. It explains the already-frozen strict M10 artifact:

`models/m10_strict_reference_catboost.cbm`

The model SHA-256 stayed byte-identical throughout M12:

`efbb86375751e901c48c2f5724514828a10f9eb8ff08ffaabf2d875e369ef86a`

The 53/53 strict final predictions were reproduced from that artifact to <1e-9 h absolute numerical difference. The older M6 freeze also remains unchanged (14/14 hashes).

## What is being explained

The frozen strict model is a 47-tree CatBoost regressor trained with MAE loss on **14 current-snapshot features**:

- position: `track_lat`, `track_lon`;
- motion: `track_sog`, `track_cog`, `track_heading`;
- vessel/context: `track_draught`, `ship_type_cat`, `flag_cat`;
- voyage/state: `destination_norm`, `nav_status_cat`;
- feed context: `track_msg_count`;
- time context: `last_hour_sin`, `last_hour_cos`, `last_dow`.

No ETA/reference field is an input feature.

CatBoost supports both model-internal feature importance (`PredictionValuesChange`) and SHAP values. `PredictionValuesChange` measures average model prediction change associated with feature changes; SHAP decomposes each individual model prediction into feature contributions plus an expected value. These are **model attribution tools, not causal maritime effects**.

Primary technical references:

- CatBoost feature importance: https://catboost.ai/docs/en/features/feature-importances-calculation
- CatBoost SHAP values: https://catboost.ai/docs/en/concepts/shap-values

## Global attribution

### Native CatBoost importance

The largest `PredictionValuesChange` importances are:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `destination_norm` | 15.63 |
| 2 | `track_lat` | 11.81 |
| 3 | `nav_status_cat` | 10.25 |
| 4 | `track_sog` | 10.24 |
| 5 | `track_cog` | 10.01 |
| 6 | `track_heading` | 8.15 |

The result is consistent with a latest-state reference target: destination/intended port context, current geography and current movement carry most of the frozen model's structure.

### SHAP on the calibration split

The calibration mean absolute SHAP ranking begins with:

| Rank | Feature | Mean |SHAP| (h) |
|---:|---|---:|
| 1 | `destination_norm` | 12.06 |
| 2 | `track_lat` | 9.96 |
| 3 | `track_sog` | 7.18 |
| 4 | `nav_status_cat` | 5.90 |
| 5 | `track_cog` | 5.27 |
| 6 | `ship_type_cat` | 4.98 |

Native CatBoost SHAP additivity was verified on train, calibration and final test. The maximum reconstruction error was `1.14e-13 h`.

The SHAP expected value is **40.41 h**. In other words, the frozen model starts from a roughly 40-hour internal baseline and the 14 features move the prediction by tens of hours around that baseline.

## Feature-group explanation

For interpretation, M12 groups the 14 frozen features into six fixed groups. On calibration, mean absolute grouped SHAP is:

| Group | Mean absolute grouped SHAP |
|---|---:|
| motion | 13.41 h |
| voyage_state | 12.39 h |
| location | 11.97 h |
| vessel_context | 6.32 h |
| time_context | 4.21 h |
| feed_context | 1.20 h |

On the final test the three dominant groups remain the same: voyage state, motion and location. This broad split-to-split stability is reassuring for interpretation, but it is not a new validation result and does not change the frozen model.

## Frozen masking stress — calibration only

M12 also performs a **no-retraining perturbation test**. One feature group at a time is replaced by train-derived typical values (numeric median, categorical mode), and the already-frozen model is re-evaluated on calibration.

This is a sensitivity test, not a causal ablation and not a feature-selection procedure.

| Group masked | Mean absolute prediction shift | Calibration MAE delta |
|---|---:|---:|
| voyage_state | 22.59 h | +9.47 h |
| location | 16.61 h | +4.76 h |
| motion | 15.10 h | +7.93 h |
| vessel_context | 9.59 h | +1.93 h |
| time_context | 7.48 h | +1.46 h |
| feed_context | 2.16 h | +1.08 h |

At individual-feature level, `destination_norm` and `track_lat` create the largest frozen prediction shifts when collapsed to train-typical values. Some individual perturbations can even reduce calibration error (for example `flag_cat`), which is precisely why M12 does **not** use masking to prune features after the final test has already been opened.

## Frozen architecture ablation inherited from M10

M12 does not retrain the alternative M10 candidates. It reuses their **pre-final calibration scores** as the only valid architecture/history ablation.

### Strict all-parseable reference task

| Candidate | Calibration MAE | Delta vs current CatBoost |
|---|---:|---:|
| CatBoost current snapshot | **339.81 h** | — |
| CatBoost current + causal history | 346.19 h | **+6.38 h** |
| destination median + fallback | 343.43 h | +3.62 h |
| Ridge current + history | 448.90 h | +109.09 h |

### Future-reference diagnostic

| Candidate | Calibration MAE | Delta vs current CatBoost |
|---|---:|---:|
| destination median + fallback | **191.49 h** | **−9.26 h** |
| CatBoost current snapshot | 200.75 h | — |
| CatBoost current + causal history | 200.97 h | +0.22 h |
| Ridge current + history | 307.53 h | +106.78 h |

This explains why M10 did not retain the history-heavy CatBoost. Causal history was not ignored by accident: it was tested before final-test opening and did not improve the calibration criterion. For the future-reference diagnostic, a simple destination prior was actually stronger than current-snapshot CatBoost.

## Why history did not clearly help this target

The evidence supports a narrow statement:

> For the supplied **latest Tracks ETA reference**, causal Positions history did not add stable calibration value beyond the current/latest-state information already available in Tracks.

This does **not** imply that history is useless for physical vessel-arrival prediction. M0–M9 showed route/trajectory history can matter for the separate physical port-entry problem. The two targets are different.

A plausible interpretation of M10/M12 together is that the latest manually maintained ETA reference is strongly associated with current destination/state priors, while older movement history may contain less information about that particular latest reported reference than it does about physical arrival.

## The heavy-tail failure is visible in the frozen model geometry

On the strict 53-row final test:

- frozen CatBoost predictions range from **−109.5 h to +93.8 h**;
- company reference targets range from **−3055.0 h to +3553.3 h**.

The model therefore operates over a prediction range of a few days, while a handful of supplied references lie months into the past/future after year resolution. SHAP confirms the model's individual contributions are mostly tens of hours, not thousands.

This explains the M11 tail behavior without changing the benchmark. For example:

- `MIDEA`: reference `+3553.3 h`, prediction `+34.8 h`, absolute error `3518.4 h`;
- `DURO`: reference `−3055.0 h`, prediction `+8.2 h`, absolute error `3063.2 h`;
- `GIOVANNI E TOMMASO`: reference `−2514.9 h`, prediction `+6.4 h`, absolute error `2521.2 h`.

In these examples the top SHAP contributions are still only on the order of 10–40 hours. The frozen predictor is therefore not secretly learning a multi-month stale-reference regime.

## What M12 does and does not establish

### Supported

- the frozen model is most influenced by destination, geography and current motion/state features;
- its attribution pattern is broadly similar across train/calibration/final splits;
- adding causal history was already tested before final opening and did not improve calibration;
- the strict heavy-tail errors are far outside the prediction scale learned by the frozen current-state model;
- all explainability results reproduce the exact frozen M10 artifact.

### Not supported

- causal statements such as “destination causes ETA by X hours”;
- feature pruning based on final-test explainability;
- post-final selection of a different model;
- claims that history is generally useless for ETA prediction;
- claims that `Tracks.eta` is observed actual arrival.

## Gate / next step

**M12 complete. No model change.**

Proceed to **M13 — Final Company Submission Reconciliation**. M13 may rewrite the company-facing story so that M10 is the primary requested exercise and M0–M9 is clearly secondary domain research, but it may not alter the frozen M10 model, split, final predictions or metrics.
