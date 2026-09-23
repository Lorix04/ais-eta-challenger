# M16D — Historical Route-Analogue Expert

## Decision

**PASS_ROUTE_SIGNAL_COMPONENT.** M16D demonstrates that causal route-shape similarity contains useful ETA signal beyond simple history aggregates. It is retained as a strong expert for M16G, but it is **not** declared a new final-test winner and the frozen 53-row M10 final set remains excluded from selection/scoring.

## Why M16D exists

M16A showed that current-snapshot CatBoost does not dominate a simple destination prior. M16B/C improved the destination representation but also showed that rare groups and stale/extreme reference ETA values can produce unstable priors. M16D therefore tests a different hypothesis: ships following similar recent AIS movement patterns may have informative historical target outcomes even when rolling summaries lose the route geometry.

Recent maritime work supports this direction. Fan et al. (Ocean Engineering, 2026) use improved Dynamic Time Warping (DTW) to associate spatial-kinematic ship movement patterns and combine historical instantaneous ETA information. AIS trajectory literature also uses DTW because it can compare trajectories under temporal misalignment rather than relying only on pointwise spatial distance.

## Leakage controls

- Population: only the 386 M16A development MMSIs (`train + calibration`).
- The 53 already-observed M10 final MMSIs remain hard-blocked.
- Outer folds are the frozen M16A folds.
- Each trajectory uses only observations `recorded_at <= history_last_at`, where `history_last_at <= Tracks.last_update`.
- Pairwise DTW matrices are target-free.
- Candidate configuration selection occurs with 4-fold inner CV inside each outer-train fold only.
- Validation target values never enter neighbour search, DTW distance, route representation, or configuration selection.
- Historical neighbour target values come only from the applicable training subset.

## Representation and candidate grid

Each recent history is resampled to 12 points over its observed causal span. Spatial coordinates use a fixed local equirectangular approximation around eastern Sicily. Kinematic variants add low-weight SOG and circular COG features.

Predeclared representations:

1. `geo_3h` — 3 h geometry only;
2. `geo_kin_3h` — 3 h geometry + SOG/COG;
3. `geo_kin_6h` — 6 h geometry + SOG/COG.

Every representation is crossed with:

- `global` vs canonical-destination-gated neighbour search;
- `K = 5` vs `K = 9`.

Total candidate configurations: **12**.

The DTW distance uses a Sakoe-Chiba band of 2 resampled steps. Prediction is a robust similarity-weighted median of the K training-neighbour targets. Destination-gated search falls back to global search when fewer than three matching destination neighbours exist.

## Direct history-aggregate baseline

M16D also reconstructs an OOF CatBoost using the M10 current features **plus all causal history aggregates**. Tree count is selected via an inner early-stopping split drawn only from the outer-train fold, followed by refit on the complete outer-train fold. This provides the direct gate comparison requested by the workflow.

## Development-only OOF results

| Model | MAE h | MedAE h | P90 AE h | P95 AE h |
|---|---:|---:|---:|---:|
| **M16D route analogue** | **194.98** | 22.33 | **292.90** | 876.35 |
| M16C hierarchical prior | 202.56 | 25.16 | 339.87 | 813.69 |
| M16A raw destination median | 202.97 | **22.27** | 330.82 | **757.75** |
| M16A current-snapshot CatBoost | 206.14 | 25.94 | 325.51 | 818.26 |
| M16D CatBoost current+history aggregates | 208.36 | 27.60 | 351.69 | 823.66 |

Route analogue gain vs history-aggregate CatBoost: **13.37 h MAE**.

Route analogue gain vs raw destination median: **7.99 h MAE**.

Paired outer-fold wins:

- route analogue vs history-aggregate CatBoost: **4/5**;
- route analogue vs raw destination median: **3/5**.

This clears the predeclared M16D gate against simple history aggregates.

## Selected configuration by outer fold

| Outer fold | Inner-selected configuration |
|---:|---|
| 0 | `geo_kin_6h__destination__k9` |
| 1 | `geo_kin_6h__destination__k9` |
| 2 | `geo_kin_3h__global__k9` |
| 3 | `geo_kin_3h__destination__k9` |
| 4 | `geo_kin_3h__destination__k5` |

The repeated selection of geometry+kinematics, frequently destination-gated, supports the intended interpretation: route shape contributes more when combined with motion state and destination semantics.

## Manual geometry audit

`m16d_neighbour_geometry_audit.png` displays one low-distance query per outer fold with its Top-3 outer-train neighbours. The associated `m16d_neighbour_audit.csv` contains 15 query-neighbour rows. This is a geometry sanity check only, not a target-based selection step.

## Important limitation

The pooled MAE/P90 result is encouraging, but the P95 tail remains worse than the raw destination prior and the route expert can inherit extreme reference ETA values from historical neighbours. Therefore M16D is **not post-hoc promoted as a final standalone model**. It is frozen as a strong, complementary expert for M16G, where blending/gating must itself be fitted only from OOF expert predictions.

A fixed untuned 50/50 diagnostic blend of route analogue and raw destination prior gives about **194.83 h MAE**. This weight was not selected or tuned and is not a promoted model.

## Files

- `reports/m16d_oof_predictions.csv`
- `reports/m16d_inner_search.csv`
- `reports/m16d_outer_selected_configs.csv`
- `reports/m16d_neighbour_details.csv`
- `reports/m16d_model_metrics.csv`
- `reports/m16d_fold_metrics.csv`
- `reports/m16d_route_support.csv`
- `reports/m16d_route_sequences.npz`
- `reports/m16d_route_distance_matrices.npz`
- `reports/m16d_neighbour_audit.csv`
- `reports/m16d_neighbour_geometry_audit.png`
- `reports/m16d_route_analogue_comparison.png`
- `reports/M16D_SUMMARY.json`
- `reports/M16D_ROUTE_ANALOGUE_FREEZE.json`

## Research references

- Fan T., Liu Q., Jing W., He Y. *Incorporation of movement pattern analysis into route searching for ship ETA prediction*. Ocean Engineering 358 (2026), 125455. DOI: 10.1016/j.oceaneng.2026.125455.
- *Development and Application of an Advanced Automatic Identification System (AIS)-Based Ship Trajectory Extraction Framework for Maritime Traffic Analysis*. Journal of Marine Science and Engineering 12(9), 1672 (2024). Uses DTW for AIS trajectory similarity and discusses limitations of purely spatial distances.

## Next step

Proceed to **M16E — Maritime / Physics Expert**. M16D artifacts and all earlier M14/M10/M6/M16A-C freezes remain immutable.
