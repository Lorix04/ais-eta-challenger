# M16C — Fold-safe Hierarchical Destination Priors

## Decision

**PASS as a complementary expert; NOT standalone-promoted.**

M16C implements the statistical layer motivated by M16B: canonical destination semantics are retained, but rare/small groups no longer directly memorize their raw target median. Every target-derived prior is trained on the relevant training fold only, with deterministic fallback and shrinkage to its parent prior.

## Leakage boundary

- Development population: **386 MMSIs** (former M10 train + calibration).
- Blocked old final: **53 MMSIs**.
- Old final used for model/config selection: **no**.
- Outer folds: the frozen M16A 5-fold assignment.
- Inner selection: **4-fold deterministic CV inside each outer-train only**.
- Outer-validation targets never choose hierarchy, minimum support, shrinkage alpha or clipping.

This mirrors the cross-fitting principle used by target encoders to prevent category-level target statistics from leaking their own outcomes into training/validation encodings.

## Prior architecture

The prediction begins at the outer-train global median and can specialize through a hierarchy built from target-free current-state information:

1. canonical destination;
2. optional destination + ship type;
3. optional destination + coarse great-circle distance band;
4. optional destination + distance + ship type + movement bucket.

At a valid child level:

`w = n / (n + alpha)`

`prior_child = w * group_median + (1 - w) * prior_parent`

A child is skipped if support is below `min_support` or the key is unseen. The group statistic is a **median**, not a mean. Candidate support/shrinkage profiles are `(2,10)`, `(3,20)` and `(5,50)`. Optional 2.5% or 5% train-only quantile guardrails are also eligible and are selected only by the inner CV.

## Outer-fold selections

| Outer fold | Inner-selected configuration | Inner MAE (h) |
|---:|---|---:|
| 0 | `full__n2__a10__clip_none` | 209.83 |
| 1 | `dest_ship__n2__a10__clip_none` | 201.78 |
| 2 | `full__n3__a20__clip_q0.05` | 191.30 |
| 3 | `full__n2__a10__clip_none` | 232.33 |
| 4 | `full__n2__a10__clip_q0.025` | 161.72 |

## Development-only OOF result

| Model | MAE (h) | MedAE (h) | P90 AE (h) | RMSE (h) |
|---|---:|---:|---:|---:|
| M16A raw destination median | 202.97 | 22.27 | 330.82 | 663.91 |
| M16A CatBoost current snapshot | 206.14 | 25.94 | 325.51 | 641.53 |
| M16B naive canonical median | 209.26 | 20.73 | 313.70 | 668.12 |
| **M16C hierarchical prior** | **202.56** | 25.16 | 339.87 | **621.15** |

M16C gains **0.41 h MAE** over the raw destination median and **3.58 h** over the reconstructed CatBoost OOF benchmark. It also removes the most extreme raw-destination prediction range (`-3742 h` minimum becomes roughly `-1448 h`).

However, this is **not a stable standalone win**: M16C beats the raw destination median in only **1/5** outer folds and its MedAE/P90 are worse. Therefore `standalone_promoted=false`.

## Complementarity diagnostic

A fixed, **untuned** 50/50 average of the M16C prior and the M16A raw destination prior yields:

- MAE: **199.86 h**;
- MedAE: 22.44 h;
- P90 AE: 334.67 h.

The weight `0.5` was not searched or selected; this is only a diagnostic showing that the experts are not identical. Residual correlation is **0.863**. Any actual ensemble-weight selection is deferred to **M16G** and must be OOF/fold-safe there.

## Interpretation

M16C resolves the statistical failure mode exposed by M16B: small canonical groups no longer inherit an extreme/stale target at full strength. The gain is modest and not fold-stable enough for standalone promotion, but the prior is more regularized and has demonstrable complementary signal. It should therefore be carried forward as one expert, not as the final challenger.

## Reproducibility artifacts

- `reports/m16c_oof_predictions.csv`
- `reports/m16c_inner_search.csv`
- `reports/m16c_outer_selected_configs.csv`
- `reports/m16c_model_metrics.csv`
- `reports/m16c_fold_metrics.csv`
- `reports/m16c_fallback_summary.csv`
- `reports/m16c_prior_comparison.png`
- `reports/M16C_SUMMARY.json`
- `reports/M16C_HIERARCHICAL_PRIOR_FREEZE.json`
- `scripts/73_m16c_build_hierarchical_priors.py`
- `scripts/74_m16c_verify.py`
- `tests/test_m16c.py`

## External methodological references

- scikit-learn TargetEncoder cross-fitting example: https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder_cross_val.html
- scikit-learn TargetEncoder API: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html
- Category Encoders TargetEncoder hierarchy/smoothing: https://contrib.scikit-learn.org/category_encoders/targetencoder.html
