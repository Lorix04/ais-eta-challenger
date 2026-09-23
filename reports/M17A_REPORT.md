# M17A — Self-Supervised / Learned Trajectory Retrieval Benchmark against M16D

## Decision

**PASS_LEARNED_RETRIEVAL_COMPONENT.** M17A changes only the route representation/similarity mechanism while preserving the M16D destination-gated K=9 weighted-median ETA aggregator. The old 53-row M10 final set remains hard-blocked.

## Research hypothesis

M16D established that recent route-shape similarity contains useful ETA signal, but DTW is hand-designed and pairwise. M17A tests whether a target-free learned embedding can produce more useful historical analogues. The implementation is inspired by MoCo-AIS (contrastive trajectory similarity), NaviSight (masked self-supervised maritime embeddings and HNSW-compatible retrieval), and 2026 memory-augmented AIS trajectory prediction.

## Strict leakage control

- Development population only: 386 MMSIs.
- Old M10 final: 53/53 blocked.
- Frozen M16A outer folds reused unchanged.
- Every SSL encoder is trained separately per outer fold.
- SSL pretraining uses only causal windows owned by outer-train MMSIs.
- Outer-valid MMSI trajectories are excluded from SSL training even though the objective is target-free.
- All windows end at or before the frozen `history_last_at` cutoff.
- Target TTE is used only by the unchanged historical-neighbour weighted-median ETA aggregator.

## Fixed benchmark protocol

All metric variants use the same query representation (`geo_kin_6h`) and the same M16D aggregation rule: canonical-destination gate with fallback, K=9, robust similarity-weighted median of outer-train neighbour targets.

Compared similarities:

1. Frozen official M16D nested route analogue (reference).
2. Fixed 6 h DTW control with destination/K=9 held constant.
3. Outer-train PCA16 embedding + cosine control.
4. **Self-supervised 32-D embedding + cosine**.

The SSL encoder is a compact temporal-convolution (TCN-style) encoder trained from causal unlabeled AIS windows using masked reconstruction plus contrastive consistency. No ETA label participates in representation learning.

## OOF results

| model                                             |   n |   mae_h |   medae_h |   rmse_h |   p90_ae_h |   p95_ae_h |   within_24h |   within_48h |   within_72h |   prediction_min_h |   prediction_max_h |   nonfinite_prediction_count |   abs_gt_1y_prediction_count |
|:--------------------------------------------------|----:|--------:|----------:|---------:|-----------:|-----------:|-------------:|-------------:|-------------:|-------------------:|-------------------:|-----------------------------:|-----------------------------:|
| m17a_ssl_masked_contrastive_cosine_destination_k9 | 386 | 190.105 |    19.044 |  628.438 |    276.643 |    779.252 |        0.560 |        0.715 |        0.749 |          -2422.871 |           2378.252 |                            0 |                            0 |
| m17a_pca16_cosine_destination_k9                  | 386 | 190.802 |    19.279 |  625.979 |    279.569 |    779.252 |        0.544 |        0.718 |        0.764 |          -2422.871 |           2378.252 |                            0 |                            0 |
| m17a_fixed_dtw_geo_kin_6h_destination_k9          | 386 | 192.565 |    20.086 |  626.215 |    291.941 |    779.252 |        0.541 |        0.707 |        0.746 |          -2424.540 |           2378.252 |                            0 |                            0 |
| m16d_official_nested_route_analogue               | 386 | 194.982 |    22.333 |  626.807 |    292.901 |    876.346 |        0.516 |        0.705 |        0.744 |          -2424.540 |           2378.252 |                            0 |                            0 |

## Fold comparison vs M16D

SSL fold wins vs frozen M16D: **5/5**.

SSL MAE gain vs frozen M16D: **4.877 h** (positive means M17A is better).
SSL P90 ratio vs M16D: **0.944**.

The stricter apples-to-apples representation comparison keeps window/gate/K/aggregator identical and changes only the distance metric: SSL improves over fixed 6 h DTW by **2.460 h MAE** and wins **4/5** folds. This isolates a real learned-representation contribution rather than attributing the full gain to SSL.

Predeclared promotion rule: pooled MAE must improve, SSL must win at least 3/5 folds, and P90 must be <= 1.05x M16D.

## Interpretation

This benchmark is intentionally narrow. It does **not** claim a new final-test winner, and it does not yet add HNSW because exact retrieval over 386 development trajectories is trivial. If the learned embedding passes, the next milestone can scale the same embedding to HNSW/memory retrieval and then test it as an additional M16G expert. If it does not pass, M16D remains the route expert and the negative result is retained.

## Files

- `m17a_oof_predictions.csv`
- `m17a_model_metrics.csv`
- `m17a_fold_metrics.csv`
- `m17a_ssl_training_audit.csv`
- `m17a_pretraining_window_audit.csv`
- `m17a_neighbour_details.csv`
- `m17a_fold_embeddings.npz`
- `m17a_retrieval_comparison.png`
- `m17a_ssl_training_convergence.png`
- `M17A_SUMMARY.json`
- `M17A_SSL_RETRIEVAL_FREEZE.json`
