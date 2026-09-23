# M17B — Historical Memory / HNSW ETA Expert

## Decision

**PASS_HNSW_SCALING_NO_MEMORY_PROMOTION.** M17B is development-only; M16J/M17A remain frozen and the 53 old-final MMSIs stay blocked.

## Method

M17A used one owner-level query trajectory per training MMSI. M17B builds a recent historical memory of causal 6 h snapshots from the last 24 h, at 1 h stride. The same outer-train-only self-supervised encoder maps those snapshots and the validation query into latent space. Canonical-destination cohorts are searched first, with global fallback. Retrieved windows are deduplicated by MMSI before the unchanged K=9 weighted-median aggregation.

Memory targets are phase-aligned: `eta_reference_dt - memory_last_at`; the known query age from `history_last_at` to `last_update` is subtracted after retrieval.

## OOF metrics

| model                      |   n |   mae_h |   medae_h |   rmse_h |   p90_ae_h |   p95_ae_h |   within_24h |   within_48h |   within_72h |   prediction_min_h |   prediction_max_h |   nonfinite_prediction_count |   abs_gt_1y_prediction_count |
|:---------------------------|----:|--------:|----------:|---------:|-----------:|-----------:|-------------:|-------------:|-------------:|-------------------:|-------------------:|-----------------------------:|-----------------------------:|
| m17a_ssl_owner_exact       | 386 | 190.105 |    19.044 |  628.438 |    276.643 |    779.252 |        0.560 |        0.715 |        0.749 |          -2422.871 |           2378.252 |                            0 |                            0 |
| m17b_owner_same_encoder    | 386 | 191.996 |    20.844 |  629.393 |    280.220 |    743.842 |        0.539 |        0.694 |        0.744 |          -2422.871 |           2378.252 |                            0 |                            0 |
| m17b_memory_snapshot_exact | 386 | 198.502 |    19.634 |  641.446 |    290.034 |    783.010 |        0.547 |        0.692 |        0.738 |          -2421.876 |           2378.253 |                            0 |                            0 |
| m17b_memory_snapshot_hnsw  | 386 | 198.554 |    20.004 |  641.431 |    290.034 |    779.950 |        0.544 |        0.692 |        0.738 |          -2421.876 |           2378.253 |                            0 |                            0 |
| m17b_memory_static_exact   | 386 | 198.868 |    20.183 |  641.481 |    282.386 |    779.252 |        0.544 |        0.694 |        0.738 |          -2422.871 |           2378.252 |                            0 |                            0 |

Exact historical-memory gain vs frozen M17A: **-8.397 h**; fold wins **2/5**; P90 ratio **1.048**.

Same-encoder isolation: memory vs M17B owner control improves by **-6.507 h**, with **2/5** fold wins and P90 ratio **1.035**.

HNSW-style owner recall@K vs exact cosine: **0.9755**; MAE delta vs exact memory **0.0515 h**.

## ANN note

The clean environment does not bundle the compiled `hnswlib` extension. M17B therefore uses a deterministic dependency-free HNSW-style reference backend: the standard random hierarchy, local M-neighbour layers built efficiently with cKDTree, greedy upper-layer descent, and ef best-first layer-0 graph search. Exact cosine is retained as the oracle, and owner recall is measured explicitly. A production `hnswlib` backend can replace this search layer without changing the memory, cohort-gating, owner-deduplication or ETA aggregation contract.

## Promotion gates

Memory promotion: beat the same-encoder owner control with at least 3/5 fold wins and P90 <= 1.05x M17A. ANN equivalence: owner recall@K >= 0.95 and |MAE_HNSW-MAE_exact| <= 1.0 h.
