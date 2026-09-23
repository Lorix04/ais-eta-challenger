# M16A — Branch Integrity + Development-only OOF Benchmark Reconstruction

## Decision

**PASS.** M16A establishes the frozen development-only comparison baseline for the Smart Hybrid Challenger. It does not reopen or replace the M10/M14 final benchmark.

## Population and leakage boundary

- source: `data/derived/m10_reference_rows.pkl.gz`;
- eligible development population: **386 MMSIs** = former M10 train (321) + calibration (65);
- prohibited population: **53 MMSIs** from the already-observed M10 final test;
- the 53 prohibited MMSIs are explicitly stored in `M16A_DEVELOPMENT_MANIFEST.json` and trigger a hard failure if they enter development code;
- five outer folds are assigned target-free from MMSI only using a frozen SHA-256 ordering salt; fold sizes are **78/77/77/77/77**.

For CatBoost, the outer validation fold is never used for early stopping. Tree count is selected from a deterministic inner split inside outer-train, then the model is refit on all outer-train rows before the OOF prediction is produced.

## Frozen pooled OOF results

| Baseline | MAE h | MedAE h | RMSE h | P90 AE h | Within 24 h | Within 48 h |
|---|---:|---:|---:|---:|---:|---:|
| Train-only destination median | 202.97 | 22.27 | 663.91 | 330.82 | 51.6% | 71.5% |
| Current-snapshot CatBoost | 206.14 | 25.94 | 641.53 | 325.51 | 47.2% | 65.3% |
| Global train median | 213.66 | 34.48 | 643.86 | 342.46 | 36.3% | 67.4% |

CatBoost beats the global median on MAE in **4/5 outer folds**, but beats the destination median in only **1/5**. The destination median therefore becomes the strongest M16A MAE baseline, while CatBoost retains lower RMSE/P90 than destination median in the pooled view. The result supports M16B: improve the semantic destination representation before adding model complexity.

## Reproducibility freeze

- two clean reruns of the M16A builder are byte-identical for all frozen artifacts;
- `reports/M16A_BENCHMARK_FREEZE.json` locks salts, fold policy, comparison models, prediction columns, metric schema and artifact hashes;
- immutable M14/M10/M6 hashes are verified by `scripts/70_m16a_verify.py`;
- M14 final ZIP SHA-256 remains `65684d016775deea6b111db1a3ad76fe442cf84f1ceb1ac17f4327ea145c553c`.

## Claim boundary

These values are **development-only OOF metrics**. They are not a new untouched final-test result and must not be compared as though M16 had a fresh independent holdout. The preferred final validation for a promoted M16 challenger remains new untouched company data or a newly issued holdout.

## Next gate

Proceed to **M16B — Canonical Destination Resolver** using the exact same frozen outer-fold assignments and the M16A baseline ledger. The resolver must be target-independent and versioned; any alias/fuzzy policy must be fixed without inspecting the old 53-row final set.
