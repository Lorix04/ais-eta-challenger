# AIS ETA Challenger

End-to-end maritime ETA research project built from AIS position and voyage data, with leakage-aware validation, mixture-of-experts modelling, uncertainty estimation, selective prediction, reproducibility controls, and a one-shot external validation on MMDEC.

> **Repository note:** raw company AIS files and the MMDEC Parquet files are intentionally not included. This repository contains derived research artifacts used for reproducibility. Keep the repository **private** unless the data owner explicitly authorizes public distribution.

## Project objective

At a vessel's `Tracks.last_update`, predict the remaining time to the company-provided `Tracks.eta` reference using only information available at or before that decision time.

The project deliberately distinguishes the company-provided AIS ETA reference from observed operational arrival events such as ATA, pilot boarding, port entry, berth arrival, or all-fast.

## Final status

The project progressed from baseline/data forensics through a frozen Mixture-of-Experts challenger, uncertainty/selective ETA, and finally a true external-domain holdout.

### Main development result

The strongest frozen development point model is **M16G — Mixture of Experts**:

| Metric | Frozen development OOF result |
|---|---:|
| MAE | **185.88 h** |
| Median absolute error | **19.34 h** |
| P90 absolute error | **290.68 h** |
| Development MMSIs | **386** |
| Previously observed final MMSIs blocked | **53 / 53** |

M16G combines four leakage-safe expert families: hierarchical destination priors, historical route analogue retrieval, a causal physics expert, and a tabular HistGradientBoosting challenger.

### Uncertainty / selective ETA

M16H adds risk and interval estimates without changing the M16G point prediction. M18F freezes a `BALANCED_80` selective policy.

On the development population, the policy automatically serves **304 / 386 (78.76%)** rows. The retained subset has **108.70 h MAE**, while the near-term `FUTURE_0_7D` diagnostic retains **91.4%** of rows and records **14.21 h MAE**.

These are selective-prediction diagnostics, not a replacement for the full-population M16G score.

## M19 — one-shot external validation

M19 uses **MMDEC v1** (`10.5281/zenodo.17491518`) as a geographically and temporally separate public AIS domain. Predictions were produced and hash-sealed **before** external ETA labels were revealed.

The sealed evaluation contains **500 unique MMSIs**.

| External result | Value |
|---|---:|
| Full external M16G MAE | **550.41 h** |
| M18F retained rows | **76 / 500 (15.2%)** |
| M18F retained MAE | **342.55 h** |
| M18F deferred MAE | **587.67 h** |
| Retained interval empirical coverage | **69.74%** |
| Risk ↔ absolute-error Spearman | **+0.326** |
| Post-hoc `FUTURE_0_7D` rows | **197** |
| Post-hoc `FUTURE_0_7D` MAE | **38.36 h** |

**External decision:** the frozen M18F selective layer does **not** pass its external gate. MMDEC is now a spent holdout and must not be used for retuning.

This negative external result is retained intentionally. It shows a substantial domain shift: destination resolution and physics eligibility degrade strongly outside the eastern-Sicily development domain.

See [`reports/M19_EXTERNAL_REPORT.md`](reports/M19_EXTERNAL_REPORT.md) and [`reports/M19_EXTERNAL_RESULT_FREEZE.json`](reports/M19_EXTERNAL_RESULT_FREEZE.json).

## What this project demonstrates

- forensic AIS data analysis and target-contract design;
- strict decision-time causality and leakage controls;
- destination normalization and hierarchical priors;
- historical trajectory / route analogue retrieval;
- physics-based ETA estimation;
- tabular ML and nested out-of-fold evaluation;
- Mixture-of-Experts and selective routing experiments;
- self-supervised AIS representation learning;
- uncertainty estimation and selective prediction;
- deterministic artifact freezes, SHA-256 verification and clean-room testing;
- blind prediction-ledger sealing before external label reveal;
- honest reporting of negative experiments and external domain shift.

## Claim boundaries

**Supported:** performance under the frozen company-reference protocol, development-only M16G/M16H evidence, causal feature construction, uncertainty diagnostics, and the one-shot MMDEC external-domain result.

**Not established:** observed ATA accuracy from `Tracks.eta`, universal unseen-port generalization, berth/all-fast prediction, superiority over an undisclosed company production model, or safety-critical navigation suitability.

Historical reported AIS ETA updates are not available in the original `Positions` data; the primary benchmark uses the company-provided latest `Tracks.eta` reference.

## Repository map

```text
src/          reusable feature, modelling, validation and holdout code
scripts/      milestone builders, scorers and verifiers
reports/      frozen reports, metrics, manifests and reproducibility artifacts
models/       frozen M10 model and M18G1 full-development scorer
config/       versioned research configuration
docs/         contracts, model card, reproducibility and detailed research history
tests/        deterministic unit / contract / milestone tests
data/derived/ frozen derived artifacts required by the private reproducibility repo
dist/         historical frozen submission packages used by integrity tests
```

## Recommended reading order

1. [`README.md`](README.md) — current project state and final results.
2. [`EXECUTIVE_SUMMARY.md`](EXECUTIVE_SUMMARY.md) — short technical overview.
3. [`reports/M19_EXTERNAL_REPORT.md`](reports/M19_EXTERNAL_REPORT.md) — external-validation result.
4. [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) — intended use and limitations.
5. [`docs/M18A_PREDICTION_CONTRACT.md`](docs/M18A_PREDICTION_CONTRACT.md) — frozen prediction semantics.
6. [`docs/M18G_EXTERNAL_PROMOTION_GATE.md`](docs/M18G_EXTERNAL_PROMOTION_GATE.md) — one-shot external gate.
7. [`docs/M18G1_FULL_DEVELOPMENT_SCORING_BUNDLE.md`](docs/M18G1_FULL_DEVELOPMENT_SCORING_BUNDLE.md) — blind scoring bundle.
8. [`docs/M19_FRESH_EXTERNAL_HOLDOUT.md`](docs/M19_FRESH_EXTERNAL_HOLDOUT.md) — MMDEC protocol.
9. [`docs/RESEARCH_HISTORY.md`](docs/RESEARCH_HISTORY.md) — detailed M0→M19 history.

## Setup

Python version is recorded in [`PYTHON_VERSION.txt`](PYTHON_VERSION.txt).

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
pytest -q
python -m compileall -q src scripts tests
```

For the optional canonical MMDEC Parquet reader:

```bash
pip install -r requirements-mmdec.txt
```

## External data

The raw datasets are deliberately absent.

For MMDEC, obtain the canonical files from Zenodo DOI `10.5281/zenodo.17491518` and verify them before use:

| File | Published MD5 |
|---|---|
| `Dataset_AIS_POS.parquet` | `12b824d26488e381680b2de90090cc2c` |
| `Dataset_AIS_SPEC.parquet` | `f1dd53064868fa078c714986991c3941` |

Do **not** use MMDEC for further tuning after the frozen M19 opening. A future cross-domain research phase requires a new independent holdout for final evaluation.

## Reproducibility and integrity

The repository contains frozen reports and SHA-256 manifests for the principal milestones. The M18G1 scorer exists to create target-free predictions on a new cohort before labels are observed. M19 records the exact pre-label prediction-ledger hash used for the one-shot external evaluation.

Full replay instructions are in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Data / repository privacy

This GitHub edition excludes:

- raw company AIS CSV/ZIP files;
- the two MMDEC Parquet files (~661 MB total);
- internal screenshot/evidence history;
- local scratch files and caches;
- personalized email / call-preparation documents.

It still contains derived artifacts from the take-home dataset for reproducibility. **Use a private GitHub repository unless public distribution has been authorized by the data owner.**
