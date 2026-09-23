# M13 — Final Company Submission Reconciliation

Date: 2026-09-21

## Decision

M13 is a **documentation and packaging reconciliation milestone only**. It does not train, tune, reselect, rescore, prune, or replace the frozen M10 model. It reorganizes the company-facing material so the requested `Tracks.eta` benchmark is unambiguously primary and the earlier physical-arrival work is clearly secondary.

## Why reconciliation was needed

The project initially explored physical port-entry ETA because the first interpretation of "arrival" was an AIS-observable port-entry event. The company later clarified that, for this exercise, the ETA already present in `Tracks` should be used as the reference/ground-truth value and that no port polygon or reconstructed arrival is required.

M10 implemented that new company-defined target as a separate frozen benchmark. M11 then explained the reference-label heavy tail; M12 explained the frozen selected model. M13 now aligns the delivery narrative with that final task definition.

## Company-facing story after M13

### Primary exercise

**Predict the latest `Tracks.eta` reference from the current vessel snapshot and causal historical `Positions` information.**

The primary evidence chain is:

1. `M10_REPORT.md` — target construction, split, candidate models, calibration selection and final test;
2. `M11_REPORT.md` — reference-quality/error forensics without deleting labels;
3. `M12_REPORT.md` — exact frozen-model attribution and pre-final architecture/history ablation.

### Secondary domain research

M0–M9 remain available as a separate study of AIS-derived physical port-entry ETA. They are not used to reinterpret the company reference target and are not presented as a superior replacement benchmark.

## Headline claims frozen by M13

The company-facing strict headline is:

- **53 final MMSIs**;
- **312.0 h MAE**;
- **25.3 h median absolute error**;
- **560.4 h P90 absolute error**;
- **43.4% within ±24 h**;
- **60.4% within ±48 h**.

The following are explicitly labelled diagnostic rather than headline replacements:

- 28.6 h MAE for the same frozen strict model on final 0–7 day future references;
- 79.1% of total final absolute error concentrated in the top five rows;
- future-reference destination-median benchmark;
- M12 SHAP/masking attribution.

## Model-freeze verification

M13 must preserve:

- M10 reference-dataset freeze;
- frozen M10 CatBoost model SHA-256;
- frozen M10 final predictions and split;
- one-time final-test opening discipline;
- all M6 frozen physical-arrival artifacts.

No post-final subset is promoted into a replacement benchmark and no final-test explanation is used to modify the model.

## Documentation changes

M13 rewrites/reconciles:

- `README.md` — company task first, clear reading order and claim boundaries;
- `EXECUTIVE_SUMMARY.md` — concise M10/M11/M12 story;
- `TAKE_HOME_REPORT.md` — full report reordered around the company-defined target;
- `docs/MODEL_CARD.md` — primary M10 model, target semantics, limitations;
- `docs/PACKAGE_GUIDE.md` — primary/secondary package boundary;
- `docs/COMPANY_SUBMISSION_GUIDE.md` — reviewer navigation;
- `docs/REPRODUCIBILITY.md` — diagnostic replay and M13 package verification.

## Claim-control table

`reports/m13_claim_map.csv` records which statements are primary, diagnostic, secondary, or explicitly not established. It is used by the M13 verifier to prevent the company-facing story from drifting beyond the frozen evidence.

## Package policy

The M13 archive is a **draft company submission**, not the final M14 freeze. It excludes:

- raw company AIS;
- high-volume row-level derived datasets;
- internal screenshots/evidence bundles;
- email/call-preparation notes;
- row-level SHAP and row-level top-error tables not needed for review.

It includes the frozen model, key aggregate diagnostics, source/tests, and the selected secondary physical-arrival report.

## Verification result

- repository tests: **93/93 passed**;
- extracted M13 draft-package tests: **73/73 passed**;
- M10 reference/model and M6 freeze hashes unchanged;
- deterministic draft archive: **142 files**;
- deterministic archive hash verified by rebuilding twice; exact SHA-256 is recorded in the generated M13 manifest;
- raw AIS, high-volume row-level derived data, internal evidence and correspondence excluded.

M14 still owns the final immutable submission freeze.

## Gate

**M13 complete when:**

1. M10/M6 freeze hashes remain unchanged;
2. company-facing docs lead with M10 and label M0–M9 secondary;
3. all headline numbers match frozen M10 outputs;
4. diagnostic subset results are clearly marked as diagnostics;
5. unsupported claims remain blocked;
6. deterministic M13 draft package builds and verifies;
7. repository tests and static checks pass.

Next allowed milestone: **M14 — Final Reproducibility & Submission Freeze**.
