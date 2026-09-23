# M14 — Final Reproducibility & Submission Freeze

Date: 2026-09-21

## Decision

M14 is the **final non-modelling milestone**. It freezes the reconciled company submission after integrity, reproducibility and clean-room checks. No model is trained, tuned, selected, rescored or replaced in M14, and the opened M10 final test is not reused for optimization.

## Frozen scientific state

The company-facing primary benchmark remains unchanged:

- 439 parseable `Tracks.eta` references / 439 MMSIs;
- deterministic 321 train / 65 calibration / 53 final split;
- frozen strict model: `catboost_current_snapshot`;
- strict final MAE: **312.0 h**;
- strict final median absolute error: **25.3 h**;
- strict final P90 absolute error: **560.4 h**.

M11 remains diagnostic-only reference/error forensics. M12 remains frozen-model explainability/ablation. M0–M9 remain a secondary physical port-entry research branch.

## Freeze invariants

`reports/M14_FINAL_FREEZE.json` records SHA-256 digests for the M10 reference dataset, frozen CatBoost model, final-prediction ledger, M10 freeze/open markers and M6 freeze file. The M14 verifier also checks all 14 hashes embedded in the M6 freeze.

The immutable scientific rule is unchanged: **no post-final-test model selection or tuning**.

## Deterministic final package

The final archive is built as `dist/ais_eta_takehome_submission_final.zip` with:

- stable sorted member order;
- fixed ZIP member timestamp (`2026-09-21 00:00:00`);
- fixed Unix file mode (`0644`);
- DEFLATE level 9 compression;
- a package-local `SUBMISSION_CONTENT_MANIFEST.json` containing SHA-256 and size for every payload file;
- a stdlib-only `VERIFY_SUBMISSION.py` that detects any missing, added or modified payload file;
- an external archive manifest and `.sha256` checksum for byte-level verification of the ZIP itself.

This follows the reproducible-build principle that stable inputs/metadata should produce byte-for-byte identical outputs, which can then be verified with cryptographic checksums.

## Confidentiality / content boundary

The final company archive excludes:

- the three raw company AIS CSV files;
- high-volume row-level derived trajectory/reference datasets;
- the row-level M10 final-prediction ledger;
- internal evidence/screenshot bundles;
- internal correspondence and personalized preparation documents;
- row-level SHAP and top-error tables not needed for review.

Compact audited configuration/annotation files required by the analytical tests remain included. The package is company-facing and is **not automatically public-safe**.

## Clean-room verification

The final archive is extracted into a fresh temporary directory and verified without relying on the working-tree path. The clean-room procedure runs:

1. `python VERIFY_SUBMISSION.py`;
2. `PYTHONPATH=src python -m pytest -q` on the included analytical tests;
3. `python -m compileall -q src scripts tests`;
4. archive content/banned-file checks;
5. deterministic ZIP metadata checks.

The full repository suite is also run before and after the M14 changes. The final package does not require company raw data for these data-free integrity/tests; a full data-backed analytical replay still requires the three original CSVs as documented in `docs/REPRODUCIBILITY.md`.

## Verification outcome

The final M14 verification completed with:

- **96/96** tests passing in the full audit repository;
- **73/73** analytical tests passing from the extracted company package;
- successful `compileall` in both the full repository and extracted package;
- exact M10 reference/model/final-prediction hashes preserved;
- **14/14** M6 frozen artifact hashes preserved;
- deterministic final ZIP SHA-256 reproduced across consecutive builds;
- package-local content-manifest verification passing with no missing, added or modified payload file.

The current audit bundle intentionally does **not** contain the raw `data/vessel_tracks.csv` / position CSVs. Therefore the raw-data-backed `scripts/50_m10_verify.py` replay is not claimed as an M14 run: invoking it correctly stops on the missing company CSV. M11 and M12 frozen-artifact verifiers do run from the retained derived/frozen artifacts, including exact reproduction of **53/53** strict M10 predictions. A complete raw-data-backed replay still requires the three original company CSVs.

The byte-level ZIP checksum is intentionally stored in the external dist manifest rather than embedded in the archive, avoiding a self-referential archive hash.

## Finality

Once the final archive SHA-256 is written to `dist/ais_eta_takehome_submission_final_manifest.json` and the clean-room verifier passes, that exact ZIP is the immutable submission artifact. Any later modification must produce a new archive/hash and must not be represented as the M14-frozen artifact.

## Gate

M14 is complete only when:

1. M10 reference/model/final-prediction and M6 freeze hashes match;
2. repository tests and static compilation pass;
3. the deterministic final archive builds twice with the same SHA-256;
4. the extracted package passes its integrity verifier, analytical tests and compilation;
5. raw/high-volume/internal-only material is absent;
6. the final archive checksum is recorded externally;
7. no model/split/final prediction changed.
