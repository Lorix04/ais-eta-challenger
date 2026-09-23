# Reproducibility runbook

## Environment

Tested environment:

- Python 3.13.5 (tested; see `PYTHON_VERSION.txt`)
- pandas 2.2.3
- numpy 2.3.5
- matplotlib 3.10.8
- shapely 2.1.2
- scikit-learn 1.8.0
- lightgbm 4.6.0
- catboost 1.2.8
- pytest 9.0.2

Install:

```bash
python -m pip install -r requirements.txt
```

## Data

Place the three company-provided files under `data/`:

```text
data/vessel_positions_part1.csv
data/vessel_positions_part2.csv
data/vessel_tracks.csv
```

Company data is intentionally excluded from the clean submission package.

## Data-free package verification

Without company data, the package can still verify frozen metrics, claim scope, source integrity and tests:

```bash
PYTHONPATH=src pytest -q
python -m compileall -q src scripts tests
```

## Data-backed replay

From the repository root after placing the company CSVs under `data/`. This is the full analytical replay and is materially slower than the data-free verification:

```bash
PYTHONPATH=src python scripts/01_provider_audit.py
PYTHONPATH=src python scripts/04_m0c_semantic_states.py
PYTHONPATH=src python scripts/05_m0c_verify.py
PYTHONPATH=src python scripts/07_m0d_catania_calls.py
PYTHONPATH=src python scripts/10_m0e_manual_audit.py
PYTHONPATH=src python scripts/13_m1_baseline_feasibility.py
PYTHONPATH=src python scripts/16_m1x_augusta_candidates.py
PYTHONPATH=src python scripts/18_m1x_augusta_audit_and_gate.py
PYTHONPATH=src python scripts/20_m2_train_only_route_model.py
PYTHONPATH=src python scripts/23_m3_physics_residual.py
PYTHONPATH=src python scripts/26_m4_uncertainty_stability.py
PYTHONPATH=src python scripts/29_m5_port_generalization.py
```

M6 is intentionally protected against accidental rescoring. The final holdout has already been opened and its frozen outputs are committed in `reports/`. Do **not** rerun `33_m6_final_eval.py` as a normal reproduction step. Review `reports/M6_FREEZE.json`, `reports/M6_FINAL_REPORT.md` and `reports/M6_FINAL_HOLDOUT_OPENED.json` instead.

## M10 company reference-ETA replay

After `M0C` has produced `data/derived/m0c_ship_states.pkl.gz`, the M10 benchmark is reproduced with:

```bash
PYTHONPATH=src python scripts/46_m10_build_reference_dataset.py
PYTHONPATH=src python scripts/47_m10_freeze.py   # run only in a fresh experiment with no existing M10 freeze
PYTHONPATH=src python scripts/48_m10_benchmark.py # opens the M10 final test once
PYTHONPATH=src python scripts/49_m10_visual_audit.py
PYTHONPATH=src python scripts/50_m10_verify.py
PYTHONPATH=src python scripts/51_m10_finalize_docs.py
```

In the submitted/frozen project, `M10_FREEZE.json` and `M10_FINAL_HOLDOUT_OPENED.json` already exist. Do **not** delete them merely to rescore the final holdout. The scripts deliberately refuse an accidental second opening.

## M11 diagnostic replay

M11 is post-hoc forensics only. It consumes the already-frozen M10 reference dataset and final predictions and must not train or select a model:

```bash
PYTHONPATH=src python scripts/54_m11_reference_eta_forensics.py
PYTHONPATH=src python scripts/55_m11_verify.py
PYTHONPATH=src python scripts/56_m11_visual_audit.py
```

The expected result is 439 reference rows, 53 frozen strict final predictions, zero rows removed from the strict benchmark, and unchanged M10/M6 freeze checks.

## Tests

```bash
PYTHONPATH=src pytest -q
python -m compileall -q src scripts tests
python scripts/00_project_check.py
# In the full audit bundle only (not the company-facing ZIP):
python scripts/37_m7_verify_delivery.py
python scripts/39_m8_red_team.py
python scripts/41_m8_verify_submission.py
```

## Manual audit dependency

M0E and M1X contain deliberately human-audited label decisions in versioned CSV files under `config/`. These are part of the experiment definition, not model-generated truth. The original visual audit artefacts are preserved under `reports/` / `evidence/` in the full audit bundle.


## Optional reported-ETA alignment input

`M4` no longer assumes a container-specific `/mnt/data` path. If the original ZIP is available, pass it explicitly with `--data-zip /path/to/data.zip`. M6 reporting is frozen and should not normally be rerun; its optional ZIP lookup can be configured with `AIS_ETA_DATA_ZIP`.

## M12 frozen explainability replay

M12 is diagnostic-only and must not refit or replace the selected M10 model:

```bash
PYTHONPATH=src python scripts/57_m12_frozen_explainability.py
PYTHONPATH=src python scripts/58_m12_verify.py
PYTHONPATH=src python scripts/59_m12_visual_audit.py
```

The expected invariants are exact reproduction of 53/53 M10 strict final predictions, unchanged M10 model/reference hashes, and unchanged M6 freeze hashes.

## M13 company-submission reconciliation

M13 changes documentation/package structure only. It can be verified with:

```bash
PYTHONPATH=src python scripts/60_m13_verify_reconciliation.py
python scripts/61_m13_build_submission.py
python scripts/62_m13_verify_submission.py
```

The M13 archive was a draft. M14 performs the final clean-room verification and immutable submission freeze.

## M14 final submission verification

In the full audit bundle:

```bash
PYTHONPATH=src python scripts/63_m14_verify_pre_freeze.py
python scripts/64_m14_build_final_submission.py
python scripts/65_m14_verify_final_submission.py
```

Inside the extracted final company package, the stdlib-only integrity check is:

```bash
python VERIFY_SUBMISSION.py
PYTHONPATH=src python -m pytest -q
python -m compileall -q src scripts tests
```

`SUBMISSION_CONTENT_MANIFEST.json` hashes every payload file. The external full-audit manifest `dist/ais_eta_takehome_submission_final_manifest.json` records the SHA-256 of the final ZIP itself. The final archive deliberately excludes raw company AIS and high-volume row-level derived data, so a complete data-backed replay still requires the three original company CSVs.
