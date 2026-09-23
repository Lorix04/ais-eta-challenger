# Change, Test and Screenshot Protocol

This protocol is mandatory for every material modification to the project.

## Before editing

1. Read `WORKFLOW.md` and `PROJECT_STATE.json`.
2. State the change objective and milestone.
3. Confirm that the change does not cross a workflow gate prematurely.
4. Record the pre-change Git status.

## During editing

1. Keep the change scoped to one auditable objective when possible.
2. Add or update tests together with code.
3. Do not overwrite previous evidence bundles.
4. Never use future information in causal feature code.

## After editing

Create `evidence/YYYYMMDD_HHMM_<slug>/` and save:

- `CHANGE.md`
- `git_diff.patch`
- `commands.txt`
- `test_output.txt`
- `01_changes.png`
- `02_tests.png`
- `03_result.png`

Then update `PROJECT_STATE.json` if the current task, next task, blocker, status or gate changed.

## Screenshot content

### 01_changes.png
Must visibly show at least:
- files changed;
- relevant diff/stat;
- change identifier or purpose.

### 02_tests.png
Must visibly show:
- exact verification command(s);
- pass/fail status;
- important counts/warnings.

### 03_result.png
Use the most informative result for the change:
- map for geospatial changes;
- event timeline for semantic-state changes;
- table/metric output for data transformations;
- model metrics/plots for modelling changes;
- project-state verification for documentation-only changes.

## Required verification by change type

| Change type | Minimum verification |
|---|---|
| Documentation/workflow | project contract test + JSON parse + Python compile |
| Cleaning/data contract | unit tests + sample data smoke + full-data summary comparison |
| Geospatial/geofence | unit tests + known-point checks + map screenshot + manual cases |
| Port-call detector | deterministic synthetic cases + real trajectories + manual audit |
| Feature engineering | causality tests + no-future assertions + distribution checks |
| Model/training | grouped/temporal split audit + reproducibility + baseline comparison |
| Uncertainty | voyage-level calibration check + coverage/width by horizon |

## Failure handling

A failed test or negative experiment is kept in evidence. Do not delete it to make the history look clean. Fixes receive a new evidence bundle or an appended clearly dated section if they are part of the same atomic change.
