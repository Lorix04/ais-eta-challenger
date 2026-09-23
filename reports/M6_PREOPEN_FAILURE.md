# M6 pre-open failure record

The first execution attempt of `scripts/33_m6_final_eval.py` failed **before any final-holdout prediction was computed or persisted**.

Failure: the evaluator imported `add_horizon_band` from `ais_eta.m1`, whose function signature expects a time-to-arrival series rather than a dataframe. The resulting `TypeError` occurred while constructing the combined decision panel.

Evidence of no holdout opening at failure time:
- `reports/M6_FINAL_HOLDOUT_OPENED.json` did not exist;
- `reports/m6_final_predictions.csv` did not exist;
- no final metrics were produced.

The original pre-open freeze was preserved as `M6_FREEZE_v1_preopen_failed.json`. The import was corrected to the already-used `ais_eta.m2.add_horizon_band`, tests/compile checks were rerun, and a second immutable freeze was created before any final scoring.
