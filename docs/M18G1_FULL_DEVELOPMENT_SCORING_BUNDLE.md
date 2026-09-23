# M18G1 Full-Development Scoring Bundle

The trusted serialized scorer is `models/m18g1_full_development_scoring_bundle.joblib` (SHA-256 `682619085b60beed4ea25084e492fdd1516e30f230bcfd1c5e5bc12c73348078`). Only load this repository-generated joblib file; joblib/pickle files are executable serialization formats.

## Required sequence before any fresh labels

1. Receive a **fresh unlabeled** row table and matching causal state history.
2. Freeze/hash a provenance manifest for that cohort.
3. Run `python scripts/111_m18g1_score_blind.py --rows <rows> --states <states> --output <prediction.csv> --holdout-id <id> --provenance-sha256 <sha> --opening-manifest <manifest.json>`.
4. Independently verify bundle, ledger and manifest hashes.
5. Only then may the label ledger be revealed and `scripts/109_m18g_evaluate_external_holdout.py` be run once.

The blind scorer rejects `abs_error_h, covered_80, error_h, eta_reference_dt, eta_reference_raw, reference_eta_status, signed_error_h, target_tte_h` if present in fresh rows. The full schema is frozen in `reports/M18G1_SCORING_INPUT_SCHEMA.json`.

The operational refit is not a new development score and must not replace M16G's frozen nested-OOF **185.882697 h MAE** claim. It exists solely to produce pre-label predictions for unseen rows.
