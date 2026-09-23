# M15 — Interview / Technical Defense Preparation

Date: 2026-09-21

## Decision

M15 is an **internal, post-freeze, non-modelling milestone**. It prepares a technically defensible interview narrative from the already-frozen M10–M14 evidence. It does not train, tune, rescore, prune, replace or rebuild the M10 model or the immutable M14 company submission.

## Defense architecture

The interview material is organized around five layers:

1. **Target semantics:** distinguish company `Tracks.eta` reference prediction from observed physical arrival.
2. **Causal dataset contract:** one row per MMSI; historical information only up to `Tracks.last_update`; target never used as feature.
3. **Evaluation protocol:** target-free deterministic MMSI split 321/65/53; selection on calibration; final opened once.
4. **Result interpretation:** strict MAE 312.0 h, MedAE 25.3 h, P90 560.4 h, with M11 heavy-tail/reference-quality diagnostics kept post-hoc.
5. **Reproducibility and claim discipline:** frozen hashes, explainability without retraining, deterministic M14 archive, unsupported claims explicitly blocked.

## Key defense numbers

- 439 parseable reference ETAs / 439 MMSIs.
- 321 train / 65 calibration / 53 final.
- frozen strict model: CatBoost current snapshot, 47 trees, 14 features.
- strict final: MAE 312.0 h; MedAE 25.3 h; P90 absolute error 560.4 h.
- 64/439 reference ETAs already in the past at prediction time.
- 56/439 >7 days in the future.
- top five final absolute errors explain 79.1% of total final absolute error.
- same frozen model on the post-hoc 0–7 day future-reference subset: 28.6 h MAE on 40 rows; diagnostic only.
- `destination_norm` strongest single native/mean-|SHAP| calibration feature.
- causal-history candidate did not improve strict calibration over current-snapshot CatBoost.

## External fact-check anchors

The defense notes were checked against authoritative documentation:

- USCG NAVCEN documents AIS Class A Message 5 ETA as `MMDDHHMM UTC` and destination as voyage-related data; no year is encoded in the ETA field.
- IMO MSC.74(69) describes destination and ETA as voyage-related AIS information and distinguishes it from dynamic position/motion information.
- CatBoost documents SHAP values as per-feature contributions to a prediction plus an expected value; M15 therefore explicitly forbids presenting SHAP as causal maritime evidence.
- Reproducible Builds documents byte-for-byte reproducibility and cryptographic checksum comparison, matching the M14 deterministic ZIP/hash strategy.

## Internal deliverables

- `docs/M15_TECHNICAL_DEFENSE_IT.md`: oral storyline, hard questions, whiteboard narrative, claim boundaries.
- `docs/M15_QA_BANK_IT.md`: prioritized Q&A bank.
- `docs/M15_LIVE_DEMO_CHECKLIST_IT.md`: safe commands and demo flow.
- `reports/m15_defense_map.png`: visual interview narrative.
- `reports/M15_DEFENSE_MANIFEST.json`: machine-readable defense/freeze invariants.
- `scripts/67_m15_verify_defense.py`: post-freeze integrity and content verifier.
- `scripts/68_m15_visual_audit.py`: visual defense map generator.
- `tests/test_m15.py`: automated M15 integrity/content tests.

## Frozen-artifact rule

The following M14/M10/M6 artifacts must remain byte-identical through M15:

- `dist/ais_eta_takehome_submission_final.zip`;
- `dist/ais_eta_takehome_submission_final_manifest.json`;
- `dist/ais_eta_takehome_submission_final.zip.sha256`;
- `models/m10_strict_reference_catboost.cbm`;
- `reports/m10_final_predictions.csv`;
- `reports/M6_FREEZE.json`.

M15 is complete only if these SHA-256 values match the M14 freeze and the full repository tests/compilation pass.

## Claim map for interview

### Supported primary
- `Tracks.eta` is the company exercise reference target.
- 439 parseable supervised MMSIs.
- strict final result is 312.0 h MAE / 25.3 h MedAE on 53 rows.
- selection occurred pre-final and final was opened once.
- post-final milestones did not change the frozen model.

### Supported diagnostic only
- 0–7 day final subset has 28.6 h MAE with the same frozen model.
- top five final errors explain 79.1% of total absolute error.
- `destination_norm` is the strongest single model-attribution feature.

### Not established
- superiority over the company's production/internal ETA model;
- `Tracks.eta` equivalence to observed ATA;
- unseen-port generalization;
- berth/all-fast readiness;
- safety-critical navigation suitability.

## Gate

M15 passes when:

1. all M14 immutable hashes remain identical;
2. full pytest and compileall pass after M15 changes;
3. the defense docs contain the primary benchmark, diagnostic boundaries and blocked claims;
4. no final package/model/split/prediction is rebuilt or modified;
5. evidence screenshots and change/test records are saved.
