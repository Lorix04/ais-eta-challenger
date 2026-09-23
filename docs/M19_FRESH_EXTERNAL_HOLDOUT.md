# M19 — Fresh External Holdout (MMDEC v1)

M19 is the first **external-domain** evaluation stage. It does not create another split from the Sicily development data and it must never reuse the 53 already-observed old-final MMSIs.

## Frozen external source

Source: **MMDEC: Multimodal Maritime Dataset on the English Channel**, Zenodo v1, DOI `10.5281/zenodo.17491518`. The dataset covers 1 July–30 September 2023 in the western Celtic Sea / English Channel / part of the North Sea. The published dataset contains 19,014,229 AIS position messages (25,130 MMSIs) and 13,558,007 AIS status messages (23,958 MMSIs). Message-5 records contain Destination plus ETA month/day/hour/minute.

Canonical source integrity:

- `Dataset_AIS_POS.parquet` — MD5 `12b824d26488e381680b2de90090cc2c`
- `Dataset_AIS_SPEC.parquet` — MD5 `f1dd53064868fa078c714986991c3941`

The external label remains **AIS-declared ETA**, not observed ATA/berth/all-fast. This matches the M18A target family better than using a different operational event.

## Cohort policy frozen before labels

- Message type 5 only.
- ETA component availability/range may be checked only to ensure an evaluable outcome exists; ETA magnitude is not computed in the scoring stage.
- At most 2,000 owner candidates are preselected by deterministic SHA-256 hash, independent of ETA magnitude.
- At least two valid causal position observations within the preceding 6 h.
- Last valid position no more than 60 min old.
- At most 500 final owners, ordered deterministically by the frozen selection hash.
- The decision-time Message-5 destination/draught/ship type are allowed features; ETA fields are dropped from the target-free row representation.
- Historical position states are causal (`recorded_at <= decision_time`). Decision-time destination is **not** backfilled into prior position states.

## Blinding limitation

MMDEC is public and co-locates ETA components with the static/voyage record. M19 therefore enforces **software/procedural blinding**, not provider-enforced hidden labels. The feature builder only uses ETA fields as a boolean availability/range mask and emits no ETA value or target. A stronger future company holdout with labels delivered after prediction sealing remains preferable.

## One-shot run order

1. `python scripts/112_m19_download_mmdec.py`
2. `python scripts/113_m19_prepare_mmdec_holdout.py --spec ... --positions ... --output-dir ...`
3. Hash the generated `M19_COHORT_PROVENANCE.json`.
4. `python scripts/111_m18g1_score_blind.py --rows ... --states ... --output ... --holdout-id M19_MMDEC_V1 --provenance-sha256 <sha> --opening-manifest ...`
5. **Only after step 4:** `python scripts/114_m19_reveal_mmdec_labels.py ...`
6. Run the frozen one-shot evaluator `scripts/109_m18g_evaluate_external_holdout.py` with the exact M18G acknowledgement string.
7. No retuning on MMDEC after label reveal. Any revised model needs another untouched external cohort.

## Current state

The M19 software path is implemented and tested with a synthetic MMDEC-schema fixture. The canonical Zenodo binaries could not be materialized in the current execution environment, so no real external score has been computed and the M18G holdout-opening claim remains **false**.


## Real one-shot opening (2026-09-23)

The canonical source files were supplied and checksums matched. A 500-MMSI target-free cohort was scored and SHA-256 sealed before ETA reveal. The frozen external full MAE is 550.409 h. M18F BALANCED_80 retains 15.2% with 342.545 h MAE and 69.74% empirical interval coverage, so the frozen selective gate fails. MMDEC is now a spent test set: no post-hoc retuning is permitted.
