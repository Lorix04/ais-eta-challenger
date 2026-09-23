# Model card — Company Reference-ETA Benchmark

## Model identity

**Frozen primary artifact:** `models/m10_strict_reference_catboost.cbm`
**SHA-256:** `efbb86375751e901c48c2f5724514828a10f9eb8ff08ffaabf2d875e369ef86a`

## Intended use

Predict the **company-provided ETA reference stored in `Tracks.eta`** for the technical exercise.

The target is expressed as time from `Tracks.last_update` to the parsed reference ETA. The model is intended for offline evaluation of the supplied exercise dataset, not for safety-critical navigation or deployment as an operational ETA service.

## Target semantics

`Tracks.eta` is treated as the exercise reference because the company explicitly instructed that it may be used as ground truth/reference for this task.

It is **not** relabelled as observed Actual Time of Arrival (ATA). AIS Message 5 defines ETA as voyage-related estimated information.

## Unit of supervision

Exactly one supervised row per MMSI / Tracks record. Historical `Positions` observations may be used to derive causal summaries up to `Tracks.last_update`; they are not treated as thousands of independent copies of the same ETA label.

## Frozen input features

The selected strict CatBoost uses 14 current-snapshot features:

- `track_lat`, `track_lon`;
- `track_sog`, `track_cog`, `track_heading`;
- `track_draught`;
- `ship_type_cat`, `flag_cat`;
- `destination_norm`, `nav_status_cat`;
- `track_msg_count`;
- `last_hour_sin`, `last_hour_cos`, `last_dow`.

No ETA/reference field is an input feature.

## Split and model selection

Deterministic target-free SHA-256 split of MMSI:

- train 321;
- calibration 65;
- final test 53.

Candidate selection was performed on calibration only. The final test was opened once.

Candidate models included global/destination medians, Ridge, CatBoost current snapshot and CatBoost current + causal history.

## Frozen strict final result

Calibration selected **CatBoost current snapshot**.

On 53 final-test MMSIs:

- MAE **312.0 h**;
- median absolute error **25.3 h**;
- P90 absolute error **560.4 h**;
- 43.4% within ±24 h;
- 60.4% within ±48 h.

The benchmark retains every parseable company reference, including past and far-future values.

## Reference-quality context

M11 is diagnostic only and does not change the benchmark:

- 64/439 references are already in the past at snapshot time;
- 56/439 are >7 days in the future;
- top 5 strict final errors account for 79.1% of total absolute error;
- on the same frozen model, the 40 final references 0–7 days in the future have 28.6 h MAE.

That subset result is explanatory only and is not promoted as a replacement final benchmark.

## Explainability

M12 reproduces the frozen artifact exactly and uses CatBoost native importance and SHAP as **model attribution**, not causal evidence.

`destination_norm` is the strongest single feature. The dominant groups are voyage state, current motion and current location.

A history-heavy CatBoost was tested before final opening and did not improve strict calibration over the selected current-snapshot model.

## Limitations

- The target is a latest ETA reference, not observed ATA.
- The externally developed ETA model is unavailable, so superiority/feature parity cannot be evaluated.
- Extreme past/far-future reference values dominate the mean error.
- Message-5 ETA does not encode a year; the benchmark uses a deterministic nearest-year interpretation.
- No unseen-port, berth/all-fast or safety-critical generalization claim is made.
- SHAP/feature importance does not imply maritime causality.

## Secondary model — physical port-entry research

A separate frozen branch predicts AIS-derived physical port-entry events for Catania/Augusta. It is preserved as secondary domain research and must not be presented as the primary company benchmark.

Its locked final result was 36.5 min MAE for route-kNN physics versus 37.7 min geodesic, with the paired gain interval including zero.

## M18A frozen prediction contract

M18A does not replace the primary M10/M14 artifact. It freezes the semantics that any post-freeze challenger must respect: prediction time is `Tracks.last_update`, `Tracks.eta` is target-only, historical AIS must satisfy `Positions.recorded_at <= Tracks.last_update`, and the company reference ETA is not relabelled as an observed ATA/PBP/berth/all-fast event.

For post-freeze research, M16G is the frozen development-only point champion (386 nested-OOF rows; 185.88 h MAE) with M16H as its uncertainty sidecar. The 53 old-final MMSIs are permanently blocked from further selection. A new company holdout, if supplied, is reserved for a predeclared promotion evaluation and cannot be used as another development set. See `docs/M18A_PREDICTION_CONTRACT.md`.


## M18F selective-output reliability layer

M18F does not replace either M16G or M16H. It adds a development-only abstention policy around them. A row is eligible for automatic ETA only when its frozen M16H predicted-error risk is below a threshold estimated from training-side risk scores. The advisory `BALANCED_80` profile realizes 78.76% automatic coverage on the 386-row development OOF ledger and lowers retained-row MAE to 108.70 h; this is a selective-risk result, not an accuracy result on the same complete population.

The existing M16H central interval remains an empirical marginal uncertainty interval. Because selective filtering changes the population, M18F does not claim selection-conditional conformal coverage; retained-set coverage is reported descriptively. The fresh company holdout remains unopened, so BALANCED_80 is advisory until one-time external evaluation.

## M18G frozen external promotion gate

M18G adds no model and opens no holdout. It freezes the protocol that any future fresh cohort must follow. Predictions, M16H risk/interval values and all promotion-gating slices must be produced and SHA-256 sealed **before** target labels are revealed. Exact reuse of any of the original 439 supervised `(MMSI, decision_time)` rows is prohibited.

A point challenger may replace the operational M16G baseline only if its pre-registered external evaluation passes the frozen M18B criteria (positive MAE gain with paired-bootstrap CI95 lower bound >0 h, P90 ratio <=1.02, and <=10% MAE regression in every sufficiently supported pre-label critical slice). M18F `BALANCED_80` is evaluated separately as a selective-output layer with its unchanged 61.116040125 h risk threshold. Failed models/policies cannot be retuned on the same external cohort.

At the M18G freeze there is still **no registered full-development M16G/M16H fresh-row scoring bundle** and no M18E point candidate. Therefore the gate is ready but the holdout opening is intentionally blocked; no external performance claim is made.

## Operational external-scoring form (M18G)

A full-development operational refit of the frozen M16G+M16H recipe is registered at `models/m18g1_full_development_scoring_bundle.joblib`. It is **not** a new benchmark result: the official development evidence remains the nested-OOF M16G/M16H artifacts. The operational bundle exists solely to score previously unseen, target-free rows before external labels are exposed. Its fresh-row scorer requires causal AIS history for M16D and rejects target/reference-ETA columns. External promotion remains subject to the frozen one-shot M18G gate.

## External validation status — M19

A first external-domain source has been frozen: MMDEC v1 (English Channel / western Celtic Sea / part of North Sea, July–September 2023). The adapter and blind-evaluation workflow are implemented, but canonical source binaries were not available inside the current runtime, so no real M19 external metric is reported. Synthetic adapter tests are implementation tests only and are not evidence of external ETA performance.
