# AIS ETA Challenger — Project Workflow

This file is the operational source of truth for **what we are doing now, what must be true before moving on, and what comes next**.

## 1. Project objective

Build the strongest scientifically defensible challenger that can be supported by the supplied Eastern Sicily AIS dataset.

The project is **not** defined as “train an ETA model at all costs”. The sequence is:

1. prove what the AIS feed actually represents;
2. reconstruct maritime events and a defensible arrival target;
3. quantify the number of independent port calls/voyages;
4. only then train and evaluate ETA models;
5. if trustworthy ETA labels are insufficient, pivot to a supported alternative use case (Port Intelligence, AIS Data Quality, or Vessel Behaviour/Anomaly Detection) without discarding the foundation work.

## 2. Non-negotiable methodological rules

- The statistical unit is the **voyage / port call**, not the minute-level AIS row.
- No random-row train/test split.
- No future trajectory, future destination, future ETA, future congestion state, centered rolling window, or post-event information may enter prediction-time features.
- Provider snapshots and repeated states must not be treated as independent sensor updates.
- AIS sentinel values must be normalized before modelling.
- A label is named after the **physical event actually reconstructed**. Do not call a port entrance crossing “ATA berth” unless the company confirms those semantics.
- Route prototypes/clusters must be fit on training voyages only inside each validation fold.
- External reanalysis data may be used only as an explicitly labelled oracle/explanatory experiment unless the corresponding forecast was available at prediction time.
- Report uncertainty and forecast stability, not only point MAE.

These rules are consistent with the domain research already completed: DCSA separates PBP, berth and operational service events, and the M3 trajectory work treats AIS as semantic motion episodes rather than independent points.

## 3. Evidence rule for every project modification

Every material project change must leave an auditable evidence bundle under:

`evidence/YYYYMMDD_HHMM_<change_slug>/`

The bundle must contain, at minimum:

- `CHANGE.md` — objective, rationale, files changed, expected effect, residual risks;
- `git_diff.patch` — exact code/documentation diff for the change;
- `commands.txt` — commands used to test/verify the change;
- `test_output.txt` — raw verification output;
- `01_changes.png` — screenshot showing the change/diff;
- `02_tests.png` — screenshot showing tests/checks passing;
- `03_result.png` — screenshot of the produced result when the change has a visual/data output. If the change has no meaningful visual result, this screenshot must instead show the verified project state and the reason must be recorded in `CHANGE.md`.

A change is not considered **DONE** until its evidence bundle exists.

## 4. Test hierarchy

For each modification, run the highest applicable levels:

**T0 — Static integrity**
- Python compilation/import checks.
- JSON/config parsing.
- required-file checks.

**T1 — Unit / contract tests**
- deterministic tests for functions and project contracts.

**T2 — Data smoke test**
- run on a small deterministic slice or representative sample.
- verify schema, timestamps, sentinels and expected row counts/invariants.

**T3 — Full-data integration**
- run the affected pipeline stage on the complete dataset when computationally reasonable.
- compare outputs with previous baselines and explain meaningful changes.

**T4 — Domain sanity / visual audit**
- maps, trajectories, event timelines or metric plots where geometry/behaviour is involved.
- manually inspect representative normal and edge cases.

**T5 — Leakage / validation audit**
- required for any training/evaluation change.
- verify grouping, temporal cutoffs, as-of joins, train-only fitting and no future information.

## 5. Phase plan and gates

### M0A — Workspace + data provenance
**Status: DONE (first pass)**

Goal: inventory files, schema, date coverage, vessel/AtoN categories and raw inputs.

Required outputs:
- reproducible repository structure;
- schema/coverage summary;
- raw data kept outside Git;
- project state file.

Gate: input dataset is reproducibly readable and its schema is known.

### M0B — Provider / polling forensics
**Status: FIRST PASS DONE; formal verification remains**

Goal: determine whether rows resemble raw AIS messages, resampled AIS or periodic latest-state snapshots.

Required checks:
- per-MMSI inter-row Δt distribution;
- fleet-wide timestamp synchrony;
- timestamp phase / modulo analysis;
- repeated core-state runs;
- `SOG > threshold` while position remains unchanged;
- dynamic vs static field change cadence;
- clear wording separating EVIDENCE, INFERENCE and UNKNOWN.

Output: `reports/provider_forensics.*`

Gate: we know how to represent observation clock, state-change clock and stale-risk proxies without claiming unavailable source-message age.

### M0C — AIS normalization + semantic motion states
**Status: DONE — gate passed 2026-09-18**

Goal: produce a clean causal vessel-state layer.

Required work:
- normalize AIS special/sentinel values;
- separate vessel / AtoN / SAR;
- compute state hashes and state-change age;
- causal motion features;
- stop / slow-motion / turn / gap candidates inspired by M3, but implemented audibly for this dataset;
- preserve dwell information rather than naively dropping duplicates.

Outputs:
- clean-state data contract;
- sentinel report;
- semantic-state sample/timeline;
- tests for edge cases.

Gate: a vessel timeline can be interpreted without using future data.

### M0D — Catania geometry + port-call reconstruction
**Status: DONE — gate passed 2026-09-18; 61 candidates handed to M0E**

Goal: reconstruct a defensible, explicitly named AIS-observable arrival event for Catania first.

Required work:
- document authoritative geometry sources;
- define approach / entrance / inner-port / berth-stop zones where supportable;
- implement crossing direction, hysteresis, dwell and session logic;
- distinguish transit, anchorage/waiting and true call candidates;
- never tune geometry against final ETA performance.

Outputs:
- port geometry artefacts with provenance;
- event state machine;
- candidate-call table;
- trajectory/event screenshots.

Gate: candidate events are semantically coherent on manual inspection.

### M0E — Manual event audit + effective sample size
**Status: DONE — gate passed 2026-09-18 with GO_LIMITED**

Goal: determine how many independent, trustworthy calls actually exist.

Each candidate receives:
- event type/name;
- `ground_truth_confidence` (A/B/C or equivalent);
- `[last_outside, first_inside]` uncertainty bracket where relevant;
- audit note / ambiguity reason;
- inclusion/exclusion decision.

Required metrics:
- reliable calls overall and by port;
- unique vessels;
- censored/incomplete candidates;
- ambiguous destination fraction;
- label uncertainty distribution.

**GO/NO-GO A result (2026-09-18): GO_LIMITED.**

M0E validated 51/61 candidate Catania research-gate entries. The primary operational ETA cohort contains 28 calls across 13 unique vessels (24 cargo plus one passenger, tanker, offshore-supply and survey/offshore vessel each). This is enough to proceed to physical/simple M1 baselines, but not enough for unrestricted high-capacity ML or strong superiority claims.

- Proceed to M1 baseline and split/headroom diagnostics.
- Keep high-capacity residual ML blocked until M1 demonstrates stable signal and/or independent sample size is expanded (e.g. Augusta / longer history).
- Preserve the 51-call broader truth set for Port Intelligence analyses.

### M1 — ETA baselines
**Status: DONE — gate requires ground-truth expansion**

GO/NO-GO A has passed in limited form. Begin with sample-size/split diagnostics and simple baselines only.

Current Catania primary ETA population: **28 calls / 13 unique vessels**. Repeated-vessel concentration must be reported explicitly.

Baselines considered for the ETA programme:
- geodesic distance / current SOG;
- geodesic distance / robust recent speed;
- regularized linear/GAM sanity model;
- route-aware distance / route-kNN if route families exist.

M1 executed only the simple geodesic/current-speed and geodesic/robust-speed baselines. The gate fired before regularized or route-aware models; those remain blocked/deferred rather than silently omitted.

Evaluation:
- voyage-grouped model selection;
- chronological final test where feasible;
- horizon bands for post-hoc reporting, not future-informed training sample selection;
- MAE, MedAE, RMSE, P90/P95, within 30/60/120 min.

Gate result (2026-09-18): **EXPAND_GROUND_TRUTH_BEFORE_COMPLEXITY**. On the locked development set, only 16 calls support a causal inbound-approach benchmark and only 2 calls contribute any >6h approach evidence. The robust geodesic/30-min-speed baseline is already strong under 6h (MAE 12.1 min, P90 25.9 min). Do not add route/residual capacity yet.

### M1X — Augusta ground-truth expansion
**Status: DONE — GO_M2**

Goal: increase independent event count and broaden operational regimes before model complexity. Reuse the M0D/M0E protocol, but adapt it to Augusta's roadstead, two entrances and terminal heterogeneity.

Required order:
1. authoritative/domain-backed Augusta geometry;
2. direction-aware candidate events and sessions;
3. manual audit of every candidate;
4. ETA-scope cohort definition;
5. combined Catania+Augusta effective-N and horizon-coverage review.

Gate result (2026-09-18): **GO_M2**. Augusta audit retained 60/79 research-gate entries and defined 55 ETA-eligible calls across 46 vessels. Combined with Catania: 83 ETA calls / 58 unique vessels; 12 calls have causal approach support >6h and 7 >12h. The pre-specified evidence gate passed.

Gate: enough independent calls and >6h coverage to justify M2 train-only route modelling.

### M2 — Train-only route model
**Status: DONE — GO_M3_ROUTE_SIGNAL_CONFIRMED**

Goal: replace naive straight-line distance with a credible remaining-route estimate.

Order of experiments:
1. geodesic;
2. optional water-only sanity baseline;
3. port-specific route skeleton / historical prototype fit on training voyages only;
4. route-kNN if supported.

Gate result (2026-09-18): **GO_M3_ROUTE_SIGNAL_CONFIRMED**. Strict expanding temporal OOF evaluation used 40 calls / 33 vessels while keeping 16 chronological final-test calls untouched. Route-kNN improved voyage-balanced MAE by 11.9% at <=24 h and 20.3% on cold-vessel <=24 h calls, and beat geodesic in all 6 port x temporal folds. The all-horizon gain is only 2.2%, confirming that route geometry does not explain very-long-horizon waiting. Retain geodesic as baseline and route-kNN as an M3 feature/alternative physical distance; do not call it nautical ground truth.

Gate: passed for shallow M3 residual modelling; deep sequence models and aggressive tuning remain blocked.

### M3 — Physics + residual ML + port-state features
**Status: DONE — M3_COMPLEXITY_NOT_JUSTIFIED**

Main candidate:

`T_hat = T_physics + f(X)`

with shallow/regularized CatBoost, LightGBM or XGBoost selected by grouped validation.

Feature families:
- route/progress;
- causal speed/turn dynamics;
- semantic voyage phase;
- destination confidence;
- stale-risk/data-quality;
- AIS-derived port digital state (anchorage pressure, inbound/outbound counts, recent departures, berth occupancy proxies).

Gate result (2026-09-18): **M3_COMPLEXITY_NOT_JUSTIFIED**. M3 used a second expanding temporal stack over the 40 M2 cross-fitted route predictions (10-call warm-up + 30 OOF calls) while leaving all 16 locked final calls untouched. The best residual model, shallow LightGBM, degraded <=24 h voyage-balanced MAE from 0.539 h to 0.674 h (-25.0% relative "improvement"), won only 1/3 folds, and worsened cold-vessel performance by 24.6%. Simple same-snapshot port-state proxies improved that residual model by only 0.95% and failed their consistency gate. Retain M2 route physics; reject residual ML/port-state complexity for the primary predictor.

Gate: failed for ML complexity. This is a documented stop-condition result, not a reason to tune harder.

### M4 — Uncertainty, ETA reliability and stability
**Status: DONE — `M4_CALIBRATED_HIGH_RELIABILITY_GO_M5`**

Required outputs:
- calibrated prediction interval;
- coverage and interval width;
- reported-ETA discrepancy only where temporally aligned;
- input/data reliability diagnostics;
- absolute predicted-arrival revision metrics and jitter analysis.

Gate result (2026-09-18): **passed for a scoped HIGH-reliability regime.** M4 used 24 chronological calibration calls and 16 later temporal-evaluation calls from the existing M2 cross-fitted OOF set; the separate 16 locked final calls remained untouched. A 90% simultaneous call-level band (one maximum-error conformity score per call) has ±2.485 h nominal half-width and achieved 100% whole-call coverage on 16 later HIGH-reliability calls. A causal absolute-ETA EWMA (`alpha=0.4`, selected on calibration only) reduced temporal-evaluation P90 revisions from 39.1 to 22.2 min while slightly improving <=24 h MAE. No reported AIS ETA was used because the supplied snapshot had zero rows aligned even within 60 min.

Gate: uncertainty is calibrated at the voyage level and does not rely on pseudo-independent rows.

### M5 — Port generalization / stress tests
**Status: DONE — `M5_GO_M6_SCOPE_LIMITED`**

M5 fit no new predictor and scored no locked-final calls. It stress-tested the frozen M2 route-physics + M4 reliability/stability layer on the existing cross-fitted OOF and later temporal-evaluation sets.

Results:
- Catania <=24 h: geodesic 13.6 min MAE -> route-kNN 10.0 min (26.3% gain; 10 OOF calls).
- Augusta <=24 h: 42.2 -> 37.8 min (10.4% gain; 30 OOF calls).
- Cold-vessel gains remain positive in both ports: 22.7% Catania (6 calls), 20.0% Augusta (22 calls).
- Stress failure: Augusta initial E-family (10 calls) degrades by 12.6% versus geodesic; route value is not universal across route families.
- Augusta seen-vessel subgroup (8 calls) also degrades by 13.1%, demonstrating that repeated MMSI does not imply stable route behavior.
- Pooled M4 HIGH-reliability interval retains 100% whole-call coverage on 14 later Augusta calls; Catania has only 2 later HIGH-reliability calls, so port-specific temporal uncertainty claims there are underpowered.
- Port-only 90% conformal calibration is not promoted: with 16 Augusta and 8 Catania calibration calls the finite-sample order statistic is the maximum score in each port, producing unstable small-N widths.
- Augusta ETA cohort is Levante-only; Scirocco entrance transfer is not established.
- Leave-one-port-out was intentionally not attempted because the retained route geometry is anchored to port-specific target gates; transferring those geometries would not represent a meaningful unseen-port deployment.

Gate: proceed to M6 with scope-limited claims and without modifying the predictor based on these subgroup diagnostics.

### M6 — Final take-home deliverable
**Status: DONE — FINAL HOLDOUT OPENED ONCE; CLAIMS SCOPE-LIMITED**

Deliverables:
- concise technical report / README;
- reproducible pipeline;
- methodology and leakage audit;
- model card / target definition;
- ablation table;
- uncertainty + stability evaluation;
- visual demo with map/timeline;
- explicit limitations and company-data requests;
- alternative use cases discovered from the dataset.

## 6. Current execution pointer

The machine-readable state is stored in `PROJECT_STATE.json` and must be updated after every material change.

Before starting work, read in this order:
1. `WORKFLOW.md`
2. `PROJECT_STATE.json`
3. latest `evidence/*/CHANGE.md`
4. relevant report/script for the current task.

M15 is complete. M10 remains the frozen primary company-reference benchmark; M11–M12 remain diagnostic/explainability only; M13 reconciled the company-facing story; M14 froze the immutable final submission; M15 adds only internal interview/technical-defense preparation. No model, split, final prediction or M14 submission artifact changed.

A separately versioned post-freeze research branch is now planned as **M16 — Smart Hybrid Challenger**. M16 must leave the M14 submission immutable and may use only the former M10 train+calibration population for model selection. The 53-row M10 final set has already been observed and is therefore prohibited for M16 selection, hyperparameter tuning, feature selection, architecture choice, threshold selection or ensemble weighting. Any score on those 53 rows is post-hoc only and cannot be represented as an unbiased M16 test. The preferred future confirmation is genuinely new company data.

## 7. Stop conditions / pivot rules

Pause or pivot if any of the following is observed:
- event labels are too few or too ambiguous for supervised ETA;
- Catania event semantics cannot be reconstructed reliably from available geometry;
- route clusters are unstable across train-only folds;
- a proposed feature cannot be reproduced using only information available at prediction time;
- a complexity increase does not produce stable voyage-level out-of-fold gain;
- berth/all-fast semantics are requested but no reliable operational ground truth exists.

A failed path must be documented as a result, not silently discarded.

### M7 — Submission packaging and company-call preparation
**Status: DONE — DELIVERY READY; NO MODEL CHANGE**

M7 is documentation/packaging only. It does not change the M2 point predictor, M4 reliability/uncertainty layer, final target, thresholds, split, or M6 holdout results.

Deliverables:
- recruiter-facing `README.md`;
- `TAKE_HOME_REPORT.md` with final results and claim boundaries;
- `docs/REPRODUCIBILITY.md` and pinned `requirements.txt`;
- `docs/DATA_REQUESTS.md`;
- Italian company-call preparation and email draft;
- clean submission ZIP excluding company raw/derived data and internal evidence bundles;
- automated consistency checks against frozen M6 metrics and claim scope.

Gate result: **M7_DELIVERY_READY**. The next step is discussion/data alignment, not further tuning on the current dataset.


### M8 — Red-Team, Reproducibility & Executive Packaging
**Status: DONE — NON-MODELLING HARDENING; FROZEN M6 UNCHANGED**

M8 performs no refit/rescore/tuning. It verifies all M6 freeze hashes, red-teams recruiter-facing claims, removes personalized internal docs from the company ZIP, corrects the confidentiality boundary for compact derived audit annotations, removes container-specific Python paths, adds an executive summary and separates data-free verification from data-backed replay. The next action is company clarification, not additional modelling on the same 13-day sample.


### M9 — Company Alignment & Target Reconciliation
**Status: DONE — EVENT-FAMILY TARGET ALIGNED; EXACT LABEL EQUIVALENCE PENDING**

M9 is non-modelling. It reconciles the company's written clarifications with the frozen M6/M8 project. The company target is now known to be arrival when the vessel reaches/enters the **destination-port area**. `Positions` is confirmed as historical AIS observations, `Tracks` as latest state, historical destination is available in `Positions`, historical ETA is not, and no longer AIS history is available for the exercise. The information set of the pre-existing ETA model remains unknown.

Consequences:
- the current research-gate entrance is now a **conceptually aligned proxy** for the requested event family;
- `recorded_at` is treated as observation time, while upstream message-generation/reception/provider semantics remain unknown;
- reported-AIS-ETA longitudinal comparison remains blocked;
- frozen M6 metrics are not renamed as official company-target metrics;
- the M6 final holdout is not reopened.

Remaining questions before exact target equivalence can be claimed:
1. authoritative ground-truth arrival timestamp vs reconstruction from `Positions`;
2. perimeter/polygon defining the destination-port area.

If either answer changes the label definition materially, start a **new target version and new evaluation protocol** rather than tuning or rewriting M6 post hoc.

### M10 — Company Reference-ETA Benchmark
**Status: DONE — company-defined target implemented**

Trigger: final company clarification stated that `Tracks.eta` should be used as the ground-truth/reference value for the exercise and that reconstructing physical arrival / port geometry is not required for the primary task.

Rules:
1. `Tracks.eta` is target-only and is called **reference ETA**, not observed ATA.
2. Create one supervised row per MMSI/Tracks record.
3. Historical `Positions` features must satisfy `recorded_at <= Tracks.last_update`.
4. Split by deterministic target-free MMSI hash: 70/15/15.
5. Select candidate model on calibration only; open the M10 final test once.
6. Preserve past/stale/far-future parseable reference values in the strict benchmark and report their effect explicitly.
7. M6 physical-arrival predictor and holdout remain frozen and separate.

Outputs:
- `reports/M10_FREEZE.json`
- `reports/M10_REPORT.md`
- `reports/m10_reference_dataset_summary.json`
- `reports/m10_calibration_model_comparison.csv`
- `reports/m10_final_predictions.csv`
- `reports/m10_reference_eta_quality.png`
- `reports/m10_calibration_ablation.png`
- `reports/m10_result_panel.png`
- `models/m10_strict_reference_catboost.cbm`

Result: 439 parseable ship ETA references across 439 MMSIs. Strict calibration selected current-snapshot CatBoost; the separate future-reference calibration selected a train-only destination median. Extreme stale/far reference values dominate aggregate MAE, while the 0–7 day future-reference final subset is materially more stable. The main finding is that history-heavy ML does not clearly dominate simple latest-state/destination information for the supplied reference target.

Gate: **M10 complete. No post-final-test tuning on this dataset.**

### M11 — Reference ETA Quality & Error Forensics
**Status: DONE — diagnostic-only; M10 freeze preserved**

Goal: explain the M10 strict benchmark heavy tail without deleting company reference labels or changing the model.

Rules:
1. no training, tuning, model reselection, split change or final-test reopening;
2. retain every parseable `Tracks.eta` in the strict benchmark;
3. quantify temporal status, ETA minute granularity, missing-year sensitivity, placeholder-like patterns, feature freshness and final-error concentration;
4. any subset score is post-hoc diagnostic only and cannot replace the frozen strict headline result.

Key findings:
- 64/439 (14.6%) references are already in the past at `last_update`;
- 56/439 are >7 days in the future; 19 have absolute horizon >90 days;
- AIS ETA minute values are coarse: 86.1% `:00`, 93.8% `:00/:30`, 96.1% on a 5-minute grid;
- 5 rows have <90-day separation between the chosen and second-best year interpretation because Message 5 does not encode a year;
- 7 rows have `01/01 00:00` or `01/01 01:01` placeholder-like patterns; they remain in the strict benchmark;
- on the frozen 53-row strict final test, the top 5 errors explain 79.1% of total absolute error and the top 10 explain 91.4%;
- the same frozen model has 28.6 h MAE on the 40 final references lying 0–7 days in the future, versus 312.0 h on all parseable references.

Gate: **M11 complete. Proceed to M12 explainability/ablation only; no post-final-test model optimization.**

### M12 — Frozen Model Explainability & Ablation
**Status: DONE — DIAGNOSTIC ONLY; FROZEN M10 UNCHANGED**

Goal: explain the already-selected M10 strict CatBoost without any post-final tuning.

Rules:
1. do not refit, tune, select, prune or replace the M10 model;
2. verify exact reproduction of all 53 strict final predictions from the frozen CatBoost artifact;
3. use CatBoost native feature importance and SHAP only as model attribution, not causal evidence;
4. use train-derived group masking on calibration only as a sensitivity diagnostic, never as a new selection rule;
5. reuse the pre-final M10 calibration candidate table for history/architecture ablation rather than retraining rejected candidates;
6. preserve all M10/M6 freeze hashes.

Key findings:
- `destination_norm` is the strongest single feature under both native importance and calibration mean |SHAP|;
- current motion, voyage state and location are the dominant feature groups;
- the pre-final causal-history CatBoost did not improve calibration over current-snapshot CatBoost;
- the future-reference calibration diagnostic favored the simple train-only destination median;
- strict-final predictions occupy a far narrower horizon range than the extreme stale/far company references, explaining the heavy-tail failure mode.

Gate: **M12 complete. Proceed to M13 company-facing report reconciliation only; no M10 model or split changes.**

### M13 — Final Company Submission Reconciliation
**Status: DONE — DOCUMENTATION/PACKAGING ONLY; FROZEN M10 UNCHANGED**

Goal: make the final company-facing narrative match the company's final task definition without changing the frozen model or evaluation.

Rules:
1. M10–M12 are the primary exercise evidence chain;
2. M0–M9 remain secondary physical-arrival/domain research;
3. the strict 53-row M10 final result remains the headline benchmark;
4. 0–7 day and other post-hoc subsets remain diagnostic only;
5. unsupported claims (external-model superiority, observed ATA, unseen-port generalization, berth/all-fast, safety-critical use) remain blocked;
6. no M10/M6 model, split, final prediction or freeze hash may change.

Outputs:
- reconciled `README.md`, `EXECUTIVE_SUMMARY.md`, `TAKE_HOME_REPORT.md` and `docs/MODEL_CARD.md`;
- `docs/COMPANY_SUBMISSION_GUIDE.md`;
- `reports/M13_RECONCILIATION_REPORT.md`;
- `reports/m13_claim_map.csv`;
- deterministic `dist/ais_eta_takehome_submission_m13_draft.zip`.

Gate: **M13 complete. Proceed to M14 final clean-room reproducibility + immutable submission freeze; no post-final modelling.**

### M14 — Final Reproducibility & Submission Freeze
**Status: DONE — FINAL PACKAGING/INTEGRITY ONLY; FROZEN M10/M6 UNCHANGED**

Goal: convert the reconciled M13 draft into the immutable company-facing submission without reopening model selection or final evaluation.

Rules:
1. do not train, tune, refit, prune, select or rescore a model;
2. verify the M10 reference dataset, frozen CatBoost, final-prediction ledger and one-time-open marker hashes;
3. verify all 14 artifact hashes embedded in the frozen M6 record;
4. build the final ZIP with stable sorted input order and fixed archive metadata;
5. hash every payload file in `SUBMISSION_CONTENT_MANIFEST.json` and provide `VERIFY_SUBMISSION.py` inside the package;
6. keep the byte-level ZIP SHA-256 outside the archive to avoid self-referential hashing;
7. extract the final ZIP into a fresh temporary directory and rerun integrity verification, analytical tests and `compileall`;
8. exclude raw AIS, high-volume row-level derived data, internal evidence and correspondence;
9. any later edit creates a new hash and is not the M14-frozen submission.

Outputs:
- `reports/M14_FINAL_FREEZE.json`;
- `reports/M14_REPRODUCIBILITY_REPORT.md`;
- `reports/m14_result_panel.png`;
- `scripts/63_m14_verify_pre_freeze.py`;
- `scripts/64_m14_build_final_submission.py`;
- `scripts/65_m14_verify_final_submission.py`;
- `scripts/66_m14_visual_audit.py`;
- `dist/ais_eta_takehome_submission_final.zip`;
- `dist/ais_eta_takehome_submission_final_manifest.json`;
- `dist/ais_eta_takehome_submission_final.zip.sha256`.

Gate: **M14 is final when source freezes pass, repository tests/compilation pass, two consecutive deterministic builds have the same SHA-256, and the extracted final package passes integrity + pytest + compilation. No modelling change is allowed.**


### M15 — Interview / Technical Defense Preparation
**Status: DONE — INTERNAL POST-FREEZE PREPARATION; M14/M10/M6 UNCHANGED**

Goal: prepare a rigorous oral/whiteboard/live-demo defense from frozen evidence without modifying the scientific result or immutable company submission.

Rules:
1. do not train, tune, refit, prune, select or rescore any model;
2. do not rebuild or alter the M14 final submission ZIP, manifest or checksum;
3. preserve exact SHA-256 for M10 model, M10 final prediction ledger and M6 freeze;
4. keep the strict 53-row result (312.0 h MAE; 25.3 h MedAE; 560.4 h P90) as the headline;
5. keep M11 0–7d and error-concentration results explicitly diagnostic/post-hoc;
6. explain SHAP as frozen-model attribution, not causal inference;
7. explicitly block external-model superiority, observed-ATA equivalence, unseen-port, berth/all-fast and safety-critical claims;
8. provide safe demo commands that do not require absent raw company CSVs or rewrite frozen artifacts.

Outputs:
- `docs/M15_TECHNICAL_DEFENSE_IT.md`;
- `docs/M15_QA_BANK_IT.md`;
- `docs/M15_LIVE_DEMO_CHECKLIST_IT.md`;
- `reports/M15_TECHNICAL_DEFENSE_REPORT.md`;
- `reports/M15_DEFENSE_MANIFEST.json`;
- `reports/m15_defense_map.png`;
- `scripts/67_m15_verify_defense.py`;
- `scripts/68_m15_visual_audit.py`;
- `tests/test_m15.py`.

Gate: **M15 is complete only when the immutable M14/M10/M6 hashes are unchanged, the full repository suite and compileall pass, and the defense material keeps strict, diagnostic and blocked claims clearly separated.**

### M16 — Smart Hybrid Challenger
**Status: IN PROGRESS — M16A COMPLETE; M14/M15/M10/M6 MUST REMAIN IMMUTABLE**

Goal: test whether a more domain-aware, leakage-safe system can outperform the frozen M10 current-snapshot benchmark by exploiting canonical destination semantics, hierarchical priors, historical route analogues, maritime/physics structure, expert blending and calibrated uncertainty — without using the already-observed M10 final set for any modelling decision.

This milestone is an **experimental challenger branch**, not a rewrite of the frozen company submission. M14 remains the immutable submitted package and M15 remains the defense package. M16 gets its own artefacts, metrics and claims.

Research anchors for the planned design:
- IMO SN/Circ.244 / USCG mirror: AIS destination is a free-text field and UN/LOCODE is recommended to harmonize port naming — `https://navcen.uscg.gov/sites/default/files/pdf/NAIS/IMO_SN_Circ244_AIS_UNLOCODE.pdf`;
- UNECE UN/LOCODE Italy list, including Augusta `IT AUG` and Catania `IT CTA` — `https://service.unece.org/trade/locode/it.htm`;
- CatBoost regression objectives include MAE, Quantile/MultiQuantile, Huber and LogCosh — `https://catboost.ai/docs/en/concepts/loss-functions-regression`;
- Fan et al., Ocean Engineering (2026), route searching for ship ETA using improved DTW and confidence-weighted historical ETA integration — DOI `10.1016/j.oceaneng.2026.125455`.

#### M16 non-negotiable rules
1. **Never modify M14/M15 frozen artefacts.** Verify their hashes before and after every M16 material change.
2. **Never use the 53-row M10 final set for selection.** It is excluded from training, hyperparameter tuning, feature engineering decisions, architecture selection, blending, gating, threshold selection and uncertainty calibration.
3. M16 model development uses only the former M10 **321 train + 65 calibration = 386 development rows**.
4. All target-dependent statistics, destination priors, encoders, route neighbours, cluster prototypes, scalers and ensemble weights are fitted **inside the training fold only**.
5. Historical `Positions` information must satisfy `recorded_at <= Tracks.last_update` for the supervised row being predicted.
6. Any external port/UN/LOCODE lookup is target-independent and versioned with source/provenance.
7. The strict company-reference target remains the primary target. Any 0–7 day future-reference analysis remains a diagnostic slice unless the company explicitly changes the scoring definition.
8. A more complex model is promoted only if its gain is stable out-of-fold and survives ablation/stress tests; complexity without stable gain is rejected.
9. Do not claim observed-ATA accuracy, external-model superiority, unseen-port generalization or production readiness without corresponding evidence.
10. Every material M16 change follows the existing evidence rule: `CHANGE.md`, patch, commands, raw test output and `01_changes.png` / `02_tests.png` / `03_result.png`.

#### M16 evaluation protocol — fixed before modelling
Primary development population: 386 former M10 train+calibration MMSIs.

Protocol:
- deterministic, target-free outer folds grouped by MMSI;
- preferred default: repeated 5-fold OOF evaluation with the exact fold seed/config frozen before candidate comparison;
- any inner tuning occurs only inside each outer-train partition;
- all candidate predictions are stored in one immutable OOF ledger keyed by MMSI/fold/model;
- paired error deltas are reported against the reimplemented M10 current-snapshot CatBoost recipe and train-only destination-median baseline;
- the 53-row M10 final set is not loaded by M16 training/selection scripts by default.

Primary selection metric:
- voyage/MMSI-level **MAE hours** on the strict reference target.

Mandatory secondary metrics:
- MedAE;
- RMSE;
- P90/P95 absolute error;
- within 24/48/72 h where meaningful;
- error by reference temporal status (`past`, `0–7d`, `>7d`);
- fold-to-fold gain stability;
- prediction range and pathological-output count.

Promotion gate:
- lower pooled OOF MAE than both frozen-recipe baselines;
- positive paired MAE gain in a clear majority of outer folds/repeats rather than one lucky split;
- no material collapse in MedAE/P90 or a documented reason for the trade-off;
- all leakage tests pass;
- ablation identifies where the gain comes from;
- result remains reproducible from a frozen M16 manifest.

A candidate that wins only on a post-hoc slice or only after inspecting the old 53-row final set is **not promoted**.

#### M16A — Branch integrity + benchmark reconstruction
**Status: DONE — DEVELOPMENT-ONLY OOF BASELINE FROZEN**

Goal: create the clean experimental branch and reproduce development-only versions of the frozen baselines before adding intelligence.

Work:
- snapshot/hash M14 final ZIP, M10 model, M10 split ledger, M10 final predictions and M6 freeze;
- create an M16 development manifest containing exactly the 386 allowed MMSIs;
- hard-block the 53 final MMSIs from M16 selection code;
- reimplement the M10 current-snapshot CatBoost recipe under outer-fold OOF;
- reimplement train-only global/destination-median priors under the same folds;
- freeze fold IDs, seeds, metrics and comparison schema.

Gate: OOF benchmark ledger is deterministic, no final MMSI enters development, and baseline results reproduce across two clean runs.

Result (2026-09-21): **PASS.** Exactly 386 development MMSIs are assigned to five deterministic target-free outer folds (78/77/77/77/77); all 53 old-final MMSIs are hard-blocked. Two clean rebuilds produced byte-identical M16A artifacts. Pooled OOF MAE is 202.97 h for train-only destination median, 206.14 h for the leakage-safe current-snapshot CatBoost reconstruction and 213.66 h for the global train median. CatBoost beats global median in 4/5 folds but destination median in only 1/5, so M16B destination canonicalization remains the next planned experiment. The old 53-row final set was not used for selection or scoring in M16A.

#### M16B — Canonical Destination Resolver
**Status: DONE — semantic gate passed 2026-09-21**

Goal: replace fragmented AIS destination strings with a provenance-aware semantic destination representation.

Rationale: AIS destination is manually entered/free text and official guidance recommends UN/LOCODE to reduce multiple spellings of the same port. The resolver therefore treats canonicalization as a first-class feature-engineering problem rather than a cosmetic string cleanup.

Work:
- normalize casing, whitespace, punctuation and common separators;
- exact UN/LOCODE recognition (`ITAUG`, `IT AUG`, etc.);
- curated aliases for observed high-frequency destinations;
- conservative fuzzy matching only above a frozen confidence threshold;
- explicit `UNKNOWN`, `FOR_ORDERS`, `NON_SPECIFIC` classes instead of forced port assignment;
- attach canonical port name, country, UN/LOCODE and coordinates when resolved;
- add resolver confidence and resolution-method features;
- store the lookup table/version and provenance.

Tests:
- deterministic alias fixtures;
- no target/reference-ETA use in resolution;
- ambiguity tests;
- exact fixtures for Augusta `IT AUG` and Catania `IT CTA`;
- before/after destination-cardinality and coverage report.

Gate: resolver reduces semantic fragmentation without target leakage and without silently coercing ambiguous destinations.

Result (2026-09-21): **PASS semantic gate.** On the frozen 386-row M16A development population, the target-free resolver reduces destination cardinality from 248 raw categories to 169 canonical categories (-31.9%), resolves 158 rows (40.9%) to catalogued ports, preserves country-level `MALTA` as ambiguous rather than forcing Valletta, and explicitly separates unknown/for-orders/non-specific values. A same-fold diagnostic shows that naive canonical destination medians improve MedAE (22.27 -> 20.73 h) and P90 AE (330.82 -> 313.70 h) but worsen pooled MAE (202.97 -> 209.26 h), demonstrating that semantic cleanup alone is insufficient under extreme/stale reference-ETA values. This directly motivates M16C fold-safe support thresholds, shrinkage and fallback. The 53 old-final MMSIs were not used for resolution, selection or scoring.

#### M16C — Fold-safe Hierarchical Destination Priors
**Status: DONE — COMPONENT RETAINED, NOT STANDALONE-PROMOTED**

Goal: exploit the strong destination signal with robust shrinkage/fallback rather than high-variance raw category memorization.

Candidate hierarchy:
- global prior;
- canonical destination;
- destination + vessel category;
- destination + coarse location/distance band;
- optional destination + movement-state bucket.

Rules:
- every prior is computed on outer-train only;
- minimum-support thresholds and shrinkage strength are selected inside inner training only;
- fallback path is deterministic and logged;
- median/robust estimators are preferred to mean under the observed heavy tail.

Gate: hierarchical prior beats or complements the simple train-only destination median OOF without relying on rare-category memorization.

Result (2026-09-21): **PASS as complementary expert; NOT standalone-promoted.** M16C uses nested, development-only OOF selection: each M16A outer-train fold runs a 4-fold inner search over a deliberately small set of hierarchy/support/shrinkage/clip configurations, and only the selected configuration is refit on the complete outer-train before predicting the untouched outer-validation fold. Priors use robust group medians with parent shrinkage `w=n/(n+alpha)`, deterministic low-support/unseen fallback, and optional training-only quantile guardrails. On the frozen 386-row development population, pooled MAE is 202.56 h versus 202.97 h for the M16A raw destination median, 206.14 h for the M16A current-snapshot CatBoost, and 209.26 h for the naive M16B canonical median. However, M16C beats the raw destination median in only 1/5 outer folds and regresses MedAE/P90, so it is **not** promoted as a standalone replacement. A fixed, untuned 50/50 diagnostic blend with the raw destination prior reaches 199.86 h MAE, supporting complementary signal; ensemble weights remain strictly deferred to M16G. The 53 old-final MMSIs remain hard-blocked from selection/scoring.

#### M16D — Historical Route-Analogue Expert
**Status: DONE — PASS_ROUTE_SIGNAL_COMPONENT (development-only OOF; retained for M16G, not a new final-test claim)**

Goal: test actual trajectory similarity rather than only rolling aggregates.

Work:
- extract causal trajectory windows ending at `Tracks.last_update`;
- resample each history to a common temporal/spatial representation;
- compare normalized trajectories with a route-distance family such as DTW and/or discrete Fréchet distance;
- candidate matching keys may include canonical destination, vessel class and broad origin sector;
- retrieve Top-K neighbours from **outer-train only**;
- predict with a robust similarity-weighted target statistic;
- add neighbour support, similarity gap and route-confidence features.

Ablations:
- geometry only;
- geometry + kinematics;
- destination-gated vs global search;
- different causal history lengths;
- neighbour count K.

Gate: route analogue must beat simple history aggregates on OOF and show meaningful neighbour geometry on manual trajectory audits.

Result (2026-09-21): **PASS_ROUTE_SIGNAL_COMPONENT.** A nested development-only route analogue was built from causal 3 h / 6 h AIS histories resampled to 12 points and compared with band-limited DTW. A predeclared 12-config grid crossed geometry vs geometry+kinematics, global vs canonical-destination-gated search, and K=5/9; each outer fold selected its configuration only through 4-fold inner CV. The pooled route-analogue MAE is **194.98 h**, versus **208.36 h** for a directly reconstructed CatBoost current+causal-history-aggregate baseline and **202.97 h** for the M16A raw destination median. The route expert wins **4/5** outer folds against the history-aggregate CatBoost and **3/5** against the raw destination median; P90 improves to **292.90 h**. The P95 tail remains heavy, so M16D is frozen as a strong expert for M16G rather than post-hoc promoted as a final standalone model. The 53 old-final MMSIs remain hard-blocked.

#### M16E — Maritime / Physics Expert
**Status: DONE — PASS_PHYSICS_GATING_COMPONENT (2026-09-21)**

Goal: add an interpretable ETA component that encodes remaining maritime travel rather than forcing ML to rediscover basic motion physics.

Work:
- resolve destination coordinates/port anchor only when destination confidence is adequate;
- compute geodesic lower-bound distance and test a causal local-sinuosity route-detour proxy;
- estimate robust recent effective speed from causal 180/360 minute SOG summaries;
- build physics ETA/TTE baselines with safe handling of stopped/slow vessels;
- optionally learn a residual correction on top of the physics estimate inside each fold.

Rules:
- no straight-line ETA is promoted as the final expert without route sanity checks;
- `FOR ORDERS`/unknown destination receives no fabricated physics target;
- any external routing dataset/library and version must be documented;
- reference-ETA status is target-derived and may be used only for post-hoc diagnostics, never for selection/gating.

Gate: physics expert is retained only if it adds independent OOF signal or improves interpretability/gating.

Result (2026-09-21): **PASS_PHYSICS_GATING_COMPONENT.** M16E hard-blocks the 53 old-final MMSIs and uses the frozen M16A outer folds. A predeclared 12-config grid crosses 180/360 minute causal speed windows, geodesic vs local-sinuosity distance, and none/global/destination-shrunk residual correction; each outer fold selects only through 4-fold inner CV. Physics eligibility is target-free and requires a catalog-resolved destination with confidence >=0.99, at least 1 nm great-circle remaining distance and recent median SOG >=2 kn. Coverage is **115/386 (29.8%)**. On eligible OOF rows, physics reaches **161.48 h MAE / 2.64 h MedAE / 93.33 h P90**, versus **168.39 h / 8.82 h / 108.96 h** for M16D route analogue, and physics wins paired absolute error on **69.6%** of eligible rows. A fixed target-free hard gate (physics when eligible, M16D otherwise) reaches **192.93 h MAE**, improving M16D by **2.06 h** with **3/5** outer-fold wins. The selected distance mode is geodesic in all folds; local sinuosity was tested but not promoted, and no chart/water-mask routing dataset is claimed. `FUTURE_0_7D` performance is reported only as a target-derived diagnostic and is not used to select or gate the expert. M16E is retained for M16G, not promoted as a replacement for the frozen M14 submission.

#### M16F — Tabular Challenger Panel
**Status: COMPLETE — PASS_TABULAR_COMPLEMENT_FOR_M16G**

Goal: verify that gains come from representation/architecture rather than assuming CatBoost is optimal.

Candidate families, subject to dependency/reproducibility checks:
- M10-recipe CatBoost;
- ExtraTrees / RandomForest robust sanity models;
- HistGradientBoosting;
- SVR on a carefully scaled numeric representation;
- optional LightGBM/XGBoost only if introducing the dependency is justified.

Loss/target experiments:
- MAE baseline;
- Huber/LogCosh robustness where supported;
- Quantile/MultiQuantile for uncertainty experiments.

Rules:
- small, predeclared search spaces;
- nested tuning only;
- no brute-force hyperparameter sweep justified by the small independent N;
- deep sequence models remain deferred unless effective sample size increases materially.

Gate: retain only models with stable OOF contribution beyond the representation improvements.

Result (2026-09-21): **PASS_TABULAR_COMPLEMENT_FOR_M16G.** Five families were compared on the frozen M16A outer folds with a common target-free representation and 10 predeclared configurations. Each family selects its configuration only through 3-fold inner CV; the 53 old-final MMSIs remain hard-blocked and no M16C/D/E expert prediction is used as a feature. HistGradientBoosting is the best M16F family at **197.28 h pooled OOF MAE / 27.16 h MedAE / 292.84 h P90**, improving the frozen M16A current-snapshot CatBoost by **8.87 h** and winning **5/5** outer folds. The enhanced-representation CatBoost is close at **197.87 h MAE**, indicating that representation/robust loss choice accounts for most of the gain rather than a large algorithm-family advantage. M16D route analogue remains better on pooled MAE at **194.98 h**, although M16F HistGradientBoosting wins **3/5** folds against it. M16F is retained as the best tabular expert for M16G, not promoted as a replacement for the frozen submission or the route expert.

#### M16G — Mixture of Experts / OOF Stacking
**Status: DONE — PASS_MOE_STABLE_GAIN (2026-09-21)**

Goal: combine experts according to information quality instead of forcing one predictor to handle every AIS regime.

Candidate experts:
- hierarchical destination prior;
- route-analogue expert;
- physics/residual expert;
- best current-state tabular model.

Gating inputs may include only prediction-time information such as:
- destination-resolution confidence;
- route-neighbour support/similarity;
- recent-motion quality;
- stale-state proxies;
- vessel category;
- geography/distance support.

Rules:
- stacker/gating model is trained only on **OOF predictions from inner training data**;
- no base learner may produce in-sample predictions for the rows used to fit the stacker;
- compare learned gating against simple non-negative weighted blending and median blending;
- preserve expert predictions for full ablation.

Gate: mixture must deliver stable paired OOF gain over the best single expert and expose interpretable expert weights/selection behaviour.

Result (2026-09-21): **PASS_MOE_STABLE_GAIN.** For each frozen M16A outer fold, M16G regenerates the M16C prior, M16D route, M16E physics-gate and M16F HistGradientBoosting predictions as **inner-OOF base predictions using only the current outer-train** before fitting/selecting the meta strategy. The meta panel compares median, mean, normalized non-negative NNLS, RandomForest/ExtraTrees/HistGradientBoosting hard gating and logistic soft gating. The selected nested mixture reaches **185.88 h pooled OOF MAE / 19.34 h MedAE / 290.68 h P90**, versus **192.93 h MAE** for the best frozen single operational expert (M16E physics-gate/route fallback): **7.04 h gain**, **4/5 outer-fold wins**. The hard-gating path preserves the best-single prediction exactly on **49.7%** of rows; among rows where it actually switches prediction, it wins **54.6%**. The 53 old-final MMSIs remain hard-blocked. A globally pooled RandomForest gate scores lower post-hoc, but it is not promoted because the official M16G result is the per-outer-fold candidate selected only by inner meta-CV.

#### M16H — Probabilistic ETA + Confidence
**Status: DONE — `PASS_PROBABILISTIC_CALIBRATION_CONFIDENCE`**

Goal: accompany the point prediction with calibrated uncertainty and reliability indicators.

Outputs considered:
- P10/P50/P90 or equivalent quantiles;
- conformal residual interval calibrated strictly inside development folds;
- confidence tier based on destination, route and data-quality support.

Metrics:
- empirical interval coverage;
- interval width;
- coverage by destination-confidence and temporal-status groups;
- pinball loss for quantile predictions where applicable.

Gate: uncertainty is calibrated from held-out/OOF residuals and never from the old M10 final set.

Result (2026-09-21): **PASS_PROBABILISTIC_CALIBRATION_CONFIDENCE.** M16H preserves the frozen M16G point prediction exactly as P50 and calibrates an empirical central-80% residual/conformal-style interval using only development OOF residuals. Inside each M16A outer-train, a small HistGradientBoosting error model is cross-fitted on prediction-time-only expert/support features; interval scale candidates (`global_abs`, adaptive 12/24/48/72 h floors) are selected by inner-fold coverage/sharpness and only then applied to the outer validation fold. Pooled coverage is **80.05%** at nominal 80%, with **159.42 h mean width** versus **221.26 h** for a global symmetric residual interval (**27.9% narrower**) and 4/5 outer folds at >=75% empirical coverage. Confidence terciles are assigned from cross-fitted predicted absolute error and show ordered typical error: HIGH **3.32 h MedAE**, MEDIUM **19.17 h**, LOW **84.40 h**. Subgroup coverage is diagnostic only; low-confidence/reference-pathology regimes remain heavy-tailed, and no conditional-coverage guarantee is claimed. All 53 old-final MMSIs remain blocked.

#### M16I — Red-team, ablation and stress testing
**Status: DONE — `PASS_RED_TEAM_WITH_HEAVY_TAIL_CAVEAT`**

Required stress tests completed:
- structural counterfactual removal of canonical-destination path, route expert, physics expert and longitudinal-history path (diagnostic only; no post-red-team re-tuning);
- cold/rare destination support computed from each row's outer-train only;
- unknown/`FOR_ORDERS`/ambiguous destination regimes;
- vessel-category subgroups;
- stale/past and extreme-future reference rows as **target-derived diagnostics only**;
- low route-support / route-fallback regimes;
- frozen-fold sensitivity, leave-one-fold-out sensitivity, deterministic stratified paired bootstrap;
- top-error/gain concentration, target-magnitude trimming and winsorised stress;
- M16H coverage by confidence/risk/subgroup and prediction-pathology/fallback audit.

Result (2026-09-22): M16G keeps **185.88 h MAE** versus **192.93 h** for the best frozen single operational expert (**+7.04 h** pooled gain), wins **4/5** outer folds, and the gain stays positive after leaving out each fold (**5/5**), trimming the top 5% of rows by absolute target magnitude (**+9.66 h**), 95% winsorisation (**+4.07 h**), and removing the largest net-gain destination `NON_SPECIFIC` (**+2.49 h**). M16H pooled central-80% coverage remains **80.05%** and HIGH/MEDIUM/LOW MedAE remains ordered.

Red-team caveat: a deterministic stratified paired bootstrap gives **P(gain>0)=92.18%** but a **95% interval of [-2.02, +20.16] h**, and the five largest positive-gain rows contribute **51.0%** of total positive gain. Therefore the challenger survives the core robustness tests but does **not** earn a clean 95%-confidence superiority claim. The correct interpretation is a promising development-only challenger with material heavy-tail sensitivity; a new untouched company holdout is still required.

Gate outcome: **PASS_RED_TEAM_WITH_HEAVY_TAIL_CAVEAT.** M16 may proceed to M16J freeze/documentation, but M14 remains the official frozen submission and M16 must not be presented as proven superior on unseen data.

#### M16J — Freeze and company-facing challenger note
**Status: DONE — `PROMOTE_CHALLENGER_FOR_NEW_UNTOUCHED_HOLDOUT_ONLY` (2026-09-22)**

If the promotion gate passes:
- freeze M16 code, folds, OOF ledger, model artefacts and content hashes;
- write `reports/M16_REPORT.md` with strict separation between OOF development evidence and any post-hoc old-final analysis;
- produce a concise `docs/M16_COMPANY_NOTE_IT.md` explaining the design choice in interview language;
- create architecture/ablation/result visuals;
- preserve M14 as the official frozen submission unless the company explicitly asks for an updated challenger submission.

If the promotion gate fails:
- keep M16 as a documented negative result;
- report which intelligent components failed and why;
- do not replace the simpler M10/M14 solution merely because M16 is more sophisticated.

Preferred external confirmation: score the frozen M16 challenger on **new, untouched company data** or a newly issued holdout. That is the only clean way to make a fresh generalization claim after the original M10 final set has been observed.

Gate: **M16 closes only with a deterministic evidence bundle, passing repository tests, verified M14/M10/M6 immutability, a frozen OOF comparison ledger and an explicit PASS/FAIL promotion decision.**

Result (2026-09-22): M16 is frozen as a **development-only challenger for new untouched holdout validation**. The official nested OOF point estimate remains **185.88 h MAE**, versus **192.93 h** for the best single operational expert and **202.97 h** for the original destination-median development baseline. M16H keeps **80.05%** pooled coverage for the nominal central-80% interval. M16I robustness tests remain positive under leave-one-fold-out, trimming and winsorisation, but the paired-bootstrap 95% gain interval crosses zero and gain concentration remains material. Therefore M16 is promoted for external validation, **not** as a proven unseen-data winner and **not** as a replacement for the frozen M14 submission. A fresh company holdout is required for any new generalization/superiority claim.



## M17 — Post-freeze representation / retrieval research branch

M17 is strictly separate from the frozen M16J challenger and official M14 submission. The 53 already-observed M10 final MMSIs remain permanently unavailable for selection, representation learning, thresholding or tuning. Any M17 result is development-only until scored once on a newly issued untouched holdout.

### M17A — Self-Supervised / Learned Trajectory Retrieval Benchmark against M16D
**Status: DONE — `PASS_LEARNED_RETRIEVAL_COMPONENT` (2026-09-22)**

Hypothesis: M16D proved that historical route similarity matters, but DTW is hand-designed and pairwise. A target-free learned trajectory embedding may retrieve more useful historical analogues while remaining compatible with approximate-nearest-neighbour indexing.

Predeclared isolation protocol:
- reuse frozen M16A 5 outer folds;
- hard-block all 53 old-final MMSIs;
- pretrain one encoder per outer fold using **only causal windows owned by outer-train MMSIs**;
- do not allow outer-valid trajectories into SSL pretraining even without labels;
- use the same 6 h geometry+SOG/COG query sequence for every metric;
- fix retrieval to canonical-destination gating, `K=9`, and the unchanged M16D similarity-weighted-median target aggregator;
- compare fixed DTW, an outer-train PCA16 cosine control, and a self-supervised cosine embedding;
- no post-hoc configuration search.

Self-supervised encoder: compact TCN-style temporal encoder to a 32-D latent vector, trained on 3,076 causal unlabeled AIS windows with masked reconstruction plus contrastive consistency. Exact cosine is used at N=386; the embedding is HNSW/FAISS-compatible if retrieval is later scaled.

Predeclared promotion gate: SSL pooled OOF MAE must beat frozen M16D, win at least 3/5 outer folds, and keep P90 <=1.05x frozen M16D.

Result:
- frozen M16D official: **194.98 h MAE / 292.90 h P90**;
- fixed same-config 6 h DTW: **192.57 h MAE**;
- PCA16 cosine: **190.80 h MAE**;
- **SSL32 cosine: 190.11 h MAE / 19.04 h MedAE / 276.64 h P90**;
- SSL wins **5/5** folds vs frozen M16D;
- in the stricter apples-to-apples comparison, SSL beats fixed same-config DTW by **2.46 h MAE** and wins **4/5** folds.

Interpretation: the learned representation adds real route-retrieval signal beyond DTW, but M17A is still development-only. It is retained as a candidate expert for a later memory/HNSW or MoE experiment, not as a new company-facing final winner.


## M17B — Historical Memory / HNSW ETA Expert — DONE

Status: **PASS_HNSW_SCALING_NO_MEMORY_PROMOTION**.

M17B tested whether the passing M17A learned representation improves further when retrieval is expanded from one owner-level trajectory to a memory of recent historical snapshots. The protocol remains development-only and uses the frozen M16A outer folds; the 53 old-final MMSIs remain hard-blocked.

Protocol:
- recent memory: causal 6 h windows from the last 24 h, 1 h stride;
- 3,334 memory snapshots across 380 development MMSIs;
- outer-train-only lightweight SSL encoder, with same-encoder owner baseline;
- canonical-destination cohort gate with global fallback;
- retrieve windows, deduplicate by owner MMSI, K=9 weighted median;
- snapshot-aligned TTE labels are converted back to the original `last_update` target scale;
- exact cosine is the scientific oracle; deterministic HNSW-style ANN is audited against it.

Result:
- frozen M17A SSL owner retrieval: **190.11 h MAE**;
- M17B same-encoder owner control: **192.00 h MAE**;
- M17B expanded snapshot memory exact: **198.50 h MAE**;
- M17B HNSW-style memory: **198.55 h MAE**;
- expanded memory wins only **2/5** folds vs M17A and is **not promoted**;
- HNSW owner recall@K vs exact: **97.55%**;
- HNSW MAE delta vs exact: **0.051 h**, so ANN scaling equivalence passes.

Interpretation: overlapping recent snapshots add redundancy/trajectory-phase noise when attached to one later declared ETA target. HNSW is useful as a scalable retrieval backend, but simply increasing memory density does not improve ETA accuracy. The next research direction should add stronger route-phase/semantic structure, not more snapshots.

Next: **M17C — Sicily Historical Maritime Corridor / Knowledge Graph vs M16E Physics**.

## M17C — Sicily Historical Maritime Corridor / Knowledge Graph vs M16E Physics — DONE

Status: **PASS_CORRIDOR_SIGNAL_COMPONENT**.

M17C tests whether an AIS-only directed historical corridor graph adds information beyond M16E's great-circle-distance / recent-speed physics expert. The 53 old-final MMSIs remain permanently hard-blocked and M17C is development-only.

Protocol:
- fixed local grid resolution: 0.20° (geohash-4-like local granularity);
- AIS transition events only; no ETA target enters topology, edge weights or support;
- per outer fold, every outer-validation MMSI is excluded from graph construction;
- per query, only transitions timestamped at or before that row's `last_update` are visible;
- directed edges preserve historical travel direction;
- robust edge speed hierarchy: edge+ship-type+6h-time-bin → edge+ship-type → edge+6h-time-bin → edge-global;
- Dijkstra path time plus short source/destination snap legs;
- graph support requires source/destination within 40 nm of observed graph nodes;
- frozen M16E prediction is the apples-to-apples comparator; unsupported rows retain the frozen M16E hard-gate/route fallback.

Result:
- 6,531 usable transition events from 883 non-final MMSIs;
- M16E physics-eligible: 115 rows;
- causal graph-supported: 25 rows (21.7% of physics-eligible);
- M17C graph on supported rows: **410.475 h MAE** vs **410.698 h** for M16E physics on the exact same rows;
- paired absolute-error win share: **60.0%**;
- full graph-gated system: **192.911 h MAE** vs **192.926 h** frozen M16E hard gate, gain **0.014 h**, wins **3/5 folds**;
- target-derived `FUTURE_0_7D` diagnostic only: graph **1.775 h MAE** vs physics **1.849 h** on 16 supported rows.

Interpretation: the graph supplies a real but very small structural signal. It is useful because it replaces straight-line travel with empirically observed directed corridors and contextual segment speeds, but local AIS coverage limits destination support and the declared-ETA heavy tail dominates pooled MAE. M17C is therefore retained as an MoE candidate feature/expert, not promoted as a standalone replacement.

Next: **M17D — frozen M17A + M17C extension of the development-only Mixture of Experts**, or semantic-key-point intent research if the combined gate fails.

## M17D — Frozen M16G MoE extension with M17A learned retrieval + M17C graph — DONE

**Status:** `NO_M17D_PROMOTION`.

M17D tested whether the two post-freeze signals that individually added structural information could improve the frozen M16G development-only Mixture of Experts. The expert panel was expanded from four to six signals:

1. M16C hierarchical destination prior;
2. M16D DTW route analogue;
3. M16E physics-gate / route fallback;
4. M16F HistGradientBoosting;
5. M17A self-supervised learned trajectory retrieval;
6. M17C historical corridor-graph gate.

The stacking protocol is strict nested OOF. For each M16A outer fold, all six expert predictions used to train/select the meta layer are regenerated inner-OOF inside that outer-train partition. The M17A encoder is re-trained only on causal inner-train MMSI trajectories; the M17C graph excludes both the current outer-valid and inner-valid owners. The 53 M10 old-final MMSIs remain hard-blocked.

Predeclared promotion gate: >=1.0 h pooled MAE gain vs frozen M16G, >=3/5 fold wins, >=50% win share among rows whose prediction changes, and P90 <=1.05x frozen M16G.

Result:

- frozen M16G: **185.883 h MAE**, P90 **290.680 h**;
- M17D nested extended MoE: **196.200 h MAE**, P90 **301.166 h**;
- pooled gain: **-10.317 h** (negative = worse);
- fold wins: **2/5**;
- changed-row win share: **45.57%**;
- P90 ratio: **1.0361**.

Interpretation: the new experts are informative on their own, but unconstrained meta-selection over six experts is too unstable for only 386 labeled development voyages. M16G remains the frozen best mixture. M17A remains the strongest new route-similarity component; M17C remains a small structural graph signal. M17D is retained as a negative integration result and must not be promoted.

Possible next research, only if explicitly requested: constrained target-free gating/residual correction around M17A, latent declared-ETA noise/regime modelling, or semantic navigational-intent/key-point representation. Any superiority claim still requires a fresh untouched holdout.

## M17E — Constrained M16G ↔ M17A Selective Router — DONE

**Status:** `NO_M17E_PROMOTION`.

M17E deliberately reduces routing complexity after the negative M17D six-expert experiment. The router has only two actions: keep the frozen M16G prediction or switch to frozen M17A learned retrieval. M16G is reconstructed inside each outer-train from inner-OOF base experts and verified against its frozen outer prediction; M17A is regenerated inner-OOF with causal outer/inner-validation owner exclusion. Runtime router features are target-free.

Predeclared gate: >=1.0 h MAE gain vs M16G, >=3/5 fold wins, >=52% wins on changed rows, P90 no worse than M16G, <=35% switch share, and <=15 h worst-fold regression.

Result: M16G **185.88 h** -> M17E **185.32 h** MAE (gain **0.57 h**), 3/5 fold wins, 55.6% changed-row wins, 2.3% switched rows and P90 ratio 0.963. The router improves all observed switched-fold comparisons without a fold regression, but the pooled gain is below 1 h, therefore it is **not promoted** and M16G remains the frozen champion.

Next preferred branch: **M17F — latent target-noise / declared-ETA regime modelling**; alternative M17G — stronger SSL trajectory representation.



### M17F — Latent Target-Noise / Declared-ETA Regime Model
**Status: DONE — NO_M17F_PROMOTION**

Goal: test whether M16G residual heavy tails can be explained by a latent declared-ETA quality/regime variable that is learned from training residuals but predicted at runtime only from target-free signals.

Protocol:
- 386 development MMSIs only; 53 old-final MMSIs hard-blocked;
- reconstruct M16G inner-OOF inside each outer-train;
- candidates fixed before benchmark: no correction, robust Huber residual correction, 2/3-component Student-t residual mixtures with logistic target-free gating;
- true/derived ETA regime is never a runtime feature;
- promotion requires >=1 h MAE gain, >=3/5 fold wins, P90 <=1.02x M16G, trimmed-95% MAE non-worse, <=15 h worst-fold regression.

Result: every outer fold selected `always_m16g`. Frozen M16G and M17F both score 185.883 h MAE. Huber gives 186.334 h; Student-t mixtures degrade further (~193.15 h for two regimes, 211.15 h for three). Heavy-tail residuals exist, but the available runtime features do not predict a stable declared-ETA noise regime. Keep M16G; do not promote M17F.

## M17G — Stronger Self-Supervised AIS Encoder: MoCo / Masked Autoencoder vs frozen M17A — DONE

**Status:** `PASS_STRONGER_SSL_ENCODER_COMPONENT`.

M17G isolates the representation-learning question left open by M17A. The downstream retrieval contract is frozen: `geo_kin_6h`, canonical-destination gate, K=9 and the M16D similarity-weighted median ETA aggregator. Only the target-free trajectory encoder changes. Each encoder is retrained separately inside each M16A outer fold using only causal windows owned by outer-train MMSIs; the 53 old-final MMSIs remain blocked and outer-valid trajectories are excluded from SSL pretraining.

Predeclared candidates:
- `moco_tcn64`: momentum query/key TCN encoders, 64-D embeddings, FIFO negative queue, InfoNCE;
- `masked_transformer_mae64`: compact 2-layer Transformer, 64-D embeddings, 50% masked-timestep reconstruction with SmoothL1.

Predeclared component promotion gate: >=1.0 h MAE gain vs frozen M17A, >=3/5 fold wins, and P90 <=1.03x M17A.

Results:
- frozen M17A SSL32: **190.105 h MAE**, P90 **276.643 h**;
- M17G MoCo-TCN64: **188.287 h MAE**, gain **1.819 h**, **3/5** fold wins, P90 **263.718 h** -> **PASS**;
- M17G Masked-Transformer64: **190.910 h MAE**, gain **-0.805 h**, **2/5** fold wins, P90 **278.126 h** -> no promotion.

Interpretation: stronger contrastive representation learning adds route-retrieval signal while preserving the same ETA aggregation mechanism. The masked Transformer does not improve this small-data/local-window retrieval task. M17G is retained as a stronger retrieval component only; M16G remains the best overall promoted development-only system at 185.88 h MAE. Any superiority claim still requires a fresh untouched holdout.

## M17H — Conservative M16G integration of M17G MoCo retrieval — DONE

**Status:** `NO_M17H_PROMOTION`.

M17H intentionally keeps the M16G ensemble at four experts. It replaces the M16D route analogue one-for-one with the passing M17G MoCo-TCN64 retrieval component and uses the same MoCo retrieval as the fallback for the M16E physics gate. The meta candidate for each outer fold is frozen to the strategy selected by M16G before M17G existed; it is refit only on the new M17H inner-OOF four-expert matrix and is not re-selected.

Predeclared gate: >=1.0 h MAE gain vs frozen M16G, >=3/5 fold wins, >=50% win share among changed rows, P90 <=1.02x M16G, and <=15 h worst-fold regression.

Result: frozen M16G **185.883 h MAE** -> M17H **192.176 h MAE** (gain **-6.294 h**), only **2/5** fold wins, **47.40%** changed-row win share, P90 ratio **1.050**, worst-fold regression **29.590 h**. The gate therefore fails. M17G remains a stronger standalone retrieval component, but direct substitution into the frozen M16G architecture does not improve the overall system.

Interpretation: component-level improvement does not automatically transfer through a previously selected ensemble/gating architecture. M16G remains the promoted development-only champion. The scientifically preferred next action is a fresh untouched company holdout rather than continued recombination on the same 386 development labels.

## M18 — Evidence-First Generalization & Data Ceiling

M18 begins only after M17H. Its purpose is to stop repeated model selection on the same 386-row development population and determine what actually limits generalization before adding external information.

### M18A — Prediction Contract & Frozen Baseline — DONE

**Status:** `PREDICTION_CONTRACT_FROZEN` (2026-09-23).

Contract:
- primary task remains the company-provided `Tracks.eta` reference;
- decision time is `Tracks.last_update`;
- target is `target_tte_h = parsed Tracks.eta - Tracks.last_update`;
- AIS year inference remains the frozen M10 nearest-year rule;
- history is causal only when `Positions.recorded_at <= Tracks.last_update`;
- the supplied target is not silently reinterpreted as ATA, Pilot Boarding Place, port entry, berth or all-fast;
- target/status/error derivatives and post-decision information are forbidden as prediction-time inputs;
- all 53 old-final MMSIs remain permanently blocked from post-final selection;
- a future fresh holdout is an evaluation lockbox and cannot be used for feature engineering, threshold tuning or model selection.

Frozen baseline:
- M16G remains point champion: **185.883 h MAE / 19.342 h MedAE / 290.680 h P90**, N=386 nested OOF;
- M16H remains the uncertainty sidecar around the unchanged M16G P50;
- M14/M10 remains the official company submission;
- no model or prediction artifact is changed by M18A.

Artifacts: `docs/M18A_PREDICTION_CONTRACT.md`, `reports/M18A_PREDICTION_CONTRACT.json`, `reports/M18A_BASELINE_FREEZE.json`, `reports/M18A_REPORT.md`.

Next: **M18B — Validation Redesign**, followed by target reliability and residual attribution before any weather/routing/port-operations integration.
### M18B — Validation Redesign — DONE

**Status:** `VALIDATION_PROTOCOL_FROZEN` (2026-09-23).

M18B changes evaluation governance only. It keeps the M18A prediction contract, M16G point champion and M16H uncertainty sidecar unchanged. The 386 development MMSIs are assigned to five contiguous near-equal-count blocks by `Tracks.last_update` (78/77/77/77/77), without splitting identical timestamps. Three expanding forward folds test blocks 2, 3 and 4; a six-hour embargo before each test start matches the longest causal AIS-history window used by the route representation. The strict forward test union contains 231 unique MMSIs, each tested exactly once, with zero train/test owner overlap.

Important: the existing M16G OOF ledger was produced under the legacy balanced hash folds. Temporal slicing of that ledger is retained only as a chronology diagnostic and is **not** called forward-temporal generalization. Any future M18 candidate must refit all train-derived preprocessing, aggregates, experts, meta-learning and calibration inside each M18B TRAIN partition.

Cross-port generalization remains withheld: destination text is only a diagnostic proxy under the M18A company-reference ETA target and does not establish an authoritative arrival port/event. The 53 old-final MMSIs remain blocked and the future external lockbox is still unopened. Its promotion gate is predeclared before labels: positive MAE gain with paired owner-level 95% bootstrap lower bound >0 h, P90 ratio <=1.02, and no >10% regression in predeclared critical slices with n>=20.

Next: **M18C — Target & Label Reliability Audit**.

## M18C — Target & Label Reliability Audit (2026-09-23)

M18C is diagnostic-only and preserves the M18A/M18B contract. All 386 development labels remain in the strict benchmark and the 53 old-final MMSIs remain blocked. The audit shows that the 279 `FUTURE_0_7D` references have 20.26 h frozen M16G OOF MAE and contribute only 7.9% of total absolute error, while the remaining 107 rows contribute 92.1%. This is evidence of strong target-regime heterogeneity, not permission to filter the benchmark. Historical Catania/Augusta AIS research-gate events are retained only as non-authoritative post-hoc alignment evidence; official ATA/PBP/berth/all-fast ground truth is still unavailable. See `reports/M18C_REPORT.md`.


## M18D — Residual Error Attribution

1. Load the frozen M18C 386-row forensics ledger and frozen M16G/M16H outputs.
2. Attach only previously frozen decision-time diagnostics from M16B/M16D/M16E and the four M16G expert predictions.
3. Keep two separate scopes: strict 386-row benchmark and secondary 279-row `FUTURE_0_7D` diagnostic slice.
4. Quantify categorical slice error, numeric rank associations, quantile slices, expert disagreement, M16H confidence separation and hindsight existing-expert oracle regret.
5. Never use target-derived reliability status, residuals, coverage or oracle expert identity as runtime features.
6. Freeze a Missing Information Matrix; association is not causal proof and absent variables such as weather/port operations are marked untestable.
7. Preserve M18A target semantics, M18B validation, M18C labels, the 53-owner old-final block and the unopened fresh external holdout.
8. M18E may test one information family at a time; the first controlled data hypothesis is destination/port intent + navigable routing, while authoritative operational ground truth is pursued separately before any target change.

## M18E — External Information Ablation Lab

1. Keep M16G/M16H and M18A-D frozen.
2. Admit an external source only if its exact artifact is locally pinned, checksummed, licensed/provenanced and available at or before `decision_time`.
3. Change one information family at a time. Blocked sources are unmeasured, never encoded as zero-gain experiments.
4. Reproduce historical component ablations only as diagnostics; they are not new M18E promotions.
5. Keep target-derived `FUTURE_0_7D` slices diagnostic-only.
6. Do not open the fresh holdout and never use the 53 old-final rows for selection.
7. Run `python scripts/106_m18e_build_external_ablation_lab.py`, then `python scripts/106b_m18e_verify.py`, then the complete pytest/compile/verifier regression.

## M18F — Uncertainty & Selective ETA

1. Keep frozen M16G P50 and frozen M16H risk/interval outputs unchanged.
2. Reconstruct M16H risk inside each M16A outer fold; derive each selective threshold only from outer-train cross-fitted risk scores.
3. Freeze target service levels at 100%, 90%, 80%, 70%, 60% and 50%. Validation targets must never choose whether a row is retained.
4. Use `BALANCED_80` as the advisory operating profile: `AUTO_ETA_WITH_UNCERTAINTY` when accepted, otherwise `DEFER_LOW_CONFIDENCE`.
5. Report strict risk–coverage metrics and only then inspect target-derived near-term slices diagnostically.
6. Report M16H interval coverage on retained rows as empirical audit only; do not claim selection-conditional conformal validity.
7. Freeze the pre-holdout 80th-percentile development risk threshold before any fresh lockbox opening; never retune it on holdout labels.
8. Preserve all M18A-E freezes, the 53-owner old-final block and the unopened fresh holdout.
9. Run `python scripts/107_m18f_build_uncertainty_selective_eta.py`, `python scripts/107b_m18f_verify.py`, focused tests, then the complete pytest/compile/verifier regression.

## M18G — Frozen External Promotion Gate — DONE

1. Keep M18A-F, M16G/M16H, the 53 old-final rows and the fresh holdout frozen.
2. Freeze all external promotion criteria **before** any holdout labels are revealed.
3. For point-model promotion, require positive MAE gain, paired owner-row bootstrap CI95 lower bound > 0 h (10,000 reps; seed 18,023), candidate P90 / baseline P90 <= 1.02, and no >10% MAE regression in any predeclared prediction-time critical slice with n >= 20.
4. Freeze critical slices to vessel seen/unseen, M16H confidence tier, physics eligibility and destination-resolution status. Target-derived horizon/reference-status slices are diagnostic only and cannot determine promotion.
5. For M18F `BALANCED_80`, keep the pre-holdout M16H risk threshold at 61.116040125 h; require 70–90% realized coverage, retained MAE below full M16G MAE, deferred MAE above retained MAE, positive risk/error rank association, and >=78% empirical retained interval coverage. No selection-conditional conformal guarantee is claimed.
6. Before labels: freeze holdout provenance/schema, register scorer/candidate hashes, create a target-free prediction ledger, reject any exact overlap with the 439 original supervised sample keys, and SHA-256 seal the prediction ledger/opening manifest.
7. Only then may labels be revealed and `scripts/109_m18g_evaluate_external_holdout.py` run once. The same holdout cannot be used for retuning after a failure.
8. Current opening status is **BLOCKED** because no fresh holdout was supplied, no point candidate was promoted in M18E, and no full-development M16G/M16H fresh-row scoring bundle is registered yet. This is a readiness blocker, not an external performance result.

## M18G — Full-development blind scorer registration

1. Keep the historical M18G gate and all M16/M17/M18A-F freezes immutable.
2. Aggregate each operational hyperparameter/model-family choice only from already-frozen outer-fold selections using deterministic plurality voting; do not use fresh-holdout labels.
3. Refit M16C/M16D/M16E/M16F, M16G meta gating and M16H risk on all 386 development owners. Train M16G meta on official cross-fitted base-expert predictions and train M16H risk on official nested-OOF M16G errors.
4. Freeze the serialized scorer, input schema, source-freeze hashes and label-free smoke ledger.
5. On receipt of a fresh **unlabelled** cohort, first seal provenance, then run `scripts/111_m18g1_score_blind.py`; the resulting ledger and opening manifest must be SHA-256 sealed before label reveal.
6. Only after those hashes are fixed may the one-shot `scripts/109_m18g_evaluate_external_holdout.py` evaluation be run. Historical M16G 185.882697 h OOF MAE is not replaced by the operational refit.

## M19 — Fresh External Holdout (MMDEC v1)

Frozen order: source bytes + published MD5 verification -> target-free cohort/provenance -> blind M18G1 prediction ledger + SHA-256 -> post-seal ETA label reveal -> one-shot M18G evaluation. Development rows, the 53 old-final owners, and synthetic fixtures are forbidden substitutes for the real external cohort. The public source supports procedural/software blinding only because AIS Message-5 ETA fields are co-located with voyage fields. No retuning is allowed after MMDEC label reveal.


### M19 external opening result — frozen

The real MMDEC v1 holdout has been opened once. The sealed 500-row prediction ledger predates label reveal. The operational M16G baseline records 550.409 h full MAE; M18F BALANCED_80 fails its frozen external gate. **Do not retune, reslice for promotion, or re-open MMDEC.** Post-hoc label-regime diagnostics are descriptive only. Any revised system requires a new untouched external cohort.
