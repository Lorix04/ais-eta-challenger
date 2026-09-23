# Take-home report — Company Reference-ETA Benchmark

## 1. Exercise definition

The company clarified that **`Tracks.eta` should be used as the ground-truth/reference value for the technical exercise**. Accordingly, this report treats the latest ETA contained in `Tracks` as the supervised reference target.

This is intentionally called **`reference_eta`**, not observed Actual Time of Arrival (ATA). AIS Message 5 defines ETA as voyage-related estimated information; the company instruction is followed exactly without silently changing the target into a reconstructed physical arrival.

The earlier physical port-entry research (M0–M9) is preserved as a separate secondary branch and is summarized later in this report.

## 2. Dataset formulation

The supplied data contains two relevant roles:

- `Positions`: historical AIS observations over time for each vessel;
- `Tracks`: latest available vessel state, including the exercise ETA reference.

The primary M10 dataset contains **439 parseable ship ETA references across 439 MMSIs**. The modelling unit is one MMSI / Tracks record, not each minute-level historical row.

For vessel `i`, the target is:

`target_tte_h = parsed(Tracks.eta_i) - Tracks.last_update_i`

Historical features are constructed only from `Positions` observations satisfying:

`recorded_at <= Tracks.last_update`

The ETA itself is never a feature.

## 3. Reference-ETA parsing and audit

AIS Message-5 ETA encodes month/day/hour/minute but not a year. The implementation therefore resolves the nearest valid previous/current/next year relative to `Tracks.last_update` using a deterministic rule fixed before final evaluation.

Across the 439 parseable references:

| Reference status | Rows |
|---|---:|
| Past >24 h | 51 |
| Past 1–24 h | 10 |
| Past <1 h | 3 |
| Future 0–7 d | 319 |
| Future 7–14 d | 31 |
| Future 14–30 d | 17 |
| Future >30 d | 8 |

The strict company benchmark deliberately retains all parseable references. No stale-looking, past or far-future values are deleted after looking at performance.

## 4. Features

The candidate feature sets were built from information available at the latest snapshot time.

### Current snapshot

- latitude / longitude;
- SOG / COG / heading;
- draught;
- navigation status;
- vessel type / flag;
- destination string normalized to `destination_norm`;
- message count / feed context;
- hour-of-day and day-of-week encodings.

### Optional causal history

Historical `Positions` was summarized using only observations at or before `Tracks.last_update`, including recent movement and state-history aggregates. The latest matching Position has median age **0.55 min** relative to `Tracks.last_update`; P95 is **174.7 min**.

## 5. Validation protocol

The 439 MMSIs are split using a deterministic SHA-256 hash of MMSI only:

- train: **321**;
- calibration: **65**;
- final test: **53**.

The split is target-free. Candidate models and the selection rule were frozen before opening the final test.

The fixed candidate suite was:

1. global train median;
2. destination-specific train median with global fallback;
3. Ridge using current + causal-history features;
4. CatBoost using current snapshot;
5. CatBoost using current snapshot + causal history.

No broad hyperparameter search or post-final-test tuning was used.

## 6. Calibration model selection

### Strict all-parseable reference task

| Candidate | Calibration MAE |
|---|---:|
| **CatBoost current snapshot** | **339.8 h** |
| destination median + fallback | 343.4 h |
| CatBoost current + history | 346.2 h |
| global train median | 353.4 h |
| Ridge current + history | 448.9 h |

The strict model selected before final opening was **CatBoost current snapshot**.

### Future-reference diagnostic

For a separate diagnostic that excludes references already in the past, the calibration winner was the simple **destination median + global fallback** (191.5 h MAE), ahead of current CatBoost (200.7 h).

This diagnostic did not change the strict primary model.

## 7. Final strict result

The strict benchmark includes every parseable company reference ETA.

On the locked 53-row final test, the frozen current-snapshot CatBoost obtained:

- **MAE: 312.0 h**;
- **median absolute error: 25.3 h**;
- **P90 absolute error: 560.4 h**;
- **43.4% within ±24 h**;
- **60.4% within ±48 h**.

The difference between mean and median is substantial and is investigated explicitly rather than hidden.

## 8. M11 — reference quality and error forensics

M11 performs no training or model selection. It explains the strict benchmark distribution while retaining every label.

Key findings:

- **64/439 (14.6%)** references are already in the past at `last_update`;
- **56/439** are more than 7 days in the future;
- **19/439** have absolute horizon greater than 90 days;
- **86.1%** of ETA values are exactly on `:00` and **93.8%** on `:00` or `:30`;
- five rows have less than 90 days between the chosen and second-best year interpretation;
- seven rows have `01/01 00:00` or `01/01 01:01` placeholder-like patterns, but remain in the strict benchmark;
- on the strict final test, the top 5 errors account for **79.1%** of total absolute error and the top 10 for **91.4%**.

For context, the **same frozen strict CatBoost** has **28.6 h MAE** and **18.1 h median absolute error** on the 40 final references that fall 0–7 days in the future. This is post-hoc diagnostic evidence only; it does not replace the strict benchmark.

The intended interpretation is not “the company target is invalid”. It is that the latest reported/reference ETA has observable quality variation that materially affects aggregate error statistics.

## 9. M12 — frozen model explainability

M12 reloads the exact frozen CatBoost artifact and reproduces all **53/53** final predictions. It does not refit, prune or reselect the model.

The strongest single feature under both CatBoost native importance and calibration mean absolute SHAP is **`destination_norm`**. Current latitude, SOG, navigation status, COG and vessel type are also important.

Grouped SHAP on calibration shows the dominant information families are:

| Feature group | Mean absolute grouped SHAP |
|---|---:|
| motion | 13.41 h |
| voyage state | 12.39 h |
| location | 11.97 h |
| vessel context | 6.32 h |
| time context | 4.21 h |
| feed context | 1.20 h |

A no-retraining masking sensitivity check on calibration also shows the largest prediction shifts when voyage-state/destination, location and motion information are collapsed to train-typical values.

Importantly, the history-heavy CatBoost had already been tested before final opening and did not improve strict calibration: **346.2 h** versus **339.8 h** for current-snapshot CatBoost. On the future-reference diagnostic, history was essentially tied with current CatBoost while the destination median was better.

The supported statement is therefore narrow: **for the supplied latest `Tracks.eta` reference, causal historical Positions summaries did not add stable calibration value beyond the latest-state information already available.** This is not a general claim that vessel history is useless for physical ETA prediction.

## 10. Why extreme references dominate the strict MAE

The frozen CatBoost predicts within a range of roughly **−109.5 h to +93.8 h** on the final set. The supplied strict reference targets span roughly **−3055 h to +3553 h**.

A few examples illustrate the tail:

| Vessel | Reference TTE | Frozen prediction | Absolute error |
|---|---:|---:|---:|
| MIDEA | +3553.3 h | +34.8 h | 3518.4 h |
| DURO | −3055.0 h | +8.2 h | 3063.2 h |
| GIOVANNI E TOMMASO | −2514.9 h | +6.4 h | 2521.2 h |

The model's SHAP contributions remain on the scale of tens of hours; it is not learning a multi-month stale-reference regime. These rows remain in the strict benchmark as supplied.

## 11. Secondary branch — physical port-entry ETA research

Before the company's final reference-target clarification, the project investigated a separate maritime question: time to an AIS-derived physical port-entry event for Catania and Augusta.

That branch includes:

- AIS/provider and sentinel-value forensics;
- semantic motion-state construction;
- Catania/Augusta port-call reconstruction and manual audit;
- train-only historical route retrieval;
- grouped/temporal validation;
- an explicit stop condition that rejected residual ML when it worsened validation;
- reliability, uncertainty and ETA-stability diagnostics.

The retained physical predictor was train-only route-kNN remaining distance divided by robust trailing speed. On its own locked chronological holdout, **route-kNN MAE was 36.5 min versus 37.7 min geodesic**, and the paired bootstrap interval for the gain included zero.

This branch is useful domain analysis and supports alternative use cases, but it is **not** substituted for the company-defined M10 reference benchmark.

## 12. What is supported

- exact implementation of the company instruction to use `Tracks.eta` as reference;
- one supervised sample per MMSI/reference record;
- causal use of `Positions` history;
- calibration-only selection and one-time final evaluation;
- explicit label-quality/error forensics without post-hoc deletion;
- frozen-model attribution and pre-final architecture ablation;
- separate physical-arrival research with its own frozen evaluation.

## 13. What is not established

- `Tracks.eta` as observed ATA;
- superiority over the externally developed ETA model, whose inputs/protocol are unknown;
- unseen-port generalization;
- berth/all-fast prediction;
- safety-critical navigational suitability;
- the secondary research gate is not an **official PBP** (Pilot Boarding Place) or official port boundary;
- a causal interpretation of SHAP/feature importance;
- that history is generally useless for ETA prediction.

## 14. Reproducibility and delivery

The company-facing archive excludes the raw AIS files and high-volume row-level derived data. With the three supplied CSVs placed under `data/`, the project documents the replay sequence in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

The M10 final holdout has already been opened and is protected from accidental rescoring. M11 and M12 are diagnostic-only milestones and preserve the frozen model/split.

M14 performs the **final submission freeze** without retraining or rescoring. The final ZIP is built with sorted inputs and fixed archive metadata, contains a SHA-256 manifest for every payload file, and includes a stdlib-only integrity verifier. The archive is then extracted into a fresh directory where integrity verification, analytical tests and static compilation are rerun. The final ZIP itself is identified by an external SHA-256 manifest.

## 15. Alternative use cases identified

The same foundation can support:

- port-call / turnaround intelligence;
- anchorage and congestion proxies;
- AIS data-quality and stale-state scoring;
- route/behaviour anomaly detection;
- destination resolution and confidence;
- ETA discrepancy/reliability once historical **reported AIS ETA** becomes available at aligned timestamps.

## References

Primary domain/model references are listed in [`REFERENCES.md`](REFERENCES.md).

## M16D — Historical Route-Analogue Expert

Post-freeze research branch M16D is complete. On the frozen 386-row development population, a nested OOF DTW route-analogue expert reaches **194.98 h MAE** and **292.90 h P90 AE**, improving on the direct CatBoost current+history-aggregate OOF baseline (208.36 h MAE) and on the M16A raw destination median (202.97 h MAE). The old 53-row M10 final set was not used for selection or scoring. Because the P95 tail remains heavy, the route expert is retained for M16G rather than presented as a replacement for the frozen M14 submission. See `reports/M16D_REPORT.md`.

## M16E — Maritime / Physics Expert

Post-freeze M16E is complete. A leakage-safe physics expert uses catalog-resolved destination coordinates, great-circle remaining distance, causal recent median SOG and an optional local-sinuosity route-detour proxy. It is applicable to **115/386 development rows (29.8%)**. On those eligible rows it reaches **161.48 h MAE, 2.64 h MedAE and 93.33 h P90 AE**, versus **168.39 h, 8.82 h and 108.96 h** for the M16D route analogue; paired absolute error is lower for physics on **69.6%** of eligible rows. A fixed target-free hard gate (physics when eligible, M16D otherwise) reaches **192.93 h pooled OOF MAE**, a **2.06 h** improvement over M16D with **3/5** outer-fold wins. The 53 old-final MMSIs remain hard-blocked. Great-circle distance is explicitly treated as a lower bound/proxy, not a navigational route distance; no water-mask/chart routing dataset is claimed. See `reports/M16E_REPORT.md`.
## M16F — Tabular Challenger Panel

Post-freeze M16F is complete. Five tabular families share the same target-free representation built from current AIS state, causal compact history, M16B canonical-destination metadata and M16E geometry/data-quality features; no M16C/D/E expert prediction and no target-derived `reference_eta_status` is used as a feature. Ten configurations are predeclared and each family is selected only through 3-fold inner CV inside the frozen M16A outer folds. HistGradientBoosting is the best tabular family at **197.28 h pooled OOF MAE, 27.16 h MedAE and 292.84 h P90 AE**, versus **206.14 h** for the frozen M16A current-snapshot CatBoost. It wins **5/5 outer folds** against that baseline. The M16F CatBoost on the same enhanced representation reaches **197.87 h MAE**, so most of the improvement appears to come from representation/robustness rather than a dramatic family-specific advantage. M16D route analogue remains the best pooled single expert at **194.98 h MAE**; M16F HistGradientBoosting nevertheless wins 3/5 folds against it and is retained as the tabular component for M16G. The 53 old-final MMSIs remain hard-blocked. See `reports/M16F_REPORT.md`.

## M16G — Mixture of Experts / OOF Stacking

Post-freeze M16G is complete and passes its development-only stability gate. Four frozen expert families are combined: M16C hierarchical destination prior, M16D route analogue, M16E physics-gate/route fallback and M16F HistGradientBoosting. For every outer fold, expert predictions used to train the meta learner are regenerated as inner-OOF predictions inside that outer-train; no in-sample base prediction and no old-final MMSI is used. The selected nested mixture reaches **185.88 h OOF MAE, 19.34 h MedAE and 290.68 h P90 AE**, versus **192.93 h MAE** for the strongest frozen single operational expert, a **7.04 h gain** with **4/5 fold wins**. Exact ties are common because conservative hard gates often preserve the baseline expert; among rows where M16G actually switches prediction, the switch improves absolute error **54.6%** of the time. This is development-only challenger evidence and does not reopen the M14 submission. See `reports/M16G_REPORT.md`.


## M16H — Probabilistic ETA + Confidence (experimental post-freeze challenger)
M16H leaves the M16G nested-OOF point estimate unchanged and calibrates a central-80% development-only interval from OOF residuals. Row-level interval scale is driven by a cross-fitted absolute-error model using prediction-time-only expert disagreement/support features; the scale-floor candidate is chosen inside each outer-train. The resulting pooled empirical coverage is 80.05%, with 159.42 h mean width versus 221.26 h for global residual calibration. HIGH/MEDIUM/LOW confidence is derived from predicted error terciles and orders typical absolute error, but the LOW regime and stale/far-future reference-ETA groups remain strongly heavy-tailed. This uncertainty analysis does not reopen or modify the M14 company submission.


## M16I — Red-team / ablation conclusion
M16I deliberately tries to falsify the development-only M16G/M16H result rather than improve it. The 53 already-observed M10 final MMSIs remain hard-blocked. Structural counterfactuals remove physics, route, canonical-destination and longitudinal-history paths without post-red-team retuning; subgroup audits cover cold/rare destinations, `FOR_ORDERS`/unknown destinations, vessel categories, route support, confidence tiers and target-derived reference-ETA status diagnostics. The pooled M16G gain versus M16E is 7.04 h and survives leave-one-fold-out, target-magnitude trimming, winsorisation and largest-destination removal. But the 95% paired-bootstrap interval crosses zero and gain concentration remains high. The frozen conclusion is `PASS_RED_TEAM_WITH_HEAVY_TAIL_CAVEAT`: the architecture is worth carrying forward, while a fresh untouched company holdout is required before claiming superior generalisation.

## Post-submission challenger (M16, development-only)

After freezing the official M14 submission, a separate M16 research branch tested whether domain decomposition could improve the development benchmark without reusing the already-observed 53-row M10 final set. The nested OOF mixture reaches 185.88 h MAE versus 192.93 h for the best single operational expert. Robustness checks are encouraging but not definitive: the 95% paired-bootstrap gain interval crosses zero and improvement is partially concentrated in the heavy tail. M16 is therefore a frozen challenger for a newly issued holdout, not a replacement for the official M14 result.


## Post-freeze M17A learned retrieval benchmark

M17A is a separate development-only research branch after the M16J freeze. A compact target-free self-supervised TCN encoder was trained separately inside each frozen outer fold on causal AIS windows from outer-train MMSIs only. With the M16D destination gate, K=9 and weighted-median ETA aggregation held fixed, learned 32-D cosine retrieval reaches **190.11 h OOF MAE / 19.04 h MedAE / 276.64 h P90**, versus **192.57 h** for the same-config DTW control and **194.98 h** for frozen M16D. It wins 4/5 folds against same-config DTW and 5/5 against frozen M16D. The result is promising representation evidence only; M14 remains the official submission and M16J remains the frozen challenger pending a fresh untouched company holdout.


### M17B post-freeze research update

M17B expanded M17A into 3,334 recent causal historical memory snapshots and tested exact cosine versus a deterministic HNSW-style ANN backend. The memory itself **did not improve ETA accuracy** (198.50 h MAE vs M17A 190.11 h), so it is not promoted. The ANN layer did reproduce exact retrieval closely (97.55% owner recall@K; 0.051 h MAE delta), so HNSW is retained only as a scalability mechanism. The result strengthens the next hypothesis: route-phase/graph structure matters more than simply adding overlapping historical snapshots.

## Post-freeze M17D research note
M17D tested a six-expert nested mixture by adding M17A self-supervised learned retrieval and M17C historical corridor-graph gating to frozen M16G. Meta-selection used only inner-OOF expert predictions and did not use the 53 old-final rows. The expanded mixture produced 196.20 h development OOF MAE versus 185.88 h for M16G, so the predeclared promotion gate failed. This is a useful negative result: added expert diversity does not automatically improve a small-label mixture, and unrestricted gating can increase variance. The official M14 submission and frozen M16J/M16G challenger remain unchanged.

## M17E — Constrained M16G ↔ M17A Selective Router — DONE

**Status:** `NO_M17E_PROMOTION`.

M17E deliberately reduces routing complexity after the negative M17D six-expert experiment. The router has only two actions: keep the frozen M16G prediction or switch to frozen M17A learned retrieval. M16G is reconstructed inside each outer-train from inner-OOF base experts and verified against its frozen outer prediction; M17A is regenerated inner-OOF with causal outer/inner-validation owner exclusion. Runtime router features are target-free.

Predeclared gate: >=1.0 h MAE gain vs M16G, >=3/5 fold wins, >=52% wins on changed rows, P90 no worse than M16G, <=35% switch share, and <=15 h worst-fold regression.

Result: M16G **185.88 h** -> M17E **185.32 h** MAE (gain **0.57 h**), 3/5 fold wins, 55.6% changed-row wins, 2.3% switched rows and P90 ratio 0.963. The router improves all observed switched-fold comparisons without a fold regression, but the pooled gain is below 1 h, therefore it is **not promoted** and M16G remains the frozen champion.

Next preferred branch: **M17F — latent target-noise / declared-ETA regime modelling**; alternative M17G — stronger SSL trajectory representation.



## M17F update
M17F tested a leakage-safe latent target-noise / declared-ETA regime model on the 386 development MMSIs. All five outer folds selected no correction; M16G remains the best promoted development model at 185.88 h MAE. The experiment is retained as negative evidence that the observed heavy-tail residual regime is not stably predictable from the currently available runtime features.

## Post-freeze M17G research note
M17G is not part of the official M14 submission. It tests stronger self-supervised route representations on the 386-row development population only. With the M17A retrieval contract held fixed, MoCo-TCN64 reaches 188.29 h OOF MAE versus 190.11 h for frozen M17A and passes the predeclared component gate; masked-Transformer64 does not. M16G remains the best overall promoted development-only system at 185.88 h MAE.

## Post-freeze M17H research note
M17H tested the most conservative possible integration of the passing M17G MoCo retrieval component: the M16G panel remained at four experts, M16D route/fallback was replaced one-for-one by M17G MoCo, and the outer-fold meta strategy was frozen to the one selected by M16G rather than re-selected. The result is negative: 192.18 h development OOF MAE versus 185.88 h for frozen M16G, with 2/5 fold wins. This shows that the standalone M17G gain does not transfer through a simple frozen-architecture substitution. M16G remains the promoted development-only challenger; all M17 work remains post-freeze research and the 53 old-final rows remain untouched.

---

## Final research addendum — M19 fresh external holdout

After freezing the development pipeline, M19 evaluated the operational M16G+M16H scorer on a blind, target-free cohort drawn from **MMDEC v1**. The prediction ledger was hash-sealed before label reveal and then evaluated exactly once through the frozen M18G gate.

On 500 unique external MMSIs, full-population MAE is **550.41 h**. The frozen M18F `BALANCED_80` policy retains 76/500 rows (15.2%) at **342.55 h MAE**, so the selective layer does not validate externally. A post-hoc `FUTURE_0_7D` diagnostic contains 197 rows at **38.36 h MAE**. The external domain also shows severe degradation in destination resolution and physics eligibility, explaining why the development-domain routing/physics machinery transfers poorly zero-shot.

No model or threshold is retuned on MMDEC after this result. The dataset is now treated as a spent holdout. Any future cross-domain work must be developed without using MMDEC as a final test and evaluated on another independent holdout.
