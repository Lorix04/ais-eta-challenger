# Executive summary — Company Reference-ETA Benchmark

## Task

The company clarified that, for the exercise, **the ETA contained in `Tracks` should be used as the reference/ground-truth value**. The primary task is therefore to predict that latest reference ETA from the current vessel state and causal historical information in `Positions`.

The project uses one supervised example per MMSI. `Tracks.eta` is target-only; historical features use only `Positions.recorded_at <= Tracks.last_update`.

## Dataset and validation

There are **439 parseable ship ETA references across 439 MMSIs**. The deterministic target-free split is **321 train / 65 calibration / 53 final test**. Candidate models were selected on calibration only, then the final test was opened once.

## Final strict benchmark

Calibration selected **CatBoost on the current/latest Tracks snapshot**.

| Metric | Final test |
|---|---:|
| Rows | 53 |
| MAE | **312.0 h** |
| Median absolute error | **25.3 h** |
| P90 absolute error | 560.4 h |
| Within ±24 h | 43.4% |
| Within ±48 h | 60.4% |

The large mean/median gap is a property of the supplied reference distribution, not hidden by post-hoc filtering.

## Why the mean is so large

M11 keeps all strict labels and explains the heavy tail:

- 64/439 reference ETAs are already in the past at snapshot time;
- 56/439 are more than 7 days in the future;
- the top 5 final-test errors explain **79.1%** of total absolute error;
- the same frozen model has **28.6 h MAE** on the 40 final references 0–7 days in the future.

That 0–7 day number is diagnostic only; the official strict result remains 312.0 h MAE.

## What the frozen model learned

M12 reproduces all 53 final predictions from the frozen CatBoost artifact and uses native importance/SHAP only for explanation. `destination_norm` is the strongest single feature; voyage state, current motion and current location dominate the grouped attribution.

A CatBoost with additional causal history had already been tested before final opening and did **not** improve calibration. For the future-reference diagnostic, a simple train-only destination median was the calibration winner. The evidence therefore does not support “more history / more ML is automatically better” for this particular latest-state reference target.

## Secondary domain work

Before the final target clarification, M0–M9 studied a different question: **physical port-entry ETA** reconstructed from AIS for Catania and Augusta. That branch remains frozen as additional maritime-domain work. It includes provider/data-quality forensics, port-call reconstruction, train-only historical routing, leakage-resistant validation, uncertainty and stability analysis.

It is not used to overwrite or reinterpret the company-defined M10 benchmark.

## Bottom line

The submission provides:

- an exact implementation of the company-defined `Tracks.eta` benchmark;
- a frozen train/calibration/final evaluation protocol;
- explicit diagnosis of reference-label quality rather than silent filtering;
- model explainability without post-final tuning;
- a separate domain-aware physical-arrival investigation showing what changes when the target itself changes;
- an M14 deterministic final submission freeze with per-file SHA-256 integrity verification and clean-room package tests.

No claim is made that the submission beats the externally developed ETA model because its information set and evaluation protocol were not supplied.
Historical **reported AIS ETA** updates are not available in `Positions`; the exercise therefore predicts the latest `Tracks.eta` reference rather than evaluating a longitudinal reported-ETA series.


## Final reproducibility status

M14 is packaging/integrity only. It re-verifies the frozen M10/M6 hashes, builds the final archive deterministically, and validates the extracted package with its content-manifest verifier, analytical tests and static compilation. No model, split or final prediction is changed.

## M16D — Historical Route-Analogue Expert

Post-freeze research branch M16D is complete. On the frozen 386-row development population, a nested OOF DTW route-analogue expert reaches **194.98 h MAE** and **292.90 h P90 AE**, improving on the direct CatBoost current+history-aggregate OOF baseline (208.36 h MAE) and on the M16A raw destination median (202.97 h MAE). The old 53-row M10 final set was not used for selection or scoring. Because the P95 tail remains heavy, the route expert is retained for M16G rather than presented as a replacement for the frozen M14 submission. See `reports/M16D_REPORT.md`.

## M16E — Maritime / Physics Expert

Post-freeze M16E is complete. A leakage-safe physics expert uses catalog-resolved destination coordinates, great-circle remaining distance, causal recent median SOG and an optional local-sinuosity route-detour proxy. It is applicable to **115/386 development rows (29.8%)**. On those eligible rows it reaches **161.48 h MAE, 2.64 h MedAE and 93.33 h P90 AE**, versus **168.39 h, 8.82 h and 108.96 h** for the M16D route analogue; paired absolute error is lower for physics on **69.6%** of eligible rows. A fixed target-free hard gate (physics when eligible, M16D otherwise) reaches **192.93 h pooled OOF MAE**, a **2.06 h** improvement over M16D with **3/5** outer-fold wins. The 53 old-final MMSIs remain hard-blocked. Great-circle distance is explicitly treated as a lower bound/proxy, not a navigational route distance; no water-mask/chart routing dataset is claimed. See `reports/M16E_REPORT.md`.
## M16F — Tabular Challenger Panel

Post-freeze M16F is complete. Five tabular families share the same target-free representation built from current AIS state, causal compact history, M16B canonical-destination metadata and M16E geometry/data-quality features; no M16C/D/E expert prediction and no target-derived `reference_eta_status` is used as a feature. Ten configurations are predeclared and each family is selected only through 3-fold inner CV inside the frozen M16A outer folds. HistGradientBoosting is the best tabular family at **197.28 h pooled OOF MAE, 27.16 h MedAE and 292.84 h P90 AE**, versus **206.14 h** for the frozen M16A current-snapshot CatBoost. It wins **5/5 outer folds** against that baseline. The M16F CatBoost on the same enhanced representation reaches **197.87 h MAE**, so most of the improvement appears to come from representation/robustness rather than a dramatic family-specific advantage. M16D route analogue remains the best pooled single expert at **194.98 h MAE**; M16F HistGradientBoosting nevertheless wins 3/5 folds against it and is retained as the tabular component for M16G. The 53 old-final MMSIs remain hard-blocked. See `reports/M16F_REPORT.md`.

## M16G — Mixture of Experts / OOF Stacking

Post-freeze M16G is complete and passes its development-only stability gate. Four frozen expert families are combined: M16C hierarchical destination prior, M16D route analogue, M16E physics-gate/route fallback and M16F HistGradientBoosting. For every outer fold, expert predictions used to train the meta learner are regenerated as inner-OOF predictions inside that outer-train; no in-sample base prediction and no old-final MMSI is used. The selected nested mixture reaches **185.88 h OOF MAE, 19.34 h MedAE and 290.68 h P90 AE**, versus **192.93 h MAE** for the strongest frozen single operational expert, a **7.04 h gain** with **4/5 fold wins**. Exact ties are common because conservative hard gates often preserve the baseline expert; among rows where M16G actually switches prediction, the switch improves absolute error **54.6%** of the time. This is development-only challenger evidence and does not reopen the M14 submission. See `reports/M16G_REPORT.md`.


## M16H post-freeze challenger uncertainty
The M16G point predictor remains unchanged. A development-only cross-fitted error-risk + residual/conformal-style wrapper reaches 80.05% empirical coverage for a nominal central-80% interval while reducing mean width from 221.26 h (global residual baseline) to 159.42 h. Confidence tiers are useful for typical-error triage (MedAE 3.32 h HIGH, 19.17 h MEDIUM, 84.40 h LOW), but heavy reference-ETA pathologies remain and subgroup coverage is not guaranteed.


## M16I red-team conclusion
The post-freeze M16 challenger survives fair fold, target-magnitude trim, winsorised and destination-removal stresses: M16G remains 7.04 h better in pooled MAE than the best frozen single expert, all five leave-one-fold-out views retain positive gain, and M16H remains at 80.05% pooled coverage. The red-team also finds material heavy-tail sensitivity: the paired-bootstrap 95% interval crosses zero and five rows contribute 51.0% of total positive gain. Therefore M16I freezes `PASS_RED_TEAM_WITH_HEAVY_TAIL_CAVEAT`; no unseen-data superiority claim is made and the M14 submission remains official.

## Post-freeze M16 challenger

A separate leakage-controlled challenger branch was developed after the official M14 freeze. M16 combines destination semantics, route analogues, physics, robust tabular prediction, nested OOF mixture-of-experts and calibrated confidence. The official development-only MoE result is 185.88 h MAE, but M16I red-team analysis retains a heavy-tail caveat because the paired-bootstrap 95% gain interval crosses zero. Therefore M16 is frozen for **new untouched holdout validation only**; it does not replace M14.


## M17A post-freeze research result

M17A is a separate development-only research branch after the M16J freeze. A compact target-free self-supervised TCN encoder was trained separately inside each frozen outer fold on causal AIS windows from outer-train MMSIs only. With the M16D destination gate, K=9 and weighted-median ETA aggregation held fixed, learned 32-D cosine retrieval reaches **190.11 h OOF MAE / 19.04 h MedAE / 276.64 h P90**, versus **192.57 h** for the same-config DTW control and **194.98 h** for frozen M16D. It wins 4/5 folds against same-config DTW and 5/5 against frozen M16D. The result is promising representation evidence only; M14 remains the official submission and M16J remains the frozen challenger pending a fresh untouched company holdout.


### M17B research update

M17B expanded M17A into 3,334 recent causal historical memory snapshots and tested exact cosine versus a deterministic HNSW-style ANN backend. The memory itself **did not improve ETA accuracy** (198.50 h MAE vs M17A 190.11 h), so it is not promoted. The ANN layer did reproduce exact retrieval closely (97.55% owner recall@K; 0.051 h MAE delta), so HNSW is retained only as a scalability mechanism. The result strengthens the next hypothesis: route-phase/graph structure matters more than simply adding overlapping historical snapshots.

**M17C post-freeze research:** a target-free historical maritime corridor graph was evaluated against M16E physics. It supports 25/115 physics-eligible development cases and yields a small 0.223 h MAE improvement on those rows; a conservative graph gate improves the full development OOF MAE by only 0.014 h. The result is retained as a structural component, not a new final winner.

**M17D post-freeze integration:** the frozen M16G mixture was expanded with M17A learned-retrieval and M17C corridor-graph experts, with both new signals regenerated inner-OOF inside every outer-train. The extension did not generalize: **196.20 h MAE** vs **185.88 h** for frozen M16G, only **2/5** fold wins and **45.6%** changed-row win share. Decision: `NO_M17D_PROMOTION`; retain M16G as the best development-only mixture and preserve M17A/M17C as independent research components.

## M17E — Constrained M16G ↔ M17A Selective Router — DONE

**Status:** `NO_M17E_PROMOTION`.

M17E deliberately reduces routing complexity after the negative M17D six-expert experiment. The router has only two actions: keep the frozen M16G prediction or switch to frozen M17A learned retrieval. M16G is reconstructed inside each outer-train from inner-OOF base experts and verified against its frozen outer prediction; M17A is regenerated inner-OOF with causal outer/inner-validation owner exclusion. Runtime router features are target-free.

Predeclared gate: >=1.0 h MAE gain vs M16G, >=3/5 fold wins, >=52% wins on changed rows, P90 no worse than M16G, <=35% switch share, and <=15 h worst-fold regression.

Result: M16G **185.88 h** -> M17E **185.32 h** MAE (gain **0.57 h**), 3/5 fold wins, 55.6% changed-row wins, 2.3% switched rows and P90 ratio 0.963. The router improves all observed switched-fold comparisons without a fold regression, but the pooled gain is below 1 h, therefore it is **not promoted** and M16G remains the frozen champion.

Next preferred branch: **M17F — latent target-noise / declared-ETA regime modelling**; alternative M17G — stronger SSL trajectory representation.



## M17F update
M17F tested a leakage-safe latent target-noise / declared-ETA regime model on the 386 development MMSIs. All five outer folds selected no correction; M16G remains the best promoted development model at 185.88 h MAE. The experiment is retained as negative evidence that the observed heavy-tail residual regime is not stably predictable from the currently available runtime features.

### M17G post-freeze research update
A stronger target-free MoCo-TCN trajectory encoder improves the frozen M17A retrieval component from 190.11 h to 188.29 h development OOF MAE under the exact same K=9 destination-gated weighted-median ETA aggregation. The masked-Transformer alternative does not improve M17A. This is a component-level development result only; M16G remains the overall promoted development champion and M14 remains the official submission.

## M17H conservative MoCo integration
M17H keeps M16G at four experts and freezes the per-fold M16G meta strategy, replacing only the DTW route expert and physics route fallback with the passing M17G MoCo-TCN64 retrieval signal. The integration does not pass promotion: **192.18 h MAE vs 185.88 h for frozen M16G**, with only **2/5 fold wins**. The result is retained as negative evidence; M16G remains the best promoted development-only system and a fresh untouched holdout is now preferred over further recombination on the same 386 labels.

---

## Final external-validation addendum — M19

The post-freeze research programme culminated in a one-shot external-domain evaluation on **500 unique MMSIs from MMDEC v1**. The M18G1 prediction ledger was generated and SHA-256 sealed before external ETA labels were revealed. The frozen scorer records **550.41 h full-population MAE** on this shifted domain. The frozen M18F `BALANCED_80` selective layer retains **15.2%** of rows at **342.55 h MAE** and therefore does **not** pass its external gate. A pre-existing post-hoc semantic slice, `FUTURE_0_7D`, contains 197 rows and records **38.36 h MAE**; it remains diagnostic only.

MMDEC is now a **spent holdout** and is not used for retuning. The result establishes a real cross-domain generalization limitation rather than being hidden or optimized away. See `reports/M19_EXTERNAL_REPORT.md`.
