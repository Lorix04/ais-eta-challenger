# External references used in the project

Primary / authoritative sources are preferred for domain semantics.

1. **DCSA Port Call Standard** — standardized operational timestamps and port-call services.  
   https://dcsa.org/standards/port-call

2. **DCSA Port Call 2.0 Use Cases** — explicit ETA/RTA/PTA/ATA semantics for berth and Pilot Boarding Place, plus pilotage/towage/mooring events.  
   https://reference.dcsa.org/content/standards/releases/port-call/v2/pc-v2-use-cases

3. **IMO Just-In-Time Arrival Guide overview** — PBP arrival depends on berth, fairway and nautical-service availability.  
   https://www.imo.org/en/mediacentre/pages/whatsnew-1502.aspx

4. **USCG NAVCEN AIS Class A reports** — official reporting cadence and field/sentinel semantics based on ITU-R M.1371.  
   https://www.navcen.uscg.gov/ais-class-a-reports

5. **Evmides et al. (2024)**, *Enhancing Prediction Accuracy of Vessel Arrival Times Using Machine Learning* — AIS ETA work with route-grouped splitting and external PCS arrival truth.  
   https://www.mdpi.com/2077-1312/12/8/1362

6. **Barber & Pananjady (2026)**, *Predictive inference for time series: why is split conformal effective despite temporal dependence?* — limits/analysis of conformal prediction under temporal dependence.  
   https://proceedings.mlr.press/v313/barber26a.html

7. **AdSP Mare di Sicilia Orientale — Porto di Augusta** — official description of outer/inner roadstead and two entrances.  
   https://www.adspmaresiciliaorientale.it/porto-di-augusta/

8. **AdSP Mare di Sicilia Orientale — Porto di Catania** — official port/berth/fondale information used as domain context.  
   https://www.adspmaresiciliaorientale.it/porto-di-catania/

## M10 target semantics

- **USCG NAVCEN — AIS Class A Ship Static and Voyage Related Data (Message 5)**: ETA is transmitted as `MMDDHHMM UTC` and does not contain a year. https://www.navcen.uscg.gov/ais-class-a-static-voyage-message-5
- **Evmides et al. (2024), Journal of Marine Science and Engineering 12(8):1362**: recent AIS ETA benchmark using multiple ML models and route-aware splitting; useful as methodological context, not directly comparable to the company-reference target used in M10. https://www.mdpi.com/2077-1312/12/8/1362

## M11 reference-quality semantics

- **IMO Resolution A.1106(29)**, *Revised guidelines for the onboard operational use of shipborne AIS*: departure, destination and ETA are manually entered by the OOW at voyage start and when changes occur. https://www.navcen.uscg.gov/sites/default/files/pdf/ais/references/IMO_A1106_29_Revised_guidelines.pdf
- **USCG NAVCEN — AIS Message 5**: ETA is encoded as `MMDDHHMM UTC` without a year. https://www.navcen.uscg.gov/ais-class-a-static-voyage-message-5
- **A Reproducible Workflow for AIS-Based ETA Forecasting (2026)**: recent AIS ETA workflow documenting inconsistent manually entered destination fields and the need for structured preprocessing. https://www.mdpi.com/2571-9394/8/4/70

## M12 frozen-model explainability

- **CatBoost — Feature importances**: official definitions of `PredictionValuesChange`, `LossFunctionChange`, SHAP and interaction importance. https://catboost.ai/docs/en/features/feature-importances-calculation
- **CatBoost — ShapValues**: SHAP contribution semantics and additivity for CatBoost models. https://catboost.ai/docs/en/concepts/shap-values

## M16B — AIS destination semantics / UN/LOCODE

- IMO SN/Circ.244, *Guidance on the use of the UN/LOCODE in the Destination Field in AIS Messages* — explains that AIS destination is free text, documents multiple spellings of the same location, and recommends UN/LOCODE: https://www.navcen.uscg.gov/sites/default/files/pdf/NAIS/IMO_SN_Circ244_AIS_UNLOCODE.pdf
- UNECE / UN/CEFACT UN/LOCODE country and vocabulary tables used as provenance for the compact M16B observed-port catalog: https://service.unece.org/trade/locode/ and https://vocabulary.uncefact.org/unlocode
- USCG AIS Encoding Guide — voyage-related ETA/destination encoding guidance and `origin>destination` examples: https://www.navcen.uscg.gov/contact/ais_encoding_guide

## M16C — Fold-safe hierarchical priors / cross-fitting

- scikit-learn, *Target Encoder’s Internal Cross fitting* — shows why category target statistics for training rows must be learned out-of-fold to avoid leakage/overfitting: https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder_cross_val.html
- scikit-learn, `TargetEncoder` — documents shrinkage toward the global target mean, empirical-Bayes smoothing, and cross-fitting in `fit_transform`: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html
- Category Encoders, `TargetEncoder` — documents minimum leaf support, smoothing and optional hierarchy for continuous-target encodings: https://contrib.scikit-learn.org/category_encoders/targetencoder.html

## M16D route-analogue references

- Fan, T.; Liu, Q.; Jing, W.; He, Y. (2026). *Incorporation of movement pattern analysis into route searching for ship ETA prediction*. Ocean Engineering 358, 125455. https://doi.org/10.1016/j.oceaneng.2026.125455
- *Development and Application of an Advanced Automatic Identification System (AIS)-Based Ship Trajectory Extraction Framework for Maritime Traffic Analysis* (2024). Journal of Marine Science and Engineering 12(9), 1672. https://www.mdpi.com/2077-1312/12/9/1672

## M16E maritime / physics references

- IMO Resolution MSC.74(69), *Recommendation on Performance Standards for a Universal Shipborne Automatic Identification System (AIS)* — AIS dynamic information includes position, course over ground and speed over ground; destination and ETA are voyage-related data. https://wwwcdn.imo.org/localresources/en/OurWork/Safety/Documents/AIS/Resolution%20MSC.74%2869%29.pdf
- USCG NAVCEN, *AIS Encoding Guide* — voyage ETA/destination encoding guidance, including ETA in UTC and UN/LOCODE destination semantics. https://navcen.uscg.gov/sites/default/files/pdf/AIS/AISGuide.pdf
- Wang, Y.; Zhang, X.; Guo, Y. (2025), *Predicting estimated time of arrival for ships: A frequency-based approach considering met-ocean factors*, Ocean Engineering 337, 121873 — describes indirect ETA modelling as a combination of ship-speed prediction and geographical-distance calculation. https://doi.org/10.1016/j.oceaneng.2025.121873
- Port Digital Twin Development for Decarbonization (2023), Journal of Marine Science and Engineering 11(9), 1777 — describes ETA calculation from current vessel position and remaining route distance, with historical-route estimation. https://www.mdpi.com/2077-1312/11/9/1777

### M16F implementation references
- CatBoost regression objectives and metrics (MAE, Huber, Quantile): https://catboost.ai/docs/en/concepts/loss-functions-regression
- scikit-learn ensemble API (ExtraTrees, RandomForest, HistGradientBoosting): https://scikit-learn.org/stable/api/sklearn.ensemble.html
- scikit-learn SVR API: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html

### M16G stacking / meta-learning

- scikit-learn, `StackingRegressor`: the final estimator is trained from cross-validated base-estimator predictions. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html
- scikit-learn example, *Combine predictors using stacking*: out-of-fold base predictions are used to learn a meta-model; simple convex/SuperLearner-like combinations are discussed as an interpretable comparison. https://scikit-learn.org/stable/auto_examples/ensemble/plot_stack_predictors.html


### M16H uncertainty references
- CatBoost documentation, regression objectives including Quantile and MultiQuantile: https://catboost.ai/docs/en/concepts/loss-functions-regression
- Kuchibhotla, A. K. (2020), *Exchangeability, Conformal Prediction, and Rank Tests*, arXiv:2005.06095.
- Barber, R. F., Candès, E. J., Ramdas, A., Tibshirani, R. J. (2022), *Conformal prediction beyond exchangeability*, arXiv:2202.13415.


### M16I red-team / robustness references
- Oliveira, R. I.; Orenstein, P.; Ramos, T.; Romano, J. V. (2024), *Split Conformal Prediction and Non-Exchangeable Data*, JMLR 25(225):1–38 — discusses coverage under non-exchangeability and the limits of classical exchangeability assumptions: https://jmlr.org/papers/v25/23-1553.html
- Barber, R. F.; Candès, E. J.; Ramdas, A.; Tibshirani, R. J. (2022), *Conformal prediction beyond exchangeability* — motivates stress testing under distribution shift/non-exchangeability: https://arxiv.org/abs/2202.13415
- Zhou, Z.; Zhang, X.; Tao, C.; Yang, Y. (2026), *Conformal Prediction Assessment: A Framework for Conditional Coverage Evaluation and Selection* — motivates explicit subgroup/conditional coverage diagnostics rather than relying only on pooled marginal coverage: https://arxiv.org/abs/2603.27189

## M16J validation / reproducibility references

- scikit-learn, `StackingRegressor`: final estimators should be trained on cross-validated base-estimator predictions; using predictions from models fitted on the same rows creates high overfitting risk. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html
- Reproducible Builds, cryptographic checksums: byte-identical build outputs can be verified with cryptographic hashes. https://reproducible-builds.org/docs/checksums/
- NIST AI RMF / AIRC: test, evaluation, verification and validation (TEVV) should include model validation and assessment, with ongoing evaluation appropriate to deployment context. https://www.nist.gov/itl/ai-risk-management-framework and https://airc.nist.gov/


## M17A self-supervised / learned retrieval references

- Song, R. et al. (2026), *MoCo-AIS: A Contrastive Learning Framework for Similarity Computation of Vessel Trajectories* — self-supervised momentum-contrast vessel trajectory embeddings and scalable learned similarity retrieval. https://arxiv.org/abs/2606.17978
- Korupoju, A. K. (2026), *NaviSight* — public Apache-2.0 research prototype using masked-autoencoder maritime representation learning, 128-D embeddings and cohort-aware HNSW similarity search over large AIS corpora. https://github.com/Korunil/navisight
- Koo, W.; Chang, S.; Kim, H. (2026), *AIS-Based Vessel Trajectory Prediction Using Memory-Augmented Neural Networks* — external trajectory memory and retrieval improves AIS trajectory forecasting over non-memory baselines. https://arxiv.org/abs/2606.06311

M17A does not copy third-party source code; these works motivate the target-free representation/retrieval design.


## M17B ANN / memory references

- Malkov, Y. A., & Yashunin, D. A. — *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*. IEEE TPAMI. https://arxiv.org/abs/1603.09320
- nmslib/hnswlib — reference HNSW implementation and Python bindings. https://github.com/nmslib/hnswlib
- Koo, W., Chang, S., & Kim, H. — *AIS-Based Vessel Trajectory Prediction Using Memory-Augmented Neural Networks* (2026). https://arxiv.org/abs/2606.06311

## M17C graph / corridor research references
- Neofytos Dimitriou (2026), *Historical Knowledge Graphs for Global Maritime Estimated Time of Arrival*, arXiv:2605.18408. AIS-only directed historical graph with context-specific speed distributions and hierarchical fallback.
- Maohan Liang et al. (2024), *Estimation of vessel link-level travel time distribution: A directed network-driven approach*, Ocean Engineering 313, 119371. Directed AIS network and link-level travel-time distribution estimation.
- Seong-Won Choi et al. (2026), *A route planning method for small ships based on historical AIS data*, Ocean Engineering 343, 123372. Historical frequency/direction route graph with navigational constraints.

## M17D stacking / ensemble references
- scikit-learn, *StackingRegressor*: final estimators are trained on cross-validated base-estimator predictions; using in-sample/prefit predictions from the same training data carries high overfitting risk. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html
- scikit-learn, *Combine predictors using stacking*: stacking uses out-of-fold base predictions and can exploit partially uncorrelated model errors. https://scikit-learn.org/stable/auto_examples/ensemble/plot_stack_predictors.html
- Chu, Yan & Wang (2025), *Vessel arrival time to port prediction via a stacked ensemble approach: Fusing port call records and AIS data*, Transportation Research Part C 176, 105128. https://doi.org/10.1016/j.trc.2025.105128
- *High-accuracy prediction of vessels' estimated time of arrival in seaports: A hybrid machine learning approach* (2025), Maritime Transport Research 8, 100133. https://doi.org/10.1016/j.martra.2025.100133

## M17F target-noise / robust-regression research
- Jiang, G., Lei, F., Wang, W. (2026). *A multi-model dynamic filtering of label noise for regression*. Pattern Recognition 169, 111887. https://doi.org/10.1016/j.patcog.2025.111887
- *Robust soft sensor development based on Dirichlet process mixture of regression model for multimode processes* (2025). Chemometrics and Intelligent Laboratory Systems 267, 105550. https://doi.org/10.1016/j.chemolab.2025.105550
- Qiu, Z. et al. (2026). *Robust Student’s t mixture distribution-based Kalman filter for non-stationary heavy-tailed noise*. Journal of the Franklin Institute 363(8), 108667. https://doi.org/10.1016/j.jfranklin.2026.108667

## M17G stronger self-supervised trajectory representation

- Song R. et al. (2026), **MoCo-AIS: A Contrastive Learning Framework for Similarity Computation of Vessel Trajectories**, arXiv:2606.17978. https://arxiv.org/abs/2606.17978
- Korupoju A.K. (2026), **NaviSight**, self-supervised masked-autoencoder maritime representation learning and HNSW retrieval. https://github.com/Korunil/navisight
- Al-Falouji G. et al. (2026), **Representation Learning for Maritime Vessel Behaviour: A Three-Stage Pipeline for Robust Trajectory Embeddings**, Journal of Marine Science and Engineering 14(5), 507. https://www.mdpi.com/2077-1312/14/5/507

## M17H references
- Song R, Alam MM, Sadeghi Z, et al. *MoCo-AIS: A Contrastive Learning Framework for Similarity Computation of Vessel Trajectories* (2026), arXiv:2606.17978. https://arxiv.org/abs/2606.17978
- scikit-learn documentation, *Combine predictors using stacking*: meta estimators should be trained from cross-validated base predictions; conservative/regularized combinations reduce unnecessary degrees of freedom. https://scikit-learn.org/stable/auto_examples/ensemble/plot_stack_predictors.html

## M18A prediction-contract / validation references

- IMO Compendium, *Just In Time Arrival* data set — distinguishes Estimated/Actual/Planned/Requested arrival timestamps and defines Pilot Boarding Place and Berth as explicit maritime locations: https://imocompendium.imo.org/public/IMO-Compendium/Current/DS/Just%20In%20Time%20Concept/d11.htm
- IMO / GreenVoyage2050, *Just In Time Arrival Guide* — stresses that an arrival timestamp is ambiguous unless its location is defined (for example Pilot Boarding Place vs Berth): https://wwwcdn.imo.org/localresources/en/OurWork/PartnershipsProjects/Documents/GIA-just-in-time-hires.pdf
- U.S. Coast Guard NAVCEN, *AIS Class A Ship Static and Voyage Related Data (Message 5)* — ETA field is `MMDDHHMM` UTC, with month/day/hour/minute and no year: https://www.navcen.uscg.gov/ais-class-a-static-voyage-message-5
- *A Reproducible Workflow for AIS-Based ETA Forecasting: Evaluating the Influence of Data Preprocessing and Machine Learning Model Selection* (2026) — keeps complete voyages within one partition to reduce leakage and produce more realistic ETA evaluation: https://www.mdpi.com/2571-9394/8/4/70
## M18B validation-redesign references

- Guerreiro et al. (2026), *A Reproducible Workflow for AIS-Based ETA Forecasting: Evaluating the Influence of Data Preprocessing and Machine Learning Model Selection* — voyage-grouped train/validation/test partitions, independent held-out voyages and a separate chronological robustness split. https://www.mdpi.com/2571-9394/8/4/70
- scikit-learn, *Cross-validation: evaluating estimator performance — Time series data* — IID KFold/ShuffleSplit can be inappropriate for autocorrelated time series; future-facing splits and an optional gap are provided by `TimeSeriesSplit`. https://scikit-learn.org/dev/modules/cross_validation.html#time-series-split
- MAPEX (2026), *Map Exploitation for Vision-Based Ship Trajectory Prediction* — uses disjoint monthly train/test data and encounter-level grouping because temporally adjacent AIS trajectory windows are strongly correlated. https://www.mdpi.com/2079-8954/14/5/536


### M18D residual attribution context
- Marreiros, J. et al. (2026). *A Reproducible Workflow for AIS-Based ETA Forecasting: Evaluating the Influence of Data Preprocessing and Machine Learning Model Selection*. Forecasting 8(4), 70. https://www.mdpi.com/2571-9394/8/4/70
- Chu, Z., Yan, R., Wang, S. (2025). *Vessel arrival time to port prediction via a stacked ensemble approach: Fusing port call records and AIS data*. Transportation Research Part C 176, 105128. https://www.sciencedirect.com/science/article/pii/S0968090X25001329
- *High-accuracy prediction of vessels’ estimated time of arrival in seaports: A hybrid machine learning approach* (2025). Maritime Transport Research 8, 100133. https://www.sciencedirect.com/science/article/pii/S2666822X2500005X

## M18E external-information sources (reviewed 2026-09-23)

- UNECE / UN/CEFACT, UN/LOCODE Publications: https://unlocode.unece.org/publications/ — official production release page plus continuously updated pre-release; a moving pre-release must be locally frozen before modelling.
- UNECE / UN/CEFACT, UN/LOCODE data attributes: https://unlocode.unece.org/docs/data-attributes/ — semantics/encoding of LOCODE fields and coordinates.
- IHO, "The S-100 framework is now operational" (2026-03-10): https://iho.int/en/the-s-100-framework-is-now-operational — operational S-101/S-102/S-104/S-111/S-124/S-128/S-129 specifications; standard availability is not treated as proof of local Sicily dataset availability.
- IHO, ENC & ECDIS: https://iho.int/en/enc-ecdis — operational edition dates for S-101/S-102/S-111.
- genthalili/searoute-py: https://github.com/genthalili/searoute-py — research sea-route proxy candidate; upstream explicitly states it is not for navigational routing, so M18E requires a pinned network artifact before evaluation.
- IMO/GreenVoyage2050 Port Call Optimization Guide announcement: https://greenvoyage2050.imo.org/port-call-optimization-guide-launched-at-imo/ — operational port-call semantics reference; not a substitute for historical timestamped port-operation data.

## M18F uncertainty / selective prediction references

- Sokol, A.; Moniz, N.; Chawla, N. V. (2026), *Conformalized selective regression*, Discover Data 4, Article 14. https://doi.org/10.1007/s44248-026-00113-2 — formalizes regression with a reject option and evaluates the risk–coverage trade-off using conformalized uncertainty.
- Gao, Z.; Candès, E. et al. (2025), *Confidence on the focal: conformal prediction with selection-conditional coverage*, Journal of the Royal Statistical Society Series B 87(4):1239–1259. https://academic.oup.com/jrsssb/article/87/4/1239/8113856 — shows why marginal conformal coverage need not automatically remain valid after data-driven selection and motivates explicit selection-aware claims.
- Kwon, H.; Kim, D.-J. (2026), *Conformal selective prediction with cost aware deferral for safe clinical triage under distribution shift*, Scientific Reports 16, 10016. https://www.nature.com/articles/s41598-026-40637-w — recent example of combining calibrated uncertainty, explicit deferral and risk–coverage evaluation under temporal shift. Domain differs from maritime ETA; M18F uses it only as methodological context.

## M18G external-validation / leakage-control references

- scikit-learn, *Common pitfalls and recommended practices — Data leakage*: https://scikit-learn.org/dev/common_pitfalls.html — test data must not influence fitting, preprocessing or model choices; used as methodological support for blind pre-label scoring/sealing.
- scikit-learn, *Nested versus non-nested cross-validation*: https://scikit-learn.org/1.5/auto_examples/model_selection/plot_nested_cross_validation_iris.html — reusing the same data for selection and performance estimation produces optimistic bias; supports the separation between M18 development and external promotion.
- Riley et al., BMJ (2024), *Evaluation of clinical prediction models (part 2): how to undertake an external validation study*: https://www.bmj.com/content/384/bmj-2023-074820 — domain-general methodology for evaluating a previously specified prediction model in genuinely separate data and reporting subgroup performance. The application domain is clinical, not maritime; M18G uses only the validation-design principles.

## M18G full-development scoring-bundle methodology

- scikit-learn, *Common pitfalls and recommended practices — Data leakage*: https://scikit-learn.org/dev/common_pitfalls.html — preprocessing and fitting must be learned from training data only; the external test must not influence model choices.
- Oliveira, Orenstein, Ramos & Romano (2024), *Split Conformal Prediction and Non-Exchangeable Data*, JMLR 25(225):1–38. https://jmlr.org/papers/v25/23-1553.html — used as context for why external temporal/spatiotemporal shift may weaken nominal conformal claims even when the development calibration is valid.
- Bao et al. (2025), *CAP: A General Algorithm for Online Selective Conformal Prediction with FCR Control*, JMLR 26(287):1–74. https://jmlr.org/papers/v26/24-0452.html — methodological context for post-selection coverage; M18G/M18F therefore keep retained-set coverage empirical rather than claiming automatic selection-conditional validity.

## M19 external validation source

- Averty, T., Nasios, I., Ray, C., & Piliouras, N. (2026). *MMDEC: Multimodal maritime dataset on the English Channel*. Data in Brief, 65, 112629. DOI: 10.1016/j.dib.2026.112629.
- MMDEC v1 dataset, Zenodo DOI: 10.5281/zenodo.17491518. Required M19 files: `Dataset_AIS_POS.parquet` and `Dataset_AIS_SPEC.parquet`.
- EMSA CISE AIS Message 5 mapping: ETA is encoded as month/day/hour/minute UTC and Destination is a free-text voyage field.
