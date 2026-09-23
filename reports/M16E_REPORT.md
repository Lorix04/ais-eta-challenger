# M16E — Maritime / Physics Expert

## Decision

**PASS_PHYSICS_GATING_COMPONENT**. M16E is retained as an interpretable gating expert, not promoted as a new final model.

## Method

The expert uses M16B catalog-resolved destination coordinates, great-circle remaining distance, causal recent median SOG, and an optional local-sinuosity route-detour proxy. It tests 12 predeclared configurations (180/360 minute speed window × geodesic/local-sinuosity distance × none/global/destination-shrunk residual correction). Each outer fold selects its configuration only with 4-fold inner CV. The 53 old-final MMSIs are never used.

No S-57/S-100 chart, bathymetric water mask, commercial routing API, or future trajectory is used; therefore the distance term is explicitly a lower-bound/proxy rather than a navigational route distance.

## OOF result

- Physics-eligible coverage: **115/386 (29.8%)**.
- Physics MAE on eligible rows: **161.48 h** vs M16D route **168.39 h**.
- Physics MedAE: **2.64 h** vs route **8.82 h**.
- Physics wins paired absolute error on **69.6%** of eligible rows.
- Fixed target-free hard gate (physics when eligible, M16D otherwise): **192.93 h MAE** vs route **194.98 h**, gain **2.06 h**, with **3/5** outer-fold wins.

## Interpretation

The physics signal is highly informative when a canonical port and meaningful recent motion exist. The remaining tail is dominated by the fact that the company target is a reported/reference AIS ETA and may be stale or already in the past; a physical travel-time model cannot logically reproduce negative/stale TTE values. `reference_eta_status` is therefore used only after scoring for diagnostics and never for model selection or gating.

M16G should use physics eligibility, distance, route support, alignment and recent-motion quality as target-free gating features rather than forcing physics onto every row.
