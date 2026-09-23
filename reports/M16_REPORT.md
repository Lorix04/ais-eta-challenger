# M16 — Smart Hybrid Challenger: final freeze

## Final decision

**PROMOTE_CHALLENGER_FOR_NEW_UNTOUCHED_HOLDOUT_ONLY**

M16 is frozen as a **development-only challenger for fresh external validation**. It does **not** replace the M14 official submission, and it must not be described as proven superior on unseen company data. The 53 already-observed M10 final MMSIs were hard-blocked from every M16 selection decision.

## Why M16 exists

M16 tests whether domain structure can improve on a single tabular predictor without reopening the observed final set. The architecture combines a canonical destination path, hierarchical destination priors, historical route analogues, a maritime/physics gate, a robust tabular expert, nested OOF mixture-of-experts, and calibrated uncertainty/confidence.

## Development-only point result

- destination-median baseline: **202.97 h MAE**
- current-snapshot CatBoost baseline: **206.14 h MAE**
- route analogue: **194.98 h MAE**
- physics-gate + route fallback: **192.93 h MAE**
- best tabular challenger (HistGradientBoosting): **197.28 h MAE**
- frozen nested MoE: **185.88 h MAE / 19.34 h MedAE / 290.68 h P90**
- gain vs best frozen single operational expert: **7.04 h**, with **4/5** outer-fold wins
- gain vs the destination-median M16A baseline: **17.09 h**

## Robustness and caveat

The gain remains positive after leaving out each outer fold (5/5), after removing the top 5% of rows by absolute target magnitude (+9.66 h), after 95% winsorisation (+4.07 h), and after removing the largest net-gain destination (+2.49 h).

The paired bootstrap remains the reason not to overclaim: **P(gain>0)=92.18%**, but the **95% gain interval is [-2.02, 20.16] h**, and the five largest positive-gain rows account for **51.0%** of total positive gain. The correct claim is therefore “promising, structured challenger”, not “proven winner”.

## Probabilistic output

M16H leaves the M16G point estimate unchanged as P50 and adds a central-80% interval plus confidence tier. Pooled empirical coverage is **80.05%**, with **159.42 h** mean width, **27.9%** narrower than the global residual baseline. Confidence tiers separate typical error strongly, but low-confidence/reference-pathology regimes remain heavy-tailed and are not given a conditional-coverage guarantee.

## Strict evidence separation

**M16 development evidence:** all M16A–I model selection, gating, interval selection, ablation and red-team work uses only the 386 former M10 train+calibration MMSIs.

**Old M10 final:** 53 MMSIs were already observed before M16. They are prohibited from M16 selection and are not used here to create a new headline score.

**Official submission:** M14 remains the official frozen company submission. Its checksum is re-verified by M16J.

## What would establish a fresh generalization claim

Score the frozen M16 challenger once on **new, untouched company data or a newly issued holdout** with target semantics fixed in advance. Compare the frozen M16 point predictor and intervals against the agreed baseline with the same metric and report paired uncertainty. No M16 re-tuning should occur after that new holdout is revealed.

## Deliverables

- `reports/m16j_challenger_comparison.csv` — frozen OOF comparison ledger
- `reports/m16j_claim_register.csv` — claims and allowed wording
- `reports/m16j_architecture.png` — final architecture
- `reports/m16j_result_progression.png` — result progression
- `reports/m16j_ablation_summary.png` — structural ablation summary
- `docs/M16_COMPANY_NOTE_IT.md` — concise company-facing explanation
- `reports/M16_CONTENT_MANIFEST.json` — SHA-256 manifest of frozen M16 payload
- `dist/ais_eta_m16_challenger_freeze.zip` — deterministic M16 challenger freeze archive, **not** the official M14 submission
