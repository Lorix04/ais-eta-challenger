# M16G — Mixture of Experts / OOF Stacking

## Decision

**PASS_MOE_STABLE_GAIN**

M16G combines the frozen M16C hierarchical prior, M16D route analogue, M16E physics hard-gate and M16F HistGradientBoosting expert. For every outer fold, the four base experts are **regenerated as inner-OOF predictions using only that outer-train partition** before any meta learner is fitted or selected. Outer-validation expert predictions come from the already frozen prior-milestone OOF ledgers. The 53 old-final MMSIs remain hard-blocked.

## Result

- selected nested-mixture MAE: **185.88 h**
- MedAE: **19.34 h**
- P90 absolute error: **290.68 h**
- best single operational expert (M16E physics-gate/route fallback): **192.93 h MAE**
- pooled gain vs best single: **7.04 h**
- fold wins vs best single: **4/5**
- paired row-level win share (all rows): **27.5%**
- exact-tie share (gate preserves best single): **49.7%**
- win share among rows where M16G actually changes the prediction: **54.6%**

## Meta panel

Compared: row-wise median, equal mean, non-negative normalized NNLS, RandomForest hard gate, ExtraTrees hard gate, HistGradientBoosting hard gate, and logistic soft gating. Candidate selection is performed only inside the current outer-train using the regenerated inner-OOF expert matrix.

## Interpretation

The gate requires more than a pooled-number improvement: at least 1.0 h pooled MAE gain, at least 3/5 fold wins, and at least 50% paired wins among rows where the mixture actually changes the best-single prediction; exact ties are neutral. M16G remains development-only evidence and does not alter the official M14 frozen submission.
