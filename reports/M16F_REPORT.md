# M16F — Tabular Challenger Panel

## Decision

**PASS_TABULAR_COMPLEMENT_FOR_M16G**

M16F compares five predeclared tabular families on the frozen M16A outer folds. Hyperparameters are selected only by 3-fold inner CV on each outer-train partition. The 53 already-observed M10 final MMSIs are hard-blocked.

## Best nested-OOF tabular family

- best family: **hist_gb**
- pooled MAE: **197.28 h**
- MedAE: **27.16 h**
- P90 absolute error: **292.84 h**
- gain vs frozen M16A current CatBoost: **8.87 h**
- fold wins vs frozen M16A current CatBoost: **5/5**
- gain vs M16D route analogue: **-2.29 h**
- fold wins vs M16D route analogue: **3/5**

## Interpretation

The panel tests whether a different tabular inductive bias adds stable development-only signal once target-free destination/physics-quality representation is available. M16F does **not** use M16C/M16D/M16E expert predictions as features; those remain reserved for M16G stacking. A family is retained only if it is stable against the frozen M16A current-state CatBoost, not merely because of one pooled score.

## Scope

This is a post-freeze development experiment, not a reopened final result. No claim about the old 53-row final is made.
