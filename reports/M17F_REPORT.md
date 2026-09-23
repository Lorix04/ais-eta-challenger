# M17F — Latent Target-Noise / Declared-ETA Regime Model

## Decision

**NO_M17F_PROMOTION**

M17F models residual regimes only inside training folds. The latent regime labels/posteriors may use training residuals, but runtime gating uses target-free features only. M16G is reconstructed inner-OOF inside each outer-train and remains the frozen baseline on outer validation.

## Result

- M16G frozen MAE: **185.883 h**
- M17F MAE: **185.883 h**
- Gain: **0.000 h**
- M17F MedAE: **19.342 h**
- M17F P90: **290.680 h**
- Fold wins: **0/5**
- Trimmed-95% MAE: M16G **64.689 h**, M17F **64.689 h**
- Max fold regression: **0.000 h**

Promotion gate was fixed before the benchmark: >=1 h MAE gain, >=3/5 fold wins, P90 <=1.02x M16G, trimmed-95% MAE no worse, and <=15 h worst-fold regression.
