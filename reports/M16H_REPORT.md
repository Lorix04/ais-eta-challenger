# M16H — Probabilistic ETA + Confidence

## Decision

**PASS_PROBABILISTIC_CALIBRATION_CONFIDENCE**

M16H does **not** change the M16G point predictor. `P50` is byte-for-value identical to `pred_m16g_selected_h`. The uncertainty layer is calibrated only from development OOF residuals; all 53 old M10 final MMSIs remain hard-blocked.

## Method

For each frozen M16A outer fold, an error model is cross-fitted inside the outer-train using only prediction-time features from the four frozen expert predictions and their support/quality signals. Candidate interval scalings are then selected with leave-one-inner-fold calibration inside the outer-train. The selected scale is conformalized from held-out OOF absolute residuals at nominal central coverage **80%** and applied once to the outer-validation rows.

`P10/P90` are therefore empirical central-80% interval endpoints around the frozen M16G `P50`; they are **not** separately trained conditional quantile regressors.

## Result

- pooled empirical coverage: **80.05%**
- pooled mean width: **159.42 h**
- pooled median width: **107.16 h**
- global symmetric residual baseline coverage: **81.09%**
- global symmetric residual baseline mean width: **221.26 h**
- adaptive mean-width reduction: **27.9%**
- stable outer folds with >=75% empirical coverage: **4/5**
- M16G point MAE remains **185.88 h**

## Confidence tiers

Confidence is assigned from cross-fitted predicted absolute error, never from the row's true error. Outer-train OOF risk terciles define HIGH/MEDIUM/LOW thresholds. Typical error is strongly ordered: HIGH MedAE **3.32 h**, MEDIUM **19.17 h**, LOW **84.40 h**. Extreme stale/reference-label errors can still occur even in HIGH confidence, so confidence must be interpreted as relative operational reliability rather than a conditional guarantee.

## Scope and limitations

Conformal-style validity is marginal and relies on exchangeability-like assumptions. AIS/reference-ETA drift, rare destinations and label pathologies can violate those assumptions. Subgroup coverage is reported diagnostically and is not claimed to be guaranteed. `reference_eta_status` is target-derived and appears only in post-hoc diagnostics after calibration is frozen; it is never a confidence or interval input.

The M14 company submission and all M10/M6/M16A-G freezes remain immutable.
