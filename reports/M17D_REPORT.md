# M17D — Extend Frozen Mixture of Experts with M17A + M17C

## Decision

**NO_M17D_PROMOTION**

M17D extends the frozen M16G expert panel with two post-freeze signals: M17A learned trajectory retrieval and M17C causal corridor/knowledge-graph gating. The 53 old-final MMSIs remain hard-blocked.

## Leakage-safe stacking protocol

The final/meta learner is trained only on cross-fitted base predictions, matching the standard stacking principle that the meta-model must learn from out-of-fold base predictions rather than in-sample fits. For every M16A outer fold, all six experts are regenerated inner-OOF inside the outer-train. The M17A encoder is trained only on inner-train MMSI trajectories, while the M17C graph excludes both outer-valid and inner-valid owners. Outer-validation uses only frozen OOF predictions from M16C/D/E/F, M17A and M17C.

## Result

- Frozen M16G MAE: **185.883 h**
- M17D extended MoE MAE: **196.200 h**
- MAE gain vs M16G: **-10.317 h** (positive = better)
- M17D MedAE: **19.646 h**
- M17D P90 absolute error: **301.166 h**
- P90 ratio vs M16G: **1.0361**
- Fold wins vs M16G: **2/5**
- Changed-row win share: **45.57%**

Predeclared promotion gate: >=1 h pooled MAE gain, >=3/5 fold wins, >=50% wins among rows whose prediction changes, and P90 <=1.05x frozen M16G.

## Interpretation

A pass means the two new post-freeze signals add stable information beyond M16G under the same development-only nested protocol. A fail is retained as a valid negative result: M17A and M17C remain independently frozen components but are not allowed to replace the frozen M16G mixture without meeting the predeclared gate. No result here reopens the official M14 final submission or the already-observed 53-row M10 final set.
