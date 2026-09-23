# M17E — Constrained M16G ↔ M17A Selective Router

## Decision

**NO_M17E_PROMOTION**

M17E limits the router to exactly two actions: keep frozen M16G or switch to frozen M17A learned retrieval. No additional expert is available to the router. Runtime features are target-free; outer validation is untouched during fitting/selection; M16G is reconstructed from inner-OOF base experts and M17A is re-trained only on inner-train causal AIS trajectories.

## Result

- Frozen M16G MAE: **185.883 h**
- Frozen M17A MAE: **190.105 h**
- M17E selective-router MAE: **185.317 h**
- MAE gain vs M16G: **0.566 h**
- M17E MedAE: **19.307 h**
- M17E P90 AE: **279.890 h**
- Fold wins: **3/5**
- Switch share: **2.33%**
- Changed-row win share: **55.56%**
- Max single-fold regression: **0.000 h**

Promotion requires >=1 h pooled gain, >=3/5 fold wins, >=52% wins among changed rows, P90 no worse than M16G, <=35% switch share and <=15 h worst-fold regression.
