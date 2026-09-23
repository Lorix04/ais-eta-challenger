# M17H — Conservative M16G integration of M17G MoCo retrieval

## Decision

**NO_M17H_PROMOTION**

M17H does not enlarge the ensemble. It preserves the four-expert M16G architecture, replacing the M16D route expert one-for-one with the passing M17G MoCo-TCN64 retrieval component and using the same MoCo route as fallback when M16E physics is not eligible. The meta strategy itself is frozen per outer fold to the candidate selected by M16G before M17G existed; only that fixed strategy is refit on the new inner-OOF four-expert matrix.

## Result

- frozen M16G MAE: **185.883 h**
- M17H MAE: **192.176 h**
- pooled gain: **-6.294 h**
- fold wins: **2/5**
- M16G P90: **290.680 h**
- M17H P90: **305.223 h**
- changed-row win share: **47.40%**
- worst fold regression: **29.590 h**

## Leakage control

The 53 old-final MMSIs remain blocked. MoCo is retrained inside every inner fold using only causal AIS windows from inner-train owners. Inner-valid and outer-valid owners are excluded from SSL pretraining. No ETA labels are used by the encoder.

## Interpretation

This is an intentionally conservative substitution test. A gain can be attributed to the stronger route representation more cleanly than in M17D because expert count and meta-strategy family are not expanded or re-selected.
