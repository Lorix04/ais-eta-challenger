# M18C — Target & Label Reliability Audit

## Decision

**TARGET/LABEL AUDIT FROZEN — NO MODEL CHANGE, NO RELABELING, NO ROW DELETION.**

M18C audits the 386 development labels under the frozen M18A prediction contract. M16G, M16H, the 53-owner old-final block, M18B forward splits and the unopened fresh external holdout are unchanged.

## Main finding

The overall M16G development MAE is **185.88 h**, but that aggregate is dominated by reference-label regimes outside the near-term future window.

- `FUTURE_0_7D`: **279/386 rows (72.3%)**, M16G MAE **20.26 h**, only **7.9%** of total absolute error.
- All other regimes: **107/386 rows (27.7%)**, M16G MAE **617.74 h**, **92.1%** of total absolute error.
- Past references: **57 rows (14.8%)**.
- Future references beyond 7 days: **50 rows**; beyond 30 days: **7 rows**.
- Placeholder-like ETA patterns: **5**.
- Year-resolution margin <90 d: **4 rows**.

This does **not** justify deleting hard rows or reporting 20.26 h as the official score. It shows that the 185.88 h aggregate mixes very different label regimes and therefore should not be interpreted as pure model error.

## Historical AIS-derived proxy alignment

The Catania/Augusta research ledgers overlap **39** development MMSIs. For all of them, the validated research-gate events available in the finite dataset occur before the corresponding `Tracks.last_update`; **0** owners have a validated research-gate event at/after decision time. Consequently these events cannot serve as future actual-arrival ground truth for the M18A task.

Post-hoc only, the company ETA timestamp lies within 6 h of a historical research-gate event for **15/39** overlapping owners and within 7 days for **33/39**. This is evidence worth investigating for stale/carry-over ETA semantics, but it is not authoritative proof that those historical events are the intended labels.

## Granularity

- Exact-hour ETA fraction: **85.8%**.
- Minute 00/30: **93.5%**.
- Five-minute grid: **95.9%**.

The minute-level granularity is tiny compared with the hundreds/thousands of hours in the extreme regimes, so timestamp rounding alone cannot explain the heavy tail.

## Policy frozen by M18C

1. Keep all 386 labels in the strict primary benchmark; no post-hoc filtering.
2. Treat `FUTURE_0_7D` and the reliability tiers only as secondary diagnostics, never as predictor features or training weights.
3. Do not reinterpret `Tracks.eta` as ATA, PBP, berth arrival or all-fast.
4. Do not substitute the M0/M1X research-gate events for company labels.
5. Authoritative port-call/ATA data remains the missing evidence needed to resolve target semantics conclusively.
6. M18D may use these frozen regimes to attribute residual error, but cannot change the primary target contract.
