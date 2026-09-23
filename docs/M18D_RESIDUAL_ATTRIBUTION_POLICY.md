# M18D Residual Attribution Policy

M18D is diagnostic-only. It operates on frozen M16G OOF predictions and does not change M18A target semantics, M18B validation, M18C labels, M16G/M16H or the 53-row old-final block.

## Interpretation rules

- Always report the strict 386-row benchmark separately from the 279-row `FUTURE_0_7D` diagnostic slice.
- Do not call subgroup associations causal effects.
- Do not expose `target_tte_h`, `reference_eta_status`, M18C reliability tiers, residuals, hindsight oracle identity or coverage outcomes as runtime predictors.
- The existing-expert oracle is only a lower-bound/headroom diagnostic; it is target-dependent and non-deployable.
- Missing variables such as weather and port operations must be marked untestable rather than assigned invented importance.
- M18E experiments must add one information family at a time and preserve the M18A/M18B lockbox rules.

## Frozen headline diagnostics

- strict M16G MAE: 185.883 h
- near-term M16G MAE: 20.261 h
- near-term M16H predicted-error rank correlation: 0.600
- near-term existing-expert oracle gap: 10.115 h (diagnostic only)

## External research context

- Marreiros et al. (2026), *Forecasting*: reproducible AIS ETA workflow; leakage-free voyage grouping and anomalous/loitering behaviour as deployment constraints: https://www.mdpi.com/2571-9394/8/4/70
- Chu, Yan & Wang (2025), *Transportation Research Part C*: arrival-time prediction fusing AIS and port-call records and analyzing vessel-reported ETA accuracy: https://www.sciencedirect.com/science/article/pii/S0968090X25001329
- Maritime Transport Research (2025): ETA feature-importance analysis highlights speed, distance, course and vessel type: https://www.sciencedirect.com/science/article/pii/S2666822X2500005X
