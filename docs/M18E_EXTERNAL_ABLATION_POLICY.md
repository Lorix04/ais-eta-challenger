# M18E External Information Ablation Policy

M18E is an evidence-first laboratory around the frozen M16G/M16H system. It does **not** open the fresh holdout.

## Admission gate for a new external source

A source may enter a predictive ablation only when: (1) the exact artifact is locally pinned and checksummed; (2) its licence/provenance are recorded; (3) each feature is available no later than `decision_time`; (4) missingness is explicit; (5) the ablation changes one information family at a time; and (6) evaluation uses the frozen M18A/M18B contract.

Blocked sources are **not** assigned a zero gain. They are unmeasured. Target-derived slices such as `FUTURE_0_7D` remain diagnostic only.

## Current result

The strict retrospective destination-canonicalization ablation changes MAE by **-6.29 h** (positive is better) and P90 by **17.11 h**. This is mixed evidence, so canonicalization remains infrastructure rather than a standalone accuracy promotion.

On the already-frozen near-term physics-eligible component (n=101), external destination geometry + causal motion improves the route expert by **6.73 h MAE**. Because this is a mixed historical component and not a newly isolated M18E source, it is supporting evidence only.
