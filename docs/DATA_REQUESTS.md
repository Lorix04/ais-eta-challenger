# Company clarification status after M9

## Resolved by the company

1. **ETA event family:** arrival means the vessel reaches/enters the area of the destination port.
2. **`Positions`:** historical AIS observations over time for each vessel.
3. **`Tracks`:** most recent available vessel state.
4. **`recorded_at`:** timestamp associated with each observation in `Positions`. Upstream generation/reception/provider semantics were not supplied.
5. **Historical destination:** available in `Positions`.
6. **Historical ETA:** not available in `Positions`; ETA is latest-state information in `Tracks`.
7. **Longer history:** no additional AIS history is available for this exercise.
8. **Existing ETA-model inputs:** unknown because that model was not developed by the company.
9. **Specific business horizon/metric:** none specified beyond ETA to the destination-port arrival.

## Remaining P0 questions

1. **Ground-truth source:** is there an authoritative actual-arrival timestamp, or should arrival be reconstructed from `Positions`?
2. **Port-area geometry:** what perimeter/polygon defines entry into the destination-port area, or is defining it part of the exercise?

These two answers determine whether the current research-gate labels are merely conceptually aligned or exactly target-aligned.

## Important limitations now confirmed

- A longitudinal comparison against reported AIS ETA cannot be constructed from the supplied files without backfilling future/latest-state information.
- The exercise must be evaluated within the supplied historical window; no extra months of AIS are available.
- An apples-to-apples feature-parity comparison with the pre-existing ETA model cannot be established from the information currently available.
