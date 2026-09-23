# M9 — Company Alignment & Target Reconciliation

## Status

**ALIGNMENT APPLIED — FROZEN M6 UNCHANGED — TWO TARGET-LABEL QUESTIONS STILL PENDING**

M9 reconciles the take-home with the company's written clarifications. No model was retrained, no route library was refit, no threshold was changed, and the final chronological holdout was not reopened.

## Main conclusion

The company's requested ETA event is now known at the event-family level: **arrival when the vessel reaches/enters the area of the destination port**.

That makes the existing Catania/Augusta research-gate entrance target **conceptually aligned** with the requested use case. Exact equivalence is not yet established because the company has not yet specified:

1. the source of the actual-arrival ground truth; and
2. the perimeter/polygon used to define the destination-port area.

Therefore M6 results remain historical **research-gate** results. They are not retrospectively relabelled as official company-target metrics.

## Data-semantics reconciliation

The company clarified that `Positions` contains historical AIS observations and `Tracks` contains the latest state. This confirms the project's decision not to backfill `Tracks.ETA` into historical `Positions` rows.

The company also clarified that `recorded_at` is the timestamp associated with each observation in `Positions`, but did not provide upstream timestamp provenance. M9 therefore removes any recruiter-facing wording that sounds like the company confirmed a latest-state polling architecture for `Positions`. The ~60–61 s synchrony/repetition pattern remains an empirical dataset finding only.

Historical destination exists in `Positions`; historical ETA does not. Consequently the reported-AIS-ETA longitudinal benchmark remains unavailable without inventing data.

No longer AIS history is available for the exercise, and the company does not know the information set used by the pre-existing ETA model because it was developed externally. This preserves the existing claim boundary: the take-home cannot establish apples-to-apples superiority over that model.

## Current evidence posture

### Better aligned than before

- target family: destination-port area entry;
- `Positions`/`Tracks` roles;
- causal use of historical destination;
- explicit impossibility of historical ETA comparison;
- fixed evidence limit of the supplied history.

### Still pending

- authoritative arrival ground truth vs AIS-derived reconstruction;
- exact destination-port area geometry.

## Next decision

If the follow-up confirms that arrival is reconstructed from `Positions` using a perimeter compatible with the current research gates, the existing results can be described as directly target-aligned with an explicit geometry caveat.

If the follow-up supplies a different official perimeter or official arrival timestamps, create a **new target version** and a new evaluation plan. Do not mutate the frozen M6 result after the fact.
