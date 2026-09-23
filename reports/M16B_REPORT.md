# M16B — Canonical Destination Resolver

## Decision

**PASS semantic-resolution gate; do not promote the naive canonical destination median as the M16 challenger.**

M16B implements a target-free, provenance-aware resolver for the AIS destination field. It reduces semantic fragmentation from **248 raw destination categories to 169 canonical categories (-31.9%)** on the frozen 386-row development population while hard-blocking all 53 already-observed M10 final MMSIs.

The resolver directly resolves **158/386 rows (40.9%)** to catalogued ports and explicitly separates `UNKNOWN`, `FOR_ORDERS`, non-specific operational text and ambiguous geographies instead of forcing them to a port. Route expressions such as `ITSAL>ITCTA` are interpreted using the terminal/next-port token, consistent with the IMO `origin>destination` convention.

## Resolver contract

Inputs used by the resolver:

- `destination_norm` only;
- versioned file `data/static/m16b_destination_catalog.csv`.

Inputs **not** used:

- reference ETA or `target_tte_h`;
- split labels;
- MMSI identity;
- position, SOG, route history or future observations.

The catalog records canonical UN/LOCODE, port name, country, optional coordinates, aliases and source provenance. Exact catalogued UN/LOCODEs receive confidence 1.00, curated aliases 0.99, conservative fuzzy matches require a frozen 0.94 threshold and 0.04 winner margin, and syntactically UN/LOCODE-like codes absent from the compact catalog are formatting-normalized but explicitly marked `unlocode_like_unverified` rather than asserted as verified ports.

## Important ambiguity behavior

`MALTA` is **not** mapped to `MTMLA`. Malta contains multiple ports (including Valletta/Grand Harbour and Marsaxlokk), so a country-level destination is preserved as `AMBIGUOUS:MALTA`. Conversely, explicit `MTMLA`, `MT MLA` and `VALLETTA` resolve to Valletta.

`FOR ORDER`, `FOR ORDERS`, `ORDER` and strings containing that concept are represented as the explicit non-port class `FOR_ORDERS`. `UNKNOWN` remains `UNKNOWN`. Ocean/operational descriptors such as `ATLANTIC OCEAN`, `WORKSITE`, `IN PORT` and `MARE` are not coerced to a port.

## Development-only OOF diagnostic

For diagnosis only, M16B reruns the simple train-only destination-median prior under the **same frozen M16A 5-fold assignments**, replacing raw destination with the canonical destination key. This is not M16C hierarchical shrinkage and it is not a new untouched final result.

| Model | OOF MAE h | OOF MedAE h | OOF P90 AE h |
|---|---:|---:|---:|
| M16A raw destination median | 202.97 | 22.27 | 330.82 |
| M16B naive canonical destination median | 209.26 | 20.73 | 313.70 |

Semantic consolidation **improves median error and P90**, but worsens pooled MAE by **6.29 h**. Inspection shows why: the reference-ETA target has extreme/stale values inside some semantically correct destination groups. For example, a previously unseen raw spelling can fall back to the stable global median in M16A, while canonicalization connects it to a small train group containing an extreme stale target and therefore inherits an unstable median.

This is evidence **against** using an unshrunk destination lookup as the challenger. It directly motivates M16C: fold-safe hierarchical priors with minimum support, shrinkage and deterministic fallback.

## Key examples

- `ITAUG`, `IT AUG`, `AUGUSTA` -> `ITAUG` / Augusta.
- `ITCTA`, `CATANIA SICILY`, route terminal in `ITSAL>ITCTA` -> `ITCTA` / Catania.
- `GIGIB`, `GI GIB`, `GIBRALTAR`, observed `GBGIB` alias -> `GIGIB` / Gibraltar.
- `EGPSD`, `EG PSD`, `PORT SAID`, high-confidence `PORTSAID` fuzzy match -> `EGPSD` / Port Said.
- `MESSINA>MILAZZO` -> terminal `MILAZZO` -> `ITMLZ`.
- `MALTA` -> `AMBIGUOUS:MALTA`, not Valletta.

## Gate conclusion

M16B passes its intended gate because it substantially reduces semantic fragmentation, is deterministic, target-free, and leaves ambiguous/non-port inputs explicit. The fact that the naive canonical median does not improve MAE is itself a useful result: **better semantics require robust statistical shrinkage under this noisy reference-ETA target.**

Next: **M16C — Fold-safe Hierarchical Destination Priors.**
