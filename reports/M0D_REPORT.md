# M0D — Catania geometry and candidate port-call reconstruction

**Status:** DONE — candidate-event layer complete; no row is final ground truth.

## Objective

M0D builds a reproducible and auditable Catania arrival-event layer on top of the causal M0C vessel states. It deliberately does **not** claim to recover an official ATA, Pilot Boarding Place (PBP), port-limit crossing or all-fast time.

The event reconstructed here is named `CATANIA_RESEARCH_GATE_INBOUND` / `CATANIA_RESEARCH_GATE_OUTBOUND`. The research gate is a versioned cross-section used to generate candidate sessions for the M0E manual audit.

## Geometry provenance

Authoritative/domain sources consulted on 2026-09-18:

1. **Istituto Idrografico della Marina — Elenco Fari / Catania aids.** Published rounded coordinates identify light 2802 (`Molo di Levante, estr`) at 37°29.0' N, 15°06.0' E and light 2804 (`Nuova darsena traghetti, banchina n°31, estr`) at 37°29.3' N, 15°05.7' E. Internal aids 2808 and 2812 are also listed.
2. **Autorità di Sistema Portuale del Mare di Sicilia Orientale — Porto di Catania.** Official material names Nuova Darsena, Molo di Mezzogiorno, Molo Crispi, Sporgente Centrale, Molo di Levante and Piazzale Triangolare and publishes berth/depth information.
3. **Official 1:2000 “Planimetria stato di fatto”.** The authority repository visibly labels Molo Foraneo, Porto Nuovo, Porto Vecchio, Sporgente Centrale, Molo di Mezzogiorno, Nuova Darsena Commerciale and berth numbering.
4. **2026 operational context.** AdSP material documents continuing restrictions/works around Molo di Levante after January 2026 weather damage and ongoing breakwater works. This is another reason not to represent the research polygon as immutable official navigation truth.

The two rounded IIM light positions are connected to form `catania_research_geometry_v1`. This is an **INFERENCE / research proxy**, not an official boundary.

## Frozen geometry v1

- west gate anchor: `(15.0950, 37.4883333333)` — IIM light 2804 rounded position;
- east gate anchor: `(15.1000, 37.4833333333)` — IIM light 2802 rounded position;
- inner-harbour polygon: analysis-only dwell-confirmation envelope;
- broad local analysis envelope: `(15.06, 37.44, 15.14, 37.53)`.

The geometry is frozen before ETA development. Any change requires a new version and an external/domain justification, never downstream ETA MAE.

## Event logic

The candidate state machine is:

```text
SEA / APPROACH
      ↓
CATANIA_RESEARCH_GATE_INBOUND
      ↓
PORT-SIDE SESSION
      ├─ sustained inner-harbour stop → STOP_CONFIRMED_CALL_CANDIDATE
      └─ no sustained stop          → ENTRY_WITHOUT_CONFIRMED_STOP / GATE_CROSSING_ONLY
      ↓
CATANIA_RESEARCH_GATE_OUTBOUND
```

Key rules:

- crossing must intersect the **finite** research gate, not merely change side of an infinite line;
- immediately previous/current observations only are used to detect a crossing;
- crossing intervals above 300 s or position-jump candidates are rejected;
- every crossing preserves `[last_before, first_after]` plus the midpoint proxy;
- outbound/re-entry excursions within 15 min are hysteresis-merged;
- a 15 min sustained inner-port stop is supporting evidence for a call candidate;
- `nav_status=moored/anchor` is supporting evidence, not a required truth condition;
- the roadstead/anchorage concept is deliberately a `roadstead_wait_proxy`, because no authoritative Catania anchorage polygon was verified;
- destination is **supporting evidence only** and is restricted to the pre-entry / entry-time information set; future post-entry destination updates are not used;
- unclosed sessions are separated into genuine dataset-end right censoring vs unresolved missing outbound observations.

Future observations may confirm that an M0D *label candidate* behaved like a port call. They are not exported as prediction-time features, and `final_ground_truth` remains false for every M0D row.

## Full-run results

Input: M0C ship-state layer, 2026-04-03 to 2026-04-16.

| Measure | Result |
|---|---:|
| AIS rows inside broad Catania analysis envelope | 142,287 |
| distinct vessels in envelope | 58 |
| finite-gate crossing events | 136 |
| vessels with gate events | 40 |
| inbound / outbound events | 70 / 66 |
| candidate port-side sessions | 61 |
| vessels represented in candidate sessions | 34 |
| stop-confirmed call candidates | 54 |
| entry without confirmed stop | 7 |
| M0D candidate quality A / C | 46 / 15 |
| dataset-end right-censored sessions | 5 |
| unresolved missing-outbound sessions | 12 |
| pre-entry roadstead-wait proxy | 4 |
| entry-time/pre-entry destination support for Catania | 37 |
| final ground-truth rows | **0** |

The provider cadence is visible in the event brackets: among the 136 gate events, median crossing bracket is **61 s**, P95 is **61 s**, maximum **62 s**. This is measurement bracketing in the provider observation clock; physical crossing uncertainty can be larger when state freshness is poor.

## Verification

Automated M0D verification confirms:

- all crossing points lie on the finite research-gate segment;
- event direction is only `INBOUND` or `OUTBOUND`;
- all crossing brackets are ordered and ≤300 s;
- all candidate sessions require manual audit;
- no M0D row is final ground truth;
- geometry features are prefix-invariant on 12 sampled vessels;
- selected crossing events are prefix-invariant when future rows are removed (10/10 checks).

The visual audit deliberately includes four different failure/behaviour modes: a normal closed stop-confirmed call, a call with pre-entry roadstead waiting, a very short entry without confirmed stop, and an unresolved missing-outbound session. Unclosed tracks are visually capped after 12 h so missing observation tails do not draw misleading long trajectories.

## What M0D establishes

**EVIDENCE from the data:** a strong, repeatable AIS corridor crosses the research gate; 136 direction-aware finite-segment events can be reconstructed; most event brackets are one provider sampling interval; many inbound sessions subsequently show persistent inner-port stopping.

**INFERENCE:** the research gate is a useful Catania arrival cross-section for candidate generation. It is not equivalent to the company’s unknown ETA target until that target is defined.

**UNKNOWN:** official company ATA semantics; official PBP; legal/operational port-limit line; exact anchorage geometry; berth/all-fast timestamps; reasons for waiting; whether all missing outbound cases are provider coverage gaps, AIS gaps or vessel behaviour.

## M0E handoff

M0E must manually audit all 61 candidate sessions before any supervised ETA dataset exists. Each row needs at least:

- keep / exclude;
- audited event type;
- ground-truth confidence A/B/C;
- ambiguity reason;
- verification of the inbound crossing and subsequent port behaviour;
- right-censor / missing-outbound interpretation;
- notes on service craft, transit, shifting, repeated calls or geometry edge cases.

Only after that audit will the project compute effective independent call count and reach GO/NO-GO A.
