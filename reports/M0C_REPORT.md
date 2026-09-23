# M0C — AIS normalization and semantic motion states

**Status:** PASS
**Dataset:** 969,084 ship snapshots / 1,129 MMSI / 2026-04-03 to 2026-04-16
**Purpose:** build a causal, auditable vessel-state layer before any port-call or ETA modelling.

## What M0C changes

M0C keeps every provider snapshot and adds normalized AIS values, observation/state-change clocks, position-implied kinematics, semantic motion candidates, observation-gap flags and stale-state diagnostics. It does **not** create port calls or ETA labels.

The implementation lives in `src/ais_eta/m0c.py`; the full-data build is `scripts/04_m0c_semantic_states.py`; full-data verification is `scripts/05_m0c_verify.py`.

## AIS normalization findings

Official AIS Class-A semantics distinguish unavailable/special values from real measurements. In this dataset, M0C found:

| Field condition | Rows | Share |
|---|---:|---:|
| SOG decoded `102.3` (raw 1023 unavailable) | 48 | 0.005% |
| COG `>=360` unavailable | 97,059 | 10.016% |
| Heading `511` unavailable | 256,354 | 26.453% |
| Draught `<=0` unavailable | 24,390 | 2.517% |
| Nav status `15` undefined | 21,107 | 2.178% |
| Destination missing/blank | 228,945 | 23.625% |

There are zero SOG rows at decoded `102.2`; if they appear in future data they are retained as the AIS capped value “102.2 kn or higher”, not treated as unavailable.

Source semantics: USCG Navigation Center / ITU-R M.1371 descriptions for Class-A position reports and Message 5.

## Observation clock vs state-change clock

The layer explicitly separates:

- `recorded_at`: provider observation clock;
- `observation_dt_s`: time since the preceding provider row for the MMSI;
- `dynamic_state_changed`: whether cleaned `(lat, lon, SOG, COG, heading, nav_status)` changed;
- `dynamic_state_age_s`: time since the last **observed** dynamic-state change.

`dynamic_state_age_s` is not called AIS message age because the source-message timestamp is unavailable.

On the full dataset, **46.65%** of ship rows repeat the preceding cleaned dynamic state. This is consistent with the provider-snapshot hypothesis established in M0B, but does not by itself prove the provider architecture.

## Semantic candidate states

M0C emits candidate states/events, not maritime truth:

| Output | Rows / events |
|---|---:|
| STOPPED rows | 588,700 |
| MOVING rows | 237,194 |
| SLOW_MOTION rows | 50,533 |
| KINEMATIC_CONFLICT rows | 77,372 |
| POSITION_JUMP rows | 1,318 |
| stop starts | 11,773 |
| slow starts | 5,765 |
| turn candidates | 3,809 |
| provider observation gaps | 7,578 |

A `KINEMATIC_CONFLICT` means reported SOG indicates movement while position-implied speed over the provider interval is nearly zero. The 77,372 conflicts are a substantial data-quality/staleness signal and reinforce the need to avoid treating every minute row as a fresh sensor update.

Observation gaps break stop/slow episode continuity: dwell duration is never silently bridged across an interval in which the vessel was not observed.

## Stale-risk proxy

`stale_risk_level` is a deterministic diagnostic, not a probability:

- LOW: 856,475 rows (88.38%)
- MEDIUM: 35,163 rows (3.63%)
- HIGH: 77,446 rows (7.99%)

Long stationary periods with low SOG and `AT_ANCHOR`/`MOORED` status are protected from automatic HIGH classification. This avoids equating “unchanged” with “stale” when a vessel is plausibly stationary.

## Causality verification

M0C has no centered windows, future interpolation, future route, future destination or post-event features.

A prefix-invariance integration test was run on 12 vessels: features computed on each trajectory prefix were identical to the same rows computed with future observations present. **12/12 passed.** Unit tests also check prefix invariance directly.

## Full-data verification

`python scripts/05_m0c_verify.py` passed the following invariants:

- 969,084 rows preserved;
- 1,129 MMSI preserved;
- repeated snapshots preserved;
- normalized sentinel values do not survive in clean fields;
- per-MMSI observation time is nondecreasing;
- stop duration resets after provider observation gaps;
- full derived layer is readable and schema-complete;
- prefix invariance passes.

The local full layer is `data/derived/m0c_ship_states.pkl.gz`; `data/` remains gitignored because it contains private/large material.

## Visual audit

`reports/m0c_semantic_timeline.png` shows IEVOLI STAR (MMSI 215872000) over a compact approach/motion window. Reported SOG and position-implied speed generally track each other when positions update; candidate slow/stop transitions appear during deceleration, and a kinematic-conflict marker captures a snapshot with movement reported but no position change.

## M0C gate decision

**PASS.** We now have a causal vessel-state layer suitable for M0D Catania geometry and port-call reconstruction.

Important residual caveats:

1. semantic thresholds are engineering defaults, not universal maritime constants;
2. `observation_gap_candidate` is a provider-series gap, not proven loss of raw AIS communication;
3. `stale_risk_level` is a proxy, not source-message latency;
4. `motion_state` is an observation-level candidate state and must not be treated as a ground-truth voyage phase without M0D/M0E validation;
5. nav status and destination remain human-entered/operational signals of imperfect reliability.

## External methodological references

- USCG Navigation Center, *Class A AIS Position Report (Messages 1, 2, and 3)*: official decoded field semantics and Class-A reporting cadence.
- M3-Archimedes, *AIS-trajectory-annotation*: stop, slow-motion, turn, gap and noise event taxonomy; used as methodological inspiration only, not copied into this implementation.
