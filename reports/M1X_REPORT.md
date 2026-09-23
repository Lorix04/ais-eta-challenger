# M1X — Augusta ground-truth expansion and evidence gate

**Status:** complete  
**Target semantics:** `ACTUAL_VALIDATED_AUGUSTA_RESEARCH_GATE_ENTRANCE`  
**Important:** this is a versioned AIS-derived **research-gate** event. It is not claimed to be an official ATA, Pilot Boarding Place, port-limit event, berth arrival or all-fast timestamp.

## Why M1X existed

M1 on Catania found a strong simple <6 h physical baseline but too little independent long-horizon evidence: 28 primary ETA calls, 13 unique vessels, only 2 calls with any causal inbound-approach evidence beyond 6 h. The correct next move was therefore to expand ground truth rather than increase model capacity.

Augusta is a useful expansion port because the official port description separates outer and inner roadstead/Porto Megarese and documents two entrances. The Istituto Idrografico Portolano names the entrances Levante and Scirocco and gives distinct navigation rules. This makes Augusta a materially different operational regime from Catania.

## Geometry and provenance

The frozen geometry is documented in `docs/M1X_AUGUSTA_GEOMETRY.md` and `config/augusta_geometry_v1.geojson`.

- **Levante:** research cross-section anchored to rounded Istituto Idrografico light-list positions near the Diga Settentrionale / Diga Centrale opening.
- **Scirocco:** explicit research proxy anchored to official breakwater reference coordinates. It is **not** represented as an official regulatory entrance line.
- **Inner Porto Megarese proxy:** analysis-only polygon used to confirm port-side stops; not legal port limits.

Any retained Scirocco event is capped at confidence **B** until authoritative entrance-line geometry is obtained.

## Candidate reconstruction

From the M0C causal state layer:

- 361,191 ship observations fell inside the Augusta analysis envelope;
- 134 vessels were observed in that envelope;
- 152 finite research-gate crossing events were detected across 76 vessels;
- 79 candidate sessions were reconstructed across 62 vessels;
- 75 candidate inbound events used Levante and 4 used Scirocco.

Candidate sessions remained non-ground-truth until the audit.

## Manual/structured audit

All **79** candidates were reviewed through the nine-page trajectory atlas plus confirmation-continuity diagnostics.

Audit principles were frozen around the target semantics:

1. a gate crossing alone is insufficient;
2. a persistent port-side stop must be confirmed within the reconstructed session;
3. confirmation that only appears after a very long observation gap is not trustworthy enough;
4. a missing outbound does not automatically invalidate the entry if post-entry confirmation is contiguous;
5. Scirocco retained events are confidence B at most because the gate is a research proxy.

Outcome:

- **60** validated Augusta research-gate entries;
- **19** excluded ambiguous/non-call candidates;
- retained confidence: **33 A / 27 B**;
- retained entrance mix: **57 Levante / 3 Scirocco**;
- median crossing bracket: **61 s**; P95 approximately **61 s** on the provider observation clock.

Main exclusion causes:

- 8 candidates: no confirmed stop within the reconstructed session;
- 10 candidates: confirmation required a long observation gap;
- 1 candidate: unnamed + sparse + unresolved confirmation.

## ETA-feasibility cohort

Port-call truth and ETA scope are deliberately separate.

Instead of inventing vessel-type labels for every Augusta MMSI, M1X uses a conservative **behavioral long-range approach scope**. A validated call is primary-ETA eligible only when:

- ground-truth confidence is A/B;
- the vessel is named;
- at least 30 pre-entry observations exist in the previous 6 h;
- the observed pre-entry track reaches at least 15 km from the target entrance;
- the MMSI is not already known from the Catania audit to be a local/non-primary role.

This produces:

- **55 Augusta ETA-eligible calls**;
- **46 unique Augusta ETA vessels**;
- top-five vessel share only **21.8%**, far less concentrated than the Catania-only cohort;
- all 55 primary Augusta calls enter through Levante; the three retained Scirocco calls remain useful Port Intelligence truth but do not satisfy the long-range ETA scope.

The scope is operational/behavioral, not a claim that every included ship belongs to a specific commercial class.

## Causal decision-time coverage

A 15-minute UTC wall-clock panel was built using only the latest state available at or before each decision time. Input quality checks and progress toward the **call-specific target gate** are causal. Prefix-invariance passed **10/10** spot checks.

Augusta alone:

- 55 calls with causal inbound-approach support;
- **10** calls with some support >6 h;
- **5** calls >12 h;
- **3** calls >24 h.

Catania + Augusta:

- **83 ETA-eligible calls**;
- **58 unique vessels**;
- **74 calls** with at least one causal inbound-approach decision point;
- **12 calls** with >6 h approach support;
- **7 calls** with >12 h support;
- **5 calls** with >24 h support.

Only one MMSI appears in primary ETA cohorts at both ports, so the expansion genuinely increases vessel diversity rather than just duplicating the same services.

## Pre-specified M1X gate

Before evaluating the combined decision panel, the engineering sufficiency gate was frozen as:

- at least 50 combined primary ETA calls;
- at least 25 unique vessels;
- at least 10 calls with causal approach evidence >6 h;
- at least 5 calls with causal approach evidence >12 h.

These are project engineering thresholds, **not scientific constants or statistical guarantees**.

Observed:

- 83 calls — PASS;
- 58 unique vessels — PASS;
- 12 calls >6 h — PASS;
- 7 calls >12 h — PASS.

**Gate result: `GO_M2`.**

This does not justify high-capacity ML. It only justifies the next planned experiment: a train-only route representation compared against the geodesic baseline.

## Leakage controls

M1X explicitly checks that:

- source observation time never exceeds decision time;
- decision time is always earlier than ground truth;
- decision grid is wall-clock anchored, not ATA-anchored;
- current distance/progress use only observed prefixes;
- route modelling has **not** yet been fitted;
- post-entry information is used only for ex-post label validation;
- prefix-invariance holds on sampled Augusta calls.

## Verification

- `pytest`: 34 passed;
- M1X verification: 16/16 checks passed;
- prefix invariance: 10/10;
- all 79 audit decisions present;
- no unconfirmed-stop candidate retained;
- retained Scirocco confidence capped at B.

## Decision

Proceed to **M2 — train-only route model**.

M2 remains intentionally narrow: compare geodesic remaining distance with a port-specific route skeleton / empirical route prototype fitted inside training folds only. Residual boosting and port-state ML remain blocked until route-aware distance demonstrates stable voyage-level value.
