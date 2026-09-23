# M0E — Manual Catania event audit and effective sample size

**Status:** DONE. GO/NO-GO A = **GO_LIMITED** for M1 baselines only.

## Objective

M0E converts the 61 M0D candidate sessions into an explicit human-reviewed audit ledger. It separates two questions that must not be conflated:

1. **Is this a trustworthy AIS-derived Catania research-gate arrival event?**
2. **Does this vessel/event belong in the primary ETA benchmark population?**

A valid tug, fishing-vessel or yacht port call remains useful Port Intelligence ground truth, but it is not automatically mixed with commercial/operational voyage ETA training.

The retained event is deliberately named:

`ACTUAL_VALIDATED_CATANIA_RESEARCH_GATE_ENTRANCE`

It is **not** called official ATA, ATA berth, PBP arrival or all-fast.

## Audit protocol

All 61 candidate sessions were reviewed using:

- the finite-gate crossing bracket and direction from M0D;
- six hours of pre-entry trajectory context;
- post-entry port-side trajectory and inner-harbour stop behaviour;
- observed outbound where available;
- session observation coverage and major observation gaps;
- right-censoring vs unresolved missing outbound;
- entry-time destination support (never backfilled from future Message 5 state);
- vessel role/type as a population-scope check, not as proof of the event itself;
- a seven-page visual audit atlas covering every candidate.

Future observations are allowed here only to **confirm the historical label**. M0E fields are not prediction-time model features.

### Confidence rubric

- **A** — coherent inbound crossing plus prompt/continuous port-call confirmation; normal gaps do not affect event interpretation.
- **B** — event is still credible, but coverage gaps, missing outbound or sparse session evidence reduce confidence.
- **C / excluded** — no confirmed port stop, brief gate excursion, or insufficient contiguous evidence to defend the event as a call.

The manual decisions are versioned in `config/catania_m0e_manual_audit.csv`. Vessel-role scope is separately versioned in `config/catania_m0e_vessel_roles.csv`.

## Results

| Measure | Result |
|---|---:|
| M0D candidate sessions | 61 |
| Validated research-gate port-call entries | **51** |
| Excluded ambiguous/non-call events | **10** |
| Unique vessels among validated entries | 28 |
| Confidence A / B among kept events | **37 / 14** |
| Primary ETA cohort calls | **28** |
| Primary ETA cohort unique vessels | **13** |
| Primary ETA cohort cargo calls | 24 |
| Passenger / tanker / offshore-supply / survey-offshore | 1 each |
| Validated entries right-censored after arrival | 5 |
| Validated entries with missing outbound before dataset end | 7 |
| Median / P95 crossing bracket | **61 s / 62 s** |
| Validated sessions flagged sparse | 5 |

The primary ETA cohort therefore contains only **28 calls across 13 vessels**. This is the effective scale relevant to a Catania commercial/operational ETA benchmark, not 969k AIS rows.

## Why the ETA cohort is narrower than the truth set

AIS Message 5 ship-type codes distinguish, among others, fishing (30), sailing (36), pleasure craft (37), tug (52), cargo (7x) and tanker (8x). For the primary benchmark we exclude local port tugs, fishing vessels, sailing/pleasure craft and small reserved craft because their operating process is qualitatively different from the voyage-bearing commercial/operational vessels we are trying to challenge on ETA.

This is a **benchmark population decision**, not a claim that those excluded craft did not make real port calls.

A few missing/ambiguous static roles were checked externally on 2026-09-18:

- `BELLE L'ADRIATIQUE` (MMSI 205481000): passenger/cruise vessel;
- `BRITOIL CONFIDENCE` (MMSI 253000169): offshore tug/supply ship;
- `URBANO MONTI` (MMSI 247415100): survey vessel used for submarine-cable route surveys;
- `CARMELO PADRE I` (MMSI 247047710): fishing vessel;
- `KIND OF MAGIC` and `BELUGA V`: sailing vessels;
- `KERYLOS`: 11 x 3 m small/reserved craft;
- `MILLIWAYS`: treated conservatively as recreational/sailing based on its marina-oriented public AIS history; role confidence B.

These checks are used only for audit/population scope, never to backfill time-varying prediction features.

## Important audit findings

### Brief gate crossings are real false positives

`CT-047` and `CT-053` are large cargo vessels with Catania destination support, yet they spend only about 12 min and 3 min on the port side with no confirmed stop. They are excluded rather than being promoted merely because the vessel is commercial or the destination says Catania.

This is strong evidence that a simple `destination == Catania` or `point inside polygon` rule would contaminate the target.

### Missing outbound does not automatically invalidate the arrival label

`CT-054` (`EUROCARGO VALENCIA`) has a coherent inbound crossing and more than 11 h of confirmed stop evidence before the feed disappears. It is kept at confidence B. Conversely, rows with no prompt stop and severe discontinuity are excluded.

### Right censoring after arrival is not missing arrival ground truth

`CT-060` and `CT-061` enter Catania and show sustained stop behaviour before the dataset ends. Their outbound is censored, but the **entry event** is already observed. They remain valid arrival labels.

## GO/NO-GO A

**Decision: GO_LIMITED.**

Why not NO-GO:

- 51 defensible Catania arrival events exist for Port Intelligence;
- 28 belong to a coherent primary operational ETA cohort;
- the entry brackets are tight in provider observation time;
- repeated commercial vessels provide enough observations to establish physical baselines and expose leakage/validation behaviour.

Why not an unrestricted GO:

- 28 primary calls are only borderline for port-specific ML;
- they represent just 13 unique vessels;
- 24/28 are cargo, with several repeated Eurocargo/ECO vessels;
- a chronological final holdout will be high variance;
- model selection can easily overfit vessel identity or repeated service patterns.

Therefore M1 may proceed with **physics and simple baselines only**, using voyage grouping and vessel-holdout stress tests. High-capacity residual ML, extensive hyperparameter search and strong superiority claims remain blocked until M1 confirms usable headroom and/or additional port/history data increases independent sample size.

## Outputs

- `reports/catania_m0e_audited_calls.csv` — full 61-row audit ledger;
- `reports/catania_m0e_eta_primary_cohort.csv` — 28-call primary ETA cohort;
- `reports/catania_m0e_session_diagnostics.csv` — audit-only coverage/gap diagnostics;
- `reports/m0e_catania_summary.json` — machine-readable summary;
- `reports/m0e_audit_atlas/page_01.png` ... `page_07.png` — visual audit of all 61 candidates;
- `reports/m0e_catania_audit_examples.png` — representative decision figure.

## Sources used for role/domain checks

- USCG Navigation Center, AIS Class A Static and Voyage Related Data (Message 5), ship-type code definitions: https://www.navcen.uscg.gov/ais-class-a-static-voyage-message-5
- AdSP Mare di Sicilia Orientale, Technical Nautical Services (pilotage, towage, mooring, boat service): https://www.adspmaresiciliaorientale.it/servizi/servizi-tecnico-nautici/
- Orange Marine, `Urbano Monti` fleet page: https://marine.orange.com/en/fleet/urbano-monti/
- Public vessel databases were used only as secondary static-role checks for missing AIS type metadata; they are not ground truth for arrival timestamps.
