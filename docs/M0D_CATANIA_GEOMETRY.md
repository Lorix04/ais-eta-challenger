# M0D — Catania geometry provenance and semantics

## What this geometry is

M0D does **not** claim that the project has recovered an official `ATA`, port-limit line, Pilot Boarding Place or harbour-master arrival line.

The primary spatial event is deliberately named:

`CATANIA_RESEARCH_GATE_INBOUND`

It is a research cross-section used to reconstruct auditable candidate port calls before the M0E manual audit.

## Authoritative sources used

1. **Istituto Idrografico della Marina / Elenco Fari** — Catania lights/aids:
   - 2802 / E1830, `Molo di Levante, estr`: published rounded position `37°29.0' N, 15°06.0' E`.
   - 2804 / E1832.5, `Nuova darsena traghetti, banchina n°31, estr`: published rounded position `37°29.3' N, 15°05.7' E`.
   - 2808 and 2812 additionally document internal Molo di Levante / Molo di Mezzogiorno aids and support the harbour layout.
   - Source used during M0D research: `Fari_22_22.pdf`, Istituto Idrografico della Marina.

2. **Autorità di Sistema Portuale del Mare di Sicilia Orientale** — official Catania port page and plan material:
   - named port areas include Nuova Darsena, Molo di Mezzogiorno, Molo Crispi, Sporgente Centrale, Molo di Levante and Piazzale Triangolare;
   - an official 1:2000 `Planimetria stato di fatto` visibly identifies Molo Foraneo, Porto Nuovo, Porto Vecchio, Sporgente Centrale, Molo di Mezzogiorno, Nuova Darsena Commerciale and berth numbering.

3. **2026 operational context**:
   - AdSP ordinances and the 2026–2028 operational plan document ongoing consolidation/maintenance of the Catania breakwater / Molo di Levante area in 2026.
   - This is a reason to avoid treating a hand-built polygon as immutable official navigation truth.

## Research gate v1

The gate connects two published hydrographic-aid coordinates:

- west anchor: `(15.0950, 37.4883333333)` — light 2804 rounded position;
- east anchor: `(15.1000, 37.4833333333)` — light 2802 rounded position.

This line intersects the strong inbound/outbound AIS corridor in the dataset and is used as a reproducible **cross-section proxy**.

It is explicitly **not** represented as:

- an official port-limit boundary;
- a Pilot Boarding Place;
- a legal harbour boundary;
- a company-defined ATA target.

If the company provides its actual target geometry, this proxy must be replaced and the event reconstruction rerun.

## Inner harbour proxy v1

The polygon in `config/catania_geometry_v1.geojson` is a conservative analysis envelope used only to ask:

> after an inbound gate crossing, did the vessel subsequently exhibit sustained low-speed / stopped behaviour on the port side?

The polygon was frozen before ETA model development and must never be tuned against ETA MAE.

## Roadstead / waiting semantics

M0D does not have an authoritative anchorage polygon for Catania. Therefore it does not label `ANCHORAGE` as ground truth.

Instead it exports `catania_roadstead_wait_proxy`, defined as low-speed / stopped or AIS-at-anchor behaviour on the sea side of the research gate, within a conservative radius from the gate. This is supporting evidence only.

## Event timing uncertainty

For every crossing the detector preserves:

- `crossing_last_before`;
- `crossing_first_after`;
- `crossing_uncertainty_s`;
- midpoint `crossing_time_proxy`.

The actual crossing lies within the bracket if the provider positions represent fresh observations. Because M0B/M0C found snapshot/staleness behaviour, this bracket can understate physical uncertainty; M0E therefore audits crossing quality rather than pretending second-level truth.

## Frozen-before-ML rule

Geometry version `catania_research_geometry_v1` is frozen for M0D/M0E. Changes require:

1. new geometry version;
2. documented external/domain reason;
3. rerun of candidate-event reconstruction;
4. no reference to downstream ETA test performance.