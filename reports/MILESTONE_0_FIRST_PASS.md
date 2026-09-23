# Milestone 0 - First forensic pass

Date: 2026-09-18

## What has been verified directly from `data.zip`

- 1,534,210 position rows total.
- 969,084 `ship` rows across 1,129 ship MMSIs.
- 563,971 AtoN rows across 46 AtoN MMSIs.
- 1,155 SAR rows across 3 MMSIs.
- Observation window: 2026-04-03 10:42:38 to 2026-04-16 12:43:53.
- 16,927 distinct `recorded_at` fleet timestamps.

## Strong evidence of periodic fleet snapshots / latest-state polling

The global sequence of distinct fleet timestamps is overwhelmingly periodic:

- 61 s between fleet snapshots: 14,293 times
- 60 s: 2,190
- 62 s: 416

At each timestamp, many independent MMSIs share exactly the same `recorded_at` value:

- median ship MMSIs per timestamp: 52
- 90th percentile: 83
- maximum: 151

Per-vessel row deltas are similarly concentrated:

- 61 s: 815,480 consecutive intervals
- 60 s: 98,955
- 62 s: 44,114

This is substantially different from the natural asynchronous cadence expected from raw AIS position messages. The most defensible current interpretation is a provider-side state table or latest-state endpoint sampled approximately every 60-61 seconds. This remains an inference until provider/source message metadata are available.

## State repetition is material

For ship rows:

- 47.40% of consecutive observations have exactly unchanged latitude and longitude.
- 46.53% repeat the full core state (position, SOG, COG, heading, nav status, destination and draught).
- 85,892 rows have unchanged position while reported SOG > 0.5 kn.
- 73,122 rows have unchanged position while reported SOG > 5 kn.

These rows cannot safely be treated as independent sensor updates.

## AIS sentinel / data-quality findings

- `heading == 511` in 26.45% of ship rows (AIS unavailable sentinel).
- `COG >= 360` in 10.02% of ship rows and must not be interpreted as a normal course angle.
- destination missing in 23.62% of ship rows.
- 523 distinct non-null destination strings are present.

Frequent destination aliases include `AUGUSTA`, `ITAUG`, `IT AUG`, `ITCTA`, `IT CTA`, `ITSPA`, `IT SPA`, and non-port semantics such as `IN PORT` and `FOR ORDERS`.

## `vessel_tracks.csv`

- 1,196 MMSIs, including 1,146 ships.
- 450 non-null ETA fields; 439 match the expected `MM/DD HH:MM` pattern.
- With nearest-year interpretation relative to `last_update`:
  - 61 ETA values are already >1 h in the past;
  - 51 are >24 h in the past;
  - 56 are >7 days in the future;
  - 28 differ from `last_update` by >30 days in absolute value.

Therefore the track snapshot ETA cannot be treated as ground truth and must never be backfilled onto historical position rows.

## Exploratory port structure - not ground truth

Low-speed clustering already reveals strong spatial structure:

- Catania: one dominant stationary cluster around approximately 37.49724 N, 15.09413 E involving 50 vessels.
- Augusta/Santa Panagia area: multiple distinct stationary clusters, confirming that a single circular 'Augusta port' target would collapse several operational areas.

A deliberately rough radius-based call detector produces many candidate episodes, including local/service traffic. These outputs are hypothesis-generation only and are intentionally **not** promoted to supervised labels.

## Current decision

Do not train ETA yet.

Next engineering task is to build a validated Catania semantic state machine around authoritative geometry and manually audit candidate calls. Only after that should `training_dataset` be created.
