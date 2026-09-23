# M0C clean-state data contract

This document defines the **causal vessel-state layer** produced by `scripts/04_m0c_semantic_states.py`.

## Scope

M0C does not create ETA labels, port calls or berth truth. It preserves every ship snapshot and adds normalized AIS fields plus past-only semantic motion candidates. Repeated provider states are **not dropped**, because their duration is useful for dwell and stale-state diagnostics.

## Raw vs normalized fields

Raw source columns remain unchanged. Normalized companion fields are added:

| Raw | Normalized | Rule |
|---|---|---|
| `sog` | `sog_clean` | `102.3` (decoded raw 1023) is unavailable; `102.2` is retained as the capped "102.2 kn or higher" value |
| `cog` | `cog_clean` | values >= 360° are unavailable |
| `heading` | `heading_clean` | 511 is unavailable |
| `lat/lon` | `lat_clean/lon_clean` | outside valid WGS84 range and AIS unavailable encodings are missing |
| `draught` | `draught_clean` | <=0 is unavailable |
| `nav_status` | `nav_status_clean`, `nav_status_name` | status 15/unknown is not treated as an operational state |
| `destination` | `destination_clean` | uppercase/trim only; **not** resolved to a port in M0C |

The decoded sentinel rules are based on AIS message semantics. They are normalization rules, not statistical imputations.

## Clocks and state persistence

- `recorded_at`: provider observation clock available in this dataset.
- `observation_dt_s`: elapsed provider-observation time since the previous row for the MMSI.
- `dynamic_state_hash`: deterministic identifier of the cleaned dynamic state `(lat, lon, SOG, COG, heading, nav_status)`.
- `dynamic_state_changed`: true when the current dynamic state differs from the preceding observed state.
- `last_dynamic_state_change_at`: most recent observed dynamic-state change **at or before** the current timestamp.
- `dynamic_state_age_s`: elapsed time since that observed state change.

`dynamic_state_age_s` is **not AIS message age**. The original message-generation timestamp is unavailable.

## Causal movement features

- `position_delta_m`: haversine distance from the previous observed position.
- `implied_speed_kn`: position delta divided by observation interval.
- `cog_delta_deg`: circular COG difference from the previous valid COG.
- `sog_delta_kn`: absolute SOG change from the previous valid SOG.

No centered windows, interpolation from a future point, future route or post-event information is used.

## Semantic candidates

These flags are candidate annotations, inspired by the event taxonomy used in maritime trajectory work. They are **not universal maritime truth** and are not port-call labels.

Default thresholds are held in `M0CConfig` and are deliberately explicit:

- `stop_candidate`: reported SOG <= 0.5 kn and implied speed <= 0.5 kn; provider observation gaps break episode continuity, so dwell time is never silently bridged across an unobserved interval;
- `slow_motion_candidate`: 0.5 < reported SOG < 3 kn and implied speed < 3 kn;
- `moving_candidate`: reported SOG >= 3 kn and implied speed >= 0.5 kn;
- `turn_candidate`: circular COG change >= 25° within <=180 s, with both speeds >=1.5 kn;
- `observation_gap_candidate`: provider-observation gap >180 s;
- `kinematic_conflict`: reported SOG >=2 kn while implied speed <=0.2 kn within <=180 s;
- `position_jump_candidate`: implied speed >80 kn within <=600 s.

The gap flag means **gap in this provider observation series**, not proven loss of raw AIS communication.

`STOP_START/STOP_END`, `SLOW_START/SLOW_END`, `TURN`, `OBS_GAP`, `KIN_CONFLICT` and `POS_JUMP` are emitted in `semantic_events` for auditability.

## Stale-risk proxy

`stale_risk_level` is a deterministic `LOW/MEDIUM/HIGH` diagnostic, not a probability and not raw message age. It combines repeated-state age and kinematic contradictions while protecting plausible long stationary periods when AIS status is `AT_ANCHOR` or `MOORED` and SOG is near zero.

## Causality contract

For any MMSI prefix ending at timestamp `t`, all M0C features for rows <= `t` must be identical whether the function receives only that prefix or the full future trajectory. Unit and integration tests enforce this prefix-invariance property.

## Local full-data output

`data/derived/m0c_ship_states.pkl.gz` is generated locally and intentionally excluded from Git because `data/` is private/large. Reproducibility comes from the script, config and reports committed to the repository.
