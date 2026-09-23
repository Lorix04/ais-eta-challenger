# Company alignment — final clarification applied in M10

Date: 2026-09-21

The company provided a final clarification after M9:

> For the exercise, use the ETA present in the dataset as the ground-truth/reference value; do not reconstruct actual arrival from Positions and no port-area polygon is required.

## Consequence

The primary exercise target is now **`Tracks.eta`**, treated as `reference_eta` throughout M10. This supersedes the M9 assumption that the physical destination-port-area event had to be reconstructed for the main exercise.

The earlier M0–M9 physical-arrival work remains valid as a separate research/challenger branch, but it is no longer the main ground truth used for the requested benchmark.

## File semantics retained from the previous clarification

- `Positions`: historical AIS observations for each vessel.
- `Tracks`: latest available vessel state.
- `recorded_at`: timestamp associated with each Positions observation; upstream generation/reception semantics remain unknown.
- historical destination: available in Positions.
- historical ETA: not available; ETA exists only in Tracks latest state.
- no longer AIS history is available for this exercise.

## Target wording

Because AIS Message 5 defines ETA as an **Estimated Time of Arrival** (`MMDDHHMM UTC`) and does not include a year, M10 uses the wording **company reference ETA** rather than Actual Time of Arrival. The nearest valid calendar year around `Tracks.last_update` is inferred deterministically for timestamp arithmetic.

## Benchmark policy

- one supervised row per MMSI/Tracks record;
- ETA never enters the feature matrix;
- Positions history is restricted to `recorded_at <= Tracks.last_update`;
- stale/past/far-future references are diagnosed explicitly;
- final model selection is calibration-only;
- M6 frozen physical-arrival results are preserved and not rewritten.
