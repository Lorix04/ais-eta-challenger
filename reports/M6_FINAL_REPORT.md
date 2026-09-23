# M6 — Final locked chronological evaluation and take-home conclusion

## Freeze discipline

The final design was frozen before scoring. The first execution attempt failed during dataframe preparation before any holdout prediction was persisted; the failure is preserved in `M6_PREOPEN_FAILURE.md`. A second immutable freeze was created, then the chronological holdout was opened once. No point-model, route, reliability, uncertainty, stabilizer, target or scope parameter was changed after final results were visible.

## What was actually tested

- 16 locked chronological calls existed: 10 Augusta and 6 Catania.
- Under the predeclared causal `scope_inbound_approach`, **13 calls were scorable**: 10 Augusta and 3 Catania.
- Three Catania calls had zero rows satisfying the frozen inbound-approach scope. They were not rescued post-hoc; they are deployment-coverage failures recorded in `m6_final_call_coverage.csv`.
- All scorable final predictions are <= 6.45 h from the research-gate target; the final set contains no >24 h evidence.

## Final point-forecast result

Primary metric: voyage-balanced MAE at <=24 h.

| Scope | Calls | Geodesic | Frozen route-kNN | Relative change |
|---|---:|---:|---:|---:|
| All scorable final | 13 | 37.7 min | **36.5 min** | +3.4% |
| Augusta | 10 | 45.8 min | 45.7 min | +0.1% |
| Catania | 3 | 11.0 min | 5.6 min | +48.9% |

The overall route gain is small (3.4%). The call-level paired bootstrap mean gain is 1.3 min with 95% interval [-2.5, 4.7] min, so the final sample **does not resolve a non-zero overall route advantage statistically**. Catania is strongly positive but only 3 scorable calls; Augusta is essentially neutral.

## Reliability and uncertainty

Frozen HIGH reliability has 12 final calls and <=24 h raw MAE 37.6 min. MEDIUM has 5 calls and 76.9 min. This supports reliability as a useful diagnostic regime separator, not as a probability of correctness.

The frozen M4 90% simultaneous call-level interval on HIGH rows achieved:
- row coverage: 89.2%
- voyage-balanced point coverage: 94.6%
- whole-call coverage: **91.7% (11/12 calls)**
- fixed nominal half-width: 149.1 min.

Given only 12 final HIGH calls, this is encouraging calibration evidence, not a universal coverage guarantee.

## Stability

The frozen causal EWMA reduces final P90 absolute-ETA revision from 27.4 to 16.3 min, but worsens <=24 h point MAE from 36.5 to 38.7 min. Therefore the raw M2 ETA remains the primary accuracy output; stabilization is an optional presentation/operations layer when lower jitter is worth a modest accuracy trade-off.

## Reported AIS ETA

The final holdout still cannot support a fair model-vs-reported-ETA comparison. There are 0 snapshot rows aligned within 60 min of final prediction observations; the closest same-MMSI reported-ETA snapshot is 128.7 min away. Backfilling it would be leakage.

## What the final holdout changed

It did **not** justify changing the model. It changed the strength of our claims:
1. route-kNN remains a defensible retained point model because development OOF was consistently positive and final overall MAE is not worse, but the locked final advantage is small and uncertain;
2. the reliability layer transfers usefully;
3. the simultaneous interval lands close to its nominal whole-call coverage on a small final sample;
4. the stabilizer is a jitter-reduction trade-off, not a final-accuracy improvement;
5. frozen operational coverage is incomplete: 3/16 calls had no eligible inbound-approach decision row;
6. long-horizon, unseen-port, Scirocco, berth/all-fast and reported-ETA-superiority claims remain unsupported.

## Recommended take-home message

The strongest contribution is not a claim that a complicated learner beats the company's ETA. It is a leakage-resistant maritime forecasting pipeline that reconstructs auditable arrival events, diagnoses provider snapshots, uses train-only historical route geometry, rejects ML complexity when it fails OOF, emits uncertainty only in a transparent reliability regime, quantifies forecast stability, and preserves failure/coverage cases rather than hiding them.

## Data to request next

1. exact internal ETA target and official ground-truth event;
2. historical Message 5 ETA/destination with source timestamps;
3. at least several months of AIS history;
4. PCS / berth / pilotage / movement event records;
5. provider timestamp semantics and raw message timestamp/type if available.

With those data, the next scientifically meaningful experiment is long-horizon and port-operations modelling—not a larger neural network on the same 13-day sample.
