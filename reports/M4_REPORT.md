# M4 — Voyage-level uncertainty, ETA reliability and forecast stability

**Status:** `M4_CALIBRATED_HIGH_RELIABILITY_GO_M5`
**Point predictor retained:** M2 train-only route-kNN distance / robust trailing speed
**Rejected M3 residual model used:** no
**Locked final chronological calls scored:** 0

## 1. Why M4 is call-level, not row-level

The M2 OOF panel contains many sequential prediction rows per vessel call. Those rows are strongly dependent and cannot be treated as independent calibration examples. M4 therefore sorts the 40 existing M2 cross-fitted calls chronologically and freezes:

- **24 earliest calls** as uncertainty/stability calibration;
- **16 later calls** as M4 temporal evaluation;
- the separate **16 locked M2 final chronological calls remain untouched**.

The primary conformity score is one number per calibration call: the **maximum absolute ETA error across the call's HIGH-reliability prediction states**. This produces a conservative simultaneous band intended to cover the full supported prediction path of a new call rather than exploiting hundreds of pseudo-independent AIS rows.

Standard conformal coverage theory is strongest under exchangeability. Our evaluation is deliberately chronological, so the observed temporal coverage is reported as an empirical stress test rather than as a universal distribution-free guarantee under distribution shift.

## 2. Causal reliability tier

`reliability_tier` is a transparent diagnostic, **not a probability of correctness**. It uses only information available at prediction time.

A prediction is `HIGH` only when all of these hold:

1. route-physics point ETA is within the supported predicted horizon (`<=24 h`);
2. the contemporaneous AIS destination supports the target port;
3. train-only route-kNN cross-track distance is `<=5 km`;
4. the provider source observation is `<=90 s` old and not `HIGH` stale-risk;
5. robust trailing speed is at least `1 kn`.

`MEDIUM` means the prediction is inside the supported horizon but only two or three of the four quality checks pass. Everything else is `LOW`.

On the later 16-call temporal evaluation:

| Tier | Rows | Calls | Voyage-balanced MAE, all observed horizons | <=24 h voyage-balanced MAE |
|---|---:|---:|---:|---:|
| HIGH | 142 | 16 | **16.4 min** | **16.4 min** |
| MEDIUM | 132 | 12 | 21.31 h | 43.1 min |
| LOW | 5 | 2 | 81.41 h | no supported <=24 h rows |

The extremely poor all-horizon MEDIUM/LOW values are not a reason to hide those cases. They expose the exact failure mode of an AIS-only physical ETA: a vessel can pass near the route/gate long before the audited call while target intent is not contemporaneously confirmed.

## 3. Voyage-level prediction interval

The primary interval is calibrated only for `HIGH` reliability. For each calibration call:

\[
S_v = \max_{t \in v,\;R_t=HIGH}
|T^{true}_t-\hat T^{M2}_t|.
\]

For nominal coverage `1-alpha`, the finite-sample split-conformal order statistic is:

\[
k=\lceil(n+1)(1-\alpha)\rceil,
\]

with `n=24` call-level scores.

### Sensitivity

| Nominal | Half-width | Temporal whole-call coverage | Voyage-balanced point coverage |
|---:|---:|---:|---:|
| 80% | 32.7 min | 50.0% | 81.3% |
| 85% | 57.0 min | 75.0% | 96.7% |
| **90%** | **149.1 min** | **100.0%** | **100.0%** |

The retained primary M4 band is therefore the **90% simultaneous call-level band**:

\[
\hat T \pm 2.485\text{ h},
\]

with the lower remaining-time bound clipped at zero. The nominal full width is 4.97 h; the observed mean effective width is 3.70 h because predictions close to arrival cannot extend below zero remaining time.

This interval is intentionally conservative. With only 24 calibration calls, the 90% quantile is the 23rd ordered call score, so one or two difficult voyages materially affect width. M4 does **not** attempt per-port or fine horizon-conditional conformal calibration because the independent-call support is too small.

## 4. ETA stability and causal smoothing

The operational object that should be stable is the **absolute predicted arrival timestamp**:

\[
\hat A_t=t+\hat T_t.
\]

M4 measures the absolute revision between consecutive predictions and ignores transitions across observation gaps longer than 2 h.

A tiny causal EWMA grid over absolute ETA timestamps was selected **only on the 24 calibration calls**: `alpha in {1.0, 0.8, 0.6, 0.4, 0.2}`. Selection minimized calibration voyage-balanced MAE at <=24 h, using P90 revision only as a tie-breaker. The selected value is **alpha=0.4**.

On the later 16 calls:

| Metric | Raw M2 route physics | Causal EWMA alpha=0.4 |
|---|---:|---:|
| <=24 h voyage-balanced MAE | 29.1 min | **28.5 min** |
| Median ETA revision | 5.55 min | **4.37 min** |
| P90 ETA revision | 39.1 min | **22.2 min** |
| Revisions >30 min | 11.8% | **8.3%** |
| Revisions >60 min | 6.7% | **4.7%** |

Within the `HIGH` reliability subset, P90 revision improves from **26.4 min to 11.0 min**, while <=24 h MAE improves from **16.4 min to 15.7 min**.

M4 therefore retains the causal smoother as an **optional presentation/stability layer**, not as a replacement ML model.

## 5. Reported AIS ETA discrepancy

The supplied `vessel_tracks.csv` contains a single latest-state ETA snapshot, not a historical Message-5 ETA series. M4 performs an explicit same-MMSI timestamp alignment check before using it.

Result on the M4 temporal evaluation:

- ETA rows aligned within **5 min:** 0;
- ETA rows aligned within **60 min:** 0;
- closest same-MMSI snapshot with an ETA: **455.95 min** away.

Therefore **no AIS-reported ETA discrepancy metric is produced**. Backfilling the snapshot ETA onto earlier prediction times would be future leakage.

## 6. Gate decision

Pre-specified M4 gate for the primary HIGH-reliability simultaneous interval:

- at least 12 independent temporal-evaluation calls;
- whole-call coverage >=90%;
- half-width <=3 h;
- no locked-final scoring.

Observed:

- 16 calls;
- 100% whole-call coverage;
- 2.485 h half-width;
- 0 locked-final calls scored.

**Decision: `M4_CALIBRATED_HIGH_RELIABILITY_GO_M5`.**

This is a scoped result. It does **not** mean uncertainty is calibrated for ambiguous destination intent, very long horizons, berth arrival or all-fast. In those regimes M4 explicitly withholds the primary interval rather than presenting false precision.

## 7. Outputs

- `m4_call_roles.csv` — 24 calibration / 16 temporal-evaluation calls;
- `m4_calibration_call_scores.csv` — one conformity score per calibration call;
- `m4_interval_sensitivity.csv` — 80/85/90% whole-call sensitivity;
- `m4_reliability_metrics.csv` — HIGH/MEDIUM/LOW diagnostics;
- `m4_stabilizer_calibration_grid.csv` — causal smoothing selection on calibration only;
- `m4_stability_metrics.csv` — later temporal stability evaluation;
- `m4_temporal_eval_predictions.csv` — point ETA, reliability, interval and causal smoothed ETA;
- `m4_reported_eta_nearest_snapshots.csv` — proof that the supplied ETA snapshot is not temporally aligned;
- `m4_summary.json` / `m4_gate_result.json` — machine-readable gate.

## 8. Next step

Proceed to **M5 port generalization / stress tests**. The next question is not whether to add a larger model; it is whether performance, reliability and interval behavior remain coherent when separated by port, entrance complexity, vessel novelty and difficult operating regimes.
