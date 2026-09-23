# M5 — Port generalization / stress tests

**Status:** `M5_GO_M6_SCOPE_LIMITED`  
**Locked final chronological calls scored:** 0

M5 does not fit a new predictor. It stress-tests the retained M2 route-physics model and the fixed M4 reliability/stability layer using only already-cross-fitted OOF data and the later M4 temporal-evaluation block.

## Main result

- **CATANIA:** <=24 h voyage-balanced MAE 13.6 -> 10.0 min (26.3% route gain) over 10 OOF calls.
- **AUGUSTA:** <=24 h voyage-balanced MAE 42.2 -> 37.8 min (10.4% route gain) over 30 OOF calls.

The route signal therefore survives separately in both seen ports. This is a *within-port* claim, not an unseen-port claim.

## Cold-vessel stress

- **CATANIA:** 6 cold-vessel calls, 14.6 -> 11.3 min (22.7% gain).
- **AUGUSTA:** 22 cold-vessel calls, 40.7 -> 32.6 min (20.0% gain).

Cold-vessel performance is therefore not the source of the aggregate route gain.

## Stress failures / scope limits

- Augusta's initial **E route-family** is a real stress failure: 10 calls and route-kNN degrades <=24 h MAE by 12.6% versus geodesic. Do not claim route-kNN improves every route family.
- Augusta **seen-vessel** subgroup (8 calls) degrades by 13.1%; MERSEY SPIRIT / multi-stage behavior is a major contributor. Repeated MMSI does not imply route stability.
- Catania has only 2 later M4 temporal-evaluation calls, so port-specific temporal uncertainty claims for Catania are underpowered.
- Augusta ETA ground truth is Levante-only; Scirocco entrance generalization is not established.
- Leave-one-port-out was intentionally **not attempted**: the retained route representation is anchored to port-specific gates, so transferring Augusta route geometry to Catania (or vice versa) would test an invalid geometry rather than true unseen-port generalization.

## Uncertainty by port

- **AUGUSTA:** pooled M4 interval whole-call coverage = 100.0% on 14 later HIGH-reliability calls. A port-only 90% calibration would use k=16/16 and q=888.0 min; because k=n, this is a small-N diagnostic dominated by the maximum calibration error, not a replacement for M4's pooled interval.
- **CATANIA:** pooled M4 interval whole-call coverage = 100.0% on 2 later HIGH-reliability calls. A port-only 90% calibration would use k=8/8 and q=57.0 min; because k=n, this is a small-N diagnostic dominated by the maximum calibration error, not a replacement for M4's pooled interval.

## Claim boundary

Supported: seen-port OOF route value, cold-vessel stress within Catania/Augusta, scoped M4 reliability/coverage diagnostics.  
Not established: unseen-port transfer, Augusta Scirocco, Siracusa/Santa Panagia, berth/all-fast ETA, reported-AIS-ETA superiority.

## Gate

M5 gate = **M5_GO_M6_SCOPE_LIMITED**. Core port and cold-vessel checks pass, but route-family heterogeneity and small port-specific uncertainty samples require scope-limited claims. Proceed to M6 without changing the predictor; the locked final chronological holdout is still untouched.

## External context

Recent reproducible AIS-ETA work explicitly notes that vessel-level and cross-port generalization require larger, dedicated validation rather than being inferred from row-level success. Recent port-trajectory research likewise models port phases with port-specific geospatial context. M5 follows that conservative interpretation instead of manufacturing a leave-one-port-out result from incompatible target geometries.

### Web references checked for M5 context

- *A Reproducible Workflow for AIS-Based ETA Forecasting: Evaluating the Influence of Data Preprocessing and Machine Learning Model Selection* (2026): https://www.mdpi.com/2571-9394/8/4/70 — explicitly notes that vessel-level generalization was not separately tested and calls for evaluation across ports/vessel categories/geographies.
- *A multi-phase trajectory approach for vessel behavior analysis in port areas by integrating AIS and geospatial data* (Ocean Engineering, 2026): https://www.sciencedirect.com/science/article/abs/pii/S0029801826033512 — reinforces that port-area semantics depend on port-specific geospatial context.
