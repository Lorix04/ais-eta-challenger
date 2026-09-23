# M2 — Train-only Route Model

**Status:** DONE — `GO_M3_ROUTE_SIGNAL_CONFIRMED`
**Target semantics:** validated research-gate entrance events only; not official PBP, ATA berth, or all-fast.

## Question

Does historical route geometry add reproducible ETA signal beyond the M1 geodesic-distance / robust-speed baseline, without using any validation voyage geometry to fit the route representation?

## Design

M2 deliberately avoids global route clustering. The route representation is fitted independently inside each expanding temporal fold and port.

- Catania keeps the six-call M1 chronological holdout untouched.
- Augusta locks its last ten calls as a separate chronological holdout.
- Development history is split into three expanding temporal validation blocks per port.
- Each fold route model uses only calls whose arrival occurs strictly before the validation block.
- The final holdout is never scored in M2.
- Prediction-time speed remains the causal 30-minute rolling median with a 1 kn floor.

The route experiment contains three distances:

1. **Geodesic:** direct distance to the versioned research gate.
2. **Sector prototype:** median train-only route skeleton using fixed approach sectors (`NNE`, `E`, `SSE`, `OTHER`).
3. **Route-kNN:** median remaining distance from the three historical train-only paths nearest the current vessel position.

No route hyperparameter search was performed. The route rings, sector boundaries, `k=3`, warm-up sizes, holdouts, and M2 gate were frozen before the formal run.

## How the empirical routes are built

For each training call, the pre-arrival trajectory is quality-filtered to remove invalid positions, position-jump candidates and HIGH stale-risk states. Repeated provider positions do not add geometry.

The path is represented by the **last well-observed inward crossing** of fixed distance rings around the target gate. This deliberately suppresses early loops and loitering when constructing a navigation corridor. Ring interpolation is accepted only across observation gaps <=10 minutes.

At prediction time the validation vessel is projected onto a route learned only from historical training calls. Estimated remaining route distance is:

`connector distance to route + along-route distance to gate`

and is floored by the geodesic distance so the empirical route cannot claim a physically shorter-than-straight path.

## Evaluation population

- Combined ETA calls frozen: **83**
- Locked final chronological calls: **16**
- Warm-up history calls: **23**
- Calls assigned to expanding OOF blocks: **44**
- Calls with causal inbound-approach decision points in OOF: **40**
- OOF unique vessels: **33**
- Cold-vessel OOF calls relative to their fold: **28**
- OOF prediction points: **556**
- Final-holdout prediction points used by M2: **0**

## Results

### Voyage-balanced MAE

| Scope | Geodesic | Sector prototype | Route-kNN | Route-kNN improvement vs geodesic |
|---|---:|---:|---:|---:|
| <=6 h | 0.382 h (22.9 min) | 0.343 h | **0.320 h (19.2 min)** | **16.2%** |
| <=12 h | 0.501 h (30.0 min) | 0.464 h | **0.432 h (25.9 min)** | **13.6%** |
| <=24 h | 0.584 h (35.0 min) | 0.546 h | **0.514 h (30.8 min)** | **11.9%** |
| all OOF | 5.270 h | 5.172 h | **5.154 h** | **2.2%** |

The large all-horizon MAE is still driven by a handful of very-long-horizon waiting / multi-stage cases. Route geometry cannot explain days of latent operational waiting.

### Cold-vessel stress test (<=24 h)

- Geodesic MAE: **0.586 h (35.1 min)**
- Route-kNN MAE: **0.467 h (28.0 min)**
- Improvement: **20.3%**

This matters because the route signal is not limited to memorising the same MMSI's previous calls.

### Fold consistency

Route-kNN has lower all-horizon voyage-balanced MAE than geodesic in **6/6 port x temporal folds**.

This is stronger evidence than a single pooled metric, although the absolute size of the gain varies substantially by fold.

## Visual route sanity audit

The fold-3 maps expose an important modelling lesson. A single port-wide median prototype can form a geometric compromise between distinct northern/eastern/southern route families and can therefore bend through areas that no individual vessel actually follows. The sector prototypes are more coherent, while route-kNN avoids averaging incompatible route families altogether. This visual finding is consistent with the quantitative result that route-kNN is the strongest M2 route representation.

These maps are diagnostic empirical AIS route maps, not nautical charts or legal route recommendations.

## Route-distance diagnostics

Across OOF decision points:

- median `route-kNN / geodesic` distance ratio: **1.128**;
- median kNN cross-track distance to retrieved historical paths: **2.11 km**.

The empirical route therefore adds a modest navigation-distance penalty rather than inventing radically longer paths.

## M2 gate

The pre-specified gate required:

- >=10% MAE improvement at <=24 h;
- route-kNN improvement in at least 2/3 of port x temporal folds;
- >=5% improvement for cold-vessel calls at <=24 h;
- no >5% overall OOF MAE degradation.

Observed:

- <=24 h improvement: **11.9%**;
- fold wins: **6/6**;
- cold-vessel <=24 h improvement: **20.3%**;
- overall OOF improvement: **2.2%**.

**Gate result: `GO_M3_ROUTE_SIGNAL_CONFIRMED`.**

## Interpretation

M2 supports a nuanced conclusion:

- historical route geometry contains real, train-only, out-of-fold signal;
- route-kNN is more useful than a single median prototype;
- the benefit is strongest in the navigation-dominated <=24 h regime;
- route distance does **not** solve long-horizon waiting/port-operations uncertainty;
- geodesic distance remains an important baseline and should remain available to M3;
- route-kNN should enter M3 as an alternative physical distance / feature, not be presented as a nautical ground-truth route.

This is compatible with recent maritime research that extracts regular AIS route patterns and uses movement-pattern similarity for route searching and ETA estimation. The project implementation remains intentionally simpler because the effective sample is only O(10^2) calls.

## Leakage / integrity audit

Verified automatically:

- no final-holdout call is scored;
- no validation call appears in its route training set;
- every route model's latest training arrival precedes the validation block;
- every route-kNN neighbor is a training call for that fold;
- no validation call retrieves itself;
- route distances are never below geodesic distance;
- all OOF predictions are finite.

## Next step

Proceed to **M3 — Physics + residual ML + port-state features**, still using shallow/regularized models and fold-local feature construction. The M2 result does not justify Transformers, sequence networks, or large hyperparameter searches.
