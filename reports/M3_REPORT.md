# M3 — Physics + residual ML + port-state ablation

**Status:** DONE — `M3_COMPLEXITY_NOT_JUSTIFIED`  
**Retained production candidate after M3:** M2 train-only route-kNN physical ETA.  
**Rejected for now:** residual Ridge/Huber/LightGBM correction and AIS-only port-state feature block.

## Question

After M2 established reproducible train-only route signal, does a shallow tabular model add stable voyage-level ETA accuracy beyond:

`T_physics = D_route-kNN / robust trailing speed`?

A second question was whether synchronized AIS snapshots contain enough port-state information to improve that residual model.

## Why M3 uses a second cross-fitted stack

M3 does **not** train a residual learner on in-sample route predictions. It starts from the 40 M2 out-of-fold calls. Every M2 route-kNN prediction used here was already generated from a route model that excluded the validation call.

Those 40 calls are then ordered by arrival time:

- first 10 calls: stacking warm-up;
- next 30 calls: three contiguous expanding temporal validation blocks of 10 calls each;
- training sizes are therefore 10, 20 and 30 calls;
- 16 locked M2 chronological final-test calls remain completely untouched.

This is conservative but avoids giving the residual learner a base prediction that was artificially optimized using the same call's future path.

## Prediction-time features

The core feature set contains no raw MMSI, vessel name, session id, ground-truth confidence, true remaining time or horizon label.

Families:

- M2 cross-fitted route-physics ETA;
- geodesic and route-kNN remaining distance;
- route/geodesic ratio and kNN cross-track distance;
- trailing 30-minute SOG and current SOG;
- causal 30-minute progress to gate;
- COG-to-gate angular difference;
- stale-state / provider-age information;
- stop duration, turn flag and semantic motion state;
- draught and navigational status;
- conservative current destination-support flag;
- port / entrance / route-family categorical context;
- time-of-day sine/cosine.

Training weights are equal by call and also balance the broad horizon bands occupied by each call, preventing long minute-by-minute traces or the final approach from dominating the loss.

## Port-state ablation

A separate feature block is constructed only from vessels observed in the **same provider snapshot** as the prediction source observation:

- vessel counts within 5 / 15 / 30 km of the port reference;
- stopped vessels within 5 / 15 km;
- slow and moving vessels within 15 km;
- slow/stopped pressure within 15 km.

The target vessel is subtracted from the counts. These are explicitly **AIS traffic proxies**, not berth availability or a true queue.

## Fixed candidate models

No hyperparameter sweep was run.

1. direct Ridge;
2. route-physics + residual Ridge;
3. route-physics + residual Huber;
4. route-physics + shallow residual LightGBM;
5. the same residual LightGBM + port-state proxies as a separate ablation.

LightGBM is fixed at shallow depth / small leaves with L1 regression loss and regularization. Validation targets are never used for early stopping.

## Evaluation population

- M2 cross-fitted calls available to M3: **40**;
- stacking warm-up: **10 calls**;
- M3 OOF validation: **30 calls**;
- M3 OOF unique vessels: **26**;
- prediction rows: **420**;
- cold-vessel OOF calls relative to their M3 fold: **23**;
- locked final chronological calls scored: **0**.

## Main result

### Voyage-balanced MAE

| Model | <=6 h | <=12 h | <=24 h | All OOF |
|---|---:|---:|---:|---:|
| **M2 route physics** | **0.291 h / 17.5 min** | **0.430 h / 25.8 min** | **0.539 h / 32.3 min** | **6.067 h** |
| Direct Ridge | 4.071 h | 4.369 h | 4.485 h | 7.945 h |
| Residual Ridge | 4.102 h | 4.398 h | 4.504 h | 7.967 h |
| Residual Huber | 2.394 h | 2.649 h | 2.756 h | 6.094 h |
| Residual LightGBM | 0.440 h | 0.576 h | **0.674 h / 40.4 min** | 6.214 h |
| Residual LightGBM + port-state | 0.420 h | 0.567 h | 0.667 h / 40.0 min | 6.206 h |

The best core ML candidate is residual LightGBM, but it is **25.0% worse** than route physics at <=24 h.

### Fold consistency at <=24 h

| M3 temporal fold | Route physics | Residual LightGBM | + port-state |
|---:|---:|---:|---:|
| 1 | **0.782 h** | 1.171 h | 1.116 h |
| 2 | **0.255 h** | 0.290 h | 0.296 h |
| 3 | 0.580 h | **0.559 h** | 0.589 h |

Residual LightGBM wins only **1 / 3** folds.

### Cold-vessel stress test

At <=24 h:

- route physics: **0.653 h / 39.2 min**;
- residual LightGBM: **0.813 h / 48.8 min**;
- relative change: **24.6% worse**.

### Paired call bootstrap

For the 30 M3 OOF calls at <=24 h, define call-level gain as:

`MAE(route physics) - MAE(residual LightGBM)`.

Observed:

- mean gain: **-0.135 h (-8.1 min)**;
- 95% cluster/call bootstrap interval: **[-0.275 h, -0.012 h]**;
- bootstrap probability that mean gain is positive: **1.25%**.

In this development population, the evidence points toward degradation rather than merely inconclusive improvement.

## Port-state result

Port-state proxies slightly improve the residual LightGBM itself:

- residual LightGBM <=24 h: 0.674 h;
- + port-state: 0.667 h;
- relative gain: **0.95%**.

But the pre-specified port-state gate required >=2% <=24 h gain and improvement in >=2/3 temporal folds. Observed fold wins: **1/3**.

**Decision:** `DROP_PORT_STATE_FOR_NOW`.

This does not prove port congestion is unimportant. It says that these simple AIS snapshot proxies, over only ~13 days, do not provide sufficiently stable incremental signal in the current sample.

## M3 gate

Pre-specified requirements:

- >=5% <=24 h MAE improvement over route physics;
- win >=2/3 temporal folds;
- >=5% cold-vessel <=24 h improvement;
- no >5% all-horizon degradation;
- positive gain must not be dominated by a single call.

Observed for the best core candidate:

- <=24 h improvement: **-25.0%**;
- fold wins: **1/3**;
- cold-vessel improvement: **-24.6%**;
- all-horizon improvement: **-2.4%**;
- top positive-call gain share: **27.0%**.

**Gate: `M3_COMPLEXITY_NOT_JUSTIFIED`.**

## Interpretation

M3 is a useful negative result.

The M2 route-physics model already captures most of the reliable navigation-dominated signal available in this short dataset. The residual learner has only tens of independent calls and sees heterogeneous behaviours, including operational waiting that is not identified by AIS motion features. Added flexibility therefore increases variance and hurts temporal/cold-vessel generalization.

The correct engineering response is **not** to tune LightGBM harder, try 300 Optuna trials or move to a Transformer. The project's stop rule says that a complexity increase without stable voyage-level OOF gain must be documented and rejected.

Recent maritime ETA work shows that tree ensembles can be strong when datasets, labels and operational features support them; other physics-informed maritime work also motivates combining physical structure with data-driven correction. Those papers justify testing the hypothesis, not assuming the correction must help this dataset. Our M3 experiment falsifies that hypothesis for the current sample/protocol.

## Retained model after M3

For the current take-home evidence base, retain:

`ETA = train-only route-kNN remaining distance / robust trailing speed`

with geodesic physics as the simpler reference baseline.

Do **not** retain the M3 residual model or the port-state block in the primary point predictor.

## Next step

Proceed to **M4 uncertainty / reliability / stability using the retained M2 route-physics predictor**, not the rejected M3 ML model.

This is consistent with the intent of the M3 gate: uncertainty should be built around the model that actually survived grouped temporal validation. M4 must still keep the locked final calls untouched until the final evaluation protocol is frozen.
