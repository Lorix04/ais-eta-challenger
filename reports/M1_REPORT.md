# M1 — Catania ETA baseline feasibility

**Status:** complete
**Gate outcome:** `EXPAND_GROUND_TRUTH_BEFORE_COMPLEXITY`
**Ground-truth event:** `ACTUAL_VALIDATED_CATANIA_RESEARCH_GATE_ENTRANCE` (research event, not official ATA/PBP/berth/all-fast)

## 1. Question M1 answers

M1 does **not** ask which ML algorithm wins. It asks whether the 28-call Catania primary cohort supports a defensible ETA benchmark beyond simple physics, and whether there is enough independent evidence to justify route/residual ML.

The design intentionally stays under the M0E complexity ceiling:

- no high-capacity ML;
- no future-informed fixed-horizon sampling;
- no route prototype fit on all voyages;
- no use of the locked chronological final-test metrics for model selection.

AIS2ETA/Evmides et al. is used only as a methodological reference for keeping points from the same route/call together during evaluation. Their data scale and target are not transferred to this project.

## 2. Frozen evaluation design

### 2.1 Call-level split

The 28 audited primary ETA calls are ordered by ground-truth entrance time.

- **Development:** first 22 calls, 13 unique vessels.
- **Locked chronological final test:** last 6 calls, 5 unique vessels, 12–16 April 2026.
- **Cold-vessel warning:** all 6 final-test calls belong to vessels already present in development. The chronological test therefore measures future repeated-service traffic, not unseen-vessel generalization.

Within development, two fold assignments are frozen:

- 4-fold `session_cv_fold`, all rows from one call kept together;
- 4-fold `vessel_cv_fold`, all calls from the same MMSI kept together.

The top five vessels account for **64.3%** of all 28 calls. Six vessels appear only once.

## 3. Causal prediction-time panel

Predictions are scheduled on a **15-minute UTC wall-clock grid**. The grid is not anchored to the true arrival time, so selecting a row does not encode the target horizon.

At each decision time, the feature snapshot is the latest provider observation at or before that time. A prediction-time row is considered input-quality-eligible only when:

- provider observation age <= 180 s;
- M0C stale-risk is not `HIGH`;
- no kinematic-conflict flag;
- no position-jump flag;
- valid position;
- vessel is outside the research gate.

Three causal scopes are retained separately:

1. `scope_conditional_target_known`: destination Catania is assumed known from an external planning context;
2. `scope_declared_destination`: the **current** AIS destination string supports Catania;
3. `scope_inbound_approach`: current state is fresh, within 120 km, and 30-minute past-only progress toward the gate is >=0.5 kn.

The third scope is the cleanest AIS-only ETA scope, but it covers fewer calls.

### Cadence sensitivity

15 and 30 minutes were both allowed by the workflow before M1. Development-only sensitivity gives:

| cadence | approach calls | approach points | <6h calls | <6h robust MAE | <6h P90 |
|---:|---:|---:|---:|---:|---:|
| 15 min | 16 | 80 | 16 | **12.1 min** | **25.9 min** |
| 30 min | 11 | 38 | 11 | 19.7 min | 51.1 min |

The 15-minute cadence is retained because it preserves more independent calls in the final approach; the final chronological test remains unscored.

## 4. Physical baselines

Distance is the static geodesic distance to the frozen Catania research-gate midpoint. This is intentionally still a weak route model; M2 would be responsible for train-only route geometry if data support it.

Baselines:

- `B0`: geodesic distance / current SOG, only when SOG >=0.5 kn;
- `B1`: geodesic distance / max(current SOG, 1 kn);
- `B2`: geodesic distance / max(trailing 30-min median SOG, 1 kn).

The 30-minute median is past-only and passes prefix-invariance checks.

## 5. Main development results

### 5.1 Final approach (<6h), causal inbound-approach scope

For `B2`, on **16 independent development calls**:

- voyage-balanced MAE: **12.1 min**;
- voyage-balanced median absolute error: **5.8 min**;
- voyage-balanced P90 absolute error: **25.9 min**;
- within 30 min: **92.8%**;
- within 60 min: **95.3%**;
- within 120 min: **100%**.

This is already a strong baseline for the research-gate event. It leaves limited room for a complex model to create a credible improvement at short horizon without a larger test set.

### 5.2 Long horizon

The same causal approach scope has only:

- 0 calls at 6–12 h;
- 0 calls at 12–24 h;
- 2 calls at 24–48 h;
- 1 call at 48–72 h;
- 1 call beyond 72 h.

Only **two unique development calls** contribute any approach-state prediction beyond 6 h. Those calls include loitering/multi-stage behaviour; `CT-041` (MERSEY SPIRIT) dominates the heavy tail.

Accordingly, the all-horizon B2 mean absolute error is **5.69 h** even though the median absolute error is only about **5.8 min**. This is not a contradiction: the distribution is extremely heavy-tailed and under-sampled at long horizon.

### 5.3 Destination availability

The current AIS destination explicitly supports Catania for prediction-time rows in **17 development calls**. That scope is still concentrated close to arrival: 16 of those 17 calls have at least one point under 6 h, while long-horizon coverage remains sparse.

This means the current dataset cannot support a strong claim about “when ML beats reported ETA at 24/48/72 h”. Historical Message-5 ETA is also absent, so the reported-ETA comparison remains blocked.

## 6. What M1 falsified

M1 falsifies the idea that 28 audited Catania calls automatically provide a useful long-horizon ETA benchmark.

The effective sample size collapses further once we require causal, recent, inbound-useful states:

- 28 primary calls exist;
- 22 are development calls;
- only 16 development calls support the causal inbound-approach benchmark;
- only 2 of those have any usable evidence beyond 6 h.

It also shows that the final chronological test cannot measure cold-vessel generalization because all five final-test vessels have been seen in development.

## 7. Gate decision

**Decision: `EXPAND_GROUND_TRUTH_BEFORE_COMPLEXITY`.**

Do **not** start residual boosting, route-kNN tuning, sequence models or strong superiority claims from Catania alone.

The next highest-value action is to expand independent ground truth to **Augusta**, where the domain is harder (roadstead, multiple entrances, terminal heterogeneity). After Augusta is audited, reassess whether M2 train-only route modelling has enough independent calls to be scientifically useful.

This is a stop-condition success, not a project failure: M1 identified exactly where additional model capacity would become overfitting rather than evidence.

## 8. Locked-test policy

The six chronological final-test calls are frozen in `catania_m1_call_splits.csv`. M1 reports their composition only. Their baseline/model error metrics must remain unused until a later model is frozen.

## 9. Reproducibility artefacts

- `catania_m1_call_splits.csv`
- `catania_m1_decision_panel.csv`
- `catania_m1_scope_coverage.csv`
- `catania_m1_baseline_metrics.csv`
- `catania_m1_horizon_metrics.csv`
- `catania_m1_call_metrics.csv`
- `catania_m1_cadence_sensitivity.csv`
- `m1_summary.json`
- `m1_call_concentration.png`
- `m1_horizon_coverage.png`
- `m1_baseline_error_vs_horizon.png`

## 10. External methodological references

- Evmides et al. (2024), *Enhancing Prediction Accuracy of Vessel Arrival Times Using Machine Learning*, JMSE 12(8):1362 — route-grouped evaluation and AIS2ETA codebase.
- USCG Navigation Center, AIS Class-A reports / ITU-R M.1371 semantics — reference for AIS dynamic fields and reporting cadence used in earlier data-quality milestones.
