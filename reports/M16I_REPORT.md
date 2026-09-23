# M16I — Red-team, Ablation and Stress Testing

## Decision

**PASS_RED_TEAM_WITH_HEAVY_TAIL_CAVEAT**

M16I does not tune M16G/M16H. It audits the frozen 386-row development OOF ledger and keeps all 53 old M10 final MMSIs blocked.

## Robustness of the M16G point gain

- M16G MAE: **185.88 h**
- best single M16E MAE: **192.93 h**
- pooled gain: **7.04 h**
- fold wins: **4/5**; leave-one-fold-out gain remains positive in **5/5** cases
- after removing the top 5% of rows by absolute target magnitude: **9.66 h** gain
- 95% winsorised paired gain: **4.07 h**
- after removing largest net-gain destination `NON_SPECIFIC`: **2.49 h** gain
- stratified paired bootstrap P(gain>0): **92.2%**; 95% interval **[-2.02, 20.16] h**

The red-team therefore finds a real central robustness signal, but not a clean 95% statistical separation. Positive improvement is also concentrated: the five largest positive-gain rows account for **51.0%** of total positive gain. This is preserved as a limitation, not hidden.

## Structural ablations

The required `no physics`, `no route`, `no canonical destination`, and `no longitudinal history` tests are deterministic counterfactual path stresses. They do **not** retrain or reselect a challenger after observing red-team results. Therefore they answer dependency/sensitivity questions, not 'what is the best model without component X?'. See `m16i_ablation_metrics.csv`.

## Subgroup and uncertainty stress

Audits cover cold/rare canonical destination support computed from each row's outer-train, `FOR_ORDERS`/unknown/ambiguous destination classes, vessel-category groups, target-derived stale/future status for diagnostics only, low route support, route fallback mode and M16H confidence tiers. Calibration is also checked across predicted-error-risk quintiles.

## Interpretation

M16G survives the fair robustness tests based on folds, target-magnitude trimming/winsorisation and destination removal, and M16H retains nominal pooled coverage. However, heavy tails remain dominant and the paired-bootstrap 95% interval crosses zero. The appropriate claim is therefore **development-only promising challenger with a heavy-tail caveat**, not proven superiority. A new untouched company holdout remains required for a fresh generalisation claim.
