# M18B Validation Protocol

M18B freezes how future M18 candidates must be evaluated under the M18A prediction contract. It does **not** retrain or replace M16G.

## Primary supervised unit

One `Tracks` target row per MMSI. Therefore the primary benchmark has 386 unique supervised owners, not many target rows per voyage. AIS position histories remain causal input sequences and must not be treated as independent labels.

## Forward-temporal design

Five contiguous near-equal-count blocks are formed from `Tracks.last_update`; equal timestamps remain in the same block. Test blocks are 2, 3 and 4. Each fold trains only on prior blocks, removes the six hours immediately preceding the test start, and never trains on future blocks.

Strict forward membership is frozen in `reports/m18b_forward_membership.csv`. Any candidate claiming M18B forward performance must rebuild every train-derived component inside those TRAIN rows. Re-slicing old OOF predictions is diagnostic only.

## Cross-port limitation

No leave-one-port-out claim is authorized yet. `Tracks.eta` is a company reference ETA and destination text is not an authoritative port-call event label. Destination seen/unseen is retained only as a diagnostic proxy until M18C or an external authoritative target establishes the event/port.

## Fresh holdout

The future company lockbox remains unopened. Before opening, freeze input hashes/schema, executable full-development recipes, target semantics, metrics, critical slices and the promotion gate. After labels are exposed there is no retuning.

## External methodological references

- Guerreiro et al. (2026), *A Reproducible Workflow for AIS-Based ETA Forecasting* — voyage-grouped partitions, independent held-out test and chronological robustness analysis: https://www.mdpi.com/2571-9394/8/4/70
- scikit-learn cross-validation guidance — time-series data should be evaluated on future observations rather than IID folds; `TimeSeriesSplit` supports a gap/embargo: https://scikit-learn.org/dev/modules/cross_validation.html#time-series-split
- MAPEX (2026) — disjoint monthly test periods and encounter-level grouping motivated by temporal correlation in AIS trajectories: https://www.mdpi.com/2079-8954/14/5/536
