# M17C — Sicily Historical Maritime Corridor / Knowledge Graph vs M16E Physics

## Decision

**PASS_CORRIDOR_SIGNAL_COMPONENT**.

M17C finds a small but repeatable target-free historical-corridor signal on the subset where the AIS-derived directed graph connects the current position to the resolved destination. It is retained as a **component**, not promoted as a standalone replacement for M16E or M17A.

## Method

A directed 0.20° local maritime graph is built from AIS transitions only. For every outer-validation query, all old-final MMSIs and the complete set of outer-validation MMSIs are excluded from graph construction. The graph is additionally cut at the query's `last_update`, so no future AIS transition can influence that query.

Each directed edge stores a robust historical speed with a deterministic hierarchy: edge+ship-type+6h-time-bin → edge+ship-type → edge+6h-time-bin → edge-global. Dijkstra travel time across historical water corridors is combined with short source/destination snap legs. No ETA target is used in topology, edge weights, support or graph gating.

This is an AIS-derived corridor graph, **not** an ENC/S-57/S-100 authoritative navigation route and not a bathymetric safety planner.

## Results

- Development population: **386**; old final hard-blocked: **53/53**.
- M16E physics-eligible rows: **115**.
- Causal graph-supported rows: **25** (21.7% of physics-eligible).
- M17C graph MAE on supported rows: **410.47 h**.
- M16E physics MAE on the exact same rows: **410.70 h**.
- Supported-row MAE gain: **0.223 h**; paired absolute-error wins: **60.0%**.
- Full target-free graph gate (graph when supported, frozen M16E hard-gate otherwise): **192.911 h MAE** vs **192.926 h** for frozen M16E hard-gate, gain **0.014 h**, wins **3/5** folds.

## Interpretation

The historical graph adds a real but very small increment over great-circle physics. Its strongest value is structural: it replaces straight-line distance with empirically observed directed corridors and context-conditioned segment speeds while remaining target-free. Coverage is limited because the provided AIS snapshot covers Eastern Sicily / nearby Mediterranean corridors, whereas many declared destinations (Gibraltar, Port Said, Northern Italy, etc.) lie outside the locally observed graph.

The declared/reference ETA heavy tail still dominates pooled MAE. `reference_eta_status` is therefore retained only in a diagnostic table and is never used for graph construction or gating.

## Next implication

Do not densify snapshot memory again. If M17 continues, the graph can be offered as an additional MoE feature/expert alongside M17A learned retrieval, but any meaningful company-facing superiority claim still requires a new untouched holdout.
