"""M16I red-team, ablation and stress-test utilities.

M16I does not tune or replace the frozen M16G/M16H predictors.  It audits
those frozen development-only OOF predictions under deterministic structural
ablations, subgroup stress tests, robust trimming/winsorisation and paired
bootstrap resampling.  The old M10 final set remains blocked.
"""
from __future__ import annotations

from typing import Iterable, Sequence
import numpy as np

M16I_VERSION = "m16i-red-team-ablation-stress-v1-20260922"
M16I_BOOTSTRAP_SEED = 20260922
M16I_BOOTSTRAP_REPS = 5000


def mae(y: Sequence[float], p: Sequence[float]) -> float:
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    return float(np.mean(np.abs(y - p)))


def support_bucket(n: int) -> str:
    n = int(n)
    if n <= 0: return "COLD_0"
    if n <= 2: return "RARE_1_2"
    if n <= 5: return "LOW_3_5"
    return "SUPPORTED_6PLUS"


def selected_path_counterfactual(
    meta_ids: Sequence[str], selected_experts: Sequence[str],
    expert_rows: Sequence[dict[str, float]], available_order: Sequence[str],
    hard_fallback: str = "median",
) -> np.ndarray:
    """Apply a frozen-path structural counterfactual without re-optimising.

    Hard-gate rows keep their selected expert if available.  If the selected
    expert has been ablated, they fall back deterministically to the median of
    the remaining expert predictions.  Blend rows are recomputed from only the
    remaining expert set.  This is intentionally a *stress diagnostic*, not a
    retrained challenger.
    """
    out: list[float] = []
    avail = tuple(available_order)
    if not avail:
        raise ValueError("at least one counterfactual expert is required")
    for meta, sel, row in zip(meta_ids, selected_experts, expert_rows):
        vals = np.asarray([float(row[k]) for k in avail], dtype=float)
        if not np.isfinite(vals).all():
            raise ValueError("non-finite counterfactual expert prediction")
        if str(meta) == "median_blend" or str(sel).startswith("BLEND_MEDIAN"):
            pred = float(np.median(vals))
        elif str(meta) == "mean_blend" or str(sel).startswith("BLEND_MEAN"):
            pred = float(np.mean(vals))
        elif str(sel) in row and str(sel) in avail:
            pred = float(row[str(sel)])
        elif hard_fallback == "median":
            pred = float(np.median(vals))
        else:
            raise ValueError(f"unknown hard fallback: {hard_fallback}")
        out.append(pred)
    return np.asarray(out, dtype=float)


def paired_bootstrap_gain(
    gains: Sequence[float], folds: Sequence[int], *, reps: int = M16I_BOOTSTRAP_REPS,
    seed: int = M16I_BOOTSTRAP_SEED,
) -> np.ndarray:
    """Deterministic stratified paired bootstrap of MAE gain per row."""
    g = np.asarray(gains, dtype=float); f = np.asarray(folds, dtype=int)
    if len(g) != len(f) or not np.isfinite(g).all():
        raise ValueError("invalid paired bootstrap inputs")
    rng = np.random.default_rng(int(seed))
    unique = sorted(np.unique(f).tolist())
    out = np.empty(int(reps), dtype=float)
    groups = [np.flatnonzero(f == k) for k in unique]
    for r in range(int(reps)):
        idx = np.concatenate([rng.choice(a, size=len(a), replace=True) for a in groups])
        out[r] = float(np.mean(g[idx]))
    return out


def red_team_gate(*, pooled_gain_h: float, fold_wins: int, leave_one_fold_positive: int,
                  trimmed_5pct_gain_h: float, winsorized_95_gain_h: float,
                  leave_top_destination_gain_h: float, bootstrap_positive_probability: float,
                  bootstrap_ci95_low_h: float, top5_positive_gain_share: float,
                  m16h_coverage: float, confidence_ordered: bool) -> str:
    core = (
        pooled_gain_h > 0
        and int(fold_wins) >= 4
        and int(leave_one_fold_positive) == 5
        and trimmed_5pct_gain_h > 0
        and winsorized_95_gain_h > 0
        and leave_top_destination_gain_h > 0
        and bootstrap_positive_probability >= 0.90
        and 0.78 <= m16h_coverage <= 0.86
        and bool(confidence_ordered)
    )
    if not core:
        return "FAIL_RED_TEAM_PROMOTION_GATE"
    if bootstrap_ci95_low_h > 0 and top5_positive_gain_share < 0.50:
        return "PASS_RED_TEAM_STABLE"
    return "PASS_RED_TEAM_WITH_HEAVY_TAIL_CAVEAT"
