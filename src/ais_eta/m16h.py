"""M16H development-only probabilistic ETA and confidence utilities.

M16H keeps the frozen M16G point prediction as P50 and calibrates an empirical
central interval from development-only OOF residuals.  A cross-fitted error
model estimates row-level difficulty using only prediction-time features; a
split-conformal-style normalized residual quantile then turns that difficulty
score into an adaptive interval.  The 53 already-observed M10 final MMSIs are
never eligible for calibration or selection.

The interval is an empirical development-only central 80% interval.  It should
not be described as a conditional-coverage guarantee: conformal guarantees are
marginal and rely on exchangeability assumptions that may not hold under AIS
or reference-ETA distribution shift.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

M16H_VERSION = "m16h-probabilistic-eta-confidence-v1-20260921"
M16H_NOMINAL_COVERAGE = 0.80
M16H_SCALE_FLOORS_H = (12.0, 24.0, 48.0, 72.0)
M16H_CANDIDATES = ("global_abs",) + tuple(f"adaptive_scale_{int(x)}h" for x in M16H_SCALE_FLOORS_H)
M16H_MIN_INNER_COVERAGE = 0.76
M16H_MIN_POOLED_COVERAGE = 0.78
M16H_MAX_POOLED_COVERAGE = 0.86
M16H_MIN_SHARPNESS_GAIN = 0.10
M16H_MIN_STABLE_FOLDS = 4
M16H_STABLE_FOLD_COVERAGE = 0.75

FORBIDDEN_FEATURE_NAMES = {
    "target_tte_h", "reference_eta", "reference_eta_status", "abs_error_h", "error_h",
}


@dataclass(frozen=True)
class IntervalResult:
    lower_h: np.ndarray
    p50_h: np.ndarray
    upper_h: np.ndarray
    half_width_h: np.ndarray
    scale_q: float


def conformal_upper_quantile(scores: Sequence[float], coverage: float = M16H_NOMINAL_COVERAGE) -> float:
    """Finite-sample upper order statistic used by split conformal.

    Uses k = ceil((n+1)*coverage), clipped to [1, n].
    """
    a = np.asarray(scores, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        raise ValueError("no finite calibration scores")
    if not (0 < coverage < 1):
        raise ValueError("coverage must be in (0,1)")
    a.sort()
    k = min(len(a), max(1, math.ceil((len(a) + 1) * float(coverage))))
    return float(a[k - 1])


def build_error_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Prediction-time-only features used to estimate expected absolute error."""
    bad = [c for c in frame.columns if c.lower() in FORBIDDEN_FEATURE_NAMES]
    if bad:
        raise ValueError(f"target-derived confidence features are forbidden: {bad}")

    expert_cols = [
        "pred_m16c_prior_h", "pred_m16d_route_h", "pred_m16e_physics_gate_h", "pred_m16f_tabular_h",
    ]
    required = expert_cols + [
        "pred_m16g_selected_h", "m16c_deepest_support", "m16c_deepest_weight", "m16c_fallback_count",
        "m16d_neighbour_count", "m16d_nearest_distance", "m16d_median_neighbour_distance",
        "m16d_similarity_gap", "m16d_gate_used", "physics_eligible", "resolution_confidence",
        "physics_distance_gc_nm", "physics_course_alignment_deg", "physics_recent_speed_max_kn",
        "m16g_selected_meta_id",
    ]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"missing M16H feature columns: {missing}")

    out = pd.DataFrame(index=frame.index)
    for col in expert_cols + ["pred_m16g_selected_h"]:
        out[col] = pd.to_numeric(frame[col], errors="coerce")
    out["expert_mean_h"] = out[expert_cols].mean(axis=1)
    out["expert_std_h"] = out[expert_cols].std(axis=1)
    out["expert_range_h"] = out[expert_cols].max(axis=1) - out[expert_cols].min(axis=1)
    out["mix_minus_expert_median_h"] = out["pred_m16g_selected_h"] - out[expert_cols].median(axis=1)

    numeric = [
        "m16c_deepest_support", "m16c_deepest_weight", "m16c_fallback_count",
        "m16d_neighbour_count", "m16d_nearest_distance", "m16d_median_neighbour_distance",
        "m16d_similarity_gap", "resolution_confidence", "physics_distance_gc_nm",
        "physics_course_alignment_deg", "physics_recent_speed_max_kn",
    ]
    for col in numeric:
        out[col] = pd.to_numeric(frame[col], errors="coerce")
    out["physics_eligible"] = frame["physics_eligible"].astype(bool).astype(float)

    gate = frame["m16d_gate_used"].fillna("UNKNOWN").astype(str)
    for label in ("destination", "destination_fallback_global", "global"):
        out[f"route_gate_{label}"] = gate.eq(label).astype(float)

    meta = frame["m16g_selected_meta_id"].fillna("UNKNOWN").astype(str)
    for label in sorted(meta.unique()):
        out[f"meta_{label}"] = meta.eq(label).astype(float)

    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)


def error_model() -> HistGradientBoostingRegressor:
    """Small robust error model; fixed before M16H outer-fold scoring."""
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        max_leaf_nodes=7,
        min_samples_leaf=25,
        l2_regularization=10.0,
        learning_rate=0.05,
        max_iter=120,
        random_state=42,
        early_stopping=False,
    )


def fit_error_model(features: pd.DataFrame, abs_error_h: Sequence[float]) -> HistGradientBoostingRegressor:
    y = np.asarray(abs_error_h, dtype=float)
    if len(features) != len(y):
        raise ValueError("error-model feature/target row mismatch")
    if np.any(~np.isfinite(y)) or np.any(y < 0):
        raise ValueError("absolute-error target must be finite and non-negative")
    return error_model().fit(features, np.log1p(y))


def predict_error(model: HistGradientBoostingRegressor, features: pd.DataFrame) -> np.ndarray:
    return np.maximum(0.0, np.expm1(np.asarray(model.predict(features), dtype=float)))


def assign_confidence_tier(train_oof_risk_h: Sequence[float], risk_h: Sequence[float]) -> tuple[np.ndarray, float, float]:
    train = np.asarray(train_oof_risk_h, dtype=float)
    risk = np.asarray(risk_h, dtype=float)
    if np.any(~np.isfinite(train)) or np.any(~np.isfinite(risk)):
        raise ValueError("confidence risk must be finite")
    q1, q2 = np.quantile(train, [1 / 3, 2 / 3], method="linear")
    tier = np.where(risk <= q1, "HIGH", np.where(risk <= q2, "MEDIUM", "LOW"))
    return tier.astype(object), float(q1), float(q2)


def candidate_floor(candidate_id: str) -> float | None:
    if candidate_id == "global_abs":
        return None
    if candidate_id.startswith("adaptive_scale_") and candidate_id.endswith("h"):
        return float(candidate_id.removeprefix("adaptive_scale_").removesuffix("h"))
    raise KeyError(candidate_id)


def calibrate_interval(
    train_abs_error_h: Sequence[float],
    train_risk_h: Sequence[float],
    valid_p50_h: Sequence[float],
    valid_risk_h: Sequence[float],
    candidate_id: str,
    coverage: float = M16H_NOMINAL_COVERAGE,
) -> IntervalResult:
    err = np.asarray(train_abs_error_h, dtype=float)
    risk_tr = np.asarray(train_risk_h, dtype=float)
    p50 = np.asarray(valid_p50_h, dtype=float)
    risk_va = np.asarray(valid_risk_h, dtype=float)
    if candidate_id not in M16H_CANDIDATES:
        raise KeyError(candidate_id)
    if len(err) != len(risk_tr):
        raise ValueError("calibration error/risk row mismatch")
    floor = candidate_floor(candidate_id)
    if floor is None:
        q = conformal_upper_quantile(err, coverage)
        half = np.full(len(p50), q, dtype=float)
    else:
        train_scale = floor + np.maximum(risk_tr, 0.0)
        valid_scale = floor + np.maximum(risk_va, 0.0)
        q = conformal_upper_quantile(err / train_scale, coverage)
        half = q * valid_scale
    return IntervalResult(p50 - half, p50, p50 + half, half, float(q))


def interval_metrics(target_h: Sequence[float], lower_h: Sequence[float], upper_h: Sequence[float], p50_h: Sequence[float]) -> dict[str, float]:
    y = np.asarray(target_h, dtype=float)
    lo = np.asarray(lower_h, dtype=float)
    hi = np.asarray(upper_h, dtype=float)
    p = np.asarray(p50_h, dtype=float)
    if not (len(y) == len(lo) == len(hi) == len(p)):
        raise ValueError("interval metric row mismatch")
    width = hi - lo
    covered = (y >= lo) & (y <= hi)
    ae = np.abs(y - p)
    return {
        "coverage": float(np.mean(covered)),
        "mean_width_h": float(np.mean(width)),
        "median_width_h": float(np.median(width)),
        "p90_width_h": float(np.quantile(width, 0.90)),
        "mae_h": float(np.mean(ae)),
        "medae_h": float(np.median(ae)),
        "p90_ae_h": float(np.quantile(ae, 0.90)),
    }


def promotion_gate(*, pooled_coverage: float, adaptive_mean_width_h: float, global_mean_width_h: float, stable_folds: int, high_medae_h: float, medium_medae_h: float, low_medae_h: float) -> str:
    sharp_gain = 1.0 - float(adaptive_mean_width_h) / float(global_mean_width_h)
    ordered = float(high_medae_h) <= float(medium_medae_h) <= float(low_medae_h)
    if (
        M16H_MIN_POOLED_COVERAGE <= float(pooled_coverage) <= M16H_MAX_POOLED_COVERAGE
        and sharp_gain >= M16H_MIN_SHARPNESS_GAIN
        and int(stable_folds) >= M16H_MIN_STABLE_FOLDS
        and ordered
    ):
        return "PASS_PROBABILISTIC_CALIBRATION_CONFIDENCE"
    return "FAIL_PROBABILISTIC_CALIBRATION_GATE"
