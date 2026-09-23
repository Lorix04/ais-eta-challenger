"""M18F uncertainty and selective-ETA utilities.

M18F is a development-only reliability layer around frozen M16G point ETA and
frozen M16H uncertainty.  It does not change the point predictor, relabel the
target, open the fresh holdout, or reuse the 53 blocked old-final owners.

The selective policy is evaluated with nested, fold-safe risk thresholds: for
an outer validation fold, the uncertainty/error-risk threshold is estimated
only from cross-fitted risk scores inside that outer fold's training owners.
The retained subset is therefore selected without using validation targets.

M16H's empirical central interval remains marginal.  After selective
abstention, its coverage is reported only as an empirical retained-set audit;
M18F does not claim selection-conditional conformal validity.
"""
from __future__ import annotations

from collections.abc import Iterable
import hashlib
from pathlib import Path

import numpy as np

M18F_VERSION = "m18f-uncertainty-selective-eta-v1-20260923"
M18F_EXPECTED_DEV_ROWS = 386
M18F_EXPECTED_OLD_FINAL_ROWS = 53
M18F_COVERAGE_TARGETS = (1.00, 0.90, 0.80, 0.70, 0.60, 0.50)
M18F_BALANCED_TARGET_COVERAGE = 0.80
M18F_MIN_BALANCED_REALIZED_COVERAGE = 0.70
M18F_MAX_BALANCED_REALIZED_COVERAGE = 0.90
M18F_MIN_NEAR_TERM_RETENTION = 0.85
M18F_MIN_RETAINED_EMPIRICAL_INTERVAL_COVERAGE = 0.78


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def risk_threshold(train_risk_h: Iterable[float], target_coverage: float) -> float:
    """Target-free uncertainty threshold estimated from training risk scores.

    Lower risk means more reliable.  ``target_coverage`` is the desired share
    retained under the training risk distribution, not a guarantee about a
    future distribution.  ``method='higher'`` makes the finite-sample rule
    deterministic and mildly conservative with respect to retained count.
    """
    risk = np.asarray(list(train_risk_h), dtype=float)
    risk = risk[np.isfinite(risk)]
    if risk.size == 0:
        raise ValueError("no finite training risk scores")
    if not (0 < float(target_coverage) <= 1):
        raise ValueError("target_coverage must be in (0,1]")
    if float(target_coverage) == 1.0:
        return float("inf")
    return float(np.quantile(risk, float(target_coverage), method="higher"))


def selective_metrics(
    y_true: Iterable[float],
    point_pred: Iterable[float],
    lower_h: Iterable[float],
    upper_h: Iterable[float],
    accepted: Iterable[bool],
) -> dict[str, float | int]:
    y = np.asarray(list(y_true), dtype=float)
    p = np.asarray(list(point_pred), dtype=float)
    lo = np.asarray(list(lower_h), dtype=float)
    hi = np.asarray(list(upper_h), dtype=float)
    a = np.asarray(list(accepted), dtype=bool)
    if not (y.shape == p.shape == lo.shape == hi.shape == a.shape) or y.size == 0:
        raise ValueError("equal-shape non-empty arrays required")
    if not np.isfinite(y).all() or not np.isfinite(p).all() or not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError("metrics require finite values")
    n = int(y.size)
    k = int(a.sum())
    ae = np.abs(p - y)
    total_abs = float(ae.sum())
    out: dict[str, float | int] = {
        "rows": n,
        "retained_rows": k,
        "deferred_rows": int(n - k),
        "realized_coverage": float(k / n),
    }
    if k == 0:
        out.update({
            "retained_mae_h": float("nan"),
            "retained_medae_h": float("nan"),
            "retained_p90_ae_h": float("nan"),
            "retained_interval_empirical_coverage": float("nan"),
            "retained_mean_interval_width_h": float("nan"),
            "retained_abs_error_share": 0.0,
            "deferred_mae_h": float(np.mean(ae)),
        })
        return out

    kept_ae = ae[a]
    covered = (y >= lo) & (y <= hi)
    width = hi - lo
    out.update({
        "retained_mae_h": float(np.mean(kept_ae)),
        "retained_medae_h": float(np.median(kept_ae)),
        "retained_p90_ae_h": float(np.quantile(kept_ae, 0.90)),
        "retained_interval_empirical_coverage": float(np.mean(covered[a])),
        "retained_mean_interval_width_h": float(np.mean(width[a])),
        "retained_abs_error_share": float(kept_ae.sum() / total_abs) if total_abs > 0 else 0.0,
        "deferred_mae_h": float(np.mean(ae[~a])) if (~a).any() else float("nan"),
    })
    return out


def selective_gate(
    *,
    balanced_realized_coverage: float,
    balanced_retained_mae_h: float,
    full_mae_h: float,
    fold_wins: int,
    fold_count: int,
    near_term_retention: float,
    retained_empirical_interval_coverage: float,
) -> str:
    """Predeclared promotion gate for an advisory selective-output layer."""
    if (
        M18F_MIN_BALANCED_REALIZED_COVERAGE <= float(balanced_realized_coverage) <= M18F_MAX_BALANCED_REALIZED_COVERAGE
        and float(balanced_retained_mae_h) < float(full_mae_h)
        and int(fold_wins) == int(fold_count)
        and float(near_term_retention) >= M18F_MIN_NEAR_TERM_RETENTION
        and float(retained_empirical_interval_coverage) >= M18F_MIN_RETAINED_EMPIRICAL_INTERVAL_COVERAGE
    ):
        return "PASS_SELECTIVE_ETA_ADVISORY_LAYER"
    return "NO_SELECTIVE_ETA_PROMOTION"


def validate_m18f_audit(audit: dict) -> None:
    if audit.get("milestone") != "M18F":
        raise AssertionError("wrong M18F milestone")
    if audit.get("version") != M18F_VERSION:
        raise AssertionError("wrong M18F version")
    scope = audit.get("scope", {})
    required_false = [
        "fresh_holdout_opened", "old_final_used", "m16g_modified", "m16h_modified",
        "target_relabeling", "strict_rows_deleted", "selection_uses_validation_target",
        "selection_conditional_conformal_guarantee_claimed",
    ]
    for key in required_false:
        if scope.get(key) is not False:
            raise AssertionError(f"M18F scope flag must be false: {key}")
    if int(scope.get("development_population", -1)) != M18F_EXPECTED_DEV_ROWS:
        raise AssertionError("M18F development population drift")
    if int(scope.get("blocked_old_final_population", -1)) != M18F_EXPECTED_OLD_FINAL_ROWS:
        raise AssertionError("M18F old-final population drift")
    if float(audit.get("balanced_policy", {}).get("target_coverage", -1)) != M18F_BALANCED_TARGET_COVERAGE:
        raise AssertionError("M18F balanced target coverage drift")
    if audit.get("balanced_policy", {}).get("point_prediction_source") != "frozen_M16G":
        raise AssertionError("M18F must retain frozen M16G P50")
