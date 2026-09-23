"""M18G frozen external-promotion gate utilities.

M18G does not open a holdout and does not tune a model.  It freezes the one-shot
external-validation contract that must be satisfied before any fresh holdout can
be used to promote a point predictor or externally validate the M18F selective
ETA layer.

The key separation is operational:
1. predictions / risk scores / prediction-time slices are generated and sealed
   before labels are available;
2. labels are revealed only after the prediction ledger hash is frozen;
3. the evaluator runs once under predeclared thresholds;
4. failed candidates are not retuned on the same holdout.
"""
from __future__ import annotations

from collections.abc import Iterable
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

M18G_VERSION = "m18g-frozen-external-promotion-gate-v1-20260923"
M18G_EXPECTED_DEV_ROWS = 386
M18G_EXPECTED_OLD_FINAL_ROWS = 53

# Frozen in M18B.
M18G_BOOTSTRAP_REPS = 10_000
M18G_BOOTSTRAP_SEED = 18_023
M18G_POINT_MAX_P90_RATIO = 1.02
M18G_POINT_MAX_SLICE_REGRESSION_RATIO = 1.10
M18G_POINT_MIN_CRITICAL_SLICE_N = 20

# Frozen in M18F.
M18G_SELECTIVE_RISK_THRESHOLD_H = 61.116040125
M18G_SELECTIVE_MIN_COVERAGE = 0.70
M18G_SELECTIVE_MAX_COVERAGE = 0.90
M18G_SELECTIVE_MIN_INTERVAL_COVERAGE = 0.78

M18G_CRITICAL_SLICE_COLUMNS = (
    "slice_vessel_seen",
    "slice_confidence_tier",
    "slice_physics_eligible",
    "slice_destination_resolved",
)

M18G_REQUIRED_PREDICTION_COLUMNS = (
    "sample_key",
    "mmsi",
    "decision_time",
    "baseline_pred_h",
    "m16h_risk_h",
    "m16h_lower_h",
    "m16h_upper_h",
    *M18G_CRITICAL_SLICE_COLUMNS,
)
M18G_REQUIRED_LABEL_COLUMNS = ("sample_key", "target_tte_h")
M18G_FORBIDDEN_PREDICTION_COLUMNS = (
    "target_tte_h",
    "reference_eta_status",
    "eta_reference_dt",
    "abs_error_h",
    "signed_error_h",
)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_decision_time(value) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat().replace("+00:00", "Z")


def sample_key(mmsi: int, decision_time) -> str:
    payload = f"{int(mmsi)}|{canonical_decision_time(decision_time)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def regression_metrics(y_true: Iterable[float], pred: Iterable[float]) -> dict[str, float | int]:
    y = np.asarray(list(y_true), dtype=float)
    p = np.asarray(list(pred), dtype=float)
    if y.shape != p.shape or y.size == 0:
        raise ValueError("equal-shape non-empty arrays required")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("finite y/pred required")
    err = p - y
    ae = np.abs(err)
    return {
        "n": int(len(y)),
        "mae_h": float(np.mean(ae)),
        "medae_h": float(np.median(ae)),
        "rmse_h": float(np.sqrt(np.mean(err**2))),
        "p90_ae_h": float(np.quantile(ae, 0.90)),
        "mean_signed_error_h": float(np.mean(err)),
    }


def paired_bootstrap_mae_gain(
    y_true: Iterable[float],
    baseline_pred: Iterable[float],
    candidate_pred: Iterable[float],
    *,
    reps: int = M18G_BOOTSTRAP_REPS,
    seed: int = M18G_BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    """Paired owner-row bootstrap for baseline MAE minus candidate MAE."""
    y = np.asarray(list(y_true), dtype=float)
    b = np.asarray(list(baseline_pred), dtype=float)
    c = np.asarray(list(candidate_pred), dtype=float)
    if not (y.shape == b.shape == c.shape) or y.size == 0:
        raise ValueError("equal-shape non-empty arrays required")
    if not np.isfinite(y).all() or not np.isfinite(b).all() or not np.isfinite(c).all():
        raise ValueError("finite y/baseline/candidate required")
    if int(reps) < 100:
        raise ValueError("at least 100 bootstrap reps required")
    delta = np.abs(b - y) - np.abs(c - y)
    rng = np.random.default_rng(int(seed))
    out = np.empty(int(reps), dtype=float)
    # Batched indices avoid a potentially large reps x n allocation.
    step = 512
    for start in range(0, int(reps), step):
        k = min(step, int(reps) - start)
        idx = rng.integers(0, len(delta), size=(k, len(delta)))
        out[start:start + k] = delta[idx].mean(axis=1)
    lo, hi = np.quantile(out, [0.025, 0.975])
    return {
        "n": int(len(delta)),
        "reps": int(reps),
        "seed": int(seed),
        "mae_gain_h": float(np.mean(delta)),
        "ci95_lower_h": float(lo),
        "ci95_upper_h": float(hi),
    }


def critical_slice_audit(
    frame: pd.DataFrame,
    *,
    target_col: str = "target_tte_h",
    baseline_col: str = "baseline_pred_h",
    candidate_col: str = "candidate_pred_h",
    slice_columns: tuple[str, ...] = M18G_CRITICAL_SLICE_COLUMNS,
    min_n: int = M18G_POINT_MIN_CRITICAL_SLICE_N,
) -> pd.DataFrame:
    rows: list[dict] = []
    for col in slice_columns:
        if col not in frame.columns:
            raise ValueError(f"missing predeclared critical slice column: {col}")
        for value, g in frame.groupby(col, dropna=False, sort=True):
            if len(g) < int(min_n):
                continue
            b = regression_metrics(g[target_col], g[baseline_col])
            c = regression_metrics(g[target_col], g[candidate_col])
            if b["mae_h"] == 0:
                ratio = 1.0 if c["mae_h"] == 0 else float("inf")
            else:
                ratio = float(c["mae_h"] / b["mae_h"])
            rows.append({
                "slice_column": col,
                "slice_value": str(value),
                "n": int(len(g)),
                "baseline_mae_h": float(b["mae_h"]),
                "candidate_mae_h": float(c["mae_h"]),
                "candidate_to_baseline_mae_ratio": ratio,
                "passes_max_regression": bool(ratio <= M18G_POINT_MAX_SLICE_REGRESSION_RATIO),
            })
    return pd.DataFrame(rows)


def evaluate_point_candidate(frame: pd.DataFrame) -> dict:
    for col in ("target_tte_h", "baseline_pred_h", "candidate_pred_h"):
        if col not in frame.columns:
            raise ValueError(f"missing point-evaluation column: {col}")
    baseline = regression_metrics(frame.target_tte_h, frame.baseline_pred_h)
    candidate = regression_metrics(frame.target_tte_h, frame.candidate_pred_h)
    boot = paired_bootstrap_mae_gain(frame.target_tte_h, frame.baseline_pred_h, frame.candidate_pred_h)
    slices = critical_slice_audit(frame)
    p90_ratio = float(candidate["p90_ae_h"] / baseline["p90_ae_h"]) if baseline["p90_ae_h"] > 0 else float("inf")
    slices_pass = bool(len(slices) > 0 and slices.passes_max_regression.astype(bool).all())
    passed = bool(
        candidate["mae_h"] < baseline["mae_h"]
        and boot["mae_gain_h"] > 0
        and boot["ci95_lower_h"] > 0
        and p90_ratio <= M18G_POINT_MAX_P90_RATIO
        and slices_pass
    )
    return {
        "decision": "PROMOTE_POINT_CANDIDATE" if passed else "KEEP_FROZEN_M16G",
        "baseline": baseline,
        "candidate": candidate,
        "paired_bootstrap": boot,
        "candidate_p90_ratio_vs_baseline": p90_ratio,
        "critical_slice_count": int(len(slices)),
        "critical_slices_all_pass": slices_pass,
        "critical_slice_rows": slices.to_dict(orient="records"),
    }


def evaluate_selective_layer(frame: pd.DataFrame, *, risk_threshold_h: float = M18G_SELECTIVE_RISK_THRESHOLD_H) -> dict:
    required = ["target_tte_h", "baseline_pred_h", "m16h_risk_h", "m16h_lower_h", "m16h_upper_h"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"missing selective-evaluation columns: {missing}")
    y = frame.target_tte_h.to_numpy(float)
    p = frame.baseline_pred_h.to_numpy(float)
    risk = frame.m16h_risk_h.to_numpy(float)
    lo = frame.m16h_lower_h.to_numpy(float)
    hi = frame.m16h_upper_h.to_numpy(float)
    if not all(np.isfinite(x).all() for x in (y, p, risk, lo, hi)):
        raise ValueError("selective evaluation requires finite values")
    keep = risk <= float(risk_threshold_h)
    n = len(frame)
    k = int(keep.sum())
    ae = np.abs(p - y)
    full_mae = float(ae.mean())
    retained_mae = float(ae[keep].mean()) if k else float("nan")
    deferred_mae = float(ae[~keep].mean()) if k < n else float("nan")
    interval_cov = float(np.mean((y[keep] >= lo[keep]) & (y[keep] <= hi[keep]))) if k else float("nan")
    rho = float(spearmanr(risk, ae).statistic) if n >= 3 else float("nan")
    coverage = float(k / n)
    passed = bool(
        M18G_SELECTIVE_MIN_COVERAGE <= coverage <= M18G_SELECTIVE_MAX_COVERAGE
        and k > 0 and k < n
        and retained_mae < full_mae
        and deferred_mae > retained_mae
        and interval_cov >= M18G_SELECTIVE_MIN_INTERVAL_COVERAGE
        and np.isfinite(rho) and rho > 0
    )
    return {
        "decision": "VALIDATE_M18F_SELECTIVE_LAYER" if passed else "DO_NOT_EXTERNALLY_VALIDATE_M18F_LAYER",
        "frozen_risk_threshold_h": float(risk_threshold_h),
        "n": int(n),
        "retained_rows": k,
        "deferred_rows": int(n-k),
        "realized_coverage": coverage,
        "full_mae_h": full_mae,
        "retained_mae_h": retained_mae,
        "deferred_mae_h": deferred_mae,
        "retained_interval_empirical_coverage": interval_cov,
        "risk_abs_error_spearman": rho,
        "selection_conditional_conformal_guarantee_claimed": False,
    }


def validate_prediction_ledger(frame: pd.DataFrame, *, require_candidate: bool) -> None:
    missing = [c for c in M18G_REQUIRED_PREDICTION_COLUMNS if c not in frame.columns]
    if missing:
        raise AssertionError(f"prediction ledger missing columns: {missing}")
    bad = [c for c in M18G_FORBIDDEN_PREDICTION_COLUMNS if c in frame.columns]
    if bad:
        raise AssertionError(f"prediction ledger contains target/post-label columns: {bad}")
    if require_candidate and "candidate_pred_h" not in frame.columns:
        raise AssertionError("registered point candidate requires candidate_pred_h")
    if len(frame) == 0 or frame.sample_key.duplicated().any():
        raise AssertionError("prediction ledger requires non-empty unique sample_key")
    expected = [sample_key(m, t) for m, t in zip(frame.mmsi, frame.decision_time)]
    if list(frame.sample_key.astype(str)) != expected:
        raise AssertionError("prediction ledger sample_key mismatch")
    numeric = ["baseline_pred_h", "m16h_risk_h", "m16h_lower_h", "m16h_upper_h"]
    if require_candidate:
        numeric.append("candidate_pred_h")
    for col in numeric:
        if not np.isfinite(pd.to_numeric(frame[col], errors="coerce").to_numpy(float)).all():
            raise AssertionError(f"prediction ledger non-finite: {col}")


def validate_label_ledger(frame: pd.DataFrame) -> None:
    missing = [c for c in M18G_REQUIRED_LABEL_COLUMNS if c not in frame.columns]
    if missing:
        raise AssertionError(f"label ledger missing columns: {missing}")
    if len(frame) == 0 or frame.sample_key.duplicated().any():
        raise AssertionError("label ledger requires non-empty unique sample_key")
    if not np.isfinite(pd.to_numeric(frame.target_tte_h, errors="coerce").to_numpy(float)).all():
        raise AssertionError("label ledger target_tte_h must be finite")


def validate_gate_contract(gate: dict) -> None:
    if gate.get("milestone") != "M18G" or gate.get("version") != M18G_VERSION:
        raise AssertionError("wrong M18G milestone/version")
    scope = gate.get("scope", {})
    for key in ["fresh_holdout_opened", "holdout_labels_observed", "post_holdout_retuning_allowed"]:
        if scope.get(key) is not False:
            raise AssertionError(f"M18G scope flag must be false: {key}")
    if int(scope.get("development_population", -1)) != M18G_EXPECTED_DEV_ROWS:
        raise AssertionError("M18G development population drift")
    if int(scope.get("blocked_old_final_population", -1)) != M18G_EXPECTED_OLD_FINAL_ROWS:
        raise AssertionError("M18G old-final population drift")
    pg = gate.get("point_model_gate", {})
    if int(pg.get("paired_bootstrap_reps", -1)) != M18G_BOOTSTRAP_REPS:
        raise AssertionError("M18G bootstrap reps drift")
    if int(pg.get("paired_bootstrap_seed", -1)) != M18G_BOOTSTRAP_SEED:
        raise AssertionError("M18G bootstrap seed drift")
    if float(pg.get("max_p90_ratio", -1)) != M18G_POINT_MAX_P90_RATIO:
        raise AssertionError("M18G P90 threshold drift")
    if float(pg.get("max_critical_slice_regression_ratio", -1)) != M18G_POINT_MAX_SLICE_REGRESSION_RATIO:
        raise AssertionError("M18G slice threshold drift")
    sg = gate.get("selective_eta_gate", {})
    if not np.isclose(float(sg.get("frozen_risk_threshold_h", -1)), M18G_SELECTIVE_RISK_THRESHOLD_H):
        raise AssertionError("M18G M18F threshold drift")
