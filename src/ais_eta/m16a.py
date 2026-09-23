"""M16A development-only OOF benchmark utilities.

M16A is a post-freeze experimental branch.  Its central invariant is that the
53 already-observed M10 final-test MMSIs can never enter model selection or OOF
benchmark construction.  Only the former M10 train + calibration population is
eligible for development.
"""
from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
import pandas as pd

M16A_OUTER_SALT = "M16A_OUTER_V1_20260921"
M16A_INNER_SALT = "M16A_INNER_ES_V1_20260921"
M16A_OUTER_FOLDS = 5
M16A_INNER_FOLDS = 6


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def balanced_hash_folds(
    mmsis: Iterable[int | str],
    n_splits: int = M16A_OUTER_FOLDS,
    salt: str = M16A_OUTER_SALT,
) -> dict[int, int]:
    """Assign balanced, deterministic, target-free folds from MMSI only.

    MMSIs are ordered by SHA-256(salt:MMSI) and assigned round-robin.  The
    method is deterministic, nearly perfectly balanced, and independent of
    labels, timestamps and features.
    """
    ids = sorted({int(x) for x in mmsis})
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    ordered = sorted(ids, key=lambda x: sha256_text(f"{salt}:{x}"))
    return {mmsi: rank % n_splits for rank, mmsi in enumerate(ordered)}


def assert_development_only(df: pd.DataFrame, forbidden_mmsis: Iterable[int | str]) -> None:
    """Hard fail if an old M10 final-test MMSI appears in development data."""
    forbidden = {int(x) for x in forbidden_mmsis}
    present = {int(x) for x in df["mmsi"].tolist()}
    overlap = sorted(present & forbidden)
    if overlap:
        preview = ",".join(str(x) for x in overlap[:10])
        raise AssertionError(f"M16A hard block: {len(overlap)} old-final MMSI(s) present: {preview}")
    if "split" in df.columns and df["split"].eq("final_test").any():
        raise AssertionError("M16A hard block: split=final_test present in development frame")


def extended_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float | int]:
    y = np.asarray(list(y_true), dtype=float)
    p = np.asarray(list(y_pred), dtype=float)
    if len(y) != len(p):
        raise ValueError("y_true/y_pred length mismatch")
    nonfinite = ~np.isfinite(p)
    finite_p = p[np.isfinite(p)]
    ae = np.abs(y - p)
    return {
        "n": int(len(y)),
        "mae_h": float(np.mean(ae)),
        "medae_h": float(np.median(ae)),
        "rmse_h": float(np.sqrt(np.mean((y - p) ** 2))),
        "p90_ae_h": float(np.quantile(ae, 0.90)),
        "p95_ae_h": float(np.quantile(ae, 0.95)),
        "within_24h": float(np.mean(ae <= 24)),
        "within_48h": float(np.mean(ae <= 48)),
        "within_72h": float(np.mean(ae <= 72)),
        "prediction_min_h": float(np.min(finite_p)) if len(finite_p) else float("nan"),
        "prediction_max_h": float(np.max(finite_p)) if len(finite_p) else float("nan"),
        "nonfinite_prediction_count": int(nonfinite.sum()),
        "abs_gt_1y_prediction_count": int(np.sum(np.abs(finite_p) > 24 * 365)),
    }
