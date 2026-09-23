"""M18E external-information ablation lab utilities.

M18E is an evidence-first ablation layer around the frozen M16G/M16H system.
It does not open the fresh holdout and it does not silently ingest web data.
External candidates are only eligible for modelling when their source artifact
is pinned, reproducible and available at the prediction decision time.

The milestone also re-expresses two already-frozen historical component tests
as explicit ablations:
* M16B raw destination vs external/curated canonical destination semantics;
* M16D route analogue vs M16E physics component on rows where external port
  coordinates + causal motion make the physics expert eligible.

Those retrospective component ablations are diagnostic evidence, not a new
M18E model promotion.
"""
from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

M18E_VERSION = "m18e-external-information-ablation-lab-v1-20260923"
M18E_EXPECTED_DEV_ROWS = 386
M18E_EXPECTED_OLD_FINAL_ROWS = 53
M18E_BOOTSTRAP_SEED = 1805
M18E_BOOTSTRAP_DRAWS = 4000

ALLOWED_SOURCE_STATUSES = {
    "RUN_FROZEN_RETROSPECTIVE",
    "READY_NEW_DATA_PINNED",
    "BLOCKED_DATASET_NOT_PINNED",
    "BLOCKED_NO_HISTORICAL_DECISION_TIME_DATA",
    "CONTROL_REDUNDANT_WITH_BASELINE",
}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def simple_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float | int]:
    y = np.asarray(list(y_true), dtype=float)
    p = np.asarray(list(y_pred), dtype=float)
    if y.shape != p.shape or y.size == 0:
        raise ValueError("non-empty equal-shape arrays required")
    ae = np.abs(p - y)
    return {
        "rows": int(y.size),
        "mae_h": float(np.mean(ae)),
        "medae_h": float(np.median(ae)),
        "p90_ae_h": float(np.quantile(ae, 0.90)),
        "p95_ae_h": float(np.quantile(ae, 0.95)),
        "within_24h": float(np.mean(ae <= 24.0)),
        "within_48h": float(np.mean(ae <= 48.0)),
    }


def paired_gain_summary(
    y_true: Iterable[float],
    baseline_pred: Iterable[float],
    candidate_pred: Iterable[float],
    *,
    seed: int = M18E_BOOTSTRAP_SEED,
    draws: int = M18E_BOOTSTRAP_DRAWS,
) -> dict[str, float | int | list[float]]:
    """Paired owner-level bootstrap of absolute-error gain.

    Positive gain means candidate has lower absolute error than baseline.
    Each M18A supervised row is already one owner/MMSI, so row bootstrap is
    owner-level bootstrap for this frozen benchmark.
    """
    y = np.asarray(list(y_true), dtype=float)
    b = np.asarray(list(baseline_pred), dtype=float)
    c = np.asarray(list(candidate_pred), dtype=float)
    if not (y.shape == b.shape == c.shape) or y.size == 0:
        raise ValueError("non-empty equal-shape arrays required")
    gain = np.abs(b - y) - np.abs(c - y)
    rng = np.random.default_rng(seed)
    n = len(gain)
    means = np.empty(draws, dtype=float)
    for i in range(draws):
        idx = rng.integers(0, n, size=n)
        means[i] = float(np.mean(gain[idx]))
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {
        "rows": int(n),
        "mean_abs_error_gain_h": float(np.mean(gain)),
        "median_abs_error_gain_h": float(np.median(gain)),
        "win_fraction": float(np.mean(gain > 0)),
        "tie_fraction": float(np.mean(np.isclose(gain, 0.0, atol=1e-12))),
        "paired_bootstrap_ci95_h": [float(lo), float(hi)],
        "bootstrap_draws": int(draws),
        "bootstrap_seed": int(seed),
    }


def validate_source_registry(registry: pd.DataFrame) -> None:
    required = {
        "source_id", "information_family", "source_name", "source_url",
        "source_authority", "data_version", "decision_time_safe",
        "artifact_pinned", "status", "m18e_action", "notes",
    }
    missing = required - set(registry.columns)
    if missing:
        raise ValueError(f"M18E source registry missing columns: {sorted(missing)}")
    if registry["source_id"].duplicated().any():
        raise ValueError("duplicate M18E source_id")
    bad = set(registry["status"].astype(str)) - ALLOWED_SOURCE_STATUSES
    if bad:
        raise ValueError(f"invalid M18E source status: {sorted(bad)}")
    ready = registry["status"].eq("READY_NEW_DATA_PINNED")
    if ready.any() and (~registry.loc[ready, "artifact_pinned"].astype(bool)).any():
        raise AssertionError("READY_NEW_DATA_PINNED requires artifact_pinned=true")
    if ready.any() and (~registry.loc[ready, "decision_time_safe"].astype(bool)).any():
        raise AssertionError("READY_NEW_DATA_PINNED requires decision_time_safe=true")


def validate_ablation_audit(audit: dict) -> None:
    if audit.get("milestone") != "M18E":
        raise AssertionError("wrong M18E milestone")
    if audit.get("version") != M18E_VERSION:
        raise AssertionError("wrong M18E version")
    scope = audit.get("scope", {})
    required_false = [
        "fresh_holdout_opened", "old_final_used", "m16g_modified",
        "m16h_modified", "target_relabeling", "strict_rows_deleted",
        "new_external_model_promoted",
    ]
    for k in required_false:
        if scope.get(k) is not False:
            raise AssertionError(f"M18E scope flag must be false: {k}")
    if int(scope.get("development_population", -1)) != M18E_EXPECTED_DEV_ROWS:
        raise AssertionError("M18E development population drift")
    if int(scope.get("blocked_old_final_population", -1)) != M18E_EXPECTED_OLD_FINAL_ROWS:
        raise AssertionError("M18E old-final population drift")
    decision = str(audit.get("decision", ""))
    if not decision.startswith("NO_NEW_EXTERNAL_PROMOTION"):
        raise AssertionError("M18E must not claim unsupported external promotion")
    if audit.get("policy", {}).get("blocked_sources_are_treated_as_zero_gain") is not False:
        raise AssertionError("blocked sources must not be scored as zero-gain observations")
