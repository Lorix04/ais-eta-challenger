"""M18A prediction-contract and frozen-baseline utilities.

M18A intentionally introduces no new predictor.  It formalizes the prediction
contract around the development-only M16G champion and provides reusable guards
for decision-time causality and target-feature leakage.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
import hashlib

import pandas as pd

M18A_VERSION = "m18a-prediction-contract-frozen-baseline-v1-20260923"
M18A_PRIMARY_TASK_ID = "company_tracks_eta_reference_tte_h"
M18A_DECISION_TIME_FIELD = "Tracks.last_update"
M18A_TARGET_FIELD = "Tracks.eta"

# Exact or derived target fields that must never enter prediction-time features.
M18A_FORBIDDEN_FEATURES = frozenset({
    "eta",
    "eta_reference_raw",
    "eta_reference_dt",
    "target_tte_h",
    "reference_eta_status",
    "prediction_tte_h",
    "abs_error_h",
    "error_h",
    "covered_80",
})


def sha256_file(path: str | Path) -> str:
    """Return a stable SHA-256 digest for a file."""
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def forbidden_prediction_features(feature_names: Iterable[str]) -> list[str]:
    """Return prediction-time feature names that violate the M18A contract."""
    bad: list[str] = []
    for name in feature_names:
        n = str(name).strip().lower()
        if n in M18A_FORBIDDEN_FEATURES:
            bad.append(str(name))
            continue
        # Conservative guard for explicit target/error derivatives.  It avoids
        # banning legitimate fields such as destination or eta-like substrings.
        if n.startswith("target_") or n.startswith("label_") or n.endswith("_target"):
            bad.append(str(name))
        elif n.startswith("future_") or n.startswith("post_decision_"):
            bad.append(str(name))
    return sorted(set(bad))


def assert_feature_contract(feature_names: Iterable[str]) -> None:
    """Raise when any target-derived/future-only feature is proposed."""
    bad = forbidden_prediction_features(feature_names)
    if bad:
        raise AssertionError(f"M18A prediction contract forbids features: {bad}")


def assert_history_not_after_decision_time(
    rows: pd.DataFrame,
    decision_time_by_owner: Mapping[int, object] | pd.Series,
    *,
    owner_col: str = "mmsi",
    timestamp_col: str = "recorded_at",
) -> None:
    """Assert every historical observation is available by its row decision time."""
    if owner_col not in rows or timestamp_col not in rows:
        raise KeyError(f"required columns missing: {owner_col}, {timestamp_col}")
    if isinstance(decision_time_by_owner, pd.Series):
        cut = decision_time_by_owner.to_dict()
    else:
        cut = dict(decision_time_by_owner)
    ts = pd.to_datetime(rows[timestamp_col], errors="coerce")
    owners = rows[owner_col].astype(int)
    decision = pd.to_datetime(owners.map(cut), errors="coerce")
    if decision.isna().any():
        missing = sorted(set(int(x) for x in owners[decision.isna()].tolist()))
        raise AssertionError(f"missing decision time for owners: {missing[:10]}")
    if ts.isna().any():
        raise AssertionError("historical row contains invalid timestamp")
    mask = ts > decision
    if mask.any():
        sample = rows.loc[mask, [owner_col, timestamp_col]].head(5).to_dict("records")
        raise AssertionError(f"post-decision history leakage detected: {sample}")


def validate_contract_dict(contract: Mapping[str, object]) -> None:
    """Validate the machine-readable M18A contract's non-negotiable fields."""
    if contract.get("milestone") != "M18A":
        raise AssertionError("M18A contract milestone mismatch")
    if contract.get("version") != M18A_VERSION:
        raise AssertionError("M18A contract version mismatch")
    task = contract.get("primary_prediction_contract")
    if not isinstance(task, Mapping):
        raise AssertionError("primary_prediction_contract missing")
    required = {
        "task_id": M18A_PRIMARY_TASK_ID,
        "decision_time": M18A_DECISION_TIME_FIELD,
        "label_source": M18A_TARGET_FIELD,
        "target_formula": "target_tte_h = parsed Tracks.eta - Tracks.last_update",
        "semantic_status": "company reference ETA; not observed ATA and not berth/all-fast arrival",
    }
    for key, expected in required.items():
        if task.get(key) != expected:
            raise AssertionError(f"contract field {key!r} mismatch")
    leakage = contract.get("leakage_boundary")
    if not isinstance(leakage, Mapping):
        raise AssertionError("leakage_boundary missing")
    if leakage.get("positions_rule") != "Positions.recorded_at <= Tracks.last_update":
        raise AssertionError("decision-time history boundary mismatch")
    holdout = contract.get("fresh_holdout_policy")
    if not isinstance(holdout, Mapping) or holdout.get("selection_use_allowed") is not False:
        raise AssertionError("fresh holdout must remain selection-blocked")
