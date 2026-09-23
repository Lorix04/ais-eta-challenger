"""M18B validation-redesign utilities.

M18B does not train or replace a predictor.  It freezes a stricter validation
protocol around the M18A prediction contract:

* one-row-per-MMSI owner separation,
* contiguous chronological blocks with no timestamp split across blocks,
* expanding forward windows with a six-hour embargo (matching the longest
  causal AIS history used by the frozen route expert),
* explicit distinction between diagnostic slicing of legacy OOF predictions
  and genuine forward-temporal evaluation that requires refitting inside each
  window,
* fresh-holdout lockbox rules and paired owner-level uncertainty.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

M18B_VERSION = "m18b-validation-redesign-v1-20260923"
M18B_TEMPORAL_BLOCKS = 5
M18B_TEST_BLOCKS = (2, 3, 4)
M18B_EMBARGO_H = 6.0
M18B_BOOTSTRAP_SEED = 18023
M18B_BOOTSTRAP_REPS = 10_000


@dataclass(frozen=True)
class ForwardWindow:
    fold_id: int
    test_block: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_end_exclusive: pd.Timestamp


def _decision_times(values: Sequence[object] | pd.Series) -> pd.Series:
    t = pd.to_datetime(pd.Series(values), errors="coerce", utc=False)
    if t.isna().any():
        raise AssertionError("M18B decision times contain invalid timestamps")
    return t


def temporal_block_boundaries(
    decision_times: Sequence[object] | pd.Series,
    *,
    n_blocks: int = M18B_TEMPORAL_BLOCKS,
) -> list[pd.Timestamp]:
    """Return near-equal-count chronological boundaries without splitting ties."""
    if n_blocks < 2:
        raise ValueError("n_blocks must be >= 2")
    t = _decision_times(decision_times)
    if len(t) < n_blocks:
        raise ValueError("not enough rows for requested temporal blocks")
    counts = t.value_counts().sort_index()
    times = list(counts.index)
    cumulative = np.cumsum(counts.to_numpy(int))
    total = int(len(t))
    out: list[pd.Timestamp] = []
    for k in range(1, n_blocks):
        target = k * total / n_blocks
        idx = int(np.searchsorted(cumulative, target, side="left"))
        if idx + 1 >= len(times):
            raise AssertionError("cannot place a non-empty chronological boundary")
        boundary = pd.Timestamp(times[idx + 1])
        if out and boundary <= out[-1]:
            raise AssertionError("temporal boundaries are not strictly increasing")
        out.append(boundary)
    return out


def assign_temporal_blocks(
    decision_times: Sequence[object] | pd.Series,
    boundaries: Sequence[object],
) -> np.ndarray:
    """Assign each decision timestamp to one contiguous chronological block."""
    t = _decision_times(decision_times)
    b = pd.to_datetime(pd.Series(list(boundaries)), errors="coerce")
    if b.isna().any() or not b.is_monotonic_increasing or b.duplicated().any():
        raise AssertionError("invalid M18B temporal boundaries")
    arr = np.searchsorted(
        b.to_numpy(dtype="datetime64[ns]"),
        t.to_numpy(dtype="datetime64[ns]"),
        side="right",
    )
    return arr.astype(int)


def build_validation_manifest(
    dev: pd.DataFrame,
    *,
    blocked_old_final_mmsi: Sequence[int],
    outer_fold_by_mmsi: Mapping[int, int] | None = None,
    canonical_destination_by_mmsi: Mapping[int, str] | None = None,
) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    """Create the frozen row-level M18B validation manifest."""
    required = {"mmsi", "split", "last_update", "target_tte_h", "reference_eta_status"}
    missing = required - set(dev.columns)
    if missing:
        raise KeyError(f"M18B missing required columns: {sorted(missing)}")
    out = dev.copy()
    if len(out) != 386 or out["mmsi"].astype(int).nunique() != 386:
        raise AssertionError("M18B primary development set must be 386 unique MMSIs")
    blocked = {int(x) for x in blocked_old_final_mmsi}
    owners = set(out["mmsi"].astype(int))
    if owners & blocked:
        raise AssertionError("M18B development manifest contains blocked old-final MMSI")
    if not out["split"].isin(["train", "calibration"]).all():
        raise AssertionError("M18B development rows must remain train/calibration only")

    out["decision_time"] = pd.to_datetime(out["last_update"], errors="coerce")
    boundaries = temporal_block_boundaries(out["decision_time"])
    out["m18b_temporal_block"] = assign_temporal_blocks(out["decision_time"], boundaries)
    out["m18b_strict_forward_test"] = out["m18b_temporal_block"].isin(M18B_TEST_BLOCKS)
    out["m18b_warmup_only"] = out["m18b_temporal_block"].isin([0, 1])
    if outer_fold_by_mmsi is not None:
        out["legacy_m16a_outer_fold"] = out["mmsi"].astype(int).map(dict(outer_fold_by_mmsi))
        if out["legacy_m16a_outer_fold"].isna().any():
            raise AssertionError("missing legacy M16A outer fold in M18B manifest")
    if canonical_destination_by_mmsi is not None:
        out["canonical_destination"] = (
            out["mmsi"].astype(int).map(dict(canonical_destination_by_mmsi)).fillna("UNKNOWN").astype(str)
        )
    cols = [
        "mmsi", "split", "decision_time", "target_tte_h", "reference_eta_status",
        "m18b_temporal_block", "m18b_strict_forward_test", "m18b_warmup_only",
    ]
    for optional in ["legacy_m16a_outer_fold", "canonical_destination"]:
        if optional in out:
            cols.append(optional)
    return out[cols].sort_values(["decision_time", "mmsi"], kind="mergesort").reset_index(drop=True), boundaries


def build_forward_membership(
    manifest: pd.DataFrame,
    *,
    embargo_h: float = M18B_EMBARGO_H,
    test_blocks: Sequence[int] = M18B_TEST_BLOCKS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create expanding-window membership and fold summaries.

    Rows in earlier blocks but inside the embargo are PURGED.  Later blocks are
    FUTURE_UNUSED.  TEST owners never appear in TRAIN because the primary M18A
    contract has one target row per MMSI.
    """
    if embargo_h < 0:
        raise ValueError("embargo_h must be non-negative")
    m = manifest.copy()
    m["decision_time"] = pd.to_datetime(m["decision_time"])
    rows: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for fold_id, test_block in enumerate(test_blocks):
        test_mask = m["m18b_temporal_block"].astype(int).eq(int(test_block))
        if not test_mask.any():
            raise AssertionError(f"empty M18B test block {test_block}")
        test_start = pd.Timestamp(m.loc[test_mask, "decision_time"].min())
        test_end = pd.Timestamp(m.loc[test_mask, "decision_time"].max())
        train_end = test_start - pd.Timedelta(hours=float(embargo_h))
        earlier = m["m18b_temporal_block"].astype(int) < int(test_block)
        train_mask = earlier & (m["decision_time"] < train_end)
        purged_mask = earlier & ~train_mask
        future_mask = m["m18b_temporal_block"].astype(int) > int(test_block)

        role = np.full(len(m), "UNUSED", dtype=object)
        role[train_mask.to_numpy()] = "TRAIN"
        role[purged_mask.to_numpy()] = "PURGED"
        role[test_mask.to_numpy()] = "TEST"
        role[future_mask.to_numpy()] = "FUTURE_UNUSED"
        part = m[["mmsi", "decision_time", "m18b_temporal_block"]].copy()
        part.insert(0, "m18b_forward_fold", fold_id)
        part["m18b_role"] = role

        train_owners = set(part.loc[part["m18b_role"].eq("TRAIN"), "mmsi"].astype(int))
        test_owners = set(part.loc[part["m18b_role"].eq("TEST"), "mmsi"].astype(int))
        if train_owners & test_owners:
            raise AssertionError("M18B owner leakage between forward train and test")

        if "canonical_destination" in m:
            train_dest = set(m.loc[train_mask, "canonical_destination"].astype(str))
            part["destination_seen_in_train"] = False
            part.loc[test_mask, "destination_seen_in_train"] = m.loc[test_mask, "canonical_destination"].astype(str).isin(train_dest).to_numpy()
        rows.append(part)
        summaries.append({
            "m18b_forward_fold": fold_id,
            "test_block": int(test_block),
            "train_n": int(train_mask.sum()),
            "purged_n": int(purged_mask.sum()),
            "test_n": int(test_mask.sum()),
            "future_unused_n": int(future_mask.sum()),
            "train_owner_n": len(train_owners),
            "test_owner_n": len(test_owners),
            "owner_overlap_n": len(train_owners & test_owners),
            "test_start": test_start,
            "test_end": test_end,
            "train_end_exclusive": train_end,
            "embargo_h": float(embargo_h),
        })
    return pd.concat(rows, ignore_index=True), pd.DataFrame(summaries)


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if y.shape != p.shape or y.ndim != 1 or len(y) == 0:
        raise ValueError("invalid regression metric vectors")
    err = p - y
    ae = np.abs(err)
    return {
        "n": int(len(y)),
        "mae_h": float(np.mean(ae)),
        "medae_h": float(np.median(ae)),
        "rmse_h": float(np.sqrt(np.mean(err ** 2))),
        "p90_ae_h": float(np.quantile(ae, 0.90)),
        "mean_signed_error_h": float(np.mean(err)),
    }


def paired_mae_gain_bootstrap(
    y_true: Sequence[float],
    baseline_pred: Sequence[float],
    candidate_pred: Sequence[float],
    *,
    reps: int = M18B_BOOTSTRAP_REPS,
    seed: int = M18B_BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    """Owner-row paired bootstrap of baseline MAE minus candidate MAE."""
    y = np.asarray(y_true, dtype=float)
    b = np.asarray(baseline_pred, dtype=float)
    c = np.asarray(candidate_pred, dtype=float)
    if not (y.shape == b.shape == c.shape) or y.ndim != 1 or len(y) < 2:
        raise ValueError("invalid paired bootstrap vectors")
    row_gain = np.abs(y - b) - np.abs(y - c)
    rng = np.random.default_rng(int(seed))
    n = len(y)
    # Keep memory bounded while remaining deterministic.
    vals = np.empty(int(reps), dtype=float)
    batch = 512
    for start in range(0, int(reps), batch):
        stop = min(start + batch, int(reps))
        idx = rng.integers(0, n, size=(stop - start, n))
        vals[start:stop] = row_gain[idx].mean(axis=1)
    return {
        "n": int(n),
        "reps": int(reps),
        "seed": int(seed),
        "observed_gain_h": float(row_gain.mean()),
        "gain_ci95_low_h": float(np.quantile(vals, 0.025)),
        "gain_ci95_high_h": float(np.quantile(vals, 0.975)),
        "probability_gain_positive": float(np.mean(vals > 0.0)),
    }


def validate_protocol_dict(protocol: Mapping[str, object]) -> None:
    if protocol.get("milestone") != "M18B" or protocol.get("version") != M18B_VERSION:
        raise AssertionError("M18B protocol identity mismatch")
    design = protocol.get("strict_forward_temporal_design")
    if not isinstance(design, Mapping):
        raise AssertionError("M18B strict_forward_temporal_design missing")
    if int(design.get("temporal_blocks", -1)) != M18B_TEMPORAL_BLOCKS:
        raise AssertionError("M18B temporal block count mismatch")
    if tuple(design.get("test_blocks", [])) != M18B_TEST_BLOCKS:
        raise AssertionError("M18B test block definition mismatch")
    if float(design.get("embargo_h", -1)) != M18B_EMBARGO_H:
        raise AssertionError("M18B embargo mismatch")
    if design.get("legacy_oof_metrics_are_forward_generalization") is not False:
        raise AssertionError("legacy OOF must not be called forward temporal validation")
    port = protocol.get("cross_port_validation")
    if not isinstance(port, Mapping) or port.get("status") != "WITHHELD_NO_AUTHORITATIVE_PORT_EVENT_LABEL":
        raise AssertionError("M18B must withhold cross-port claims under M18A target semantics")
    holdout = protocol.get("fresh_external_lockbox")
    if not isinstance(holdout, Mapping) or holdout.get("opened") is not False:
        raise AssertionError("fresh external lockbox must remain unopened")
