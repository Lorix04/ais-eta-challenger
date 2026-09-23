"""M16D leakage-safe historical route-analogue utilities.

Trajectory representations are built independently per MMSI from observations at
or before the row's causal history cutoff. Pairwise route distances are target
free. Target values enter only when an outer/inner training subset supplies the
historical neighbour outcomes used for prediction.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
import pandas as pd

M16D_VERSION = "m16d-route-analogue-v1-20260921"
M16D_INNER_SALT = "M16D_INNER_V1_20260921"
M16D_INNER_FOLDS = 4
M16D_POINTS = 12
M16D_DTW_BAND = 2


@dataclass(frozen=True)
class RouteConfig:
    representation: str
    gate: str
    k: int

    @property
    def config_id(self) -> str:
        return f"{self.representation}__{self.gate}__k{self.k}"


def candidate_configs() -> list[RouteConfig]:
    reps = ["geo_3h", "geo_kin_3h", "geo_kin_6h"]
    return [RouteConfig(r, g, k) for r in reps for g in ("global", "destination") for k in (5, 9)]


def representation_spec(name: str) -> tuple[int, bool]:
    if name == "geo_3h":
        return 3, False
    if name == "geo_kin_3h":
        return 3, True
    if name == "geo_kin_6h":
        return 6, True
    raise KeyError(name)


def resample_causal_trajectory(
    states: pd.DataFrame,
    *,
    end_at: pd.Timestamp,
    hours: int,
    n_points: int = M16D_POINTS,
    include_kinematics: bool = False,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Build one causal, fixed-length trajectory representation.

    Only valid positions with recorded_at <= end_at are eligible.  The selected
    history is resampled across its observed time span, avoiding any use of
    target/reference ETA information or future trajectory points.
    """
    g = states.copy()
    g["recorded_at"] = pd.to_datetime(g["recorded_at"])
    start_at = pd.Timestamp(end_at) - pd.Timedelta(hours=hours)
    mask = g["recorded_at"].between(start_at, pd.Timestamp(end_at), inclusive="both")
    if "position_valid" in g.columns:
        mask &= g["position_valid"].fillna(False).astype(bool)
    g = g.loc[mask].sort_values("recorded_at", kind="mergesort")
    g = g.dropna(subset=["lat_clean", "lon_clean"])
    if len(g) < 2:
        raise ValueError("insufficient causal trajectory points")

    # Deterministic duplicate timestamp handling: keep the first row.
    t = g["recorded_at"].astype("int64").to_numpy(dtype=np.int64) / 1e9
    _, keep = np.unique(t, return_index=True)
    g = g.iloc[keep].copy()
    t = t[keep]
    if len(t) < 2 or t[-1] <= t[0]:
        raise ValueError("insufficient unique trajectory timestamps")

    q = np.linspace(float(t[0]), float(t[-1]), int(n_points))
    lat = np.interp(q, t, g["lat_clean"].to_numpy(float))
    lon = np.interp(q, t, g["lon_clean"].to_numpy(float))

    # Fixed physical scaling, not learned from any development/final population.
    # Approximate local equirectangular projection around eastern Sicily.
    x_km = lon * 111.32 * math.cos(math.radians(37.0))
    y_km = lat * 111.32
    cols = [x_km / 50.0, y_km / 50.0]

    if include_kinematics:
        sog_s = pd.to_numeric(g.get("sog_clean"), errors="coerce").ffill().bfill().fillna(0.0)
        cog_s = pd.to_numeric(g.get("cog_clean"), errors="coerce").ffill().bfill().fillna(0.0)
        sog = np.interp(q, t, sog_s.to_numpy(float))
        cog = np.interp(q, t, cog_s.to_numpy(float))
        # Kinematics are deliberately lower-weight than spatial geometry.
        cols.extend([
            0.5 * sog / 10.0,
            0.25 * np.sin(np.deg2rad(cog)),
            0.25 * np.cos(np.deg2rad(cog)),
        ])

    seq = np.column_stack(cols).astype(np.float64)
    meta = {
        "raw_points": int(len(g)),
        "window_h": int(hours),
        "observed_span_h": float((t[-1] - t[0]) / 3600.0),
        "first_at": pd.to_datetime(t[0], unit="s").isoformat(),
        "last_at": pd.to_datetime(t[-1], unit="s").isoformat(),
    }
    return seq, meta


def dtw_distance(a: np.ndarray, b: np.ndarray, band: int = M16D_DTW_BAND) -> float:
    """Sakoe-Chiba band DTW with deterministic Euclidean point cost."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("DTW trajectories must be 2-D with matching feature dimensions")
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("inf")
    band = max(int(band), abs(n - m))
    inf = float("inf")
    prev = np.full(m + 1, inf, dtype=float)
    prev[0] = 0.0
    for i in range(1, n + 1):
        cur = np.full(m + 1, inf, dtype=float)
        j0 = max(1, i - band)
        j1 = min(m, i + band)
        ai = a[i - 1]
        for j in range(j0, j1 + 1):
            cost = float(np.sqrt(np.sum((ai - b[j - 1]) ** 2)))
            cur[j] = cost + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return float(prev[m] / (n + m))


def pairwise_dtw(sequences: list[np.ndarray], band: int = M16D_DTW_BAND) -> np.ndarray:
    n = len(sequences)
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i):
            d = dtw_distance(sequences[i], sequences[j], band=band)
            out[i, j] = d
            out[j, i] = d
    return out


def weighted_median(values: Iterable[float], weights: Iterable[float]) -> float:
    v = np.asarray(list(values), dtype=float)
    w = np.asarray(list(weights), dtype=float)
    if len(v) == 0 or len(v) != len(w):
        raise ValueError("weighted_median requires equally sized non-empty inputs")
    if not np.isfinite(w).all() or float(w.sum()) <= 0:
        w = np.ones_like(v)
    order = np.argsort(v, kind="mergesort")
    v = v[order]
    w = w[order]
    cdf = np.cumsum(w) / float(w.sum())
    return float(v[int(np.searchsorted(cdf, 0.5, side="left"))])


def predict_route_analogue(
    *,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    distance_matrix: np.ndarray,
    targets: np.ndarray,
    destinations: np.ndarray,
    config: RouteConfig,
) -> pd.DataFrame:
    """Predict validation rows from historical neighbours in train_indices only."""
    train_indices = np.asarray(train_indices, dtype=int)
    valid_indices = np.asarray(valid_indices, dtype=int)
    rows: list[dict] = []
    for i in valid_indices:
        candidates = train_indices
        gate_used = "global"
        if config.gate == "destination":
            same = train_indices[destinations[train_indices] == destinations[i]]
            if len(same) >= 3:
                candidates = same
                gate_used = "destination"
            else:
                gate_used = "destination_fallback_global"
        d = distance_matrix[i, candidates]
        order = np.argsort(d, kind="mergesort")[: min(config.k, len(candidates))]
        nn = candidates[order]
        nd = d[order]
        if len(nn) == 0:
            raise AssertionError("M16D produced empty neighbour set")
        scale = max(float(np.median(nd)), 1e-9)
        weights = np.exp(-nd / scale)
        pred = weighted_median(targets[nn], weights)
        second = float(nd[1]) if len(nd) > 1 else float(nd[0])
        gap = float((second - nd[0]) / max(abs(float(nd[0])), 1e-9)) if len(nd) > 1 else 0.0
        rows.append({
            "row_index": int(i),
            "prediction_h": pred,
            "neighbour_count": int(len(nn)),
            "nearest_distance": float(nd[0]),
            "median_neighbour_distance": float(np.median(nd)),
            "similarity_gap": gap,
            "gate_used": gate_used,
            "nearest_index": int(nn[0]),
            "nearest_target_h": float(targets[nn[0]]),
            "nearest_weight": float(weights[0]),
            "neighbour_indices": ";".join(str(int(x)) for x in nn),
        })
    return pd.DataFrame(rows)
