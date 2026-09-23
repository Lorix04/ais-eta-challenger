"""M16E leakage-safe maritime / physics expert utilities.

The expert is deliberately interpretable.  It uses only prediction-time
information: canonical destination coordinates from M16B, current position,
recent causal SOG summaries and recent travelled/net displacement summaries.
Target values enter only in an optional train-only residual correction.

The physics ETA is not asserted to be a navigationally authoritative route ETA.
Great-circle distance is a lower bound; the optional local-sinuosity multiplier
is only a causal route-detour proxy derived from the vessel's recent track.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import pandas as pd

from .m16a import balanced_hash_folds, extended_metrics

M16E_VERSION = "m16e-maritime-physics-v1-20260921"
M16E_INNER_FOLDS = 4
M16E_INNER_SALT = "M16E_INNER_V1_20260921"
M16E_MIN_SPEED_KN = 2.0
M16E_MAX_SPEED_KN = 25.0
M16E_MIN_DISTANCE_NM = 1.0
M16E_MIN_DEST_CONFIDENCE = 0.99
M16E_MAX_SINUOSITY = 1.35
M16E_RESIDUAL_MIN_SUPPORT = 3
M16E_RESIDUAL_ALPHA = 10.0


def haversine_nm(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorized great-circle distance in nautical miles."""
    lat1 = np.asarray(lat1, dtype=float)
    lon1 = np.asarray(lon1, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dlat = p2 - p1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * 3440.065 * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def initial_bearing_deg(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Initial great-circle bearing in degrees [0, 360)."""
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    dlon = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def circular_diff_deg(a, b) -> np.ndarray:
    """Absolute circular angular difference in [0, 180]."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add target-free M16E physics/gating features."""
    required = {
        "track_lat", "track_lon", "track_cog", "canonical_lat", "canonical_lon",
        "is_resolved_port", "resolution_confidence", "canonical_destination",
        "sog_median_180m", "sog_median_360m",
        "distance_travelled_km_180m", "net_displacement_km_180m",
        "distance_travelled_km_360m", "net_displacement_km_360m",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"M16E missing columns: {sorted(missing)}")

    x = df.copy()
    x["physics_distance_gc_nm"] = haversine_nm(
        pd.to_numeric(x["track_lat"], errors="coerce"),
        pd.to_numeric(x["track_lon"], errors="coerce"),
        pd.to_numeric(x["canonical_lat"], errors="coerce"),
        pd.to_numeric(x["canonical_lon"], errors="coerce"),
    )
    x["physics_bearing_to_port_deg"] = initial_bearing_deg(
        pd.to_numeric(x["track_lat"], errors="coerce"),
        pd.to_numeric(x["track_lon"], errors="coerce"),
        pd.to_numeric(x["canonical_lat"], errors="coerce"),
        pd.to_numeric(x["canonical_lon"], errors="coerce"),
    )
    x["physics_course_alignment_deg"] = circular_diff_deg(
        pd.to_numeric(x["track_cog"], errors="coerce"),
        x["physics_bearing_to_port_deg"],
    )

    for window in (180, 360):
        travelled = pd.to_numeric(x[f"distance_travelled_km_{window}m"], errors="coerce")
        net = pd.to_numeric(x[f"net_displacement_km_{window}m"], errors="coerce")
        ratio = travelled / net.clip(lower=1.0)
        ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(1.0, M16E_MAX_SINUOSITY)
        x[f"physics_local_sinuosity_{window}m"] = ratio.astype(float)

    s180 = pd.to_numeric(x["sog_median_180m"], errors="coerce")
    s360 = pd.to_numeric(x["sog_median_360m"], errors="coerce")
    x["physics_recent_speed_max_kn"] = pd.concat([s180, s360], axis=1).max(axis=1, skipna=True)
    x["physics_eligible"] = (
        x["is_resolved_port"].fillna(False).astype(bool)
        & (pd.to_numeric(x["resolution_confidence"], errors="coerce") >= M16E_MIN_DEST_CONFIDENCE)
        & np.isfinite(x["physics_distance_gc_nm"])
        & (x["physics_distance_gc_nm"] >= M16E_MIN_DISTANCE_NM)
        & (x["physics_recent_speed_max_kn"] >= M16E_MIN_SPEED_KN)
    )
    return x


@dataclass(frozen=True)
class PhysicsConfig:
    speed_window_min: int
    distance_mode: str
    correction: str

    @property
    def config_id(self) -> str:
        return f"sog{self.speed_window_min}__{self.distance_mode}__corr_{self.correction}"


def candidate_configs() -> list[PhysicsConfig]:
    return [
        PhysicsConfig(w, d, c)
        for w in (180, 360)
        for d in ("geodesic", "local_sinuosity")
        for c in ("none", "global_residual", "destination_residual")
    ]


def _effective_speed(x: pd.DataFrame, window: int) -> pd.Series:
    primary = pd.to_numeric(x[f"sog_median_{window}m"], errors="coerce")
    alt_window = 360 if window == 180 else 180
    alternate = pd.to_numeric(x[f"sog_median_{alt_window}m"], errors="coerce")
    speed = primary.where(primary >= M16E_MIN_SPEED_KN, alternate)
    return speed.clip(lower=M16E_MIN_SPEED_KN, upper=M16E_MAX_SPEED_KN)


def base_physics_prediction(x: pd.DataFrame, config: PhysicsConfig) -> pd.DataFrame:
    """Target-free distance/speed ETA for every row; caller applies eligibility."""
    speed = _effective_speed(x, config.speed_window_min)
    if config.distance_mode == "geodesic":
        factor = pd.Series(1.0, index=x.index, dtype=float)
    elif config.distance_mode == "local_sinuosity":
        factor = pd.to_numeric(x[f"physics_local_sinuosity_{config.speed_window_min}m"], errors="coerce").fillna(1.0)
    else:
        raise ValueError(f"unknown M16E distance mode: {config.distance_mode}")
    route_distance = pd.to_numeric(x["physics_distance_gc_nm"], errors="coerce") * factor
    pred = route_distance / speed
    return pd.DataFrame({
        "base_prediction_h": pred.astype(float),
        "effective_speed_kn": speed.astype(float),
        "distance_factor": factor.astype(float),
        "route_distance_proxy_nm": route_distance.astype(float),
    }, index=x.index)


def _residual_correction(train: pd.DataFrame, valid: pd.DataFrame, train_base: pd.Series, config: PhysicsConfig, target_col: str) -> pd.Series:
    eligible = train["physics_eligible"].fillna(False).astype(bool)
    if int(eligible.sum()) == 0:
        raise ValueError("M16E residual correction has no eligible training rows")
    residual = pd.to_numeric(train[target_col], errors="raise").astype(float) - train_base.astype(float)
    global_med = float(residual.loc[eligible].median())
    if config.correction == "none":
        return pd.Series(0.0, index=valid.index, dtype=float)
    if config.correction == "global_residual":
        return pd.Series(global_med, index=valid.index, dtype=float)
    if config.correction != "destination_residual":
        raise ValueError(f"unknown M16E correction: {config.correction}")

    stats_frame = train.loc[eligible, ["canonical_destination"]].copy()
    stats_frame["residual"] = residual.loc[eligible].to_numpy(float)
    stats = stats_frame.groupby("canonical_destination", dropna=False)["residual"].agg(["median", "count"])
    med = valid["canonical_destination"].map(stats["median"]).astype(float)
    count = valid["canonical_destination"].map(stats["count"]).astype(float)
    corr = pd.Series(global_med, index=valid.index, dtype=float)
    ok = med.notna() & count.notna() & (count >= M16E_RESIDUAL_MIN_SUPPORT)
    weight = count / (count + M16E_RESIDUAL_ALPHA)
    corr.loc[ok] = weight.loc[ok] * med.loc[ok] + (1.0 - weight.loc[ok]) * global_med
    return corr


def predict_physics_expert(train: pd.DataFrame, valid: pd.DataFrame, config: PhysicsConfig, target_col: str = "target_tte_h") -> pd.DataFrame:
    """Predict validation rows with target-free physics + train-only residual correction."""
    tr_base = base_physics_prediction(train, config)
    va_base = base_physics_prediction(valid, config)
    corr = _residual_correction(train, valid, tr_base["base_prediction_h"], config, target_col)
    pred = va_base["base_prediction_h"] + corr
    out = va_base.copy()
    out["residual_correction_h"] = corr.astype(float)
    out["prediction_h"] = pred.astype(float)
    out["physics_eligible"] = valid["physics_eligible"].astype(bool).to_numpy()
    return out


def select_config_inner_cv(
    outer_train: pd.DataFrame,
    outer_fold: int,
    configs: Sequence[PhysicsConfig] | None = None,
    target_col: str = "target_tte_h",
) -> tuple[PhysicsConfig, pd.DataFrame]:
    """Select M16E configuration using eligible rows in inner validation only."""
    configs = list(configs or candidate_configs())
    fold_map = balanced_hash_folds(
        outer_train["mmsi"], n_splits=M16E_INNER_FOLDS,
        salt=f"{M16E_INNER_SALT}:outer={int(outer_fold)}",
    )
    work = outer_train.copy()
    work["m16e_inner_fold"] = work["mmsi"].map(fold_map).astype(int)
    rows: list[dict] = []
    for config in configs:
        ys: list[float] = []
        ps: list[float] = []
        eligible_count = 0
        for inner in range(M16E_INNER_FOLDS):
            tr = work.loc[work["m16e_inner_fold"].ne(inner)].copy()
            va = work.loc[work["m16e_inner_fold"].eq(inner)].copy()
            pred = predict_physics_expert(tr, va, config, target_col=target_col)
            mask = pred["physics_eligible"].to_numpy(bool)
            eligible_count += int(mask.sum())
            ys.extend(pd.to_numeric(va.loc[mask, target_col]).astype(float).tolist())
            ps.extend(pred.loc[mask, "prediction_h"].astype(float).tolist())
        if eligible_count == 0:
            raise AssertionError("M16E candidate has zero inner eligible rows")
        m = extended_metrics(ys, ps)
        rows.append({
            "outer_fold": int(outer_fold), "config_id": config.config_id,
            "speed_window_min": int(config.speed_window_min), "distance_mode": config.distance_mode,
            "correction": config.correction, "eligible_inner_rows": int(eligible_count), **m,
        })
    search = pd.DataFrame(rows).sort_values(["mae_h", "p90_ae_h", "medae_h", "config_id"], kind="mergesort").reset_index(drop=True)
    best = next(c for c in configs if c.config_id == search.iloc[0]["config_id"])
    return best, search
