"""M16C fold-safe hierarchical destination priors.

This module builds a robust prior expert on top of the target-free M16B
canonical destination resolver.  All target-derived statistics must be fit on
training rows only.  Hyper-parameters are selected with an inner CV that lives
entirely inside each M16A outer-train fold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .m16a import balanced_hash_folds, extended_metrics

M16C_VERSION = "m16c-hierarchical-prior-v1-20260921"
M16C_INNER_FOLDS = 4
M16C_INNER_SALT = "M16C_INNER_V1_20260921"

DISTANCE_BINS_KM = (-np.inf, 25.0, 100.0, 300.0, 1000.0, np.inf)
DISTANCE_LABELS = ("D0_25", "D25_100", "D100_300", "D300_1000", "D1000_PLUS")

# Small, deliberately conservative search space.  The values are not selected
# on an outer validation fold: the complete tuple is chosen by inner CV only.
M16C_PARAMETER_PROFILES = (
    {"min_support": 2, "shrinkage_alpha": 10.0},
    {"min_support": 3, "shrinkage_alpha": 20.0},
    {"min_support": 5, "shrinkage_alpha": 50.0},
)
M16C_CLIP_QUANTILES = (None, 0.025, 0.05)

# The level names are precomputed target-free keys.  Every child contains its
# parent key, which makes the fallback path genuinely hierarchical.
M16C_HIERARCHIES: dict[str, tuple[str, ...]] = {
    "dest": ("key_dest",),
    "dest_ship": ("key_dest", "key_dest_ship"),
    "dest_distance": ("key_dest", "key_dest_distance"),
    "full": (
        "key_dest",
        "key_dest_distance",
        "key_dest_distance_ship",
        "key_dest_distance_ship_movement",
    ),
}


def _safe_str(s: pd.Series, fill: str = "UNKNOWN") -> pd.Series:
    return s.fillna(fill).astype(str)


def haversine_km(lat1: Iterable[float], lon1: Iterable[float], lat2: Iterable[float], lon2: Iterable[float]) -> np.ndarray:
    """Vectorized great-circle distance; NaN input yields NaN output."""
    lat1 = np.asarray(list(lat1), dtype=float)
    lon1 = np.asarray(list(lon1), dtype=float)
    lat2 = np.asarray(list(lat2), dtype=float)
    lon2 = np.asarray(list(lon2), dtype=float)
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dlat = p2 - p1
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * 6371.0088 * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def add_m16c_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add target-free hierarchy keys from current-state + M16B semantics."""
    required = {
        "canonical_destination", "canonical_lat", "canonical_lon",
        "track_lat", "track_lon", "track_sog", "ship_type_cat",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"M16C missing feature columns: {sorted(missing)}")

    x = df.copy()
    x["dest_key"] = _safe_str(x["canonical_destination"])
    x["ship_type_key"] = _safe_str(x["ship_type_cat"])
    x["distance_to_destination_km"] = haversine_km(
        pd.to_numeric(x["track_lat"], errors="coerce"),
        pd.to_numeric(x["track_lon"], errors="coerce"),
        pd.to_numeric(x["canonical_lat"], errors="coerce"),
        pd.to_numeric(x["canonical_lon"], errors="coerce"),
    )
    x["distance_band"] = (
        pd.cut(
            x["distance_to_destination_km"],
            bins=list(DISTANCE_BINS_KM),
            labels=list(DISTANCE_LABELS),
            include_lowest=True,
        )
        .astype(object)
        .fillna("DIST_UNKNOWN")
        .astype(str)
    )

    sog = pd.to_numeric(x["track_sog"], errors="coerce")
    x["movement_bucket"] = np.select(
        [sog.isna(), sog <= 0.5, sog <= 5.0],
        ["SOG_UNKNOWN", "STOPPED", "SLOW"],
        default="MOVING",
    )

    # Separator is an internal key delimiter; source categories are not parsed
    # from these values later, so the mapping remains deterministic/auditable.
    sep = "\x1f"
    x["key_dest"] = x["dest_key"]
    x["key_dest_ship"] = x["dest_key"] + sep + x["ship_type_key"]
    x["key_dest_distance"] = x["dest_key"] + sep + x["distance_band"]
    x["key_dest_distance_ship"] = x["key_dest_distance"] + sep + x["ship_type_key"]
    x["key_dest_distance_ship_movement"] = x["key_dest_distance_ship"] + sep + x["movement_bucket"]
    return x


@dataclass(frozen=True)
class PriorConfig:
    hierarchy: str
    min_support: int
    shrinkage_alpha: float
    clip_quantile: float | None

    @property
    def config_id(self) -> str:
        clip = "none" if self.clip_quantile is None else f"q{self.clip_quantile:g}"
        return f"{self.hierarchy}__n{self.min_support}__a{self.shrinkage_alpha:g}__clip_{clip}"


def candidate_configs() -> list[PriorConfig]:
    out: list[PriorConfig] = []
    for hierarchy in sorted(M16C_HIERARCHIES):
        for p in M16C_PARAMETER_PROFILES:
            for clip in M16C_CLIP_QUANTILES:
                out.append(PriorConfig(
                    hierarchy=hierarchy,
                    min_support=int(p["min_support"]),
                    shrinkage_alpha=float(p["shrinkage_alpha"]),
                    clip_quantile=clip,
                ))
    return out


def _group_stats(train: pd.DataFrame, key_col: str, target_col: str) -> pd.DataFrame:
    return train.groupby(key_col, dropna=False)[target_col].agg(["median", "count"])


def predict_hierarchical_prior(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    config: PriorConfig,
    target_col: str = "target_tte_h",
) -> pd.DataFrame:
    """Fit target statistics on ``train`` only and predict ``valid``.

    Child medians are shrunk toward the current parent prediction:
    ``w = n / (n + alpha)``.  If support is below the configured threshold or
    a key is unseen, the row remains at the parent level.  Optional clipping
    bounds are quantiles of the training targets only.
    """
    if config.hierarchy not in M16C_HIERARCHIES:
        raise ValueError(f"unknown M16C hierarchy {config.hierarchy!r}")
    if config.min_support < 1 or config.shrinkage_alpha < 0:
        raise ValueError("invalid M16C support/shrinkage")
    if len(train) == 0 or len(valid) == 0:
        raise ValueError("M16C train/valid must be non-empty")

    y = pd.to_numeric(train[target_col], errors="raise").astype(float)
    pred = np.full(len(valid), float(y.median()), dtype=float)
    deepest_level = np.zeros(len(valid), dtype=int)
    deepest_name = np.array(["global"] * len(valid), dtype=object)
    deepest_support = np.zeros(len(valid), dtype=float)
    deepest_weight = np.zeros(len(valid), dtype=float)
    attempted_levels = np.zeros(len(valid), dtype=int)

    if config.clip_quantile is None:
        lo, hi = -np.inf, np.inf
    else:
        q = float(config.clip_quantile)
        if not (0.0 < q < 0.5):
            raise ValueError("clip_quantile must be in (0, 0.5)")
        lo = float(y.quantile(q))
        hi = float(y.quantile(1.0 - q))

    for level_idx, key_col in enumerate(M16C_HIERARCHIES[config.hierarchy], start=1):
        if key_col not in train.columns or key_col not in valid.columns:
            raise ValueError(f"M16C key column missing: {key_col}")
        stats = _group_stats(train, key_col, target_col)
        med = valid[key_col].map(stats["median"]).to_numpy(dtype=float)
        count = valid[key_col].map(stats["count"]).to_numpy(dtype=float)
        seen = np.isfinite(count) & np.isfinite(med)
        attempted_levels[seen] += 1
        ok = seen & (count >= config.min_support)
        if not np.any(ok):
            continue
        weight = np.zeros(len(valid), dtype=float)
        if config.shrinkage_alpha == 0:
            weight[ok] = 1.0
        else:
            weight[ok] = count[ok] / (count[ok] + config.shrinkage_alpha)
        candidate = pred.copy()
        candidate[ok] = weight[ok] * med[ok] + (1.0 - weight[ok]) * pred[ok]
        candidate = np.clip(candidate, lo, hi)
        pred[ok] = candidate[ok]
        deepest_level[ok] = level_idx
        deepest_name[ok] = key_col
        deepest_support[ok] = count[ok]
        deepest_weight[ok] = weight[ok]

    fallback_count = len(M16C_HIERARCHIES[config.hierarchy]) - deepest_level
    return pd.DataFrame({
        "prediction_h": pred,
        "deepest_level": deepest_level,
        "deepest_level_name": deepest_name,
        "deepest_support": deepest_support,
        "deepest_weight": deepest_weight,
        "fallback_count": fallback_count,
    }, index=valid.index)


def select_config_inner_cv(
    outer_train: pd.DataFrame,
    outer_fold: int,
    target_col: str = "target_tte_h",
    configs: Sequence[PriorConfig] | None = None,
) -> tuple[PriorConfig, pd.DataFrame]:
    """Select one prior config using inner OOF predictions only."""
    configs = list(configs or candidate_configs())
    if not configs:
        raise ValueError("empty M16C candidate configuration set")

    fold_map = balanced_hash_folds(
        outer_train["mmsi"],
        n_splits=M16C_INNER_FOLDS,
        salt=f"{M16C_INNER_SALT}:outer={int(outer_fold)}",
    )
    work = outer_train.copy()
    work["m16c_inner_fold"] = work["mmsi"].map(fold_map).astype(int)

    rows: list[dict] = []
    for config in configs:
        y_all: list[float] = []
        p_all: list[float] = []
        for inner_fold in range(M16C_INNER_FOLDS):
            tr = work.loc[work["m16c_inner_fold"].ne(inner_fold)]
            va = work.loc[work["m16c_inner_fold"].eq(inner_fold)]
            if len(tr) == 0 or len(va) == 0:
                raise AssertionError("empty M16C inner fold")
            out = predict_hierarchical_prior(tr, va, config, target_col=target_col)
            y_all.extend(pd.to_numeric(va[target_col]).astype(float).tolist())
            p_all.extend(out["prediction_h"].astype(float).tolist())
        m = extended_metrics(y_all, p_all)
        rows.append({
            "outer_fold": int(outer_fold),
            "config_id": config.config_id,
            "hierarchy": config.hierarchy,
            "min_support": int(config.min_support),
            "shrinkage_alpha": float(config.shrinkage_alpha),
            "clip_quantile": config.clip_quantile,
            **m,
        })

    search = pd.DataFrame(rows).sort_values(
        ["mae_h", "p90_ae_h", "medae_h", "config_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    top = search.iloc[0]
    best = next(c for c in configs if c.config_id == top["config_id"])
    return best, search
