"""M16F leakage-safe tabular challenger panel utilities.

The milestone compares a small, predeclared family panel on the frozen M16A
outer folds. Hyperparameters are selected only inside each outer-train fold.
The already-observed M10 final 53 MMSIs are never eligible for selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

from .m16a import balanced_hash_folds, extended_metrics

M16F_VERSION = "m16f-tabular-panel-v1-20260921"
M16F_INNER_FOLDS = 3
M16F_INNER_SALT = "M16F_INNER_V1_20260921"

# Common target-free representation. It deliberately excludes all OOF expert
# predictions (M16C/D/E) so M16G remains the first place where experts combine.
M16F_NUMERIC_FEATURES = [
    "track_lat", "track_lon", "track_sog", "track_cog", "track_heading", "track_draught",
    "track_msg_count", "last_hour_sin", "last_hour_cos", "last_dow",
    "position_age_min", "history_n", "history_span_h", "dynamic_state_age_min",
    "hist_destination_unique", "hist_destination_matches_track", "track_vs_hist_position_gap_km",
    "sog_median_180m", "sog_std_180m", "moving_fraction_180m", "stopped_fraction_180m",
    "turn_count_180m", "distance_travelled_km_180m", "net_displacement_km_180m",
    "sog_median_360m", "sog_std_360m", "moving_fraction_360m", "stopped_fraction_360m",
    "turn_count_360m", "distance_travelled_km_360m", "net_displacement_km_360m",
    "resolution_confidence", "is_resolved_port", "is_non_specific", "is_route_expression",
    "canonical_lat", "canonical_lon", "physics_distance_gc_nm", "physics_bearing_to_port_deg",
    "physics_course_alignment_deg", "physics_recent_speed_max_kn", "physics_local_sinuosity_180m",
    "physics_local_sinuosity_360m", "physics_eligible",
]
M16F_CATEGORICAL_FEATURES = [
    "destination_norm", "canonical_destination", "resolution_method", "ship_type_cat",
    "nav_status_cat", "flag_cat", "motion_state_cat", "stale_risk_cat",
]
M16F_FEATURES = M16F_NUMERIC_FEATURES + M16F_CATEGORICAL_FEATURES


@dataclass(frozen=True)
class TabularConfig:
    family: str
    config_id: str
    params: dict[str, Any]


def candidate_configs() -> list[TabularConfig]:
    """Small predeclared panel; two configurations per family."""
    return [
        TabularConfig("catboost", "cat_mae_d4", {"loss_function": "MAE", "iterations": 160, "depth": 4, "learning_rate": 0.04, "l2_leaf_reg": 10.0}),
        TabularConfig("catboost", "cat_huber24_d4", {"loss_function": "Huber:delta=24", "iterations": 160, "depth": 4, "learning_rate": 0.04, "l2_leaf_reg": 10.0}),
        TabularConfig("extra_trees", "et_leaf2", {"n_estimators": 80, "criterion": "squared_error", "min_samples_leaf": 2, "max_features": 0.7}),
        TabularConfig("extra_trees", "et_leaf5", {"n_estimators": 80, "criterion": "squared_error", "min_samples_leaf": 5, "max_features": 1.0}),
        TabularConfig("random_forest", "rf_leaf3", {"n_estimators": 80, "criterion": "squared_error", "min_samples_leaf": 3, "max_features": 0.7}),
        TabularConfig("random_forest", "rf_leaf6", {"n_estimators": 80, "criterion": "squared_error", "min_samples_leaf": 6, "max_features": 1.0}),
        TabularConfig("hist_gb", "hgb_l1_leaf10", {"loss": "absolute_error", "learning_rate": 0.06, "max_iter": 90, "max_leaf_nodes": 15, "min_samples_leaf": 10, "l2_regularization": 1.0}),
        TabularConfig("hist_gb", "hgb_quant50_leaf20", {"loss": "quantile", "quantile": 0.5, "learning_rate": 0.06, "max_iter": 90, "max_leaf_nodes": 15, "min_samples_leaf": 20, "l2_regularization": 5.0}),
        TabularConfig("svr", "svr_rbf_c10_e12", {"kernel": "rbf", "C": 10.0, "epsilon": 12.0, "gamma": "scale"}),
        TabularConfig("svr", "svr_linear_c1_e24", {"kernel": "linear", "C": 1.0, "epsilon": 24.0}),
    ]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(M16F_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"M16F missing features: {sorted(missing)}")
    x = df.copy()
    for c in M16F_CATEGORICAL_FEATURES:
        x[c] = x[c].fillna("UNKNOWN").astype(str)
    for c in M16F_NUMERIC_FEATURES:
        if x[c].dtype == bool:
            x[c] = x[c].astype(float)
        else:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    return x


def _sklearn_preprocessor(*, scale_numeric: bool) -> ColumnTransformer:
    num_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    numeric = Pipeline(num_steps)
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, M16F_NUMERIC_FEATURES),
        ("cat", categorical, M16F_CATEGORICAL_FEATURES),
    ], remainder="drop", sparse_threshold=0.0)


def _build_sklearn_model(config: TabularConfig) -> Pipeline:
    p = dict(config.params)
    if config.family == "extra_trees":
        model = ExtraTreesRegressor(random_state=42, n_jobs=1, **p)
        return Pipeline([("prep", _sklearn_preprocessor(scale_numeric=False)), ("model", model)])
    if config.family == "random_forest":
        model = RandomForestRegressor(random_state=42, n_jobs=1, **p)
        return Pipeline([("prep", _sklearn_preprocessor(scale_numeric=False)), ("model", model)])
    if config.family == "hist_gb":
        model = HistGradientBoostingRegressor(random_state=42, early_stopping=False, **p)
        return Pipeline([("prep", _sklearn_preprocessor(scale_numeric=False)), ("model", model)])
    if config.family == "svr":
        model = SVR(cache_size=256, **p)
        return Pipeline([("prep", _sklearn_preprocessor(scale_numeric=True)), ("model", model)])
    raise ValueError(f"not a sklearn family: {config.family}")


def fit_predict(train: pd.DataFrame, valid: pd.DataFrame, config: TabularConfig, target_col: str = "target_tte_h") -> np.ndarray:
    """Fit on train only and predict valid for one predeclared config."""
    tr = prepare_features(train)
    va = prepare_features(valid)
    y = pd.to_numeric(tr[target_col], errors="raise").astype(float)
    if config.family == "catboost":
        model = CatBoostRegressor(
            random_seed=42, verbose=False, allow_writing_files=False, thread_count=1,
            **dict(config.params),
        )
        model.fit(tr[M16F_FEATURES], y, cat_features=M16F_CATEGORICAL_FEATURES, verbose=False)
        return np.asarray(model.predict(va[M16F_FEATURES]), dtype=float)
    model = _build_sklearn_model(config)
    model.fit(tr[M16F_FEATURES], y)
    return np.asarray(model.predict(va[M16F_FEATURES]), dtype=float)


def select_family_config_inner_cv(
    outer_train: pd.DataFrame,
    outer_fold: int,
    family: str,
    configs: Sequence[TabularConfig] | None = None,
    target_col: str = "target_tte_h",
) -> tuple[TabularConfig, pd.DataFrame]:
    """Select one config for one family using only inner OOF predictions."""
    configs = [c for c in (configs or candidate_configs()) if c.family == family]
    if not configs:
        raise ValueError(f"no M16F configs for family={family}")
    fmap = balanced_hash_folds(
        outer_train["mmsi"], n_splits=M16F_INNER_FOLDS,
        salt=f"{M16F_INNER_SALT}:outer={int(outer_fold)}",
    )
    work = outer_train.copy()
    work["m16f_inner_fold"] = work["mmsi"].map(fmap).astype(int)
    rows: list[dict[str, Any]] = []
    for cfg in configs:
        ys: list[float] = []
        ps: list[float] = []
        for inner in range(M16F_INNER_FOLDS):
            tr = work.loc[work["m16f_inner_fold"].ne(inner)].copy()
            va = work.loc[work["m16f_inner_fold"].eq(inner)].copy()
            pred = fit_predict(tr, va, cfg, target_col=target_col)
            ys.extend(pd.to_numeric(va[target_col], errors="raise").astype(float).tolist())
            ps.extend(pred.tolist())
        rows.append({
            "outer_fold": int(outer_fold), "family": family, "config_id": cfg.config_id,
            **extended_metrics(ys, ps),
        })
    search = pd.DataFrame(rows).sort_values(
        ["mae_h", "p90_ae_h", "medae_h", "config_id"], kind="mergesort"
    ).reset_index(drop=True)
    best_id = str(search.iloc[0]["config_id"])
    best = next(c for c in configs if c.config_id == best_id)
    return best, search


def family_names(configs: Iterable[TabularConfig] | None = None) -> list[str]:
    return sorted({c.family for c in (configs or candidate_configs())})
