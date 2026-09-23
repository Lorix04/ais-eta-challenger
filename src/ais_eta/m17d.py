"""M17D development-only extension of the frozen M16G Mixture of Experts.

M17D adds two post-freeze expert signals to the four M16G experts:
- M17A target-free self-supervised learned trajectory retrieval
- M17C causal historical corridor / graph gate

The meta learner is selected and fitted only on base predictions regenerated
inner-OOF inside each M16A outer-train partition.  The old 53-row M10 final
set is never eligible for fitting or selection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .m16a import extended_metrics

M17D_VERSION = "m17d-extended-mixture-of-experts-v1-20260922"
M17D_INNER_FOLDS = 3
M17D_INNER_SALT = "M17D_INNER_V1_20260922"
M17D_EXPERT_NAMES = (
    "prior", "route", "physics_gate", "tabular", "learned_retrieval", "corridor_graph_gate"
)
M17D_META_CANDIDATES = (
    "median_blend", "mean_blend", "nnls_convex", "rf_hard_gate",
    "extra_trees_hard_gate", "hist_gb_hard_gate", "logit_soft_gate",
)
M17D_MIN_POOLED_GAIN_H = 1.0
M17D_MIN_FOLD_WINS = 3
M17D_MIN_CHANGED_ROW_WIN_SHARE = 0.50
M17D_MAX_P90_RATIO = 1.05


@dataclass(frozen=True)
class MetaResult:
    prediction: np.ndarray
    selected_expert: np.ndarray
    weights: np.ndarray | None = None


def _ensure_finite_matrix(x: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or a.shape[1] != len(M17D_EXPERT_NAMES):
        raise ValueError(f"{name} must have shape (n, {len(M17D_EXPERT_NAMES)})")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains non-finite expert predictions")
    return a


def build_gating_features(base_predictions: np.ndarray, quality: pd.DataFrame) -> pd.DataFrame:
    """Build target-free gating features for the six-expert M17D panel."""
    b = _ensure_finite_matrix(base_predictions, "base_predictions")
    q = quality.reset_index(drop=True).copy()
    if len(q) != len(b):
        raise ValueError("quality rows must align with base predictions")
    bad_tokens = ("target", "reference_eta", "true_eta", "error", "absolute_error")
    bad = [c for c in q.columns if any(tok in c.lower() for tok in bad_tokens)]
    if bad:
        raise ValueError(f"target-derived gating columns are forbidden: {bad}")

    out = pd.DataFrame(b, columns=[f"pred_{x}_h" for x in M17D_EXPERT_NAMES])
    numeric = [
        "m16c_deepest_support", "m16c_deepest_weight", "m16c_fallback_count",
        "m16d_neighbour_count", "m16d_nearest_distance", "m16d_median_neighbour_distance",
        "m16d_similarity_gap", "physics_eligible", "resolution_confidence",
        "physics_distance_gc_nm", "physics_course_alignment_deg", "physics_recent_speed_max_kn",
        "m16e_effective_speed_kn", "m16e_distance_factor", "m16e_route_distance_proxy_nm",
        "m17a_neighbour_count", "m17a_nearest_distance", "m17a_median_neighbour_distance",
        "m17a_similarity_gap", "m17c_graph_supported", "m17c_path_edges",
        "m17c_min_path_support", "m17c_median_path_support", "m17c_contextual_edge_share",
        "m17c_source_snap_nm", "m17c_destination_snap_nm",
    ]
    for col in numeric:
        if col not in q.columns:
            # Graph details are naturally absent on unsupported rows; but the column
            # contract itself must still be explicit.
            raise ValueError(f"missing M17D quality feature: {col}")
        out[col] = pd.to_numeric(q[col], errors="coerce")

    route_gate = q.get("m16d_gate_used", pd.Series(["UNKNOWN"] * len(q))).fillna("UNKNOWN").astype(str)
    learned_gate = q.get("m17a_gate_used", pd.Series(["UNKNOWN"] * len(q))).fillna("UNKNOWN").astype(str)
    out["route_gate_destination"] = route_gate.eq("destination").astype(float)
    out["route_gate_fallback"] = route_gate.eq("destination_fallback_global").astype(float)
    out["route_gate_global"] = route_gate.eq("global").astype(float)
    out["learned_gate_destination"] = learned_gate.eq("destination").astype(float)
    out["learned_gate_fallback"] = learned_gate.eq("destination_fallback_global").astype(float)
    out["learned_gate_global"] = learned_gate.eq("global").astype(float)

    pred_cols = [f"pred_{x}_h" for x in M17D_EXPERT_NAMES]
    out["expert_mean_h"] = out[pred_cols].mean(axis=1)
    out["expert_std_h"] = out[pred_cols].std(axis=1)
    out["expert_range_h"] = out[pred_cols].max(axis=1) - out[pred_cols].min(axis=1)
    out["route_minus_tabular_h"] = out["pred_route_h"] - out["pred_tabular_h"]
    out["prior_minus_route_h"] = out["pred_prior_h"] - out["pred_route_h"]
    out["physics_gate_minus_route_h"] = out["pred_physics_gate_h"] - out["pred_route_h"]
    out["learned_minus_route_h"] = out["pred_learned_retrieval_h"] - out["pred_route_h"]
    out["graph_minus_physics_h"] = out["pred_corridor_graph_gate_h"] - out["pred_physics_gate_h"]
    out["learned_minus_tabular_h"] = out["pred_learned_retrieval_h"] - out["pred_tabular_h"]
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    return out


def _oracle_labels(base: np.ndarray, target: Sequence[float]) -> np.ndarray:
    y = np.asarray(target, dtype=float)
    if len(y) != len(base):
        raise ValueError("target/base row mismatch")
    return np.argmin(np.abs(base - y[:, None]), axis=1).astype(int)


def _classifier(candidate: str):
    if candidate == "rf_hard_gate":
        return RandomForestClassifier(
            n_estimators=300, min_samples_leaf=10, max_features=0.7,
            class_weight="balanced_subsample", random_state=42, n_jobs=1,
        )
    if candidate == "extra_trees_hard_gate":
        return ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=8, max_features=0.7,
            class_weight="balanced", random_state=42, n_jobs=1,
        )
    if candidate == "hist_gb_hard_gate":
        return HistGradientBoostingClassifier(
            max_leaf_nodes=5, min_samples_leaf=20, l2_regularization=10.0,
            learning_rate=0.05, max_iter=100, random_state=42, early_stopping=False,
        )
    if candidate == "logit_soft_gate":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, C=0.2, class_weight="balanced", random_state=42),
        )
    raise KeyError(candidate)


def fit_predict_meta(
    candidate: str,
    train_base: np.ndarray,
    train_target: Sequence[float],
    train_features: pd.DataFrame,
    valid_base: np.ndarray,
    valid_features: pd.DataFrame,
) -> MetaResult:
    if candidate not in M17D_META_CANDIDATES:
        raise KeyError(candidate)
    xb = _ensure_finite_matrix(train_base, "train_base")
    vb = _ensure_finite_matrix(valid_base, "valid_base")
    y = np.asarray(train_target, dtype=float)
    if len(y) != len(xb):
        raise ValueError("train target/base row mismatch")
    if len(train_features) != len(xb) or len(valid_features) != len(vb):
        raise ValueError("gating feature/base row mismatch")

    if candidate == "median_blend":
        return MetaResult(np.median(vb, axis=1), np.array(["BLEND_MEDIAN"] * len(vb), dtype=object))
    if candidate == "mean_blend":
        return MetaResult(np.mean(vb, axis=1), np.array(["BLEND_MEAN"] * len(vb), dtype=object))
    if candidate == "nnls_convex":
        weights, _ = nnls(xb, y)
        if float(weights.sum()) <= 0:
            weights = np.ones(xb.shape[1], dtype=float) / xb.shape[1]
        else:
            weights = weights / float(weights.sum())
        pred = vb @ weights
        label = "BLEND_NNLS:" + ",".join(f"{n}={w:.4f}" for n, w in zip(M17D_EXPERT_NAMES, weights))
        return MetaResult(pred, np.array([label] * len(vb), dtype=object), weights=weights)

    labels = _oracle_labels(xb, y)
    model = clone(_classifier(candidate)).fit(train_features, labels)
    if candidate == "logit_soft_gate":
        proba = model.predict_proba(valid_features)
        full = np.zeros((len(vb), len(M17D_EXPERT_NAMES)), dtype=float)
        full[:, np.asarray(model.classes_, dtype=int)] = proba
        pred = np.sum(vb * full, axis=1)
        selected = np.asarray(["SOFT:" + M17D_EXPERT_NAMES[int(i)] for i in np.argmax(full, axis=1)], dtype=object)
        return MetaResult(pred, selected, weights=full)

    labels_valid = np.asarray(model.predict(valid_features), dtype=int)
    pred = vb[np.arange(len(vb)), labels_valid]
    selected = np.asarray([M17D_EXPERT_NAMES[int(i)] for i in labels_valid], dtype=object)
    return MetaResult(pred, selected)


def score_candidates_inner_cv(
    base_predictions: np.ndarray,
    target: Sequence[float],
    gating_features: pd.DataFrame,
    inner_fold: Sequence[int],
    candidates: Sequence[str] = M17D_META_CANDIDATES,
) -> pd.DataFrame:
    b = _ensure_finite_matrix(base_predictions, "base_predictions")
    y = np.asarray(target, dtype=float)
    fold = np.asarray(inner_fold, dtype=int)
    if len(y) != len(b) or len(fold) != len(b) or len(gating_features) != len(b):
        raise ValueError("M17D inner-CV row mismatch")
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        yy: list[float] = []
        pp: list[float] = []
        for k in sorted(np.unique(fold)):
            tr = np.flatnonzero(fold != k)
            va = np.flatnonzero(fold == k)
            result = fit_predict_meta(
                candidate, b[tr], y[tr], gating_features.iloc[tr],
                b[va], gating_features.iloc[va],
            )
            yy.extend(y[va].tolist())
            pp.extend(result.prediction.tolist())
        rows.append({"candidate_id": candidate, **extended_metrics(yy, pp)})
    return pd.DataFrame(rows).sort_values(
        ["mae_h", "p90_ae_h", "medae_h", "candidate_id"], kind="mergesort"
    ).reset_index(drop=True)


def promotion_gate(
    frozen_m16g_metrics: dict[str, float],
    mixture_metrics: dict[str, float],
    *, fold_wins: int, changed_row_win_share: float,
) -> str:
    gain = float(frozen_m16g_metrics["mae_h"] - mixture_metrics["mae_h"])
    p90_ratio = float(mixture_metrics["p90_ae_h"] / frozen_m16g_metrics["p90_ae_h"])
    ok = (
        gain >= M17D_MIN_POOLED_GAIN_H
        and int(fold_wins) >= M17D_MIN_FOLD_WINS
        and float(changed_row_win_share) >= M17D_MIN_CHANGED_ROW_WIN_SHARE
        and p90_ratio <= M17D_MAX_P90_RATIO
    )
    return "PASS_M17D_EXTENDED_MOE" if ok else "NO_M17D_PROMOTION"
