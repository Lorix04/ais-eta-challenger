"""M17E constrained selective router between frozen M16G and M17A.

The router has exactly two actions: keep the frozen M16G prediction or switch
to the M17A learned-retrieval prediction. Runtime features are target-free.
Outer-validation rows are never used to fit or select the router.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .m16a import extended_metrics

M17E_VERSION = "m17e-constrained-selective-router-v1-20260922"
M17E_INNER_FOLDS = 4
M17E_INNER_SALT = "M17E_INNER_V1_20260922"
M17E_LABEL_MARGIN_H = 1.0
M17E_MAX_SWITCH_SHARE = 0.35
M17E_ROUTER_CANDIDATES = (
    "always_m16g",
    "logit_c005_t070",
    "logit_c020_t070",
    "logit_c020_t080",
    "logit_c100_t080",
)
M17E_MIN_GAIN_H = 1.0
M17E_MIN_FOLD_WINS = 3
M17E_MIN_CHANGED_ROW_WIN_SHARE = 0.52
M17E_MAX_P90_RATIO = 1.0
M17E_MAX_SINGLE_FOLD_REGRESSION_H = 15.0


@dataclass(frozen=True)
class RouterResult:
    prediction: np.ndarray
    switch_to_m17a: np.ndarray
    probability_m17a_better: np.ndarray


def build_router_features(
    pred_m16g: Sequence[float],
    pred_m17a: Sequence[float],
    m16g_base4: np.ndarray,
    quality: pd.DataFrame,
) -> pd.DataFrame:
    """Construct target-free features for binary selective routing."""
    g = np.asarray(pred_m16g, dtype=float)
    a = np.asarray(pred_m17a, dtype=float)
    b = np.asarray(m16g_base4, dtype=float)
    q = quality.reset_index(drop=True).copy()
    if len(g) != len(a) or len(g) != len(b) or len(g) != len(q):
        raise ValueError("M17E router feature row mismatch")
    if b.ndim != 2 or b.shape[1] != 4 or not np.isfinite(b).all():
        raise ValueError("M17E m16g_base4 must be finite shape (n,4)")
    bad_tokens=("target", "reference_eta", "true_eta", "error", "absolute_error")
    bad=[c for c in q.columns if any(tok in c.lower() for tok in bad_tokens)]
    if bad:
        raise ValueError(f"target-derived router columns are forbidden: {bad}")
    if not np.isfinite(g).all() or not np.isfinite(a).all():
        raise ValueError("non-finite frozen predictions")

    out=pd.DataFrame({
        "pred_m16g_h":g,
        "pred_m17a_h":a,
        "signed_disagreement_h":a-g,
        "abs_disagreement_h":np.abs(a-g),
        "m16g_expert_mean_h":np.mean(b,axis=1),
        "m16g_expert_std_h":np.std(b,axis=1),
        "m16g_expert_range_h":np.max(b,axis=1)-np.min(b,axis=1),
    })
    numeric=(
        "m16c_deepest_support","m16c_deepest_weight","m16c_fallback_count",
        "m16d_neighbour_count","m16d_nearest_distance","m16d_median_neighbour_distance","m16d_similarity_gap",
        "physics_eligible","resolution_confidence","physics_distance_gc_nm","physics_course_alignment_deg",
        "physics_recent_speed_max_kn","m16e_effective_speed_kn","m16e_distance_factor","m16e_route_distance_proxy_nm",
        "m17a_neighbour_count","m17a_nearest_distance","m17a_median_neighbour_distance","m17a_similarity_gap",
    )
    for col in numeric:
        if col not in q.columns:
            raise ValueError(f"missing M17E quality feature: {col}")
        out[col]=pd.to_numeric(q[col],errors="coerce")
    rg=q.get("m16d_gate_used",pd.Series(["UNKNOWN"]*len(q))).fillna("UNKNOWN").astype(str)
    ag=q.get("m17a_gate_used",pd.Series(["UNKNOWN"]*len(q))).fillna("UNKNOWN").astype(str)
    for prefix,gate in (("route",rg),("learned",ag)):
        out[f"{prefix}_gate_destination"]=gate.eq("destination").astype(float)
        out[f"{prefix}_gate_fallback"]=gate.eq("destination_fallback_global").astype(float)
        out[f"{prefix}_gate_global"]=gate.eq("global").astype(float)
    out["learned_vs_route_distance_ratio"]=(out["m17a_nearest_distance"]+1e-6)/(out["m16d_nearest_distance"]+1e-6)
    return out.replace([np.inf,-np.inf],np.nan).fillna(0.0).astype(float)


def _candidate_params(candidate: str) -> tuple[float,float] | None:
    mapping={
        "logit_c005_t070":(0.05,0.70),
        "logit_c020_t070":(0.20,0.70),
        "logit_c020_t080":(0.20,0.80),
        "logit_c100_t080":(1.00,0.80),
    }
    return mapping.get(candidate)


def router_labels(target: Sequence[float], pred_m16g: Sequence[float], pred_m17a: Sequence[float]) -> np.ndarray:
    y=np.asarray(target,float); g=np.asarray(pred_m16g,float); a=np.asarray(pred_m17a,float)
    return (np.abs(y-a)+M17E_LABEL_MARGIN_H < np.abs(y-g)).astype(int)


def fit_predict_router(
    candidate: str,
    train_features: pd.DataFrame,
    train_target: Sequence[float],
    train_m16g: Sequence[float],
    train_m17a: Sequence[float],
    valid_features: pd.DataFrame,
    valid_m16g: Sequence[float],
    valid_m17a: Sequence[float],
) -> RouterResult:
    if candidate not in M17E_ROUTER_CANDIDATES:
        raise KeyError(candidate)
    vg=np.asarray(valid_m16g,float); va=np.asarray(valid_m17a,float)
    if candidate=="always_m16g":
        return RouterResult(vg.copy(),np.zeros(len(vg),dtype=bool),np.zeros(len(vg),dtype=float))
    labels=router_labels(train_target,train_m16g,train_m17a)
    if len(np.unique(labels))<2:
        return RouterResult(vg.copy(),np.zeros(len(vg),dtype=bool),np.zeros(len(vg),dtype=float))
    C,threshold=_candidate_params(candidate)
    model=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=5000,class_weight="balanced",random_state=42))
    model.fit(train_features,labels)
    p=model.predict_proba(valid_features)[:,1]
    sw=p>=threshold
    pred=np.where(sw,va,vg)
    return RouterResult(pred,sw,p)


def score_router_candidates(
    features: pd.DataFrame,
    target: Sequence[float],
    pred_m16g: Sequence[float],
    pred_m17a: Sequence[float],
    folds: Sequence[int],
) -> pd.DataFrame:
    y=np.asarray(target,float); g=np.asarray(pred_m16g,float); a=np.asarray(pred_m17a,float); f=np.asarray(folds,int)
    rows=[]
    for cand in M17E_ROUTER_CANDIDATES:
        yy=[]; pp=[]; ss=[]
        for k in sorted(np.unique(f)):
            tr=np.flatnonzero(f!=k); va=np.flatnonzero(f==k)
            r=fit_predict_router(cand,features.iloc[tr],y[tr],g[tr],a[tr],features.iloc[va],g[va],a[va])
            yy.extend(y[va]); pp.extend(r.prediction); ss.extend(r.switch_to_m17a)
        met=extended_metrics(yy,pp); switch=float(np.mean(ss)) if ss else 0.0
        rows.append({"candidate_id":cand,**met,"switch_share":switch,"within_switch_budget":bool(switch<=M17E_MAX_SWITCH_SHARE)})
    df=pd.DataFrame(rows)
    eligible=df.loc[df["within_switch_budget"]].copy()
    if eligible.empty:
        eligible=df.loc[df["candidate_id"].eq("always_m16g")].copy()
    order=eligible.sort_values(["mae_h","p90_ae_h","switch_share","candidate_id"],kind="mergesort")
    best=str(order.iloc[0]["candidate_id"])
    df["selected"]=df["candidate_id"].eq(best)
    return df.sort_values(["selected","mae_h","candidate_id"],ascending=[False,True,True],kind="mergesort").reset_index(drop=True)


def promotion_gate(base_metrics: dict, router_metrics: dict, *, fold_wins: int, changed_row_win_share: float, switch_share: float, max_fold_regression_h: float) -> str:
    gain=float(base_metrics["mae_h"]-router_metrics["mae_h"])
    p90_ratio=float(router_metrics["p90_ae_h"]/base_metrics["p90_ae_h"])
    ok=(gain>=M17E_MIN_GAIN_H and fold_wins>=M17E_MIN_FOLD_WINS and changed_row_win_share>=M17E_MIN_CHANGED_ROW_WIN_SHARE and p90_ratio<=M17E_MAX_P90_RATIO and switch_share<=M17E_MAX_SWITCH_SHARE and max_fold_regression_h<=M17E_MAX_SINGLE_FOLD_REGRESSION_H)
    return "PASS_M17E_SELECTIVE_ROUTER" if ok else "NO_M17E_PROMOTION"
