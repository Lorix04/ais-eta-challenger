"""M17F latent target-noise / declared-ETA regime model.

Development-only residual correction on top of frozen/nested M16G.  Latent
regimes are inferred from training residuals, while runtime gating uses only
prediction-time, target-free features.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import t as student_t
from sklearn.linear_model import HuberRegressor, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .m16a import extended_metrics

M17F_VERSION = "m17f-latent-target-noise-v1-20260922"
M17F_CANDIDATES = (
    "always_m16g",
    "huber_residual",
    "latent_t2_c010",
    "latent_t2_c100",
    "latent_t3_c010",
)
M17F_STUDENT_DF = 3.0
M17F_CORRECTION_CLIP_H = 168.0
M17F_MIN_GAIN_H = 1.0
M17F_MIN_FOLD_WINS = 3
M17F_MAX_P90_RATIO = 1.02
M17F_MAX_TRIM95_REGRESSION_H = 0.0
M17F_MAX_SINGLE_FOLD_REGRESSION_H = 15.0


def build_noise_features(pred_m16g: Sequence[float], base4: np.ndarray, quality: pd.DataFrame) -> pd.DataFrame:
    """Target-free runtime features for latent regime gating/correction."""
    g=np.asarray(pred_m16g,float); b=np.asarray(base4,float); q=quality.reset_index(drop=True).copy()
    if b.ndim!=2 or b.shape[1]!=4 or len(g)!=len(b) or len(q)!=len(g):
        raise ValueError("M17F feature shape mismatch")
    bad_tokens=("target","reference_eta","true_eta","error","absolute_error","covered")
    bad=[c for c in q.columns if any(tok in c.lower() for tok in bad_tokens)]
    if bad: raise ValueError(f"target-derived M17F features forbidden: {bad}")
    out=pd.DataFrame({
        "pred_m16g_h":g,
        "abs_pred_m16g_h":np.abs(g),
        "expert_mean_h":np.mean(b,axis=1),
        "expert_std_h":np.std(b,axis=1),
        "expert_range_h":np.max(b,axis=1)-np.min(b,axis=1),
        "prior_minus_route_h":b[:,0]-b[:,1],
        "physics_minus_route_h":b[:,2]-b[:,1],
        "tabular_minus_route_h":b[:,3]-b[:,1],
    })
    numeric=(
        "m16c_deepest_support","m16c_deepest_weight","m16c_fallback_count",
        "m16d_neighbour_count","m16d_nearest_distance","m16d_median_neighbour_distance","m16d_similarity_gap",
        "physics_eligible","resolution_confidence","physics_distance_gc_nm","physics_course_alignment_deg",
        "physics_recent_speed_max_kn","m16e_effective_speed_kn","m16e_distance_factor","m16e_route_distance_proxy_nm",
    )
    for c in numeric:
        if c not in q.columns: raise ValueError(f"missing M17F quality feature: {c}")
        out[c]=pd.to_numeric(q[c],errors="coerce")
    gate=q.get("m16d_gate_used",pd.Series(["UNKNOWN"]*len(q))).fillna("UNKNOWN").astype(str)
    out["route_gate_destination"]=gate.eq("destination").astype(float)
    out["route_gate_fallback"]=gate.eq("destination_fallback_global").astype(float)
    out["route_gate_global"]=gate.eq("global").astype(float)
    return out.replace([np.inf,-np.inf],np.nan).fillna(0.0).astype(float)


@dataclass(frozen=True)
class StudentTMixture:
    weights: np.ndarray
    locs: np.ndarray
    scales: np.ndarray
    responsibilities: np.ndarray


def fit_student_t_mixture(residual: Sequence[float], k: int, *, df: float=M17F_STUDENT_DF, max_iter: int=100) -> StudentTMixture:
    """Deterministic univariate fixed-df Student-t mixture EM."""
    r=np.asarray(residual,float)
    if k not in (2,3) or len(r)<20 or not np.isfinite(r).all(): raise ValueError("invalid Student-t mixture input")
    qs=np.linspace(0.2,0.8,k); locs=np.quantile(r,qs).astype(float)
    mad=np.median(np.abs(r-np.median(r))); base_scale=max(1.0,1.4826*mad)
    scales=np.full(k,base_scale,float); weights=np.full(k,1.0/k,float)
    prev=None
    for _ in range(max_iter):
        logp=np.column_stack([np.log(max(weights[j],1e-12))+student_t.logpdf((r-locs[j])/scales[j],df=df)-np.log(scales[j]) for j in range(k)])
        logz=logsumexp(logp,axis=1); resp=np.exp(logp-logz[:,None])
        ll=float(np.sum(logz))
        for j in range(k):
            q=resp[:,j]; qsum=max(float(q.sum()),1e-9)
            delta=((r-locs[j])/max(scales[j],1e-6))**2
            latent=(df+1.0)/(df+delta)
            w=q*latent; wsum=max(float(w.sum()),1e-9)
            locs[j]=float(np.sum(w*r)/wsum)
            scales[j]=float(np.sqrt(max(np.sum(q*latent*(r-locs[j])**2)/qsum,1.0)))
            weights[j]=qsum/len(r)
        order=np.argsort(locs); locs=locs[order]; scales=scales[order]; weights=weights[order]
        if prev is not None and abs(ll-prev)<1e-6: break
        prev=ll
    logp=np.column_stack([np.log(max(weights[j],1e-12))+student_t.logpdf((r-locs[j])/scales[j],df=df)-np.log(scales[j]) for j in range(k)])
    resp=np.exp(logp-logsumexp(logp,axis=1)[:,None])
    return StudentTMixture(weights,locs,scales,resp)


def _candidate_params(candidate:str):
    if candidate=="latent_t2_c010": return 2,0.10
    if candidate=="latent_t2_c100": return 2,1.00
    if candidate=="latent_t3_c010": return 3,0.10
    return None


def fit_predict_candidate(candidate:str, train_x:pd.DataFrame, train_y:Sequence[float], train_base:Sequence[float], valid_x:pd.DataFrame, valid_base:Sequence[float]):
    if candidate not in M17F_CANDIDATES: raise KeyError(candidate)
    y=np.asarray(train_y,float); b=np.asarray(train_base,float); vb=np.asarray(valid_base,float); residual=y-b
    if candidate=="always_m16g":
        return vb.copy(), np.zeros(len(vb)), {"n_regimes":1,"regime_locs_h":[0.0]}
    if candidate=="huber_residual":
        model=make_pipeline(StandardScaler(),HuberRegressor(epsilon=1.35,alpha=1.0,max_iter=1000))
        model.fit(train_x,residual); corr=np.asarray(model.predict(valid_x),float)
        corr=np.clip(corr,-M17F_CORRECTION_CLIP_H,M17F_CORRECTION_CLIP_H)
        return vb+corr,corr,{"n_regimes":1,"regime_locs_h":[float(np.median(residual))]}
    k,C=_candidate_params(candidate)
    mix=fit_student_t_mixture(residual,k)
    hard=np.argmax(mix.responsibilities,axis=1)
    # Collapse safely if a tiny regime appears.
    counts=np.bincount(hard,minlength=k)
    if np.sum(counts>=8)<2:
        return vb.copy(),np.zeros(len(vb)),{"n_regimes":int(k),"regime_locs_h":mix.locs.tolist(),"collapsed":True}
    gate=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=5000,class_weight="balanced",random_state=42))
    gate.fit(train_x,hard)
    probs=gate.predict_proba(valid_x)
    classes=np.asarray(gate[-1].classes_,int)
    full=np.zeros((len(valid_x),k),float); full[:,classes]=probs
    corr=full@mix.locs
    corr=np.clip(corr,-M17F_CORRECTION_CLIP_H,M17F_CORRECTION_CLIP_H)
    return vb+corr,corr,{"n_regimes":int(k),"regime_locs_h":[float(x) for x in mix.locs],"regime_counts":[int(x) for x in counts]}


def score_candidates(features:pd.DataFrame,target:Sequence[float],base:Sequence[float],folds:Sequence[int])->pd.DataFrame:
    y=np.asarray(target,float); b=np.asarray(base,float); f=np.asarray(folds,int); rows=[]
    for cand in M17F_CANDIDATES:
        yy=[]; pp=[]
        for k in sorted(np.unique(f)):
            tr=np.flatnonzero(f!=k); va=np.flatnonzero(f==k)
            p,_,_=fit_predict_candidate(cand,features.iloc[tr],y[tr],b[tr],features.iloc[va],b[va])
            yy.extend(y[va]); pp.extend(p)
        met=extended_metrics(yy,pp); rows.append({"candidate_id":cand,**met})
    df=pd.DataFrame(rows).sort_values(["mae_h","p90_ae_h","candidate_id"],kind="mergesort").reset_index(drop=True)
    df["selected"]=False; df.loc[0,"selected"]=True
    return df


def trimmed_mae(y:Sequence[float],pred:Sequence[float],q:float=0.95)->float:
    y=np.asarray(y,float); p=np.asarray(pred,float); cutoff=np.quantile(np.abs(y),q); m=np.abs(y)<=cutoff
    return float(np.mean(np.abs(y[m]-p[m])))


def promotion_gate(base_metrics:dict,model_metrics:dict,*,fold_wins:int,max_fold_regression_h:float,trim95_base:float,trim95_model:float)->str:
    gain=float(base_metrics["mae_h"]-model_metrics["mae_h"]); p90_ratio=float(model_metrics["p90_ae_h"]/base_metrics["p90_ae_h"]); trim_reg=float(trim95_model-trim95_base)
    ok=(gain>=M17F_MIN_GAIN_H and fold_wins>=M17F_MIN_FOLD_WINS and p90_ratio<=M17F_MAX_P90_RATIO and trim_reg<=M17F_MAX_TRIM95_REGRESSION_H and max_fold_regression_h<=M17F_MAX_SINGLE_FOLD_REGRESSION_H)
    return "PASS_M17F_LATENT_NOISE_MODEL" if ok else "NO_M17F_PROMOTION"
