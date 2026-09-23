"""M17B leakage-safe historical memory and deterministic HNSW retrieval.

M17B scales the passing M17A learned trajectory representation from one query
embedding per development MMSI to a memory of recent historical snapshots.
Memory rows are built only for outer-train MMSIs, labels are aligned to the
memory snapshot timestamp, and retrieved windows are deduplicated by owner MMSI
before the frozen K=9 weighted-median ETA aggregation is applied.

The ANN implementation below is a compact deterministic NumPy reference HNSW
implementation.  It exists so the benchmark remains self-contained in the
submission environment; the retrieval contract is compatible with production
HNSW libraries such as hnswlib.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable

import numpy as np
import pandas as pd

from ais_eta.m16d import M16D_POINTS, resample_causal_trajectory, weighted_median

M17B_VERSION = "m17b-historical-memory-hnsw-v1-20260922"
M17B_WINDOW_H = 6
M17B_MEMORY_HORIZON_H = 24
M17B_MEMORY_STRIDE_H = 1
M17B_POINTS = M16D_POINTS
M17B_K_OWNERS = 9
M17B_MIN_DEST_OWNERS = 3
M17B_HNSW_M = 16
M17B_HNSW_EF_CONSTRUCTION = 96
M17B_HNSW_EF_SEARCH = 128
M17B_HNSW_SEED = 1702
M17B_SSL_EPOCHS = 4
M17B_PRETRAIN_WINDOWS_PER_MMSI = 4
M17B_PROMOTION_MIN_FOLD_WINS = 3
M17B_PROMOTION_MAX_P90_RATIO = 1.05
M17B_HNSW_MIN_OWNER_RECALL = 0.95
M17B_HNSW_MAX_MAE_DELTA_H = 1.0


def build_historical_memory_windows(
    states: pd.DataFrame,
    dev: pd.DataFrame,
    *,
    horizon_h: int = M17B_MEMORY_HORIZON_H,
    stride_h: int = M17B_MEMORY_STRIDE_H,
    window_h: int = M17B_WINDOW_H,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build recent causal memory snapshots and snapshot-aligned TTE labels.

    This is a vectorized equivalent of repeatedly calling the M16D causal
    resampler: each MMSI is cleaned/deduplicated once, hourly endpoints are
    located with ``searchsorted``, and each 6 h subwindow is interpolated with
    NumPy.  No target is read until after the trajectory window is constructed.
    """
    s=states.copy(); s['recorded_at']=pd.to_datetime(s['recorded_at'])
    d=dev.copy(); d['history_last_at']=pd.to_datetime(d['history_last_at']); d['last_update']=pd.to_datetime(d['last_update']); d['eta_reference_dt']=pd.to_datetime(d['eta_reference_dt'])
    grouped={int(k):g.sort_values('recorded_at',kind='mergesort') for k,g in s.groupby('mmsi',sort=False)}
    seqs=[]; rows=[]; cos37=math.cos(math.radians(37.0))
    for row in d.sort_values('mmsi',kind='mergesort').itertuples(index=False):
        mmsi=int(row.mmsi); g=grouped.get(mmsi)
        if g is None: continue
        cutoff=pd.Timestamp(row.history_last_at); lower=cutoff-pd.Timedelta(hours=int(horizon_h+window_h))
        mask=g['recorded_at'].between(lower,cutoff,inclusive='both')
        if 'position_valid' in g.columns: mask &= g['position_valid'].fillna(False).astype(bool)
        gg=g.loc[mask,['recorded_at','lat_clean','lon_clean','sog_clean','cog_clean']].dropna(subset=['lat_clean','lon_clean']).copy()
        if len(gg)<2: continue
        # Keep first duplicate timestamp exactly like M16D.
        tv=gg['recorded_at'].astype('int64').to_numpy(np.int64); _,keep=np.unique(tv,return_index=True); keep=np.sort(keep); gg=gg.iloc[keep]; tv=tv[keep]
        if len(tv)<2: continue
        t=tv.astype(np.float64)/1e9; lat=gg.lat_clean.to_numpy(float); lon=gg.lon_clean.to_numpy(float)
        sog=pd.to_numeric(gg.sog_clean,errors='coerce').ffill().bfill().fillna(0.0).to_numpy(float); cog=pd.to_numeric(gg.cog_clean,errors='coerce').ffill().bfill().fillna(0.0).to_numpy(float)
        earliest=cutoff-pd.Timedelta(hours=int(horizon_h)); desired=list(pd.date_range(earliest,cutoff,freq=f'{int(stride_h)}h'))
        if not desired or desired[-1]!=cutoff: desired.append(cutoff)
        seen_last=set()
        for end_at in desired:
            end_s=float(pd.Timestamp(end_at).value/1e9); start_s=end_s-float(window_h)*3600.0
            lo=int(np.searchsorted(t,start_s,side='left')); hi=int(np.searchsorted(t,end_s,side='right'))
            if hi-lo<2: continue
            ts=t[lo:hi]
            if ts[-1]<=ts[0]: continue
            last_ns=int(round(ts[-1]*1e9));
            if last_ns in seen_last: continue
            seen_last.add(last_ns); observed=(ts[-1]-ts[0])/3600.0
            if observed<0.25: continue
            q=np.linspace(float(ts[0]),float(ts[-1]),M17B_POINTS)
            la=np.interp(q,ts,lat[lo:hi]); lo2=np.interp(q,ts,lon[lo:hi]); sg=np.interp(q,ts,sog[lo:hi]); cg=np.interp(q,ts,cog[lo:hi])
            x=lo2*111.32*cos37; y=la*111.32
            seq=np.column_stack([x/50.0,y/50.0,0.5*sg/10.0,0.25*np.sin(np.deg2rad(cg)),0.25*np.cos(np.deg2rad(cg))]).astype(np.float32)
            last_at=pd.to_datetime(ts[-1],unit='s'); tte=(pd.Timestamp(row.eta_reference_dt)-last_at).total_seconds()/3600.0; age=(pd.Timestamp(row.last_update)-cutoff).total_seconds()/3600.0
            seqs.append(seq); rows.append({'memory_id':len(rows),'mmsi':mmsi,'memory_last_at':last_at.isoformat(),'history_cutoff_at':cutoff.isoformat(),'last_update':pd.Timestamp(row.last_update).isoformat(),'canonical_destination':str(getattr(row,'canonical_destination','UNKNOWN')),'owner_target_tte_h':float(row.target_tte_h),'snapshot_tte_h':float(tte),'query_age_h':float(age),'raw_points':int(hi-lo),'observed_span_h':float(observed)})
    if not seqs: raise AssertionError('M17B produced no historical memory windows')
    return np.stack(seqs),pd.DataFrame(rows)


def cosine_distance_to_query(vectors: np.ndarray, query: np.ndarray) -> np.ndarray:
    v = np.asarray(vectors, dtype=np.float64)
    q = np.asarray(query, dtype=np.float64)
    vn = np.linalg.norm(v, axis=1)
    qn = float(np.linalg.norm(q))
    vn = np.where(vn < 1e-12, 1.0, vn)
    qn = max(qn, 1e-12)
    return np.clip(1.0 - (v @ q) / (vn * qn), 0.0, 2.0)


def _dedupe_owner_candidates(
    memory_meta: pd.DataFrame,
    candidate_ids: Iterable[int],
    candidate_distances: Iterable[float],
    *,
    k_owners: int = M17B_K_OWNERS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep the closest memory snapshot for each MMSI, deterministically."""
    triples = sorted(
        ((float(dist), int(mid), int(memory_meta.iloc[int(mid)]["mmsi"])) for mid, dist in zip(candidate_ids, candidate_distances)),
        key=lambda x: (x[0], x[1]),
    )
    owners: list[int] = []
    mids: list[int] = []
    dists: list[float] = []
    seen: set[int] = set()
    for dist, mid, owner in triples:
        if owner in seen:
            continue
        seen.add(owner); owners.append(owner); mids.append(mid); dists.append(dist)
        if len(owners) >= int(k_owners):
            break
    return np.asarray(mids, dtype=int), np.asarray(owners, dtype=np.int64), np.asarray(dists, dtype=float)


def memory_prediction_from_candidates(
    *,
    memory_meta: pd.DataFrame,
    memory_ids: np.ndarray,
    distances: np.ndarray,
    query_age_h: float,
    label_mode: str = "snapshot_aligned",
) -> dict:
    if len(memory_ids) == 0:
        raise AssertionError("empty M17B owner-deduplicated memory")
    if label_mode == "snapshot_aligned":
        vals = memory_meta.iloc[memory_ids]["snapshot_tte_h"].to_numpy(float) - float(query_age_h)
    elif label_mode == "owner_target_static":
        vals = memory_meta.iloc[memory_ids]["owner_target_tte_h"].to_numpy(float)
    else:
        raise KeyError(label_mode)
    scale = max(float(np.median(distances)), 1e-9)
    weights = np.exp(-np.asarray(distances, dtype=float) / scale)
    pred = weighted_median(vals, weights)
    return {
        "prediction_h": float(pred),
        "neighbour_count": int(len(memory_ids)),
        "nearest_distance": float(distances[0]),
        "median_neighbour_distance": float(np.median(distances)),
        "nearest_memory_id": int(memory_ids[0]),
        "nearest_owner_mmsi": int(memory_meta.iloc[int(memory_ids[0])]["mmsi"]),
        "neighbour_memory_ids": ";".join(str(int(x)) for x in memory_ids),
        "neighbour_owner_mmsi": ";".join(str(int(x)) for x in memory_meta.iloc[memory_ids]["mmsi"].astype(int)),
    }


def exact_memory_predict(
    *,
    memory_embeddings: np.ndarray,
    memory_meta: pd.DataFrame,
    query_embedding: np.ndarray,
    query_destination: str,
    query_age_h: float,
    label_mode: str = "snapshot_aligned",
    k_owners: int = M17B_K_OWNERS,
) -> dict:
    dest = memory_meta["canonical_destination"].astype(str).to_numpy()
    owners = memory_meta["mmsi"].astype(int).to_numpy()
    same_mask = dest == str(query_destination)
    same_unique = len(set(int(x) for x in owners[same_mask]))
    if same_unique >= M17B_MIN_DEST_OWNERS:
        candidate_ids = np.flatnonzero(same_mask)
        gate = "destination"
    else:
        candidate_ids = np.arange(len(memory_meta), dtype=int)
        gate = "destination_fallback_global"
    d = cosine_distance_to_query(memory_embeddings[candidate_ids], query_embedding)
    order = np.argsort(d, kind="mergesort")
    mids, own, nd = _dedupe_owner_candidates(memory_meta, candidate_ids[order], d[order], k_owners=k_owners)
    out = memory_prediction_from_candidates(
        memory_meta=memory_meta, memory_ids=mids, distances=nd,
        query_age_h=query_age_h, label_mode=label_mode,
    )
    out.update({
        "gate_used": gate,
        "candidate_windows": int(len(candidate_ids)),
        "candidate_unique_owners": int(len(set(int(x) for x in owners[candidate_ids]))),
        "distance_evaluations": int(len(candidate_ids)),
    })
    return out


@dataclass
class _SearchResult:
    ids: np.ndarray
    distances: np.ndarray
    distance_evaluations: int


class DeterministicHNSW:
    """Fast deterministic HNSW-style cosine index for the research benchmark.

    The clean environment does not ship the compiled hnswlib extension.  To keep
    M17B self-contained and deterministic, construction uses the standard HNSW
    random hierarchy but builds each layer's local M-neighbour graph in one
    vectorized cKDTree pass over L2-normalized vectors.  Querying then follows
    the HNSW contract: greedy descent on upper layers and an ef best-first search
    on layer 0.  Exact cosine remains the oracle and recall is measured explicitly.
    """
    def __init__(self, dim: int, *, m: int = M17B_HNSW_M, ef_construction: int = M17B_HNSW_EF_CONSTRUCTION, seed: int = M17B_HNSW_SEED):
        self.dim=int(dim); self.m=int(m); self.ef_construction=int(ef_construction); self.seed=int(seed)
        self.vectors=np.empty((0,self.dim),dtype=np.float64); self.levels=np.empty(0,dtype=int); self.graphs:list[dict[int,set[int]]]=[]
        self.entry_point:int|None=None; self.max_level=-1

    @staticmethod
    def _normalize_rows(x: np.ndarray) -> np.ndarray:
        a=np.asarray(x,dtype=np.float64); n=np.linalg.norm(a,axis=1,keepdims=True); n=np.where(n<1e-12,1.0,n); return a/n

    def _dist(self,q:np.ndarray,idx:int)->float:
        return float(np.clip(1.0-float(q@self.vectors[int(idx)]),0.0,2.0))

    def add_items(self,vectors:np.ndarray)->None:
        from scipy.spatial import cKDTree
        x=np.asarray(vectors,dtype=np.float64)
        if x.ndim!=2 or x.shape[1]!=self.dim: raise ValueError('HNSW vectors have wrong shape')
        self.vectors=self._normalize_rows(x); n=len(x)
        rng=np.random.default_rng(self.seed); lm=1.0/max(math.log(max(self.m,2)),1e-9)
        u=np.maximum(rng.random(n),1e-12); self.levels=np.floor(-np.log(u)*lm).astype(int)
        self.max_level=int(self.levels.max(initial=0)); self.graphs=[]
        for level in range(self.max_level+1):
            ids=np.flatnonzero(self.levels>=level); graph={int(i):set() for i in ids}
            if len(ids)>1:
                tree=cKDTree(self.vectors[ids]); k=min(self.m+1,len(ids)); _,nn=tree.query(self.vectors[ids],k=k,workers=1)
                if k==1: nn=nn[:,None]
                for row,gid in enumerate(ids):
                    for local in np.atleast_1d(nn[row]):
                        nb=int(ids[int(local)])
                        if nb==int(gid): continue
                        graph[int(gid)].add(nb); graph.setdefault(nb,set()).add(int(gid))
            self.graphs.append(graph)
        top=np.flatnonzero(self.levels==self.max_level); self.entry_point=int(top[0]) if len(top) else 0

    def _greedy(self,q:np.ndarray,ep:int,level:int)->tuple[int,float,int]:
        cur=int(ep); curd=self._dist(q,cur); evals=1; changed=True
        while changed:
            changed=False
            for nb in sorted(self.graphs[level].get(cur,set())):
                d=self._dist(q,nb); evals+=1
                if d<curd-1e-15 or (abs(d-curd)<=1e-15 and nb<cur): cur,curd,changed=int(nb),float(d),True
        return cur,curd,evals

    def _search_layer(self,q:np.ndarray,ep:int,ef:int)->_SearchResult:
        visited={int(ep)}; d0=self._dist(q,int(ep)); evals=1
        candidates=[(d0,int(ep))]; best=[(-d0,-int(ep))]
        while candidates:
            cd,c=heapq.heappop(candidates); worst=-best[0][0]
            if len(best)>=int(ef) and cd>worst: break
            for nb in sorted(self.graphs[0].get(c,set())):
                if nb in visited: continue
                visited.add(nb); d=self._dist(q,nb); evals+=1; worst=-best[0][0]
                if len(best)<int(ef) or d<worst or (abs(d-worst)<=1e-15 and nb<-best[0][1]):
                    heapq.heappush(candidates,(d,nb)); heapq.heappush(best,(-d,-nb))
                    if len(best)>int(ef): heapq.heappop(best)
        pairs=sorted([(-a,-b) for a,b in best],key=lambda z:(z[0],z[1]))
        return _SearchResult(np.asarray([x[1] for x in pairs],int),np.asarray([x[0] for x in pairs],float),evals)

    def query(self,query:np.ndarray,*,k:int,ef_search:int=M17B_HNSW_EF_SEARCH)->_SearchResult:
        if self.entry_point is None: raise AssertionError('query on empty HNSW')
        q=self._normalize_rows(np.asarray(query,dtype=float).reshape(1,-1))[0]; ep=int(self.entry_point); evals=0
        for level in range(self.max_level,0,-1): ep,_,ev=self._greedy(q,ep,level); evals+=ev
        sr=self._search_layer(q,ep,max(int(ef_search),int(k))); n=min(int(k),len(sr.ids))
        return _SearchResult(sr.ids[:n],sr.distances[:n],evals+sr.distance_evaluations)


class CohortHNSWMemory:
    """Global + canonical-destination HNSW indices over one outer-train memory."""
    def __init__(self, embeddings: np.ndarray, meta: pd.DataFrame, *, seed: int):
        self.embeddings = np.asarray(embeddings, dtype=np.float64)
        self.meta = meta.reset_index(drop=True).copy()
        self.seed = int(seed)
        self.global_ids = np.arange(len(self.meta), dtype=int)
        self.global_index = self._build(self.global_ids, self.seed)
        self.dest_indices: dict[str, tuple[np.ndarray, DeterministicHNSW]] = {}
        for j, (dest, g) in enumerate(self.meta.groupby("canonical_destination", sort=True)):
            ids = g.index.to_numpy(int)
            owners = set(g["mmsi"].astype(int))
            if len(owners) >= M17B_MIN_DEST_OWNERS:
                self.dest_indices[str(dest)] = (ids, self._build(ids, self.seed + 1000 + j))

    def _build(self, ids: np.ndarray, seed: int) -> DeterministicHNSW:
        idx = DeterministicHNSW(self.embeddings.shape[1], seed=seed)
        idx.add_items(self.embeddings[np.asarray(ids,int)])
        return idx

    def query(self, query: np.ndarray, destination: str, *, k_owners: int = M17B_K_OWNERS) -> dict:
        if str(destination) in self.dest_indices:
            global_ids, index = self.dest_indices[str(destination)]; gate = "destination"
        else:
            global_ids, index = self.global_ids, self.global_index; gate = "destination_fallback_global"
        want = min(len(global_ids), max(64, int(k_owners) * 16))
        total_evals = 0
        while True:
            sr = index.query(query, k=want, ef_search=max(M17B_HNSW_EF_SEARCH, want)); total_evals += sr.distance_evaluations
            mids_global = global_ids[sr.ids]
            mids, owners, dists = _dedupe_owner_candidates(self.meta, mids_global, sr.distances, k_owners=k_owners)
            max_unique = len(set(self.meta.iloc[global_ids]["mmsi"].astype(int)))
            if len(mids) >= min(int(k_owners), max_unique) or want >= len(global_ids):
                break
            want = min(len(global_ids), want * 2)
        return {
            "memory_ids": mids, "owner_mmsi": owners, "distances": dists,
            "gate_used": gate, "candidate_windows": int(len(global_ids)),
            "candidate_unique_owners": int(max_unique), "distance_evaluations": int(total_evals),
        }
