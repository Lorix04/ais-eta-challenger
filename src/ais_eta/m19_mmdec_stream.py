"""Streaming adapter from canonical MMDEC Parquet bytes to frozen M19 tables.

No target magnitude is computed before blind scoring.  The SPEC scan uses ETA
components only as an availability/range mask, exactly as the frozen M19 policy
already specifies.  Position history is filtered causally to the six hours at
or before each decision time.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .m19 import (
    M19Policy, M19_SOURCE_PERIOD_END, M19_SOURCE_PERIOD_START,
    _eta_components_available, _selection_hash,
    build_mmdec_target_free_holdout, canonical_decision_time,
    reveal_mmdec_label_ledger, standardize_mmdec_spec,
)
from .m19_parquet_lite import decode_column, parse_footer

SPEC_COLUMNS = [
    "Date","MessageType","Mmsi","ShipType","Draught10thMetres","Destination",
    "EtaMonth","EtaDay","EtaHour","EtaMinute",
]
POS_BASE_COLUMNS = ["Date","Mmsi"]
POS_EXTRA_COLUMNS = [
    "MessageType","NavigationStatus","Latitude","Longitude",
    "CourseOverGroundDegrees","SpeedOverGround","TrueHeadingDegrees",
]


def _flat_map(md, rgi: int) -> dict[str,int]:
    return {
        c.meta_data.path_in_schema[-1]: i
        for i,c in enumerate(md.row_groups[rgi].columns)
        if len(c.meta_data.path_in_schema)==1
    }


def _decode_df(path: str | Path, md, rgi: int, names: list[str]) -> pd.DataFrame:
    cmap=_flat_map(md,rgi); missing=[n for n in names if n not in cmap]
    if missing: raise ValueError(f"MMDEC Parquet missing columns {missing} in row group {rgi}")
    d={n:decode_column(path,md,rgi,cmap[n]) for n in names}
    lens={len(v) for v in d.values()}
    expected=int(md.row_groups[rgi].num_rows)
    if lens != {expected}: raise AssertionError(f"row alignment failure rgi={rgi}: {lens} != {expected}")
    for n,a in d.items():
        if getattr(a,"dtype",None)==object:
            d[n]=np.array([x.decode("utf-8","replace") if isinstance(x,(bytes,bytearray)) else x for x in a],dtype=object)
    return pd.DataFrame(d)


def _scan_spec_row_group_worker(args):
    """Exact frozen M19 candidate scan for one SPEC row group.

    Kept top-level so CPU-heavy SHA-256 candidate ranking can use separate
    processes without changing the frozen selection semantics.
    """
    import hashlib
    import time as _time
    from .m19 import M19_SELECTION_SALT

    spec_path, rgi, start_ns, end_ns = args
    md = parse_footer(spec_path)
    cmap = _flat_map(md, int(rgi))
    basic = ["Date","MessageType","Mmsi","EtaMonth","EtaDay","EtaHour","EtaMinute"]
    a = {n: decode_column(spec_path, md, int(rgi), cmap[n]) for n in basic}
    date=np.asarray(a["Date"],dtype=np.int64); msg=np.asarray(a["MessageType"],dtype=float); mmsi=np.asarray(a["Mmsi"],dtype=float)
    mo=np.asarray(a["EtaMonth"],dtype=float); da=np.asarray(a["EtaDay"],dtype=float); hr=np.asarray(a["EtaHour"],dtype=float); mi=np.asarray(a["EtaMinute"],dtype=float)
    mask=(msg==5)&np.isfinite(mmsi)&(date>=int(start_ns))&(date<int(end_ns))
    mask &= np.isfinite(mo)&np.isfinite(da)&np.isfinite(hr)&np.isfinite(mi)
    mask &= (mo>=1)&(mo<=12)&(da>=1)&(da<=31)&(hr>=0)&(hr<=23)&(mi>=0)&(mi<=59)
    ix=np.flatnonzero(mask)
    if ix.size==0: return []
    mm=mmsi[ix].astype(np.int64); dt=date[ix]
    if np.any(dt % 1_000_000_000):
        stamps=[canonical_decision_time(pd.Timestamp(int(x),tz="UTC")) for x in dt]
    else:
        stamps=[_time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime(int(x)//1_000_000_000)) for x in dt]
    hashes=[hashlib.sha256(f"{M19_SELECTION_SALT}|{int(m)}|{ts}".encode()).hexdigest() for m,ts in zip(mm,stamps)]
    temp=pd.DataFrame({"j":ix,"mmsi":mm,"date_ns":dt,"selection_hash":hashes,"stamp":stamps})
    temp=temp.sort_values(["mmsi","selection_hash","date_ns"],kind="mergesort").drop_duplicates("mmsi",keep="first")
    out=[]
    for j,m,date_ns,selection_hash,stamp in temp.itertuples(index=False,name=None):
        jj=int(j)
        out.append({"_key":(str(selection_hash),str(stamp)),"rgi":int(rgi),"j":jj,"date_ns":int(date_ns),"Mmsi":int(m),
                    "EtaMonth":float(mo[jj]),"EtaDay":float(da[jj]),"EtaHour":float(hr[jj]),"EtaMinute":float(mi[jj]),
                    "selection_hash":str(selection_hash)})
    return out


def select_candidates_from_parquet(spec_path: str | Path, policy: M19Policy = M19Policy()) -> pd.DataFrame:
    """Select the exact frozen M19 candidates with deterministic CPU-parallel hashing."""
    from concurrent.futures import ProcessPoolExecutor

    md=parse_footer(spec_path)
    start_ns=int(pd.Timestamp(M19_SOURCE_PERIOD_START).value)
    end_ns=int(pd.Timestamp(M19_SOURCE_PERIOD_END).value)
    tasks=[(str(spec_path), rgi, start_ns, end_ns) for rgi in range(len(md.row_groups))]

    best: dict[int,dict]={}
    # Hash ranking dominates runtime (~3.46M eligible Message-5 rows).  Separate
    # processes preserve the exact SHA-256 selection rule while avoiding GIL
    # serialization.  executor.map yields results in row-group order, so merge
    # behavior stays deterministic.
    with ProcessPoolExecutor(max_workers=4) as ex:
        for rows in ex.map(_scan_spec_row_group_worker, tasks, chunksize=2):
            for row in rows:
                m=row["Mmsi"]; old=best.get(m)
                if old is None or row["_key"] < old["_key"]: best[m]=row
    print(f"M19 SPEC selection scan complete: {len(best)} MMSI candidates", flush=True)
    if not best: raise ValueError("no eligible MMDEC Message-5 candidates")
    winners=sorted(best.values(),key=lambda r:(r["selection_hash"],r["Mmsi"]))[:int(policy.preselect_rows)]
    by_rg: dict[int,list[dict]]={}
    for row in winners: by_rg.setdefault(int(row["rgi"]),[]).append(row)
    out=[]
    for _n,(rgi,rows) in enumerate(sorted(by_rg.items())):
        if _n % 20 == 0: print(f"M19 SPEC static recovery {_n}/{len(by_rg)}", flush=True)
        cmap=_flat_map(md,rgi)
        ship=decode_column(spec_path,md,rgi,cmap["ShipType"]); draught=decode_column(spec_path,md,rgi,cmap["Draught10thMetres"]); dest=decode_column(spec_path,md,rgi,cmap["Destination"])
        for row in rows:
            j=row["j"]; dv=dest[j]
            if isinstance(dv,(bytes,bytearray)): dv=dv.decode("utf-8","replace")
            out.append({"Date":pd.Timestamp(row["date_ns"],tz="UTC"),"MessageType":5,"Mmsi":row["Mmsi"],
                        "ShipType":ship[j],"Draught10thMetres":draught[j],"Destination":dv,
                        "EtaMonth":row["EtaMonth"],"EtaDay":row["EtaDay"],"EtaHour":row["EtaHour"],"EtaMinute":row["EtaMinute"],
                        "selection_hash":row["selection_hash"]})
    return pd.DataFrame(out).sort_values(["selection_hash","Mmsi"],kind="mergesort").reset_index(drop=True)

def positions_for_candidates_from_parquet(
    pos_path: str | Path, candidates: pd.DataFrame, policy: M19Policy = M19Policy()
) -> pd.DataFrame:
    """Read only causal history rows needed by the preselected MMSIs."""
    cand={int(r.Mmsi):pd.Timestamp(r.Date) for r in candidates.itertuples(index=False)}
    cand_set=np.array(sorted(cand),dtype=np.int64)
    decision_ns={m:int(t.value) for m,t in cand.items()}
    history_ns=int(pd.Timedelta(hours=int(policy.history_hours)).value)
    md=parse_footer(pos_path); pieces=[]
    for rgi in range(len(md.row_groups)):
        cmap=_flat_map(md,rgi)
        dates=decode_column(pos_path,md,rgi,cmap["Date"])
        mmsi=decode_column(pos_path,md,rgi,cmap["Mmsi"])
        # Date and MMSI are required/non-null in canonical MMDEC POS.
        marr=np.asarray(mmsi,dtype=np.int64); darr=np.asarray(dates,dtype=np.int64)
        ix=np.flatnonzero(np.isin(marr,cand_set,assume_unique=False))
        if ix.size==0: continue
        mm=marr[ix]; dd=darr[ix]
        dec=np.fromiter((decision_ns[int(x)] for x in mm),dtype=np.int64,count=len(mm))
        keep=(dd<=dec)&(dd>=dec-history_ns)
        ix=ix[keep]
        if ix.size==0: continue
        extra={n:decode_column(pos_path,md,rgi,cmap[n]) for n in POS_EXTRA_COLUMNS}
        d={"Date":darr[ix],"Mmsi":marr[ix]}
        for n,a in extra.items(): d[n]=np.asarray(a)[ix]
        pieces.append(pd.DataFrame(d))
    if not pieces: raise ValueError("no causal MMDEC position histories found for M19 candidates")
    return pd.concat(pieces,ignore_index=True)


def build_target_free_from_parquet(
    spec_path: str | Path, pos_path: str | Path, policy: M19Policy = M19Policy()
):
    candidates=select_candidates_from_parquet(spec_path,policy)
    positions=positions_for_candidates_from_parquet(pos_path,candidates,policy)
    # Reuse the already-frozen dataframe-level M19 builder.  The candidate
    # frame is one row/MMSI, so its deterministic reselection is identical.
    return build_mmdec_target_free_holdout(candidates,positions,policy)


def reveal_labels_from_parquet(spec_path: str | Path, cohort_locator: pd.DataFrame) -> pd.DataFrame:
    """Read ETA components only after the prediction ledger has been sealed."""
    locator=cohort_locator.copy()
    locator["_decision_ns"]=[pd.Timestamp(x).value for x in locator.decision_time]
    # pandas.itertuples() renames columns that start with an underscore, so
    # access by column values instead of tuple attributes.  This is a pure
    # post-seal label-reveal plumbing fix; it does not alter cohort selection,
    # blind predictions, hashes, or any frozen model/gate parameter.
    target=dict(zip(locator["mmsi"].astype(int), locator["_decision_ns"].astype("int64")))
    mset=np.array(sorted(target),dtype=np.int64)
    md=parse_footer(spec_path); rows=[]
    needed=["Date","MessageType","Mmsi","EtaMonth","EtaDay","EtaHour","EtaMinute"]
    for rgi in range(len(md.row_groups)):
        cmap=_flat_map(md,rgi); dates=decode_column(spec_path,md,rgi,cmap["Date"]); mmsi=decode_column(spec_path,md,rgi,cmap["Mmsi"])
        marr=np.asarray(mmsi,dtype=np.int64); darr=np.asarray(dates,dtype=np.int64)
        ix=np.flatnonzero(np.isin(marr,mset,assume_unique=False))
        if ix.size==0: continue
        mm=marr[ix]; dd=darr[ix]; want=np.fromiter((target[int(x)] for x in mm),dtype=np.int64,count=len(mm))
        ix=ix[dd==want]
        if ix.size==0: continue
        d={"Date":darr[ix],"Mmsi":marr[ix]}
        for n in needed:
            if n in ("Date","Mmsi"): continue
            d[n]=np.asarray(decode_column(spec_path,md,rgi,cmap[n]))[ix]
        rows.append(pd.DataFrame(d))
    if not rows: raise ValueError("no exact MMDEC SPEC rows found for frozen M19 cohort")
    raw=pd.concat(rows,ignore_index=True)
    return reveal_mmdec_label_ledger(raw,cohort_locator)
