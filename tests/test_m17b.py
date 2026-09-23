from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ais_eta.m17b import (
    CohortHNSWMemory, DeterministicHNSW, M17B_K_OWNERS,
    build_historical_memory_windows, exact_memory_predict,
)


def test_hnsw_is_deterministic_and_recovers_simple_neighbors():
    x=np.array([[1.,0.],[.99,.01],[0.,1.],[-1.,0.]],float)
    a=DeterministicHNSW(2,seed=7); a.add_items(x)
    b=DeterministicHNSW(2,seed=7); b.add_items(x)
    qa=a.query(np.array([1.,0.]),k=3,ef_search=16)
    qb=b.query(np.array([1.,0.]),k=3,ef_search=16)
    np.testing.assert_array_equal(qa.ids,qb.ids)
    np.testing.assert_allclose(qa.distances,qb.distances)
    assert int(qa.ids[0])==0 and int(qa.ids[1])==1


def test_memory_windows_never_cross_cutoff_and_labels_align_to_snapshot():
    t0=pd.Timestamp('2026-04-10 12:00:00')
    times=pd.date_range(t0-pd.Timedelta(hours=8),t0+pd.Timedelta(hours=1),freq='30min')
    st=pd.DataFrame({'mmsi':[1]*len(times),'recorded_at':times,'lat_clean':37+np.arange(len(times))*.001,
        'lon_clean':15+np.arange(len(times))*.001,'sog_clean':[8.]*len(times),'cog_clean':[90.]*len(times),'position_valid':[True]*len(times)})
    dev=pd.DataFrame({'mmsi':[1],'history_last_at':[t0],'last_update':[t0+pd.Timedelta(hours=1)],
        'eta_reference_dt':[t0+pd.Timedelta(hours=11)],'target_tte_h':[10.], 'canonical_destination':['ITCTA']})
    seq,meta=build_historical_memory_windows(st,dev,horizon_h=4,stride_h=1)
    assert len(seq)==len(meta)>0
    assert (pd.to_datetime(meta.memory_last_at)<=pd.to_datetime(meta.history_cutoff_at)).all()
    last=meta.sort_values('memory_last_at').iloc[-1]
    assert abs(float(last.snapshot_tte_h)-11.0)<1e-6


def test_exact_memory_deduplicates_owners():
    emb=np.array([[1,0],[.999,.001],[.9,.1],[0,1]],float)
    meta=pd.DataFrame({'mmsi':[1,1,2,3],'canonical_destination':['A']*4,'snapshot_tte_h':[10,10,20,30],'owner_target_tte_h':[9,9,19,29]})
    out=exact_memory_predict(memory_embeddings=emb,memory_meta=meta,query_embedding=np.array([1.,0.]),query_destination='A',query_age_h=1,k_owners=3)
    owners=[int(x) for x in str(out['neighbour_owner_mmsi']).split(';')]
    assert owners==[1,2,3] and out['neighbour_count']==3


def test_cohort_hnsw_respects_destination_gate():
    emb=np.array([[1,0],[.95,.05],[.9,.1],[0,1],[-.1,.9]],float)
    meta=pd.DataFrame({'mmsi':[1,2,3,4,5],'canonical_destination':['A','A','A','B','B'],
        'snapshot_tte_h':[1,2,3,4,5],'owner_target_tte_h':[1,2,3,4,5]})
    idx=CohortHNSWMemory(emb,meta,seed=4)
    out=idx.query(np.array([1.,0.]),'A',k_owners=3)
    assert out['gate_used']=='destination'
    assert set(out['owner_mmsi'])=={1,2,3}


def test_m17b_artifacts_if_present_are_development_only():
    s=ROOT/'reports/M17B_SUMMARY.json'; p=ROOT/'reports/m17b_oof_predictions.csv'
    if not s.exists() or not p.exists(): return
    summary=json.loads(s.read_text()); led=pd.read_csv(p)
    manifest=json.loads((ROOT/'reports/M16A_DEVELOPMENT_MANIFEST.json').read_text())
    assert len(led)==386 and led.mmsi.nunique()==386
    assert set(led.mmsi.astype(int)).isdisjoint(manifest['blocked_old_final_mmsi'])
    assert summary['final_test_used_for_selection'] is False
    assert summary['outer_valid_used_for_memory'] is False
    assert summary['outer_valid_used_for_ssl_pretraining'] is False


def test_project_state_can_advance_to_m17b():
    state=json.loads((ROOT/'PROJECT_STATE.json').read_text())
    if 'm17b_summary' not in state: return
    assert state['current_phase'].startswith(('M17B_','M17C_','M17D_','M17E_','M17F_','M17G_','M17H_','M18A_','M18B_','M18C_','M18D_','M18E_','M18F_','M18G_','M19_'))
    assert state['m17b_summary']['final_test_used_for_selection'] is False
