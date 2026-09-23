from __future__ import annotations
import json
from pathlib import Path
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m18g import validate_prediction_ledger
from ais_eta.m18g1 import score_fresh_rows
from ais_eta.m19 import (
    M19Policy, M19_SOURCE_DOI, M19_SOURCE_FILES, M19_VERSION,
    build_mmdec_target_free_holdout, reveal_mmdec_label_ledger,
)


def _synthetic_mmdec(n=8):
    spec=[]; pos=[]
    for j in range(n):
        m=900000000+j
        dt=pd.Timestamp('2023-07-10 12:00:00')+pd.Timedelta(minutes=j)
        spec.append(dict(Date=dt, MessageType=5, Mmsi=m, ShipType=70+j%3,
                         Draught10thMetres=80, Destination='GBSOU',
                         EtaMonth=7, EtaDay=11, EtaHour=6, EtaMinute=0))
        for k in range(15):
            t=dt-pd.Timedelta(minutes=30*(14-k))
            pos.append(dict(Date=t, MessageType=1, Mmsi=m, NavigationStatus=0,
                            Latitude=49.0+0.01*j+0.002*k,
                            Longitude=-5.0+0.01*j+0.004*k,
                            CourseOverGroundDegrees=60, SpeedOverGround=12,
                            TrueHeadingDegrees=60))
    return pd.DataFrame(spec), pd.DataFrame(pos)


def test_m19_source_contract_is_frozen():
    assert M19_VERSION.startswith('m19-mmdec-fresh-external-holdout')
    assert M19_SOURCE_DOI == '10.5281/zenodo.17491518'
    assert M19_SOURCE_FILES['Dataset_AIS_POS.parquet']['md5'] == '12b824d26488e381680b2de90090cc2c'
    assert M19_SOURCE_FILES['Dataset_AIS_SPEC.parquet']['md5'] == 'f1dd53064868fa078c714986991c3941'


def test_m19_builder_is_target_free_and_reveal_is_separate():
    spec,pos=_synthetic_mmdec()
    rows,states,loc,audit=build_mmdec_target_free_holdout(spec,pos,M19Policy(max_rows=5,preselect_rows=8))
    assert len(rows)==5 and rows.mmsi.nunique()==5
    assert len(loc)==5 and loc.sample_key.nunique()==5
    assert audit['eta_values_exposed_to_scoring_rows'] is False
    forbidden={'target_tte_h','eta_reference_dt','eta_reference_raw','reference_eta_status','eta_month','eta_day','eta_hour','eta_minute'}
    assert forbidden.isdisjoint(rows.columns)
    assert {'mmsi','recorded_at','position_valid','lat_clean','lon_clean','sog_clean','cog_clean'}.issubset(states.columns)
    labels=reveal_mmdec_label_ledger(spec,loc)
    assert len(labels)==len(rows)
    assert labels.target_tte_h.gt(0).all()
    assert set(labels.sample_key)==set(loc.sample_key)


def test_m19_rows_score_with_registered_m18g1_bundle():
    spec,pos=_synthetic_mmdec()
    rows,states,loc,_=build_mmdec_target_free_holdout(spec,pos,M19Policy(max_rows=3,preselect_rows=8))
    reg=json.loads((ROOT/'reports'/'M18G1_SCORING_BUNDLE_REGISTRATION.json').read_text())
    bundle=joblib.load(ROOT/reg['bundle_path'])
    ledger,detail=score_fresh_rows(bundle,rows,states)
    validate_prediction_ledger(ledger,require_candidate=False)
    assert len(ledger)==3
    assert 'target_tte_h' not in ledger.columns
    assert set(ledger.sample_key)==set(loc.sample_key)
    assert detail.m18f_action.isin(['AUTO_ETA_WITH_UNCERTAINTY','DEFER_LOW_CONFIDENCE']).all()


def test_m19_reveal_rejects_ambiguous_eta_for_same_message():
    spec,pos=_synthetic_mmdec()
    rows,states,loc,_=build_mmdec_target_free_holdout(spec,pos,M19Policy(max_rows=1,preselect_rows=8))
    m=int(loc.iloc[0].mmsi); dt=pd.Timestamp(loc.iloc[0].decision_time)
    base=spec.loc[spec.Mmsi.eq(m)].iloc[0].copy()
    base['Date']=dt; base['EtaMinute']=17
    bad=pd.concat([spec,pd.DataFrame([base])],ignore_index=True)
    try:
        reveal_mmdec_label_ledger(bad,loc)
    except ValueError as e:
        assert 'ambiguous' in str(e)
    else:
        raise AssertionError('ambiguous ETA rows must be rejected')
