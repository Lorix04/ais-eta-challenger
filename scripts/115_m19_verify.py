#!/usr/bin/env python3
"""Verify the M19 external holdout implementation and frozen one-shot result."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import joblib
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ais_eta.m18g import validate_prediction_ledger, sha256_file
from ais_eta.m18g1 import score_fresh_rows
from ais_eta.m19 import M19Policy, M19_SOURCE_DOI, M19_VERSION, build_mmdec_target_free_holdout


def fixture():
    spec=[];pos=[]
    for j in range(6):
        m=910000000+j; dt=pd.Timestamp('2023-08-15 08:00:00')+pd.Timedelta(minutes=j)
        spec.append(dict(Date=dt,MessageType=5,Mmsi=m,ShipType=70,Draught10thMetres=90,Destination='GBSOU',EtaMonth=8,EtaDay=16,EtaHour=2,EtaMinute=0))
        for k in range(15):
            t=dt-pd.Timedelta(minutes=30*(14-k))
            pos.append(dict(Date=t,MessageType=1,Mmsi=m,NavigationStatus=0,Latitude=49.2+0.002*k,Longitude=-4.8+0.004*k,CourseOverGroundDegrees=62,SpeedOverGround=12,TrueHeadingDegrees=62))
    return pd.DataFrame(spec),pd.DataFrame(pos)


def main()->int:
    state=json.loads((ROOT/'PROJECT_STATE.json').read_text())
    status=json.loads((ROOT/'reports'/'M19_STATUS.json').read_text())
    registry=json.loads((ROOT/'reports'/'M19_SOURCE_REGISTRY.json').read_text())
    reg=json.loads((ROOT/'reports'/'M18G1_SCORING_BUNDLE_REGISTRATION.json').read_text())
    assert str(state.get('current_phase','')).startswith('M19_')
    assert registry['version']==M19_VERSION and registry['source']['doi']==M19_SOURCE_DOI
    bundle_path=ROOT/reg['bundle_path']
    assert sha256_file(bundle_path)==reg['bundle_sha256']

    # The historical pre-opening implementation freeze remains as provenance,
    # but the current package is verified against the post-opening result freeze.
    result_freeze_path=ROOT/'reports'/'M19_EXTERNAL_RESULT_FREEZE.json'
    if status.get('external_holdout_opened'):
        assert status['external_labels_revealed'] is True
        assert status['external_score_computed'] is True
        assert status['no_retuning_on_same_holdout'] is True
        freeze=json.loads(result_freeze_path.read_text())
        assert freeze['status']=='ONE_SHOT_COMPLETED__NO_RETUNING_ON_MMDEC'
        assert freeze['rows']==500 and freeze['no_retuning_on_same_holdout'] is True
        portability_patch_path = ROOT/'reports'/'M19_PORTABILITY_PATCH.json'
        portability_patch = json.loads(portability_patch_path.read_text()) if portability_patch_path.exists() else None
        for rel,digest in freeze['artifact_sha256'].items():
            actual = sha256_file(ROOT/rel)
            if actual == digest:
                continue
            if rel == 'src/ais_eta/m19_parquet_lite.py' and portability_patch is not None:
                assert portability_patch['scientific_status'] == 'NON_SCIENTIFIC_PACKAGING_ONLY'
                assert portability_patch['historical_m19_source_sha256'] == digest
                assert portability_patch['patched_source_sha256'] == actual
                assert portability_patch['no_model_or_gate_parameter_changed'] is True
                assert portability_patch['frozen_prediction_ledger_sha256'] == freeze['prelabel_prediction_ledger_sha256']
                assert portability_patch['frozen_label_ledger_sha256'] == freeze['label_ledger_sha256']
                continue
            raise AssertionError(rel)
        opening=json.loads((ROOT/'reports'/'M19_EXTERNAL_OPENING_MANIFEST.json').read_text())
        assert opening['labels_observed_when_prediction_ledger_sealed'] is False
        assert opening['prediction_ledger_sha256']==sha256_file(ROOT/'reports'/'m19_external_blind_predictions.csv')
        result=json.loads((ROOT/'reports'/'M19_EXTERNAL_EVALUATION.json').read_text())
        assert result['rows']==500 and result['no_retuning_on_same_holdout'] is True
        assert result['selective_eta']['decision']=='DO_NOT_EXTERNALLY_VALIDATE_M18F_LAYER'
        assert abs(result['selective_eta']['full_mae_h']-550.4090836774562)<1e-9
    else:
        assert status['external_labels_revealed'] is False

    # Keep target-free adapter/scorer smoke coverage independent of the spent
    # real test set.
    spec,pos=fixture()
    rows,states,loc,audit=build_mmdec_target_free_holdout(spec,pos,M19Policy(max_rows=3,preselect_rows=6))
    assert audit['eta_values_exposed_to_scoring_rows'] is False
    assert 'target_tte_h' not in rows.columns
    bundle=joblib.load(bundle_path)
    ledger,_=score_fresh_rows(bundle,rows,states)
    validate_prediction_ledger(ledger,require_candidate=False)
    assert set(ledger.sample_key)==set(loc.sample_key)
    print('PASS M19 source contract:', M19_SOURCE_DOI)
    print('PASS M19 current phase:', state['current_phase'])
    print('PASS M19 real external holdout rows:', status.get('rows'))
    print('PASS M19 external full MAE h:', status.get('full_mae_h'))
    print('PASS M19 selective decision:', status.get('selective_decision'))
    print('PASS MMDEC spent / no-retuning rule:', status.get('no_retuning_on_same_holdout'))
    print('PASS registered M18G1 bundle SHA-256:', reg['bundle_sha256'])
    print('PASS synthetic target-free adapter -> blind scorer rows:', len(ledger))
    return 0

if __name__=='__main__': raise SystemExit(main())
