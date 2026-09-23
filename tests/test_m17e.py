from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from ais_eta.m17e import *


def _q(n):
    return pd.DataFrame({
      'm16c_deepest_support':[4]*n,'m16c_deepest_weight':[.5]*n,'m16c_fallback_count':[1]*n,
      'm16d_neighbour_count':[9]*n,'m16d_nearest_distance':[.2]*n,'m16d_median_neighbour_distance':[.4]*n,'m16d_similarity_gap':[.1]*n,'m16d_gate_used':['destination']*n,
      'physics_eligible':[1]*n,'resolution_confidence':[1]*n,'physics_distance_gc_nm':[20]*n,'physics_course_alignment_deg':[10]*n,'physics_recent_speed_max_kn':[12]*n,
      'm16e_effective_speed_kn':[11]*n,'m16e_distance_factor':[1]*n,'m16e_route_distance_proxy_nm':[21]*n,
      'm17a_neighbour_count':[9]*n,'m17a_nearest_distance':[.1]*n,'m17a_median_neighbour_distance':[.2]*n,'m17a_similarity_gap':[.3]*n,'m17a_gate_used':['destination']*n,
    })

def test_m17e_is_binary_and_conservative():
    assert M17E_ROUTER_CANDIDATES[0]=='always_m16g'
    assert M17E_MAX_SWITCH_SHARE==.35 and M17E_LABEL_MARGIN_H==1.0

def test_features_reject_target_columns():
    q=_q(4); q['target_tte_h']=1
    try: build_router_features([1]*4,[2]*4,np.ones((4,4)),q)
    except ValueError as e: assert 'target-derived' in str(e)
    else: raise AssertionError('target column must be rejected')

def test_features_include_disagreement_and_quality():
    x=build_router_features([1,2],[2,1],np.array([[1,2,3,4],[2,3,4,5]],float),_q(2))
    assert {'abs_disagreement_h','m16g_expert_std_h','m17a_nearest_distance','learned_gate_destination'} <= set(x.columns)
    assert np.isfinite(x.to_numpy()).all()

def test_always_m16g_never_switches():
    q=_q(4); x=build_router_features([1]*4,[2]*4,np.ones((4,4)),q)
    r=fit_predict_router('always_m16g',x,[1,2,3,4],[1]*4,[2]*4,x.iloc[:2],[5,6],[7,8])
    assert r.prediction.tolist()==[5,6] and not r.switch_to_m17a.any()

def test_promotion_gate_requires_all_conditions():
    b={'mae_h':185.88,'p90_ae_h':290.68}; g={'mae_h':184.0,'p90_ae_h':289.0}
    assert promotion_gate(b,g,fold_wins=3,changed_row_win_share=.53,switch_share=.2,max_fold_regression_h=10)=='PASS_M17E_SELECTIVE_ROUTER'
    assert promotion_gate(b,g,fold_wins=2,changed_row_win_share=.53,switch_share=.2,max_fold_regression_h=10)=='NO_M17E_PROMOTION'
    assert promotion_gate(b,g,fold_wins=3,changed_row_win_share=.51,switch_share=.2,max_fold_regression_h=10)=='NO_M17E_PROMOTION'
    assert promotion_gate(b,g,fold_wins=3,changed_row_win_share=.53,switch_share=.4,max_fold_regression_h=10)=='NO_M17E_PROMOTION'

def test_m17e_artifacts_if_present_are_development_only():
    s=ROOT/'reports/M17E_SUMMARY.json'; p=ROOT/'reports/m17e_oof_predictions.csv'
    if not s.exists() or not p.exists(): return
    summary=json.loads(s.read_text()); led=pd.read_csv(p); manifest=json.loads((ROOT/'reports/M16A_DEVELOPMENT_MANIFEST.json').read_text())
    assert len(led)==386 and led.mmsi.nunique()==386
    assert set(led.mmsi.astype(int)).isdisjoint(manifest['blocked_old_final_mmsi'])
    assert summary['final_test_used_for_selection'] is False
    assert summary['router_actions']==['KEEP_M16G','SWITCH_TO_M17A']
    assert summary['target_derived_router_features_used'] is False
    assert summary['gate'] in {'PASS_M17E_SELECTIVE_ROUTER','NO_M17E_PROMOTION'}
