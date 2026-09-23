from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ais_eta.m17d import (
    M17D_EXPERT_NAMES, M17D_META_CANDIDATES, build_gating_features,
    fit_predict_meta, promotion_gate,
)


def _quality(n: int) -> pd.DataFrame:
    return pd.DataFrame({
        'm16c_deepest_support':[4]*n, 'm16c_deepest_weight':[.5]*n, 'm16c_fallback_count':[1]*n,
        'm16d_neighbour_count':[9]*n, 'm16d_nearest_distance':[.2]*n, 'm16d_median_neighbour_distance':[.4]*n,
        'm16d_similarity_gap':[.1]*n, 'm16d_gate_used':['destination']*n,
        'physics_eligible':[1]*n, 'resolution_confidence':[1]*n, 'physics_distance_gc_nm':[20]*n,
        'physics_course_alignment_deg':[10]*n, 'physics_recent_speed_max_kn':[12]*n,
        'm16e_effective_speed_kn':[11]*n, 'm16e_distance_factor':[1]*n, 'm16e_route_distance_proxy_nm':[21]*n,
        'm17a_neighbour_count':[9]*n, 'm17a_nearest_distance':[.1]*n, 'm17a_median_neighbour_distance':[.2]*n,
        'm17a_similarity_gap':[.3]*n, 'm17a_gate_used':['destination']*n,
        'm17c_graph_supported':[1]*n, 'm17c_path_edges':[3]*n, 'm17c_min_path_support':[2]*n,
        'm17c_median_path_support':[4]*n, 'm17c_contextual_edge_share':[.5]*n,
        'm17c_source_snap_nm':[1]*n, 'm17c_destination_snap_nm':[2]*n,
    })


def test_m17d_has_six_predeclared_experts_and_meta_panel():
    assert M17D_EXPERT_NAMES == ('prior','route','physics_gate','tabular','learned_retrieval','corridor_graph_gate')
    assert {'median_blend','nnls_convex','rf_hard_gate','extra_trees_hard_gate','hist_gb_hard_gate','logit_soft_gate'} <= set(M17D_META_CANDIDATES)


def test_gating_features_reject_target_columns():
    b=np.tile(np.arange(1,7,dtype=float),(4,1)); q=_quality(4); q['target_tte_h']=1.0
    try: build_gating_features(b,q)
    except ValueError as e: assert 'target-derived' in str(e)
    else: raise AssertionError('target-derived gating column must be rejected')


def test_gating_features_include_new_signal_disagreement():
    b=np.tile(np.arange(1,7,dtype=float),(4,1)); out=build_gating_features(b,_quality(4))
    assert {'pred_learned_retrieval_h','pred_corridor_graph_gate_h','learned_minus_route_h','graph_minus_physics_h'} <= set(out.columns)
    assert np.isfinite(out.to_numpy()).all()


def test_simple_blends_are_finite():
    train=np.array([[1,2,3,4,5,6],[2,3,4,5,6,7],[3,4,5,6,7,8],[4,5,6,7,8,9]],float)
    valid=np.array([[2,3,4,5,6,7],[5,6,7,8,9,10]],float)
    qtr=build_gating_features(train,_quality(len(train))); qv=build_gating_features(valid,_quality(len(valid)))
    for c in ['median_blend','mean_blend','nnls_convex']:
        r=fit_predict_meta(c,train,[2,3,4,5],qtr,valid,qv)
        assert len(r.prediction)==2 and np.isfinite(r.prediction).all()


def test_promotion_gate_requires_all_predeclared_conditions():
    base={'mae_h':185.88,'p90_ae_h':290.68}
    good={'mae_h':183.0,'p90_ae_h':295.0}
    assert promotion_gate(base,good,fold_wins=3,changed_row_win_share=.51)=='PASS_M17D_EXTENDED_MOE'
    assert promotion_gate(base,good,fold_wins=2,changed_row_win_share=.51)=='NO_M17D_PROMOTION'
    assert promotion_gate(base,good,fold_wins=3,changed_row_win_share=.49)=='NO_M17D_PROMOTION'
    badp90={'mae_h':183.0,'p90_ae_h':310.0}
    assert promotion_gate(base,badp90,fold_wins=4,changed_row_win_share=.6)=='NO_M17D_PROMOTION'


def test_m17d_artifacts_if_present_are_development_only_and_inner_oof():
    s=ROOT/'reports/M17D_SUMMARY.json'; p=ROOT/'reports/m17d_oof_predictions.csv'
    if not s.exists() or not p.exists(): return
    summary=json.loads(s.read_text()); led=pd.read_csv(p)
    manifest=json.loads((ROOT/'reports/M16A_DEVELOPMENT_MANIFEST.json').read_text())
    assert len(led)==386 and led.mmsi.nunique()==386
    assert set(led.mmsi.astype(int)).isdisjoint(manifest['blocked_old_final_mmsi'])
    assert summary['final_test_used_for_selection'] is False
    assert summary['m17a_regenerated_inner_oof'] is True
    assert summary['m17c_regenerated_inner_oof'] is True
    assert summary['target_derived_gating_features_used'] is False
    assert summary['gate'] in {'PASS_M17D_EXTENDED_MOE','NO_M17D_PROMOTION'}


def test_m17d_project_state_can_advance():
    state=json.loads((ROOT/'PROJECT_STATE.json').read_text())
    if 'm17d_summary' not in state: return
    assert state['current_phase'].startswith(('M17D_','M17E_','M17F_','M17G_','M17H_','M18A_','M18B_','M18C_','M18D_','M18E_','M18F_','M18G_','M19_'))
    assert state['m17d_summary']['final_test_used_for_selection'] is False
