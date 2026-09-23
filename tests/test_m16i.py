from __future__ import annotations
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from ais_eta.m16i import paired_bootstrap_gain, red_team_gate, selected_path_counterfactual, support_bucket

def test_support_buckets_are_stable():
    assert [support_bucket(x) for x in [0,1,2,3,5,6,99]] == ['COLD_0','RARE_1_2','RARE_1_2','LOW_3_5','LOW_3_5','SUPPORTED_6PLUS','SUPPORTED_6PLUS']

def test_structural_counterfactual_preserves_available_hard_gate_and_recomputes_blend():
    meta=['rf_hard_gate','rf_hard_gate','median_blend']; sel=['prior','physics_gate','BLEND_MEDIAN']
    rows=[{'prior':1.,'route':2.,'tabular':3.},{'prior':1.,'route':2.,'tabular':3.},{'prior':1.,'route':2.,'tabular':9.}]
    p=selected_path_counterfactual(meta,sel,rows,['prior','route','tabular'])
    assert np.allclose(p,[1.,2.,2.])

def test_bootstrap_is_deterministic():
    g=np.array([1.,-1.,2.,3.]); f=np.array([0,0,1,1])
    a=paired_bootstrap_gain(g,f,reps=20,seed=7); b=paired_bootstrap_gain(g,f,reps=20,seed=7)
    assert np.array_equal(a,b)

def test_red_team_gate_distinguishes_heavy_tail_caveat():
    kw=dict(pooled_gain_h=7,fold_wins=4,leave_one_fold_positive=5,trimmed_5pct_gain_h=2,winsorized_95_gain_h=3,leave_top_destination_gain_h=2,bootstrap_positive_probability=.92,m16h_coverage=.80,confidence_ordered=True)
    assert red_team_gate(**kw,bootstrap_ci95_low_h=-1,top5_positive_gain_share=.51)=='PASS_RED_TEAM_WITH_HEAVY_TAIL_CAVEAT'
    assert red_team_gate(**kw,bootstrap_ci95_low_h=1,top5_positive_gain_share=.40)=='PASS_RED_TEAM_STABLE'

def test_if_m16i_artifacts_exist_they_are_dev_only_and_frozen():
    p=ROOT/'reports/M16I_SUMMARY.json'
    if not p.exists(): return
    s=json.loads(p.read_text()); f=json.loads((ROOT/'reports/M16I_RED_TEAM_FREEZE.json').read_text())
    a=np.genfromtxt(ROOT/'reports/m16i_row_audit.csv',delimiter=',',names=True,dtype=None,encoding='utf8')
    assert len(a)==386 and s['blocked_old_final_rows']==53 and s['final_test_used'] is False
    assert s['gate'] in {'PASS_RED_TEAM_STABLE','PASS_RED_TEAM_WITH_HEAVY_TAIL_CAVEAT'}
    for n,d in f['artifact_sha256'].items(): assert hashlib.sha256((ROOT/'reports'/n).read_bytes()).hexdigest()==d,n
    state=json.loads((ROOT/'PROJECT_STATE.json').read_text()); assert state['current_phase'].startswith(('M16I_','M16J_','M17A_','M17B_','M17C_','M17D_','M17E_','M17F_','M17G_','M17H_','M18A_','M18B_','M18C_','M18D_','M18E_','M18F_','M18G_','M19_'))
