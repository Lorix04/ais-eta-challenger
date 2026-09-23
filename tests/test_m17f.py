from pathlib import Path
import numpy as np, pandas as pd
from ais_eta.m17f import build_noise_features,fit_student_t_mixture,fit_predict_candidate,M17F_CANDIDATES

def test_m17f_candidate_contract():
    assert 'always_m16g' in M17F_CANDIDATES and any(c.startswith('latent_t2') for c in M17F_CANDIDATES)

def test_student_t_mixture_deterministic():
    x=np.r_[np.linspace(-10,10,40),np.linspace(80,120,20)]
    a=fit_student_t_mixture(x,2); b=fit_student_t_mixture(x,2)
    assert np.allclose(a.locs,b.locs) and np.allclose(a.responsibilities,b.responsibilities)

def test_runtime_features_reject_target_columns():
    b=np.ones((3,4)); q=pd.DataFrame({'target_tte_h':[1,2,3]})
    try: build_noise_features([1,2,3],b,q)
    except ValueError: pass
    else: raise AssertionError('target-derived feature accepted')

def test_always_m16g_identity():
    x=pd.DataFrame({'a':[0.,1.,2.]}); y=np.array([1.,2.,3.]); base=np.array([2.,2.,2.]); p,c,_=fit_predict_candidate('always_m16g',x,y,base,x,base)
    assert np.array_equal(p,base) and np.array_equal(c,np.zeros(3))

def test_project_state_m17f_or_later():
    import json
    s=json.loads((Path(__file__).resolve().parents[1]/'PROJECT_STATE.json').read_text())
    assert s['current_phase'].startswith(('M17F','M17G','M17H','M17I','M17J','M18','M19'))
