from pathlib import Path
import pandas as pd

from ais_eta.m6 import call_level_errors, frozen_design


def test_frozen_design_contains_fixed_m2_m4_choices():
    d = frozen_design(2.485228427881811, 0.4)
    assert d['m2_config']['knn_k'] == 3
    assert d['m2_config']['speed_floor_kn'] == 1.0
    assert d['reliability']['max_supported_pred_h'] == 24.0
    assert d['uncertainty']['simultaneous_call_level_half_width_h'] == 2.485228427881811
    assert d['stability']['alpha'] == 0.4


def test_call_level_errors_equalizes_calls():
    d = pd.DataFrame({
        'session_id':['a','a','b'], 'port':['P','P','P'], 'mmsi':[1,1,2], 'name':['A','A','B'],
        'true_tta_h':[1.,2.,1.], 'p':[2.,2.,3.]
    })
    x = call_level_errors(d, ['p'])
    assert len(x) == 2
    assert x.set_index('session_id').loc['a','p_mae_h'] == 0.5
    assert x.set_index('session_id').loc['b','p_mae_h'] == 2.0
