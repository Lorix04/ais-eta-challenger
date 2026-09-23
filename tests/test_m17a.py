from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from ais_eta.m17a import (
    M17A_RETRIEVAL_CONFIG,
    build_causal_pretraining_windows,
    cosine_distance_matrix,
    standardize_apply,
    standardize_fit,
)


def test_m17a_retrieval_protocol_changes_metric_not_eta_aggregator():
    assert M17A_RETRIEVAL_CONFIG.representation == 'geo_kin_6h'
    assert M17A_RETRIEVAL_CONFIG.gate == 'destination'
    assert M17A_RETRIEVAL_CONFIG.k == 9


def test_cosine_distance_is_symmetric_and_zero_diagonal():
    e = np.array([[1.,0.],[0.,1.],[1.,1.]])
    d = cosine_distance_matrix(e)
    assert np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0)
    assert d[0,1] > d[0,2]


def test_standardization_uses_supplied_training_statistics_only():
    tr = np.arange(2*3*2, dtype=float).reshape(2,3,2)
    va = np.full((1,3,2), 9999.0)
    m,s = standardize_fit(tr)
    m2,s2 = standardize_fit(tr.copy())
    np.testing.assert_allclose(m,m2); np.testing.assert_allclose(s,s2)
    transformed = standardize_apply(va,m,s)
    assert transformed.mean() > 100


def test_pretraining_window_builder_never_crosses_cutoff():
    t0 = pd.Timestamp('2026-04-10 12:00:00')
    states = pd.DataFrame({
        'mmsi':[1]*5,
        'recorded_at':[t0-pd.Timedelta(hours=2),t0-pd.Timedelta(hours=1),t0,t0+pd.Timedelta(minutes=1),t0+pd.Timedelta(hours=1)],
        'lat_clean':[37,37.1,37.2,99,99], 'lon_clean':[15,15.1,15.2,99,99],
        'sog_clean':[5,6,7,99,99], 'cog_clean':[10,20,30,99,99], 'position_valid':[True]*5,
    })
    dev = pd.DataFrame({'mmsi':[1], 'history_last_at':[t0]})
    w,o,a = build_causal_pretraining_windows(states, dev, max_windows_per_mmsi=3)
    assert len(w) >= 1 and set(o)=={1}
    assert (pd.to_datetime(a['window_end_at']) <= pd.to_datetime(a['history_cutoff_at'])).all()


def test_m17a_artifacts_if_present_are_dev_only_and_old_final_blocked():
    s = ROOT/'reports/M17A_SUMMARY.json'
    p = ROOT/'reports/m17a_oof_predictions.csv'
    if not s.exists() or not p.exists():
        return
    summary = json.loads(s.read_text())
    ledger = pd.read_csv(p)
    manifest = json.loads((ROOT/'reports/M16A_DEVELOPMENT_MANIFEST.json').read_text())
    assert len(ledger)==386 and ledger['mmsi'].nunique()==386
    assert set(ledger['mmsi'].astype(int)).isdisjoint(manifest['blocked_old_final_mmsi'])
    assert summary['final_test_used_for_selection'] is False
    assert summary['outer_valid_used_for_ssl_pretraining'] is False


def test_project_state_can_advance_to_m17a():
    state = json.loads((ROOT/'PROJECT_STATE.json').read_text())
    if 'm17a_summary' not in state:
        return
    assert state['current_phase'].startswith(('M17A_','M17B_','M17C_','M17D_','M17E_','M17F_','M17G_','M17H_','M18A_','M18B_','M18C_','M18D_','M18E_','M18F_','M18G_','M19_'))
    assert state['m17a_summary']['final_test_used_for_selection'] is False
    assert state['m17a_summary']['outer_valid_used_for_ssl_pretraining'] is False
