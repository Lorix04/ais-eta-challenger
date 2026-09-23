from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ais_eta.m17g import (
    M17G_CANDIDATES, M17G_RETRIEVAL_CONFIG, M17G_PROMOTION_MIN_GAIN_H,
    promotion_pass, train_candidate,
)

def test_m17g_keeps_frozen_m17a_retrieval_contract():
    assert M17G_RETRIEVAL_CONFIG.representation=='geo_kin_6h'
    assert M17G_RETRIEVAL_CONFIG.gate=='destination'
    assert M17G_RETRIEVAL_CONFIG.k==9

def test_m17g_candidates_are_predeclared_and_target_free_encoder_families():
    assert M17G_CANDIDATES==('moco_tcn64','masked_transformer_mae64')
    assert M17G_PROMOTION_MIN_GAIN_H==1.0

def test_m17g_promotion_gate_is_strict():
    assert promotion_pass(gain_h=1.1,fold_wins=3,p90_ratio=1.0)
    assert not promotion_pass(gain_h=.9,fold_wins=5,p90_ratio=.9)
    assert not promotion_pass(gain_h=2,fold_wins=2,p90_ratio=.9)
    assert not promotion_pass(gain_h=2,fold_wins=5,p90_ratio=1.04)

def test_m17g_trainers_are_deterministic_on_tiny_data():
    rng=np.random.default_rng(7)
    tr=rng.normal(size=(24,12,5)).astype('float32')
    q=rng.normal(size=(7,12,5)).astype('float32')
    for cand in M17G_CANDIDATES:
        a=train_candidate(cand,tr,q,seed=123,epochs=1).embeddings
        b=train_candidate(cand,tr,q,seed=123,epochs=1).embeddings
        assert a.shape==(7,64)
        np.testing.assert_allclose(a,b,atol=1e-7,rtol=0)
        np.testing.assert_allclose(np.linalg.norm(a,axis=1),1.0,atol=1e-6)

def test_m17g_artifacts_if_present_are_development_only():
    s=ROOT/'reports/M17G_SUMMARY.json'; p=ROOT/'reports/m17g_oof_predictions.csv'
    if not s.exists() or not p.exists(): return
    import pandas as pd
    summary=json.loads(s.read_text()); led=pd.read_csv(p); man=json.loads((ROOT/'reports/M16A_DEVELOPMENT_MANIFEST.json').read_text())
    assert len(led)==386 and led.mmsi.nunique()==386
    assert set(led.mmsi.astype(int)).isdisjoint(man['blocked_old_final_mmsi'])
    assert summary['final_test_used_for_selection'] is False
    assert summary['outer_valid_used_for_ssl_pretraining'] is False

def test_project_state_can_advance_to_m17g():
    st=json.loads((ROOT/'PROJECT_STATE.json').read_text())
    if 'm17g_summary' not in st: return
    assert st['current_phase'].startswith(('M17G_','M17H_','M17I_','M17J_','M18A_','M18B_','M18C_','M18D_','M18E_','M18F_','M18G_','M19_'))
    assert st['m17g_summary']['final_test_used_for_selection'] is False
