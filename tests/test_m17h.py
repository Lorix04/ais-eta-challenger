from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ais_eta.m17h import (
    M17H_MIN_GAIN_H, M17H_MIN_FOLD_WINS, M17H_MIN_CHANGED_ROW_WIN_SHARE,
    M17H_MAX_P90_RATIO, promotion_gate,
)


def test_m17h_gate_is_predeclared_and_strict():
    assert M17H_MIN_GAIN_H == 1.0
    assert M17H_MIN_FOLD_WINS == 3
    assert M17H_MIN_CHANGED_ROW_WIN_SHARE == 0.50
    assert M17H_MAX_P90_RATIO == 1.02
    assert promotion_gate(gain_h=1.2, fold_wins=3, changed_row_win_share=.55, p90_ratio=1.0, worst_fold_regression_h=5) == 'PASS_CONSERVATIVE_MOCO_INTEGRATION'
    assert promotion_gate(gain_h=.9, fold_wins=5, changed_row_win_share=.8, p90_ratio=.9, worst_fold_regression_h=0) == 'NO_M17H_PROMOTION'


def test_m17h_artifacts_if_present_keep_four_experts_and_block_old_final():
    s=ROOT/'reports/M17H_SUMMARY.json'; p=ROOT/'reports/m17h_oof_predictions.csv'
    if not s.exists() or not p.exists(): return
    summary=json.loads(s.read_text()); led=pd.read_csv(p); man=json.loads((ROOT/'reports/M16A_DEVELOPMENT_MANIFEST.json').read_text())
    assert len(led)==386 and led.mmsi.nunique()==386
    assert set(led.mmsi.astype(int)).isdisjoint(man['blocked_old_final_mmsi'])
    assert summary['expert_count']==4
    assert summary['route_replacement']=='M16D route -> M17G MoCo-TCN64'
    assert summary['physics_fallback_replacement']=='M16D route fallback -> M17G MoCo-TCN64 fallback'
    assert summary['meta_candidate_reselected'] is False
    assert summary['final_test_used_for_selection'] is False


def test_m17h_frozen_meta_strategy_matches_m16g_if_present():
    p=ROOT/'reports/m17h_outer_meta.csv'; q=ROOT/'reports/m16g_outer_selected_meta.csv'
    if not p.exists(): return
    a=pd.read_csv(p).sort_values('outer_fold'); b=pd.read_csv(q).sort_values('outer_fold')
    assert list(a.selected_meta_id)==list(b.selected_meta_id)
    assert a.meta_strategy_frozen_from_m16g.astype(bool).all()


def test_m17h_project_state_forward_compatible():
    st=json.loads((ROOT/'PROJECT_STATE.json').read_text())
    if 'm17h_summary' not in st: return
    assert st['current_phase'].startswith(('M17H_','M17I_','M17J_','M18','M19'))
