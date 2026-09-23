#!/usr/bin/env python3
"""Verify M17A leakage controls, benchmark gate, determinism metadata and prior freezes."""
from __future__ import annotations

import hashlib, json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ais_eta.m17a import M17A_PROMOTION_MAX_P90_RATIO, M17A_PROMOTION_MIN_FOLD_WINS

R=ROOT/'reports'

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main() -> int:
    s=json.loads((R/'M17A_SUMMARY.json').read_text())
    f=json.loads((R/'M17A_SSL_RETRIEVAL_FREEZE.json').read_text())
    l=pd.read_csv(R/'m17a_oof_predictions.csv')
    m=pd.read_csv(R/'m17a_model_metrics.csv').set_index('model')
    fm=pd.read_csv(R/'m17a_fold_metrics.csv')
    ta=pd.read_csv(R/'m17a_ssl_training_audit.csv')
    manifest=json.loads((R/'M16A_DEVELOPMENT_MANIFEST.json').read_text())
    old_final=set(int(x) for x in manifest['blocked_old_final_mmsi'])

    assert s['development_rows']==386 and s['blocked_old_final_rows']==53
    assert s['final_test_used_for_selection'] is False
    assert s['outer_valid_used_for_ssl_pretraining'] is False
    assert len(l)==386 and l['mmsi'].nunique()==386
    assert set(l['mmsi'].astype(int)).isdisjoint(old_final)
    assert len(ta)==5 and not ta['outer_valid_used_for_ssl_pretraining'].astype(bool).any()
    assert (ta['ssl_pretraining_unique_mmsi']==ta['outer_train_mmsi']).all()
    assert (ta['ssl_train_loss_end'] < ta['ssl_train_loss_start']).all()

    ssl='m17a_ssl_masked_contrastive_cosine_destination_k9'
    base='m16d_official_nested_route_analogue'
    fixed='m17a_fixed_dtw_geo_kin_6h_destination_k9'
    assert float(m.loc[ssl,'mae_h']) < float(m.loc[base,'mae_h'])
    assert float(m.loc[ssl,'mae_h']) < float(m.loc[fixed,'mae_h'])
    assert float(m.loc[ssl,'p90_ae_h']) / float(m.loc[base,'p90_ae_h']) <= M17A_PROMOTION_MAX_P90_RATIO + 1e-12
    p=fm.pivot(index='outer_fold',columns='model',values='mae_h')
    wins_base=int((p[ssl] < p[base]).sum())
    wins_fixed=int((p[ssl] < p[fixed]).sum())
    assert wins_base >= M17A_PROMOTION_MIN_FOLD_WINS
    assert wins_fixed >= M17A_PROMOTION_MIN_FOLD_WINS
    assert wins_base==int(s['ssl_fold_wins_vs_m16d'])
    assert wins_fixed==int(s['ssl_fold_wins_vs_fixed_same_config_dtw'])
    assert s['promotion_gate_passed'] is True and s['gate']=='PASS_LEARNED_RETRIEVAL_COMPONENT'

    # Frozen prior artifacts must still match the hashes captured in the M17A freeze.
    mapping={
      'm16d_summary':R/'M16D_SUMMARY.json',
      'm16d_predictions':R/'m16d_oof_predictions.csv',
      'm16d_sequences':R/'m16d_route_sequences.npz',
      'm16d_distances':R/'m16d_route_distance_matrices.npz',
      'm16j_freeze':R/'M16_FINAL_FREEZE.json',
    }
    for key,pth in mapping.items():
        assert sha(pth)==f['immutable_inputs'][key]
    for name,expected in f['artifact_sha256'].items():
        pth=R/name
        assert pth.exists(), name
        assert sha(pth)==expected, name

    # M14/M10/M6/M16J immutability contract remains authoritative.
    m16j=json.loads((R/'M16_FINAL_FREEZE.json').read_text())
    for key,rec in m16j['immutable_pre_m16_sha256'].items():
        pth=ROOT/rec['path']
        assert pth.exists(), key
        assert sha(pth)==rec['sha256'], key
    assert sha(ROOT/m16j['m16_freeze_zip'])==m16j['m16_freeze_zip_sha256']

    print('PASS M17A development-only population 386; old-final 53 hard-blocked')
    print('PASS outer-valid trajectories excluded from every SSL pretraining fold')
    print(f"PASS SSL MAE {m.loc[ssl,'mae_h']:.6f} < M16D {m.loc[base,'mae_h']:.6f}")
    print(f"PASS SSL same-config gain {m.loc[fixed,'mae_h']-m.loc[ssl,'mae_h']:.6f} h; fold wins {wins_fixed}/5")
    print(f"PASS promotion gate: {wins_base}/5 wins vs M16D; P90 ratio {m.loc[ssl,'p90_ae_h']/m.loc[base,'p90_ae_h']:.6f}")
    print('PASS prior M14/M10/M6/M16J and M16D artifacts unchanged')
    print('PASS M17A artifact hashes verified')
    return 0

if __name__=='__main__': raise SystemExit(main())
