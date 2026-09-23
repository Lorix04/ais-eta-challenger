#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'reports'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    s=json.loads((R/'M17E_SUMMARY.json').read_text()); f=json.loads((R/'M17E_SELECTIVE_ROUTER_FREEZE.json').read_text()); led=pd.read_csv(R/'m17e_oof_predictions.csv'); ssl=pd.read_csv(R/'m17e_ssl_inner_audit.csv'); sel=pd.read_csv(R/'m17e_outer_selected_router.csv'); manifest=json.loads((R/'M16A_DEVELOPMENT_MANIFEST.json').read_text())
    assert len(led)==386 and led.mmsi.nunique()==386
    assert set(led.mmsi.astype(int)).isdisjoint(manifest['blocked_old_final_mmsi'])
    assert s['blocked_old_final_rows']==53 and s['final_test_used_for_selection'] is False
    assert s['router_actions']==['KEEP_M16G','SWITCH_TO_M17A'] and s['target_derived_router_features_used'] is False
    assert s['m16g_reconstructed_nested'] is True and s['m17a_regenerated_inner_oof'] is True
    assert not ssl['inner_valid_used'].astype(bool).any() and not ssl['outer_valid_used'].astype(bool).any()
    assert float(sel['m16g_rebuild_max_abs_diff_h'].max()) <= 1e-6
    src=pd.read_csv(R/'m16g_oof_predictions.csv')[['mmsi','pred_m16g_selected_h']]; x=led.merge(src,on='mmsi'); assert np.allclose(x.pred_m16g_frozen_h,x.pred_m16g_selected_h,rtol=0,atol=1e-9)
    assert f['immutable_inputs']['m16g_freeze']==sha(R/'M16G_MIXTURE_FREEZE.json'); assert f['immutable_inputs']['m17a_freeze']==sha(R/'M17A_SSL_RETRIEVAL_FREEZE.json'); assert f['immutable_inputs']['m17d_freeze']==sha(R/'M17D_EXTENDED_MOE_FREEZE.json'); assert f['immutable_inputs']['m16j_freeze']==sha(R/'M16_FINAL_FREEZE.json'); assert f['immutable_inputs']['m14_submission']==sha(ROOT/'dist/ais_eta_takehome_submission_final.zip')
    for n,h in f['artifact_sha256'].items(): assert sha(R/n)==h,n
    assert s['gate'] in {'PASS_M17E_SELECTIVE_ROUTER','NO_M17E_PROMOTION'}
    print(f"PASS M17E development-only {len(led)}; old-final {len(manifest['blocked_old_final_mmsi'])} hard-blocked")
    print(f"PASS M16G reconstruction max diff {sel['m16g_rebuild_max_abs_diff_h'].max():.3g} h; SSL validation leakage 0/{len(ssl)} folds")
    print(f"PASS MAE {s['m16g_frozen_mae_h']:.3f} -> {s['m17e_mae_h']:.3f} h; gain {s['gain_h_vs_m16g']:.3f} h; switches {s['switch_share']*100:.2f}%")
    print(f"PASS gate {s['gate']}; folds {s['fold_wins_vs_m16g']}/5; changed-row wins {s['changed_row_win_share_vs_m16g']:.3f}; P90 ratio {s['p90_ratio_vs_m16g']:.4f}")
    return 0
if __name__=='__main__': raise SystemExit(main())
