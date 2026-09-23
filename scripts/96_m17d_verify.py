#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'reports'
def sha(p: Path)->str: return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main()->int:
    s=json.loads((R/'M17D_SUMMARY.json').read_text())
    f=json.loads((R/'M17D_EXTENDED_MOE_FREEZE.json').read_text())
    led=pd.read_csv(R/'m17d_oof_predictions.csv')
    ssl=pd.read_csv(R/'m17d_ssl_inner_audit.csv')
    graph=pd.read_csv(R/'m17d_graph_inner_audit.csv')
    m16g=pd.read_csv(R/'m16g_oof_predictions.csv')[['mmsi','pred_m16g_selected_h']]
    manifest=json.loads((R/'M16A_DEVELOPMENT_MANIFEST.json').read_text())
    assert len(led)==386 and led.mmsi.nunique()==386
    assert set(led.mmsi.astype(int)).isdisjoint(manifest['blocked_old_final_mmsi'])
    assert s['blocked_old_final_rows']==53 and s['final_test_used_for_selection'] is False
    assert s['m17a_regenerated_inner_oof'] is True and s['m17c_regenerated_inner_oof'] is True
    assert s['target_derived_gating_features_used'] is False
    assert not ssl['inner_valid_used'].astype(bool).any()
    assert int(graph['inner_valid_owner_overlap'].sum())==0
    x=led.merge(m16g,on='mmsi',suffixes=('','_source'))
    assert np.allclose(x['pred_m16g_frozen_h'],x['pred_m16g_selected_h'],rtol=0,atol=1e-9)
    assert f['immutable_inputs']['m16g_freeze']==sha(R/'M16G_MIXTURE_FREEZE.json')
    assert f['immutable_inputs']['m17a_freeze']==sha(R/'M17A_SSL_RETRIEVAL_FREEZE.json')
    assert f['immutable_inputs']['m17c_freeze']==sha(R/'M17C_CORRIDOR_GRAPH_FREEZE.json')
    assert f['immutable_inputs']['m16j_freeze']==sha(R/'M16_FINAL_FREEZE.json')
    assert f['immutable_inputs']['m14_submission']==sha(ROOT/'dist/ais_eta_takehome_submission_final.zip')
    for name,h in f['artifact_sha256'].items(): assert sha(R/name)==h,name
    assert s['gate'] in {'PASS_M17D_EXTENDED_MOE','NO_M17D_PROMOTION'}
    print(f"PASS M17D development-only {len(led)}; old-final {len(manifest['blocked_old_final_mmsi'])} hard-blocked")
    print(f"PASS inner SSL validation leakage 0/{len(ssl)} folds; graph owner overlap {int(graph['inner_valid_owner_overlap'].sum())}")
    print(f"PASS frozen M16G identity; MAE {s['m16g_frozen_mae_h']:.3f} -> {s['m17d_mae_h']:.3f} h; gain {s['gain_h_vs_m16g']:.3f} h")
    print(f"PASS gate {s['gate']}; fold wins {s['fold_wins_vs_m16g']}/5; changed-row win share {s['changed_row_win_share_vs_m16g']:.3f}")
    return 0
if __name__=='__main__': raise SystemExit(main())
