#!/usr/bin/env python3
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'reports'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    s=json.loads((R/'M17H_SUMMARY.json').read_text()); f=json.loads((R/'M17H_CONSERVATIVE_INTEGRATION_FREEZE.json').read_text()); led=pd.read_csv(R/'m17h_oof_predictions.csv'); man=json.loads((R/'M16A_DEVELOPMENT_MANIFEST.json').read_text())
    assert len(led)==386 and led.mmsi.nunique()==386 and set(led.mmsi.astype(int)).isdisjoint(man['blocked_old_final_mmsi'])
    assert s['blocked_old_final_rows']==53 and not s['final_test_used_for_selection'] and not s['meta_candidate_reselected']
    assert s['expert_count']==4 and s['m17g_regenerated_inner_oof'] and not s['outer_valid_used_for_ssl_pretraining'] and not s['target_used_for_ssl_pretraining']
    m=pd.read_csv(R/'m17h_outer_meta.csv').sort_values('outer_fold'); old=pd.read_csv(R/'m16g_outer_selected_meta.csv').sort_values('outer_fold'); assert list(m.selected_meta_id)==list(old.selected_meta_id)
    frozen=pd.read_csv(R/'m16g_oof_predictions.csv')[['mmsi','pred_m16g_selected_h']]; x=led.merge(frozen,on='mmsi'); assert np.allclose(x.pred_m16g_frozen_h,x.pred_m16g_selected_h,atol=1e-9,rtol=0)
    for k,p in [('m16g_freeze',R/'M16G_MIXTURE_FREEZE.json'),('m17g_freeze',R/'M17G_STRONG_SSL_FREEZE.json'),('m16j_freeze',R/'M16_FINAL_FREEZE.json'),('m14_submission',ROOT/'dist/ais_eta_takehome_submission_final.zip')]: assert f['immutable_inputs'][k]==sha(p),k
    for n,h in f['artifact_sha256'].items(): assert sha(R/n)==h,n
    assert s['gate'] in {'PASS_CONSERVATIVE_MOCO_INTEGRATION','NO_M17H_PROMOTION'}
    print(f"PASS M17H {len(led)} development rows; old-final 53 hard-blocked")
    print("PASS four-expert shape preserved; M16D route/fallback replaced by M17G MoCo only")
    print("PASS meta strategy frozen from M16G per outer fold; no candidate re-selection")
    print(f"PASS MAE {s['m17h_mae_h']:.3f} h vs M16G {s['m16g_mae_h']:.3f} h gain={s['gain_h_vs_m16g']:.3f} h gate={s['gate']}")
    return 0
if __name__=='__main__': raise SystemExit(main())
