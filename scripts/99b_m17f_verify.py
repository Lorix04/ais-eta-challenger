#!/usr/bin/env python3
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'reports'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 s=json.loads((R/'M17F_SUMMARY.json').read_text()); f=json.loads((R/'M17F_LATENT_NOISE_FREEZE.json').read_text()); led=pd.read_csv(R/'m17f_oof_predictions.csv'); sel=pd.read_csv(R/'m17f_outer_selected_model.csv'); man=json.loads((R/'M16A_DEVELOPMENT_MANIFEST.json').read_text())
 assert len(led)==386 and led.mmsi.nunique()==386 and set(led.mmsi.astype(int)).isdisjoint(man['blocked_old_final_mmsi'])
 assert s['blocked_old_final_rows']==53 and not s['final_test_used_for_selection'] and not s['target_derived_runtime_features_used'] and s['latent_regime_training_uses_target_residuals']
 src=pd.read_csv(R/'m16g_oof_predictions.csv')[['mmsi','pred_m16g_selected_h']]; x=led.merge(src,on='mmsi'); assert np.allclose(x.pred_m16g_frozen_h,x.pred_m16g_selected_h,atol=1e-9,rtol=0)
 assert float(sel.m16g_rebuild_max_abs_diff_h.max())<=1e-6
 for k,p in [('m16g_freeze',R/'M16G_MIXTURE_FREEZE.json'),('m17e_freeze',R/'M17E_SELECTIVE_ROUTER_FREEZE.json'),('m16j_freeze',R/'M16_FINAL_FREEZE.json'),('m14_submission',ROOT/'dist/ais_eta_takehome_submission_final.zip')]: assert f['immutable_inputs'][k]==sha(p)
 for n,h in f['artifact_sha256'].items(): assert sha(R/n)==h,n
 assert s['gate'] in {'PASS_M17F_LATENT_NOISE_MODEL','NO_M17F_PROMOTION'}
 print(f"PASS M17F {len(led)} development rows; old-final 53 hard-blocked")
 print(f"PASS M16G reconstruction max diff {sel.m16g_rebuild_max_abs_diff_h.max():.3g} h; runtime features target-free")
 print(f"PASS MAE {s['m16g_frozen_mae_h']:.3f} -> {s['m17f_mae_h']:.3f} h; gain {s['gain_h_vs_m16g']:.3f} h; folds {s['fold_wins_vs_m16g']}/5")
 print(f"PASS gate {s['gate']}; P90 ratio {s['p90_ratio_vs_m16g']:.4f}; trim95 {s['trim95_m16g_mae_h']:.3f}->{s['trim95_m17f_mae_h']:.3f}")
 return 0
if __name__=='__main__': raise SystemExit(main())
