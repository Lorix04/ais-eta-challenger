#!/usr/bin/env python3
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'reports'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    s=json.loads((R/'M17G_SUMMARY.json').read_text()); f=json.loads((R/'M17G_STRONG_SSL_FREEZE.json').read_text()); led=pd.read_csv(R/'m17g_oof_predictions.csv'); man=json.loads((R/'M16A_DEVELOPMENT_MANIFEST.json').read_text())
    assert len(led)==386 and led.mmsi.nunique()==386 and set(led.mmsi.astype(int)).isdisjoint(man['blocked_old_final_mmsi'])
    assert s['blocked_old_final_rows']==53 and not s['final_test_used_for_selection'] and not s['outer_valid_used_for_ssl_pretraining'] and not s['target_used_for_ssl_pretraining']
    src=pd.read_csv(R/'m17a_oof_predictions.csv')[['mmsi','pred_m17a_ssl_h']]; x=led.merge(src,on='mmsi'); assert np.allclose(x.pred_m17a_frozen_h,x.pred_m17a_ssl_h,atol=1e-9,rtol=0)
    assert s['retrieval_contract']=={'representation':'geo_kin_6h','gate':'destination','k':9,'aggregator':'frozen M16D similarity-weighted median'}
    for k,p in [('m17a_freeze',R/'M17A_SSL_RETRIEVAL_FREEZE.json'),('m17f_freeze',R/'M17F_LATENT_NOISE_FREEZE.json'),('m16g_freeze',R/'M16G_MIXTURE_FREEZE.json'),('m16j_freeze',R/'M16_FINAL_FREEZE.json'),('m14_submission',ROOT/'dist/ais_eta_takehome_submission_final.zip')]: assert f['immutable_inputs'][k]==sha(p),k
    for n,h in f['artifact_sha256'].items(): assert sha(R/n)==h,n
    assert s['gate'] in {'PASS_STRONGER_SSL_ENCODER_COMPONENT','NO_M17G_PROMOTION'}
    print(f"PASS M17G {len(led)} development rows; old-final 53 hard-blocked")
    print("PASS same frozen M17A retrieval/ETA aggregation contract; target-free outer-train-only SSL")
    for r in s['candidate_results']: print(f"PASS {r['candidate']} MAE={r['mae_h']:.3f} h gain={r['gain_h_vs_m17a']:.3f} h folds={r['fold_wins_vs_m17a']}/5 gate={r['promotion_gate_passed']}")
    print(f"PASS gate {s['gate']} selected={s['selected_candidate']}")
    return 0
if __name__=='__main__': raise SystemExit(main())
