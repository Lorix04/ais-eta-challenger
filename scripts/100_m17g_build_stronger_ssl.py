#!/usr/bin/env python3
"""Build M17G stronger self-supervised trajectory encoder benchmark vs frozen M17A."""
from __future__ import annotations

import argparse, hashlib, io, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from ais_eta.m16a import assert_development_only, extended_metrics
from ais_eta.m16d import predict_route_analogue
from ais_eta.m17a import build_causal_pretraining_windows, cosine_distance_matrix
from ais_eta.m17g import (
    M17G_VERSION, M17G_CANDIDATES, M17G_RETRIEVAL_CONFIG, M17G_SEED,
    M17G_PROMOTION_MIN_GAIN_H, M17G_PROMOTION_MIN_FOLD_WINS,
    M17G_PROMOTION_MAX_P90_RATIO, promotion_pass, train_candidate,
)

REF=ROOT/'data/derived/m10_reference_rows.pkl.gz'; STATES=ROOT/'data/derived/m0c_ship_states.pkl.gz'; R=ROOT/'reports'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save_npz_deterministic(path: Path, arrays: dict[str,np.ndarray]):
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as zf:
        for name in sorted(arrays):
            buf=io.BytesIO(); np.save(buf,np.asarray(arrays[name]),allow_pickle=False)
            info=zipfile.ZipInfo(f'{name}.npy',date_time=(1980,1,1,0,0,0)); info.compress_type=zipfile.ZIP_DEFLATED; info.external_attr=0o644<<16
            zf.writestr(info,buf.getvalue(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)

def prepare():
    raw=pd.read_pickle(REF); final=raw.loc[raw.split.eq('final_test')].copy(); dev=raw.loc[raw.split.isin(['train','calibration'])].copy(); assert len(dev)==386 and len(final)==53; assert_development_only(dev,final.mmsi)
    a=pd.read_csv(R/'m16a_oof_predictions.csv'); b=pd.read_csv(R/'m16b_destination_resolutions.csv'); f=pd.read_csv(R/'m17a_oof_predictions.csv')
    dev=dev.merge(a[['mmsi','m16a_outer_fold']],on='mmsi',how='left').merge(b[['mmsi','canonical_destination']],on='mmsi',how='left').merge(f[['mmsi','pred_m17a_ssl_h']],on='mmsi',how='left')
    dev=dev.sort_values('mmsi',kind='mergesort').reset_index(drop=True)
    assert not dev[['m16a_outer_fold','pred_m17a_ssl_h']].isna().any().any(); return dev,final

def worker(outer:int, outdir:Path):
    outdir.mkdir(parents=True,exist_ok=True); dev,final=prepare(); states=pd.read_pickle(STATES); states=states.loc[states.mmsi.isin(dev.mmsi)].copy(); states['recorded_at']=pd.to_datetime(states.recorded_at)
    windows,owners,_=build_causal_pretraining_windows(states,dev)
    seq=np.asarray(np.load(R/'m16d_route_sequences.npz')['geo_kin_6h'],dtype=np.float32)
    y=dev.target_tte_h.to_numpy(float); folds=dev.m16a_outer_fold.to_numpy(int); mmsis=dev.mmsi.to_numpy(int); dest=dev.canonical_destination.fillna('UNKNOWN').astype(str).to_numpy()
    tr_idx=np.flatnonzero(folds!=outer); va_idx=np.flatnonzero(folds==outer); tr_m=set(int(x) for x in mmsis[tr_idx]); va_m=set(int(x) for x in mmsis[va_idx]); cmask=np.asarray([int(x) in tr_m for x in owners],bool)
    assert not any(int(x) in va_m for x in owners[cmask]); tr_windows=windows[cmask]; tr_owners=owners[cmask]
    pred=dev.iloc[va_idx][['mmsi','target_tte_h','m16a_outer_fold']].copy().reset_index(drop=True); pred['pred_m17a_frozen_h']=dev.pred_m17a_ssl_h.to_numpy(float)[va_idx]
    audits=[]; details=[]; embeddings={}
    for ci,cand in enumerate(M17G_CANDIDATES):
        res=train_candidate(cand,tr_windows,seq,seed=M17G_SEED+outer*100+ci)
        dist=cosine_distance_matrix(res.embeddings)
        route=predict_route_analogue(train_indices=tr_idx,valid_indices=va_idx,distance_matrix=dist,targets=y,destinations=dest,config=M17G_RETRIEVAL_CONFIG)
        col=f'pred_m17g_{cand}_h'; pred[col]=route.prediction_h.to_numpy(float)
        z=route.copy(); z['candidate']=cand; z['outer_fold']=outer; z['query_mmsi']=z.row_index.map(dev.mmsi.astype(int).to_dict()); z['nearest_mmsi']=z.nearest_index.map(dev.mmsi.astype(int).to_dict()); details.append(z)
        embeddings[f'fold{outer}_{cand}_embeddings']=res.embeddings.astype(np.float32)
        audits.append({'outer_fold':outer,'candidate':cand,'outer_train_mmsi':len(tr_m),'outer_valid_mmsi':len(va_m),'ssl_pretraining_windows':len(tr_windows),'ssl_pretraining_unique_mmsi':len(set(int(x) for x in tr_owners)),'outer_valid_used_for_ssl_pretraining':False,'loss_start':res.loss_start,'loss_end':res.loss_end,'objective':res.objective})
    pred.to_csv(outdir/f'outer_{outer}_pred.csv',index=False,float_format='%.12g'); pd.DataFrame(audits).to_csv(outdir/f'outer_{outer}_audit.csv',index=False,float_format='%.12g'); pd.concat(details,ignore_index=True).to_csv(outdir/f'outer_{outer}_details.csv',index=False,float_format='%.12g'); save_npz_deterministic(outdir/f'outer_{outer}_embeddings.npz',embeddings)
    parts=[]
    for c in M17G_CANDIDATES:
        mae=float(np.mean(np.abs(pred.target_tte_h.to_numpy(float)-pred[f'pred_m17g_{c}_h'].to_numpy(float))))
        parts.append(f'{c}={mae:.3f}')
    print(f'M17G outer={outer} ' + ' '.join(parts), flush=True)

def run(output_dir:Path,reuse_workers:Path|None=None):
    output_dir.mkdir(parents=True,exist_ok=True); dev,final=prepare(); states=pd.read_pickle(STATES); states=states.loc[states.mmsi.isin(dev.mmsi)].copy(); states['recorded_at']=pd.to_datetime(states.recorded_at)
    windows,owners,audit=build_causal_pretraining_windows(states,dev); audit.to_csv(output_dir/'m17g_pretraining_window_audit.csv',index=False)
    tmp=Path(reuse_workers) if reuse_workers else output_dir/'.m17g_workers'
    if reuse_workers is None:
        if tmp.exists(): shutil.rmtree(tmp)
        tmp.mkdir(); env=dict(os.environ); env.update({'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
        for outer in range(5): subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker-fold',str(outer),'--worker-dir',str(tmp)],cwd=ROOT,env=env,check=True)
    pred=pd.concat([pd.read_csv(tmp/f'outer_{o}_pred.csv') for o in range(5)],ignore_index=True).sort_values('mmsi',kind='mergesort').reset_index(drop=True)
    train_audit=pd.concat([pd.read_csv(tmp/f'outer_{o}_audit.csv') for o in range(5)],ignore_index=True)
    details=pd.concat([pd.read_csv(tmp/f'outer_{o}_details.csv') for o in range(5)],ignore_index=True)
    emb_arrays={}
    for o in range(5):
        z=np.load(tmp/f'outer_{o}_embeddings.npz')
        for k in z.files: emb_arrays[k]=np.asarray(z[k])
    if reuse_workers is None: shutil.rmtree(tmp)
    assert len(pred)==386 and set(pred.mmsi.astype(int)).isdisjoint(set(final.mmsi.astype(int)))
    y=pred.target_tte_h.to_numpy(float); baseline=pred.pred_m17a_frozen_h.to_numpy(float); bm=extended_metrics(y,baseline)
    metrics=[{'model':'m17a_frozen_ssl32','source':'frozen OOF',**bm}]; foldrows=[]; candidate_results=[]
    for cand in M17G_CANDIDATES:
        col=f'pred_m17g_{cand}_h'; mm=extended_metrics(y,pred[col]); metrics.append({'model':f'm17g_{cand}','source':'development OOF',**mm}); wins=0
        for fold,g in pred.groupby('m16a_outer_fold'):
            b=extended_metrics(g.target_tte_h,g.pred_m17a_frozen_h); m=extended_metrics(g.target_tte_h,g[col]); gain=b['mae_h']-m['mae_h']; wins+=int(gain>0); foldrows.append({'outer_fold':int(fold),'candidate':cand,'m17a_mae_h':b['mae_h'],'m17g_mae_h':m['mae_h'],'gain_h':gain,'m17g_wins':gain>0})
        gain=bm['mae_h']-mm['mae_h']; p90r=mm['p90_ae_h']/bm['p90_ae_h']; passed=promotion_pass(gain_h=gain,fold_wins=wins,p90_ratio=p90r)
        candidate_results.append({'candidate':cand,'mae_h':mm['mae_h'],'medae_h':mm['medae_h'],'p90_ae_h':mm['p90_ae_h'],'gain_h_vs_m17a':gain,'fold_wins_vs_m17a':wins,'p90_ratio_vs_m17a':p90r,'promotion_gate_passed':passed})
    cres=pd.DataFrame(candidate_results); passing=cres.loc[cres.promotion_gate_passed].sort_values(['mae_h','p90_ae_h','candidate'])
    if len(passing): selected=str(passing.iloc[0].candidate); gate='PASS_STRONGER_SSL_ENCODER_COMPONENT'
    else: selected=None; gate='NO_M17G_PROMOTION'
    pred['pred_m17g_selected_h']=pred[f'pred_m17g_{selected}_h'] if selected else pred.pred_m17a_frozen_h
    pred.to_csv(output_dir/'m17g_oof_predictions.csv',index=False,float_format='%.12g'); pd.DataFrame(metrics).sort_values(['mae_h','p90_ae_h','model']).to_csv(output_dir/'m17g_model_metrics.csv',index=False,float_format='%.12g'); pd.DataFrame(foldrows).to_csv(output_dir/'m17g_fold_comparison.csv',index=False,float_format='%.12g'); cres.to_csv(output_dir/'m17g_candidate_summary.csv',index=False,float_format='%.12g'); train_audit.to_csv(output_dir/'m17g_ssl_training_audit.csv',index=False,float_format='%.12g'); details.to_csv(output_dir/'m17g_neighbour_details.csv',index=False,float_format='%.12g'); save_npz_deterministic(output_dir/'m17g_fold_embeddings.npz',emb_arrays)
    # plots
    vis=pd.DataFrame(metrics).sort_values('mae_h'); fig,ax=plt.subplots(figsize=(9,5)); ax.bar(range(len(vis)),vis.mae_h); ax.set_xticks(range(len(vis))); ax.set_xticklabels(['M17A frozen' if x=='m17a_frozen_ssl32' else x.replace('m17g_','').replace('_','\n') for x in vis.model]); ax.set_ylabel('OOF MAE (hours)'); ax.set_title('M17G stronger SSL encoders vs frozen M17A'); [ax.text(i,float(v)+1,f'{v:.2f}',ha='center',fontsize=9) for i,v in enumerate(vis.mae_h)]; fig.tight_layout(); fig.savefig(output_dir/'m17g_encoder_comparison.png',dpi=170); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));
    for cand,g in train_audit.groupby('candidate'): ax.plot(g.outer_fold,g.loss_start,marker='o',linestyle='--',label=f'{cand} start'); ax.plot(g.outer_fold,g.loss_end,marker='o',label=f'{cand} end')
    ax.set_xlabel('Outer fold'); ax.set_ylabel('SSL objective'); ax.set_title('M17G self-supervised convergence'); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(output_dir/'m17g_ssl_convergence.png',dpi=170); plt.close(fig)
    summary={'milestone':'M17G','version':M17G_VERSION,'gate':gate,'selected_candidate':selected,'development_rows':386,'blocked_old_final_rows':53,'final_test_used_for_selection':False,'outer_valid_used_for_ssl_pretraining':False,'target_used_for_ssl_pretraining':False,'retrieval_contract':{'representation':M17G_RETRIEVAL_CONFIG.representation,'gate':M17G_RETRIEVAL_CONFIG.gate,'k':M17G_RETRIEVAL_CONFIG.k,'aggregator':'frozen M16D similarity-weighted median'},'m17a_frozen_oof':bm,'candidate_results':candidate_results,'promotion_gate':{'min_mae_gain_h':M17G_PROMOTION_MIN_GAIN_H,'min_fold_wins':M17G_PROMOTION_MIN_FOLD_WINS,'max_p90_ratio':M17G_PROMOTION_MAX_P90_RATIO},'research_basis':['MoCo-AIS (2026) momentum-contrast trajectory embeddings','NaviSight (2026) masked-autoencoder maritime embeddings','Representation Learning for Maritime Vessel Behaviour (2026) masked self-supervised AIS encoders'],'claim':'Development-only representation benchmark; exact same ETA retrieval/aggregation contract as M17A. Any superiority claim still requires a fresh untouched holdout.'}; (output_dir/'M17G_SUMMARY.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    lines=['# M17G — Stronger Self-Supervised AIS Encoder: MoCo / Masked Autoencoder vs frozen M17A','','## Decision','',f'**{gate}**','',f'Frozen M17A MAE: **{bm["mae_h"]:.3f} h**.','']
    for r in candidate_results: lines += [f'- {r["candidate"]}: **{r["mae_h"]:.3f} h MAE**, gain {r["gain_h_vs_m17a"]:.3f} h, fold wins {r["fold_wins_vs_m17a"]}/5, P90 ratio {r["p90_ratio_vs_m17a"]:.4f}, gate={r["promotion_gate_passed"]}.']
    lines += ['','M17G changes only the target-free trajectory encoder/similarity representation. The M17A destination gate, K=9 retrieval and weighted-median ETA aggregation remain fixed. Each encoder is retrained separately inside each outer fold using only causal windows owned by outer-train MMSIs.','', 'Research references: MoCo-AIS (arXiv:2606.17978); NaviSight (github.com/Korunil/navisight); Representation Learning for Maritime Vessel Behaviour (JMSE 2026, 14(5), 507).']
    (output_dir/'M17G_REPORT.md').write_text('\n'.join(lines)+'\n')
    arts=['M17G_REPORT.md','M17G_SUMMARY.json','m17g_oof_predictions.csv','m17g_model_metrics.csv','m17g_fold_comparison.csv','m17g_candidate_summary.csv','m17g_ssl_training_audit.csv','m17g_pretraining_window_audit.csv','m17g_neighbour_details.csv','m17g_fold_embeddings.npz','m17g_encoder_comparison.png','m17g_ssl_convergence.png']
    freeze={'milestone':'M17G','version':M17G_VERSION,'gate':gate,'selected_candidate':selected,'development_population':386,'blocked_old_final_population':53,'final_test_used_for_selection':False,'outer_valid_used_for_ssl_pretraining':False,'immutable_inputs':{'m17a_freeze':sha(R/'M17A_SSL_RETRIEVAL_FREEZE.json'),'m17f_freeze':sha(R/'M17F_LATENT_NOISE_FREEZE.json'),'m16g_freeze':sha(R/'M16G_MIXTURE_FREEZE.json'),'m16j_freeze':sha(R/'M16_FINAL_FREEZE.json'),'m14_submission':sha(ROOT/'dist/ais_eta_takehome_submission_final.zip')},'artifact_sha256':{n:sha(output_dir/n) for n in arts}}; (output_dir/'M17G_STRONG_SSL_FREEZE.json').write_text(json.dumps(freeze,indent=2,sort_keys=True)+'\n')
    return summary

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',type=Path,default=R); ap.add_argument('--worker-fold',type=int); ap.add_argument('--worker-dir',type=Path); ap.add_argument('--reuse-workers',type=Path); args=ap.parse_args()
    if args.worker_fold is not None: worker(args.worker_fold,args.worker_dir); return 0
    print(json.dumps(run(args.output_dir,args.reuse_workers),indent=2,sort_keys=True)); return 0
if __name__=='__main__':
    rc=main(); sys.stdout.flush(); sys.stderr.flush(); os._exit(int(rc or 0))
