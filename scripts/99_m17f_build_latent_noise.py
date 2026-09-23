#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from ais_eta.m16a import assert_development_only,balanced_hash_folds,extended_metrics
from ais_eta.m16c import add_m16c_features,candidate_configs as c_configs,predict_hierarchical_prior
from ais_eta.m16d import candidate_configs as d_configs,predict_route_analogue
from ais_eta.m16e import candidate_configs as e_configs,predict_physics_expert
from ais_eta.m16f import candidate_configs as f_configs,fit_predict
from ais_eta.m16g import M16G_INNER_FOLDS,M16G_INNER_SALT,build_gating_features as build_m16g_features,fit_predict_meta as fit_m16g_meta,score_candidates_inner_cv as score_m16g_candidates
from ais_eta.m17f import M17F_VERSION,M17F_CANDIDATES,build_noise_features,fit_predict_candidate,score_candidates,trimmed_mae,promotion_gate

REF=ROOT/'data/derived/m10_reference_rows.pkl.gz'; R=ROOT/'reports'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare():
    raw=pd.read_pickle(REF); final=raw.loc[raw.split.eq('final_test')].copy(); dev=raw.loc[raw.split.isin(['train','calibration'])].copy(); assert len(dev)==386 and len(final)==53; assert_development_only(dev,final.mmsi)
    b=pd.read_csv(R/'m16b_destination_resolutions.csv'); e=pd.read_csv(R/'m16e_physics_feature_audit.csv'); a=pd.read_csv(R/'m16a_oof_predictions.csv')
    dev=dev.merge(b[['mmsi','canonical_destination','canonical_port_name','canonical_lat','canonical_lon','resolution_method','resolution_confidence','is_resolved_port','is_non_specific','is_route_expression']],on='mmsi',how='left')
    dev=dev.merge(e[['mmsi','physics_distance_gc_nm','physics_bearing_to_port_deg','physics_course_alignment_deg','physics_recent_speed_max_kn','physics_local_sinuosity_180m','physics_local_sinuosity_360m','physics_eligible']],on='mmsi',how='left')
    dev=dev.merge(a[['mmsi','m16a_outer_fold']],on='mmsi',how='left').sort_values('mmsi',kind='mergesort').reset_index(drop=True); dev=add_m16c_features(dev); return dev,final

def config_maps(): return ({c.config_id:c for c in c_configs()},{c.config_id:c for c in d_configs()},{c.config_id:c for c in e_configs()},{c.config_id:c for c in f_configs()})
def selected_configs(outer):
    cm,dm,em,fm=config_maps(); sc=pd.read_csv(R/'m16c_outer_selected_configs.csv'); sd=pd.read_csv(R/'m16d_outer_selected_configs.csv'); se=pd.read_csv(R/'m16e_outer_selected_configs.csv'); sf=pd.read_csv(R/'m16f_outer_selected_configs.csv')
    return cm[str(sc.loc[sc.outer_fold.eq(outer),'selected_config_id'].iloc[0])],dm[str(sd.loc[sd.outer_fold.eq(outer),'selected_config_id'].iloc[0])],em[str(se.loc[se.outer_fold.eq(outer),'selected_config_id'].iloc[0])],fm[str(sf.loc[sf.outer_fold.eq(outer)&sf.family.eq('hist_gb'),'selected_config_id'].iloc[0])]
def quality(valid,prior,route,physics):
    return pd.DataFrame({'m16c_deepest_support':prior.deepest_support.to_numpy(float),'m16c_deepest_weight':prior.deepest_weight.to_numpy(float),'m16c_fallback_count':prior.fallback_count.to_numpy(float),'m16d_neighbour_count':route.neighbour_count.to_numpy(float),'m16d_nearest_distance':route.nearest_distance.to_numpy(float),'m16d_median_neighbour_distance':route.median_neighbour_distance.to_numpy(float),'m16d_similarity_gap':route.similarity_gap.to_numpy(float),'m16d_gate_used':route.gate_used.astype(str).to_numpy(),'physics_eligible':valid.physics_eligible.astype(bool).to_numpy(),'resolution_confidence':pd.to_numeric(valid.resolution_confidence,errors='coerce').to_numpy(float),'physics_distance_gc_nm':pd.to_numeric(valid.physics_distance_gc_nm,errors='coerce').to_numpy(float),'physics_course_alignment_deg':pd.to_numeric(valid.physics_course_alignment_deg,errors='coerce').to_numpy(float),'physics_recent_speed_max_kn':pd.to_numeric(valid.physics_recent_speed_max_kn,errors='coerce').to_numpy(float),'m16e_effective_speed_kn':physics.effective_speed_kn.to_numpy(float),'m16e_distance_factor':physics.distance_factor.to_numpy(float),'m16e_route_distance_proxy_nm':physics.route_distance_proxy_nm.to_numpy(float)})
def frozen(dev):
    g=pd.read_csv(R/'m16g_oof_predictions.csv'); c=pd.read_csv(R/'m16c_oof_predictions.csv'); d=pd.read_csv(R/'m16d_oof_predictions.csv'); e=pd.read_csv(R/'m16e_oof_predictions.csv')
    base=dev[['mmsi']].merge(g[['mmsi','pred_m16c_prior_h','pred_m16d_route_h','pred_m16e_physics_gate_h','pred_m16f_tabular_h','pred_m16g_selected_h']],on='mmsi',how='left')
    q=(c[['mmsi','m16c_deepest_support','m16c_deepest_weight','m16c_fallback_count']].merge(d[['mmsi','m16d_neighbour_count','m16d_nearest_distance','m16d_median_neighbour_distance','m16d_similarity_gap','m16d_gate_used']],on='mmsi').merge(e[['mmsi','physics_eligible','resolution_confidence','physics_distance_gc_nm','physics_course_alignment_deg','physics_recent_speed_max_kn','m16e_effective_speed_kn','m16e_distance_factor','m16e_route_distance_proxy_nm']],on='mmsi'))
    q=dev[['mmsi']].merge(q,on='mmsi',how='left').drop(columns=['mmsi']); b4=np.column_stack([base.pred_m16c_prior_h,base.pred_m16d_route_h,base.pred_m16e_physics_gate_h,base.pred_m16f_tabular_h]).astype(float); return b4,q,base.pred_m16g_selected_h.to_numpy(float)
def crossfit_meta(candidate,base,y,features,fold):
    out=np.full(len(y),np.nan)
    for k in sorted(np.unique(fold)):
        tr=np.flatnonzero(fold!=k); va=np.flatnonzero(fold==k); out[va]=fit_m16g_meta(candidate,base[tr],y[tr],features.iloc[tr],base[va],features.iloc[va]).prediction
    return out

def run_worker(outer,worker_dir):
    worker_dir.mkdir(parents=True,exist_ok=True); dev,final=prepare(); y=dev.target_tte_h.to_numpy(float); mmsis=dev.mmsi.to_numpy(int); folds=dev.m16a_outer_fold.to_numpy(int); dest=dev.canonical_destination.fillna('UNKNOWN').astype(str).to_numpy(); mats=dict(np.load(R/'m16d_route_distance_matrices.npz')); cc,dc,ec,fc=selected_configs(outer)
    tr_outer=np.flatnonzero(folds!=outer); va_outer=np.flatnonzero(folds==outer); inner_map=balanced_hash_folds(mmsis[tr_outer],n_splits=M16G_INNER_FOLDS,salt=f'{M16G_INNER_SALT}:outer={outer}'); inner=np.asarray([inner_map[int(x)] for x in mmsis[tr_outer]],int)
    train_b4=np.full((len(tr_outer),4),np.nan); qparts=[]; pparts=[]
    for k in range(M16G_INNER_FOLDS):
        va_loc=np.flatnonzero(inner==k); tr_loc=np.flatnonzero(inner!=k); vi=tr_outer[va_loc]; ti=tr_outer[tr_loc]; tr=dev.iloc[ti].copy(); va=dev.iloc[vi].copy()
        prior=predict_hierarchical_prior(tr,va,cc); route=predict_route_analogue(train_indices=ti,valid_indices=vi,distance_matrix=mats[dc.representation],targets=y,destinations=dest,config=dc); physics=predict_physics_expert(tr,va,ec); tab=fit_predict(tr,va,fc); rp=route.prediction_h.to_numpy(float); pp=physics.prediction_h.to_numpy(float); pg=np.where(va.physics_eligible.astype(bool).to_numpy()&np.isfinite(pp),pp,rp)
        train_b4[va_loc]=np.column_stack([prior.prediction_h.to_numpy(float),rp,pg,tab]); qparts.append(quality(va,prior,route,physics)); pparts.append(va_loc)
    tq=pd.DataFrame(index=np.arange(len(tr_outer)))
    for pos,part in zip(pparts,qparts):
        for col in part.columns: tq.loc[pos,col]=part.reset_index(drop=True)[col].to_numpy()
    mf=build_m16g_features(train_b4,tq); searchg=score_m16g_candidates(train_b4,y[tr_outer],mf,inner); gcand=str(searchg.iloc[0].candidate_id); train_g=crossfit_meta(gcand,train_b4,y[tr_outer],mf,inner)
    frozen_b4,frozen_q,frozen_g=frozen(dev); vb4=frozen_b4[va_outer]; vq=frozen_q.iloc[va_outer].reset_index(drop=True); vmf=build_m16g_features(vb4,vq); mf,vmf=mf.align(vmf,join='outer',axis=1,fill_value=0.0); rebuilt=fit_m16g_meta(gcand,train_b4,y[tr_outer],mf,vb4,vmf).prediction; expected=json.loads((R/'M16G_SUMMARY.json').read_text())['selected_meta_by_outer_fold'][str(outer)]; diff=float(np.max(np.abs(rebuilt-frozen_g[va_outer]))); assert gcand==expected and diff<=1e-6
    tx=build_noise_features(train_g,train_b4,tq); vx=build_noise_features(frozen_g[va_outer],vb4,vq); tx,vx=tx.align(vx,join='outer',axis=1,fill_value=0.0); search=score_candidates(tx,y[tr_outer],train_g,inner); cand=str(search.loc[search.selected,'candidate_id'].iloc[0]); pred,corr,diag=fit_predict_candidate(cand,tx,y[tr_outer],train_g,vx,frozen_g[va_outer])
    out=dev.iloc[va_outer][['mmsi','target_tte_h','m16a_outer_fold']].copy().reset_index(drop=True); out['pred_m16g_frozen_h']=frozen_g[va_outer]; out['pred_m17f_h']=pred; out['m17f_correction_h']=corr; out['m17f_selected_candidate']=cand
    for c in M17F_CANDIDATES:
        p,_,_=fit_predict_candidate(c,tx,y[tr_outer],train_g,vx,frozen_g[va_outer]); out[f'pred_m17f_{c}_h']=p
    search.insert(0,'outer_fold',outer); search['selected']=search.candidate_id.eq(cand); pd.DataFrame([{'outer_fold':outer,'selected_candidate_id':cand,'m16g_meta_id':gcand,'m16g_expected_meta_id':expected,'m16g_rebuild_max_abs_diff_h':diff,'outer_train_n':len(tr_outer),'outer_valid_n':len(va_outer),'selected_on_inner_only':True,'diagnostic':json.dumps(diag,sort_keys=True)}]).to_csv(worker_dir/f'outer_{outer}_selected.csv',index=False); out.to_csv(worker_dir/f'outer_{outer}_pred.csv',index=False,float_format='%.12g'); search.to_csv(worker_dir/f'outer_{outer}_search.csv',index=False,float_format='%.12g'); print(f'M17F outer={outer} cand={cand} mae={np.mean(np.abs(y[va_outer]-pred)):.3f} base={np.mean(np.abs(y[va_outer]-frozen_g[va_outer])):.3f}',flush=True); return 0

def run(output_dir,reuse_workers=None):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True); dev,final=prepare(); tmp=Path(reuse_workers) if reuse_workers else output_dir/'.m17f_workers'
    if reuse_workers is None:
        if tmp.exists(): shutil.rmtree(tmp)
        tmp.mkdir(); env=dict(os.environ); env.update({'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
        for o in range(5): subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker-fold',str(o),'--worker-dir',str(tmp)],cwd=ROOT,env=env,check=True)
    pred=pd.concat([pd.read_csv(tmp/f'outer_{o}_pred.csv') for o in range(5)],ignore_index=True).sort_values('mmsi',kind='mergesort').reset_index(drop=True); search=pd.concat([pd.read_csv(tmp/f'outer_{o}_search.csv') for o in range(5)],ignore_index=True); sel=pd.concat([pd.read_csv(tmp/f'outer_{o}_selected.csv') for o in range(5)],ignore_index=True)
    if reuse_workers is None: shutil.rmtree(tmp)
    assert len(pred)==386 and set(pred.mmsi.astype(int)).isdisjoint(set(final.mmsi.astype(int)))
    y=pred.target_tte_h.to_numpy(float); g=pred.pred_m16g_frozen_h.to_numpy(float); p=pred.pred_m17f_h.to_numpy(float); bm=extended_metrics(y,g); mm=extended_metrics(y,p); foldrows=[]; wins=0; regs=[]
    for o in range(5):
        m=pred.m16a_outer_fold.astype(int).eq(o).to_numpy(); b=extended_metrics(y[m],g[m]); x=extended_metrics(y[m],p[m]); gain=b['mae_h']-x['mae_h']; wins+=int(gain>0); regs.append(max(0,-gain)); foldrows.append({'outer_fold':o,'m16g_mae_h':b['mae_h'],'m17f_mae_h':x['mae_h'],'gain_h':gain,'m17f_wins':gain>0})
    tb=trimmed_mae(y,g); tm=trimmed_mae(y,p); gate=promotion_gate(bm,mm,fold_wins=wins,max_fold_regression_h=max(regs),trim95_base=tb,trim95_model=tm); gain=bm['mae_h']-mm['mae_h']; p90r=mm['p90_ae_h']/bm['p90_ae_h']
    metrics=[{'model':'m17f_selected','source':'nested OOF',**mm},{'model':'m16g_frozen','source':'frozen OOF',**bm}]+[{'model':f'm17f_{c}','source':'outer OOF candidate',**extended_metrics(y,pred[f'pred_m17f_{c}_h'])} for c in M17F_CANDIDATES]
    pd.DataFrame(metrics).sort_values(['mae_h','p90_ae_h','model']).to_csv(output_dir/'m17f_model_metrics.csv',index=False,float_format='%.12g'); pred.to_csv(output_dir/'m17f_oof_predictions.csv',index=False,float_format='%.12g'); search.to_csv(output_dir/'m17f_inner_search.csv',index=False,float_format='%.12g'); sel.to_csv(output_dir/'m17f_outer_selected_model.csv',index=False); pd.DataFrame(foldrows).to_csv(output_dir/'m17f_fold_comparison.csv',index=False,float_format='%.12g')
    resid=y-g; audit=pred[['mmsi','m16a_outer_fold','target_tte_h','pred_m16g_frozen_h','pred_m17f_h','m17f_correction_h','m17f_selected_candidate']].copy(); audit['m16g_residual_h']=resid; audit['ae_m16g_h']=np.abs(resid); audit['ae_m17f_h']=np.abs(y-p); audit['gain_h']=audit.ae_m16g_h-audit.ae_m17f_h; audit.to_csv(output_dir/'m17f_residual_audit.csv',index=False,float_format='%.12g')
    fig,ax=plt.subplots(figsize=(8,5)); ax.bar(['M16G frozen','M17F latent-noise'],[bm['mae_h'],mm['mae_h']]); ax.set_ylabel('OOF MAE (hours)'); ax.set_title('M17F latent target-noise benchmark'); ax.text(0,bm['mae_h']+1,f"{bm['mae_h']:.2f}",ha='center'); ax.text(1,mm['mae_h']+1,f"{mm['mae_h']:.2f}",ha='center'); fig.tight_layout(); fig.savefig(output_dir/'m17f_noise_model_comparison.png',dpi=170); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5)); ax.scatter(g,resid,s=14,alpha=.65); ax.axhline(0,linewidth=1); ax.set_xlabel('M16G prediction (h)'); ax.set_ylabel('Residual target - prediction (h)'); ax.set_title('Heavy-tail residual structure audited by M17F'); fig.tight_layout(); fig.savefig(output_dir/'m17f_residual_structure.png',dpi=170); plt.close(fig)
    summary={'milestone':'M17F','version':M17F_VERSION,'gate':gate,'development_rows':386,'blocked_old_final_rows':53,'final_test_used_for_selection':False,'target_derived_runtime_features_used':False,'latent_regime_training_uses_target_residuals':True,'m16g_reconstructed_nested':True,'m16g_frozen_mae_h':bm['mae_h'],'m17f_mae_h':mm['mae_h'],'m17f_medae_h':mm['medae_h'],'m17f_p90_ae_h':mm['p90_ae_h'],'gain_h_vs_m16g':gain,'fold_wins_vs_m16g':wins,'p90_ratio_vs_m16g':p90r,'trim95_m16g_mae_h':tb,'trim95_m17f_mae_h':tm,'max_single_fold_regression_h':max(regs),'selected_candidate_by_outer_fold':{str(int(r.outer_fold)):str(r.selected_candidate_id) for r in sel.itertuples()},'promotion_gate':{'min_mae_gain_h':1.0,'min_fold_wins':3,'max_p90_ratio':1.02,'max_trim95_regression_h':0.0,'max_single_fold_regression_h':15.0},'claim':'Development-only latent declared-ETA noise/regime benchmark; runtime gating uses target-free features only.'}; (output_dir/'M17F_SUMMARY.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    report=f'''# M17F — Latent Target-Noise / Declared-ETA Regime Model\n\n## Decision\n\n**{gate}**\n\nM17F models residual regimes only inside training folds. The latent regime labels/posteriors may use training residuals, but runtime gating uses target-free features only. M16G is reconstructed inner-OOF inside each outer-train and remains the frozen baseline on outer validation.\n\n## Result\n\n- M16G frozen MAE: **{bm['mae_h']:.3f} h**\n- M17F MAE: **{mm['mae_h']:.3f} h**\n- Gain: **{gain:.3f} h**\n- M17F MedAE: **{mm['medae_h']:.3f} h**\n- M17F P90: **{mm['p90_ae_h']:.3f} h**\n- Fold wins: **{wins}/5**\n- Trimmed-95% MAE: M16G **{tb:.3f} h**, M17F **{tm:.3f} h**\n- Max fold regression: **{max(regs):.3f} h**\n\nPromotion gate was fixed before the benchmark: >=1 h MAE gain, >=3/5 fold wins, P90 <=1.02x M16G, trimmed-95% MAE no worse, and <=15 h worst-fold regression.\n'''; (output_dir/'M17F_REPORT.md').write_text(report)
    arts=['M17F_REPORT.md','M17F_SUMMARY.json','m17f_model_metrics.csv','m17f_oof_predictions.csv','m17f_inner_search.csv','m17f_outer_selected_model.csv','m17f_fold_comparison.csv','m17f_residual_audit.csv','m17f_noise_model_comparison.png','m17f_residual_structure.png']; freeze={'milestone':'M17F','version':M17F_VERSION,'gate':gate,'development_population':386,'blocked_old_final_population':53,'inner_selection_only':True,'final_test_used_for_selection':False,'target_derived_runtime_features_used':False,'immutable_inputs':{'m16g_freeze':sha(R/'M16G_MIXTURE_FREEZE.json'),'m17e_freeze':sha(R/'M17E_SELECTIVE_ROUTER_FREEZE.json'),'m16j_freeze':sha(R/'M16_FINAL_FREEZE.json'),'m14_submission':sha(ROOT/'dist/ais_eta_takehome_submission_final.zip')},'artifact_sha256':{n:sha(output_dir/n) for n in arts}}; (output_dir/'M17F_LATENT_NOISE_FREEZE.json').write_text(json.dumps(freeze,indent=2,sort_keys=True)+'\n'); return summary

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',type=Path,default=R); ap.add_argument('--worker-fold',type=int); ap.add_argument('--worker-dir',type=Path); ap.add_argument('--reuse-workers',type=Path); args=ap.parse_args()
    if args.worker_fold is not None: return run_worker(args.worker_fold,args.worker_dir)
    print(json.dumps(run(args.output_dir,args.reuse_workers),indent=2,sort_keys=True)); return 0
if __name__=='__main__':
    rc=main(); sys.stdout.flush(); sys.stderr.flush(); os._exit(int(rc or 0))
