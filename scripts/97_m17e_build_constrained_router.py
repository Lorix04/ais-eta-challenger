#!/usr/bin/env python3
"""Build M17E constrained selective router: frozen M16G vs M17A only."""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from ais_eta.m16a import assert_development_only, balanced_hash_folds, extended_metrics
from ais_eta.m16c import add_m16c_features, candidate_configs as c_configs, predict_hierarchical_prior
from ais_eta.m16d import candidate_configs as d_configs, predict_route_analogue
from ais_eta.m16e import candidate_configs as e_configs, predict_physics_expert
from ais_eta.m16f import candidate_configs as f_configs, fit_predict
from ais_eta.m16g import (
    M16G_EXPERT_NAMES,M16G_INNER_FOLDS,M16G_INNER_SALT,
    build_gating_features as build_m16g_features,
    fit_predict_meta as fit_m16g_meta,
    score_candidates_inner_cv as score_m16g_candidates,
)
from ais_eta.m17a import M17A_RETRIEVAL_CONFIG,M17A_SEED,build_causal_pretraining_windows,cosine_distance_matrix,train_ssl_encoder
from ais_eta.m17e import (
    M17E_VERSION,M17E_ROUTER_CANDIDATES,build_router_features,fit_predict_router,
    score_router_candidates,promotion_gate,
)

REF=ROOT/'data/derived/m10_reference_rows.pkl.gz'; STATES=ROOT/'data/derived/m0c_ship_states.pkl.gz'; R=ROOT/'reports'

def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def mae(y,p): return float(np.mean(np.abs(np.asarray(y,float)-np.asarray(p,float))))

def prepare_development():
    raw=pd.read_pickle(REF); final=raw.loc[raw.split.eq('final_test')].copy(); dev=raw.loc[raw.split.isin(['train','calibration'])].copy()
    if len(dev)!=386 or len(final)!=53: raise AssertionError('unexpected M17E cardinality')
    assert_development_only(dev,final.mmsi)
    b=pd.read_csv(R/'m16b_destination_resolutions.csv'); e=pd.read_csv(R/'m16e_physics_feature_audit.csv'); a=pd.read_csv(R/'m16a_oof_predictions.csv')
    dev=dev.merge(b[['mmsi','canonical_destination','canonical_port_name','canonical_lat','canonical_lon','resolution_method','resolution_confidence','is_resolved_port','is_non_specific','is_route_expression']],on='mmsi',how='left')
    dev=dev.merge(e[['mmsi','physics_distance_gc_nm','physics_bearing_to_port_deg','physics_course_alignment_deg','physics_recent_speed_max_kn','physics_local_sinuosity_180m','physics_local_sinuosity_360m','physics_eligible']],on='mmsi',how='left')
    dev=dev.merge(a[['mmsi','m16a_outer_fold']],on='mmsi',how='left').sort_values('mmsi',kind='mergesort').reset_index(drop=True)
    dev=add_m16c_features(dev)
    return dev,final

def _config_maps():
    return ({c.config_id:c for c in c_configs()},{c.config_id:c for c in d_configs()},{c.config_id:c for c in e_configs()},{c.config_id:c for c in f_configs()})

def _selected_configs(outer:int):
    cm,dm,em,fm=_config_maps(); sc=pd.read_csv(R/'m16c_outer_selected_configs.csv'); sd=pd.read_csv(R/'m16d_outer_selected_configs.csv'); se=pd.read_csv(R/'m16e_outer_selected_configs.csv'); sf=pd.read_csv(R/'m16f_outer_selected_configs.csv')
    cid=str(sc.loc[sc.outer_fold.eq(outer),'selected_config_id'].iloc[0]); did=str(sd.loc[sd.outer_fold.eq(outer),'selected_config_id'].iloc[0]); eid=str(se.loc[se.outer_fold.eq(outer),'selected_config_id'].iloc[0]); fid=str(sf.loc[sf.outer_fold.eq(outer)&sf.family.eq('hist_gb'),'selected_config_id'].iloc[0])
    return cm[cid],dm[did],em[eid],fm[fid]

def _quality16(valid,prior,route,physics):
    return pd.DataFrame({
      'm16c_deepest_support':prior.deepest_support.to_numpy(float),'m16c_deepest_weight':prior.deepest_weight.to_numpy(float),'m16c_fallback_count':prior.fallback_count.to_numpy(float),
      'm16d_neighbour_count':route.neighbour_count.to_numpy(float),'m16d_nearest_distance':route.nearest_distance.to_numpy(float),'m16d_median_neighbour_distance':route.median_neighbour_distance.to_numpy(float),'m16d_similarity_gap':route.similarity_gap.to_numpy(float),'m16d_gate_used':route.gate_used.astype(str).to_numpy(),
      'physics_eligible':valid.physics_eligible.astype(bool).to_numpy(),'resolution_confidence':pd.to_numeric(valid.resolution_confidence,errors='coerce').to_numpy(float),'physics_distance_gc_nm':pd.to_numeric(valid.physics_distance_gc_nm,errors='coerce').to_numpy(float),'physics_course_alignment_deg':pd.to_numeric(valid.physics_course_alignment_deg,errors='coerce').to_numpy(float),'physics_recent_speed_max_kn':pd.to_numeric(valid.physics_recent_speed_max_kn,errors='coerce').to_numpy(float),
      'm16e_effective_speed_kn':physics.effective_speed_kn.to_numpy(float),'m16e_distance_factor':physics.distance_factor.to_numpy(float),'m16e_route_distance_proxy_nm':physics.route_distance_proxy_nm.to_numpy(float),
    })

def _add_m17a_quality(q,learned):
    out=q.copy(); out['m17a_neighbour_count']=learned.neighbour_count.to_numpy(float); out['m17a_nearest_distance']=learned.nearest_distance.to_numpy(float); out['m17a_median_neighbour_distance']=learned.median_neighbour_distance.to_numpy(float); out['m17a_similarity_gap']=learned.similarity_gap.to_numpy(float); out['m17a_gate_used']=learned.gate_used.astype(str).to_numpy(); return out

def _prepare_ssl(dev):
    states=pd.read_pickle(STATES); states['recorded_at']=pd.to_datetime(states.recorded_at); ds=states.loc[states.mmsi.isin(dev.mmsi)].copy(); windows,owners,_=build_causal_pretraining_windows(ds,dev); seq=np.asarray(np.load(R/'m16d_route_sequences.npz')['geo_kin_6h'],dtype=np.float32); return windows,owners,seq

def _frozen(dev):
    g=pd.read_csv(R/'m16g_oof_predictions.csv'); a=pd.read_csv(R/'m17a_oof_predictions.csv'); nd=pd.read_csv(R/'m17a_neighbour_details.csv'); nd=nd.loc[nd.method.eq('ssl_masked_contrastive_cosine')].copy()
    c=pd.read_csv(R/'m16c_oof_predictions.csv'); d=pd.read_csv(R/'m16d_oof_predictions.csv'); e=pd.read_csv(R/'m16e_oof_predictions.csv')
    base=dev[['mmsi']].merge(g[['mmsi','pred_m16c_prior_h','pred_m16d_route_h','pred_m16e_physics_gate_h','pred_m16f_tabular_h','pred_m16g_selected_h']],on='mmsi',how='left').merge(a[['mmsi','pred_m17a_ssl_h']],on='mmsi',how='left')
    q=(c[['mmsi','m16c_deepest_support','m16c_deepest_weight','m16c_fallback_count']]
       .merge(d[['mmsi','m16d_neighbour_count','m16d_nearest_distance','m16d_median_neighbour_distance','m16d_similarity_gap','m16d_gate_used']],on='mmsi')
       .merge(e[['mmsi','physics_eligible','resolution_confidence','physics_distance_gc_nm','physics_course_alignment_deg','physics_recent_speed_max_kn','m16e_effective_speed_kn','m16e_distance_factor','m16e_route_distance_proxy_nm']],on='mmsi')
       .merge(nd[['query_mmsi','neighbour_count','nearest_distance','median_neighbour_distance','similarity_gap','gate_used']].rename(columns={'query_mmsi':'mmsi','neighbour_count':'m17a_neighbour_count','nearest_distance':'m17a_nearest_distance','median_neighbour_distance':'m17a_median_neighbour_distance','similarity_gap':'m17a_similarity_gap','gate_used':'m17a_gate_used'}),on='mmsi',how='left'))
    q=dev[['mmsi']].merge(q,on='mmsi',how='left').drop(columns=['mmsi'])
    b4=np.column_stack([base.pred_m16c_prior_h,base.pred_m16d_route_h,base.pred_m16e_physics_gate_h,base.pred_m16f_tabular_h]).astype(float)
    return b4,q,base.pred_m16g_selected_h.to_numpy(float),base.pred_m17a_ssl_h.to_numpy(float)

def _crossfit_selected_m16g(candidate,base,y,features,fold):
    out=np.full(len(y),np.nan,float)
    for k in sorted(np.unique(fold)):
        tr=np.flatnonzero(fold!=k); va=np.flatnonzero(fold==k)
        out[va]=fit_m16g_meta(candidate,base[tr],y[tr],features.iloc[tr],base[va],features.iloc[va]).prediction
    if not np.isfinite(out).all(): raise AssertionError('non-finite M16G crossfit')
    return out

def run_worker(outer:int,worker_dir:Path):
    worker_dir.mkdir(parents=True,exist_ok=True); dev,final=prepare_development(); y=dev.target_tte_h.to_numpy(float); mmsis=dev.mmsi.to_numpy(int); folds=dev.m16a_outer_fold.to_numpy(int); dest=dev.canonical_destination.fillna('UNKNOWN').astype(str).to_numpy(); matrices=dict(np.load(R/'m16d_route_distance_matrices.npz')); cc,dc,ec,fc=_selected_configs(outer); windows,owners,seq=_prepare_ssl(dev)
    tr_outer=np.flatnonzero(folds!=outer); va_outer=np.flatnonzero(folds==outer); va_outer_owners=set(map(int,mmsis[va_outer]))
    inner_map=balanced_hash_folds(mmsis[tr_outer],n_splits=M16G_INNER_FOLDS,salt=f'{M16G_INNER_SALT}:outer={outer}'); inner_fold=np.asarray([inner_map[int(x)] for x in mmsis[tr_outer]],int)
    train_base4=np.full((len(tr_outer),4),np.nan,float); train_m17a=np.full(len(tr_outer),np.nan,float); q_parts=[]; pos_parts=[]; ssl_audit=[]
    for inner in range(M16G_INNER_FOLDS):
        va_loc=np.flatnonzero(inner_fold==inner); tr_loc=np.flatnonzero(inner_fold!=inner); va_idx=tr_outer[va_loc]; tr_idx=tr_outer[tr_loc]; tr=dev.iloc[tr_idx].copy(); va=dev.iloc[va_idx].copy(); tr_owners=set(map(int,mmsis[tr_idx])); va_owners=set(map(int,mmsis[va_idx]))
        prior=predict_hierarchical_prior(tr,va,cc); route=predict_route_analogue(train_indices=tr_idx,valid_indices=va_idx,distance_matrix=matrices[dc.representation],targets=y,destinations=dest,config=dc); physics=predict_physics_expert(tr,va,ec); tab=fit_predict(tr,va,fc); route_p=route.prediction_h.to_numpy(float); physics_p=physics.prediction_h.to_numpy(float); pg=np.where(va.physics_eligible.astype(bool).to_numpy()&np.isfinite(physics_p),physics_p,route_p)
        train_base4[va_loc,:]=np.column_stack([prior.prediction_h.to_numpy(float),route_p,pg,tab])
        corpus=np.asarray([int(x) in tr_owners for x in owners],bool)
        if any(int(x) in va_owners or int(x) in va_outer_owners for x in owners[corpus]): raise AssertionError('validation owner leaked into M17E SSL')
        ssl=train_ssl_encoder(windows[corpus],seq,seed=M17A_SEED+100*outer+inner); dist=cosine_distance_matrix(ssl.embeddings); learned=predict_route_analogue(train_indices=tr_idx,valid_indices=va_idx,distance_matrix=dist,targets=y,destinations=dest,config=M17A_RETRIEVAL_CONFIG); train_m17a[va_loc]=learned.prediction_h.to_numpy(float)
        q_parts.append(_add_m17a_quality(_quality16(va,prior,route,physics),learned)); pos_parts.append(va_loc); ssl_audit.append({'outer_fold':outer,'inner_fold':inner,'pretraining_windows':int(corpus.sum()),'pretraining_owners':len(tr_owners),'inner_valid_owners':len(va_owners),'inner_valid_used':False,'outer_valid_used':False,'train_loss_start':float(ssl.train_loss_start),'train_loss_end':float(ssl.train_loss_end)})
    if not np.isfinite(train_base4).all() or not np.isfinite(train_m17a).all(): raise AssertionError('non-finite inner OOF data')
    tq=pd.DataFrame(index=np.arange(len(tr_outer)))
    for pos,part in zip(pos_parts,q_parts):
        for col in part.columns: tq.loc[pos,col]=part.reset_index(drop=True)[col].to_numpy()
    m16gf=build_m16g_features(train_base4,tq); m16g_search=score_m16g_candidates(train_base4,y[tr_outer],m16gf,inner_fold); m16g_cand=str(m16g_search.iloc[0].candidate_id); train_m16g=_crossfit_selected_m16g(m16g_cand,train_base4,y[tr_outer],m16gf,inner_fold)
    frozen_b4,frozen_q,frozen_g,frozen_a=_frozen(dev); valid_b4=frozen_b4[va_outer]; vq=frozen_q.iloc[va_outer].reset_index(drop=True); valid_m16gf=build_m16g_features(valid_b4,vq); m16gf,valid_m16gf=m16gf.align(valid_m16gf,join='outer',axis=1,fill_value=0.0); rebuilt=fit_m16g_meta(m16g_cand,train_base4,y[tr_outer],m16gf,valid_b4,valid_m16gf).prediction; max_rebuild=float(np.max(np.abs(rebuilt-frozen_g[va_outer]))); expected=json.loads((R/'M16G_SUMMARY.json').read_text())['selected_meta_by_outer_fold'][str(outer)]
    if m16g_cand!=expected or max_rebuild>1e-6: raise AssertionError(f'M16G reconstruction mismatch outer={outer}: {m16g_cand} vs {expected}, diff={max_rebuild}')
    train_rf=build_router_features(train_m16g,train_m17a,train_base4,tq); valid_rf=build_router_features(frozen_g[va_outer],frozen_a[va_outer],valid_b4,vq); train_rf,valid_rf=train_rf.align(valid_rf,join='outer',axis=1,fill_value=0.0)
    search=score_router_candidates(train_rf,y[tr_outer],train_m16g,train_m17a,inner_fold); selected=str(search.loc[search.selected,'candidate_id'].iloc[0]); result=fit_predict_router(selected,train_rf,y[tr_outer],train_m16g,train_m17a,valid_rf,frozen_g[va_outer],frozen_a[va_outer])
    pred=dev.iloc[va_outer][['mmsi','target_tte_h','m16a_outer_fold']].copy().reset_index(drop=True); pred['pred_m16g_frozen_h']=frozen_g[va_outer]; pred['pred_m17a_frozen_h']=frozen_a[va_outer]; pred['pred_m17e_router_h']=result.prediction; pred['m17e_switch_to_m17a']=result.switch_to_m17a; pred['m17e_probability_m17a_better']=result.probability_m17a_better; pred['m17e_selected_router_id']=selected
    for cand in M17E_ROUTER_CANDIDATES:
        rr=fit_predict_router(cand,train_rf,y[tr_outer],train_m16g,train_m17a,valid_rf,frozen_g[va_outer],frozen_a[va_outer]); pred[f'pred_m17e_{cand}_h']=rr.prediction
    search.insert(0,'outer_fold',outer); search['selected']=search.candidate_id.eq(selected)
    pd.DataFrame([{'outer_fold':outer,'selected_router_id':selected,'m16g_meta_id':m16g_cand,'m16g_expected_meta_id':expected,'m16g_rebuild_max_abs_diff_h':max_rebuild,'outer_train_n':len(tr_outer),'outer_valid_n':len(va_outer),'selected_on_inner_only':True}]).to_csv(worker_dir/f'outer_{outer}_selected.csv',index=False,float_format='%.12g')
    pred.to_csv(worker_dir/f'outer_{outer}_pred.csv',index=False,float_format='%.12g'); search.to_csv(worker_dir/f'outer_{outer}_search.csv',index=False,float_format='%.12g'); pd.DataFrame(ssl_audit).to_csv(worker_dir/f'outer_{outer}_ssl_audit.csv',index=False,float_format='%.12g')
    print(f'M17E worker outer={outer} router={selected} mae={mae(y[va_outer],result.prediction):.6f} m16g={mae(y[va_outer],frozen_g[va_outer]):.6f} switches={result.switch_to_m17a.mean():.3f}',flush=True); return 0

def _run_workers(tmp):
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir(parents=True); env=dict(os.environ); env.update({'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
    for outer in range(5): subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker-fold',str(outer),'--worker-dir',str(tmp)],cwd=ROOT,env=env,check=True)

def run(output_dir:Path,reuse_workers:Path|None=None):
    output_dir.mkdir(parents=True,exist_ok=True); dev,final=prepare_development(); tmp=reuse_workers or output_dir/'.m17e_workers'
    if reuse_workers is None: _run_workers(tmp)
    pred=pd.concat([pd.read_csv(tmp/f'outer_{o}_pred.csv') for o in range(5)],ignore_index=True).sort_values('mmsi',kind='mergesort').reset_index(drop=True); search=pd.concat([pd.read_csv(tmp/f'outer_{o}_search.csv') for o in range(5)],ignore_index=True); selected=pd.concat([pd.read_csv(tmp/f'outer_{o}_selected.csv') for o in range(5)],ignore_index=True); ssl=pd.concat([pd.read_csv(tmp/f'outer_{o}_ssl_audit.csv') for o in range(5)],ignore_index=True)
    if reuse_workers is None: shutil.rmtree(tmp)
    if len(pred)!=386 or pred.mmsi.nunique()!=386 or set(pred.mmsi.astype(int))&set(final.mmsi.astype(int)): raise AssertionError('M17E development ledger invalid')
    y=pred.target_tte_h.to_numpy(float); g=pred.pred_m16g_frozen_h.to_numpy(float); a=pred.pred_m17a_frozen_h.to_numpy(float); p=pred.pred_m17e_router_h.to_numpy(float); sw=pred.m17e_switch_to_m17a.astype(bool).to_numpy(); bm=extended_metrics(y,g); rm=extended_metrics(y,p); am=extended_metrics(y,a); delta=np.abs(y-g)-np.abs(y-p); changed=sw & (np.abs(a-g)>1e-12); changed_win=float(np.mean(delta[changed]>0)) if changed.any() else 0.0; switch_share=float(sw.mean())
    fold_rows=[]; wins=0; regress=[]
    for o in range(5):
        m=pred.m16a_outer_fold.astype(int).eq(o).to_numpy(); mm=extended_metrics(y[m],p[m]); bb=extended_metrics(y[m],g[m]); gain=float(bb['mae_h']-mm['mae_h']); wins+=int(gain>0); regress.append(max(0.0,-gain)); fold_rows.append({'outer_fold':o,'m17e_mae_h':mm['mae_h'],'m16g_mae_h':bb['mae_h'],'gain_h':gain,'m17e_wins':gain>0,'switch_share':float(sw[m].mean())})
    max_reg=max(regress); gate=promotion_gate(bm,rm,fold_wins=wins,changed_row_win_share=changed_win,switch_share=switch_share,max_fold_regression_h=max_reg); gain=float(bm['mae_h']-rm['mae_h']); p90_ratio=float(rm['p90_ae_h']/bm['p90_ae_h'])
    metrics=[{'model':'m17e_selective_router','source':'nested OOF',**rm},{'model':'m16g_frozen','source':'frozen OOF',**bm},{'model':'m17a_frozen','source':'frozen OOF',**am}]
    for cand in M17E_ROUTER_CANDIDATES: metrics.append({'model':f'm17e_{cand}','source':'outer OOF candidate',**extended_metrics(y,pred[f'pred_m17e_{cand}_h'])})
    pd.DataFrame(metrics).sort_values(['mae_h','p90_ae_h','model'],kind='mergesort').to_csv(output_dir/'m17e_model_metrics.csv',index=False,float_format='%.12g'); pred.to_csv(output_dir/'m17e_oof_predictions.csv',index=False,float_format='%.12g'); search.sort_values(['outer_fold','selected','mae_h'],ascending=[True,False,True]).to_csv(output_dir/'m17e_inner_router_search.csv',index=False,float_format='%.12g'); selected.sort_values('outer_fold').to_csv(output_dir/'m17e_outer_selected_router.csv',index=False,float_format='%.12g'); pd.DataFrame(fold_rows).to_csv(output_dir/'m17e_fold_comparison.csv',index=False,float_format='%.12g'); ssl.sort_values(['outer_fold','inner_fold']).to_csv(output_dir/'m17e_ssl_inner_audit.csv',index=False,float_format='%.12g')
    paired=pred[['mmsi','m16a_outer_fold','target_tte_h','pred_m16g_frozen_h','pred_m17a_frozen_h','pred_m17e_router_h','m17e_switch_to_m17a','m17e_probability_m17a_better','m17e_selected_router_id']].copy(); paired['ae_m16g_h']=np.abs(y-g); paired['ae_m17a_h']=np.abs(y-a); paired['ae_m17e_h']=np.abs(y-p); paired['gain_h']=delta; paired['router_change_wins']=delta>0; paired.to_csv(output_dir/'m17e_paired_router_audit.csv',index=False,float_format='%.12g')
    fig,ax=plt.subplots(figsize=(9,5)); names=['M16G frozen','M17E router','M17A learned']; vals=[bm['mae_h'],rm['mae_h'],am['mae_h']]; ax.bar(names,vals); ax.set_ylabel('OOF MAE (hours)'); ax.set_title('M17E constrained selective router'); [ax.text(i,v+1,f'{v:.2f}',ha='center') for i,v in enumerate(vals)]; fig.tight_layout(); fig.savefig(output_dir/'m17e_router_comparison.png',dpi=170); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,5)); fc=pd.DataFrame(fold_rows); ax.bar(fc.outer_fold.astype(str),fc.gain_h); ax.axhline(0,linewidth=1); ax.set_xlabel('Outer fold'); ax.set_ylabel('MAE gain vs M16G (hours)'); ax.set_title('M17E fold stability (positive = better)'); fig.tight_layout(); fig.savefig(output_dir/'m17e_fold_gain.png',dpi=170); plt.close(fig)
    summary={'milestone':'M17E','version':M17E_VERSION,'gate':gate,'development_rows':386,'blocked_old_final_rows':53,'final_test_used_for_selection':False,'router_actions':['KEEP_M16G','SWITCH_TO_M17A'],'target_derived_router_features_used':False,'outer_folds':5,'inner_folds':M16G_INNER_FOLDS,'m16g_reconstructed_nested':True,'m17a_regenerated_inner_oof':True,'m16g_frozen_mae_h':float(bm['mae_h']),'m17a_frozen_mae_h':float(am['mae_h']),'m17e_mae_h':float(rm['mae_h']),'m17e_medae_h':float(rm['medae_h']),'m17e_p90_ae_h':float(rm['p90_ae_h']),'gain_h_vs_m16g':gain,'fold_wins_vs_m16g':wins,'changed_row_win_share_vs_m16g':changed_win,'switch_share':switch_share,'p90_ratio_vs_m16g':p90_ratio,'max_single_fold_regression_h':float(max_reg),'selected_router_by_outer_fold':{str(int(r.outer_fold)):str(r.selected_router_id) for r in selected.itertuples()},'promotion_gate':{'min_mae_gain_h':1.0,'min_fold_wins':3,'min_changed_row_win_share':0.52,'max_p90_ratio':1.0,'max_switch_share':0.35,'max_single_fold_regression_h':15.0},'claim':'Development-only constrained binary router; no old-final use and no M16G/M17A freeze modification.'}; (output_dir/'M17E_SUMMARY.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    report=f'''# M17E — Constrained M16G ↔ M17A Selective Router\n\n## Decision\n\n**{gate}**\n\nM17E limits the router to exactly two actions: keep frozen M16G or switch to frozen M17A learned retrieval. No additional expert is available to the router. Runtime features are target-free; outer validation is untouched during fitting/selection; M16G is reconstructed from inner-OOF base experts and M17A is re-trained only on inner-train causal AIS trajectories.\n\n## Result\n\n- Frozen M16G MAE: **{bm['mae_h']:.3f} h**\n- Frozen M17A MAE: **{am['mae_h']:.3f} h**\n- M17E selective-router MAE: **{rm['mae_h']:.3f} h**\n- MAE gain vs M16G: **{gain:.3f} h**\n- M17E MedAE: **{rm['medae_h']:.3f} h**\n- M17E P90 AE: **{rm['p90_ae_h']:.3f} h**\n- Fold wins: **{wins}/5**\n- Switch share: **{switch_share*100:.2f}%**\n- Changed-row win share: **{changed_win*100:.2f}%**\n- Max single-fold regression: **{max_reg:.3f} h**\n\nPromotion requires >=1 h pooled gain, >=3/5 fold wins, >=52% wins among changed rows, P90 no worse than M16G, <=35% switch share and <=15 h worst-fold regression.\n'''; (output_dir/'M17E_REPORT.md').write_text(report)
    arts=['M17E_REPORT.md','M17E_SUMMARY.json','m17e_oof_predictions.csv','m17e_inner_router_search.csv','m17e_outer_selected_router.csv','m17e_model_metrics.csv','m17e_fold_comparison.csv','m17e_ssl_inner_audit.csv','m17e_paired_router_audit.csv','m17e_router_comparison.png','m17e_fold_gain.png']; freeze={'milestone':'M17E','version':M17E_VERSION,'gate':gate,'development_population':386,'blocked_old_final_population':53,'selection_population_rule':'M10 train+calibration only; old M10 final hard-blocked','inner_selection_only':True,'final_test_used_for_selection':False,'router_actions':['KEEP_M16G','SWITCH_TO_M17A'],'immutable_inputs':{'m16g_freeze':sha(R/'M16G_MIXTURE_FREEZE.json'),'m17a_freeze':sha(R/'M17A_SSL_RETRIEVAL_FREEZE.json'),'m17d_freeze':sha(R/'M17D_EXTENDED_MOE_FREEZE.json'),'m16j_freeze':sha(R/'M16_FINAL_FREEZE.json'),'m14_submission':sha(ROOT/'dist/ais_eta_takehome_submission_final.zip')},'artifact_sha256':{n:sha(output_dir/n) for n in arts}}; (output_dir/'M17E_SELECTIVE_ROUTER_FREEZE.json').write_text(json.dumps(freeze,indent=2,sort_keys=True)+'\n'); return summary

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',type=Path,default=R); ap.add_argument('--worker-fold',type=int); ap.add_argument('--worker-dir',type=Path); ap.add_argument('--reuse-workers',type=Path); args=ap.parse_args()
    if args.worker_fold is not None:
        if args.worker_dir is None: raise SystemExit('--worker-dir required')
        return run_worker(args.worker_fold,args.worker_dir)
    print(json.dumps(run(args.output_dir,args.reuse_workers),indent=2,sort_keys=True)); return 0
if __name__=='__main__':
    rc=main()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(int(rc or 0))
