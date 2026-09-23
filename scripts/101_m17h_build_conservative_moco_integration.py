#!/usr/bin/env python3
"""M17H: conservative M16G integration of M17G MoCo retrieval.

The M16G four-expert panel is kept at four experts.  The old M16D route expert
is replaced one-for-one by M17G MoCo-TCN64 retrieval, and the route fallback
inside the M16E physics gate is replaced by the same MoCo prediction.  The
meta strategy for each outer fold is *not re-selected*: it is frozen to the
candidate selected by M16G before M17G existed.  Base predictions used to fit
that fixed meta strategy are regenerated inner-OOF inside each outer-train.
"""
from __future__ import annotations

import argparse, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from ais_eta.m16a import assert_development_only, balanced_hash_folds, extended_metrics
from ais_eta.m16c import add_m16c_features, candidate_configs as c_configs, predict_hierarchical_prior
from ais_eta.m16e import candidate_configs as e_configs, predict_physics_expert
from ais_eta.m16f import candidate_configs as f_configs, fit_predict
from ais_eta.m16g import (
    M16G_INNER_FOLDS, M16G_INNER_SALT, M16G_META_CANDIDATES,
    build_gating_features, fit_predict_meta,
)
from ais_eta.m17a import build_causal_pretraining_windows, cosine_distance_matrix
from ais_eta.m17g import M17G_RETRIEVAL_CONFIG, M17G_SEED, train_candidate
from ais_eta.m16d import predict_route_analogue
from ais_eta.m17h import (
    M17H_VERSION, M17H_MIN_GAIN_H, M17H_MIN_FOLD_WINS,
    M17H_MIN_CHANGED_ROW_WIN_SHARE, M17H_MAX_P90_RATIO,
    M17H_MAX_WORST_FOLD_REGRESSION_H, promotion_gate,
)

REF=ROOT/'data/derived/m10_reference_rows.pkl.gz'
STATES=ROOT/'data/derived/m0c_ship_states.pkl.gz'
R=ROOT/'reports'


def sha(p:Path)->str: return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mean_abs(y,p)->float: return float(np.mean(np.abs(np.asarray(y,float)-np.asarray(p,float))))


def prepare_development():
    raw=pd.read_pickle(REF); final=raw.loc[raw.split.eq('final_test')].copy(); dev=raw.loc[raw.split.isin(['train','calibration'])].copy()
    if len(dev)!=386 or len(final)!=53: raise AssertionError('unexpected M17H dev/final cardinality')
    assert_development_only(dev,final.mmsi)
    b=pd.read_csv(R/'m16b_destination_resolutions.csv'); e=pd.read_csv(R/'m16e_physics_feature_audit.csv'); a=pd.read_csv(R/'m16a_oof_predictions.csv')
    bcols=['mmsi','canonical_destination','canonical_lat','canonical_lon','resolution_method','resolution_confidence','is_resolved_port','is_non_specific','is_route_expression']
    ecols=['mmsi','physics_distance_gc_nm','physics_bearing_to_port_deg','physics_course_alignment_deg','physics_recent_speed_max_kn','physics_local_sinuosity_180m','physics_local_sinuosity_360m','physics_eligible']
    dev=dev.merge(b[bcols],on='mmsi',how='left').merge(e[ecols],on='mmsi',how='left').merge(a[['mmsi','m16a_outer_fold','pred_train_destination_median_h','pred_catboost_current_snapshot_h']],on='mmsi',how='left')
    dev=dev.sort_values('mmsi',kind='mergesort').reset_index(drop=True); dev=add_m16c_features(dev)
    if dev.m16a_outer_fold.isna().any(): raise AssertionError('M17H missing frozen outer fold')
    return dev,final


def _selected_configs(outer:int):
    cm={c.config_id:c for c in c_configs()}; em={c.config_id:c for c in e_configs()}; fm={c.config_id:c for c in f_configs()}
    sc=pd.read_csv(R/'m16c_outer_selected_configs.csv'); se=pd.read_csv(R/'m16e_outer_selected_configs.csv'); sf=pd.read_csv(R/'m16f_outer_selected_configs.csv')
    cid=str(sc.loc[sc.outer_fold.eq(outer),'selected_config_id'].iloc[0]); eid=str(se.loc[se.outer_fold.eq(outer),'selected_config_id'].iloc[0]); fid=str(sf.loc[sf.outer_fold.eq(outer)&sf.family.eq('hist_gb'),'selected_config_id'].iloc[0])
    return cm[cid],em[eid],fm[fid]


def _quality(valid:pd.DataFrame, prior:pd.DataFrame, route:pd.DataFrame, physics:pd.DataFrame)->pd.DataFrame:
    # Reuse the frozen M16G feature schema: m16d_* names now carry the replacement
    # route expert's target-free MoCo retrieval diagnostics.
    return pd.DataFrame({
        'm16c_deepest_support':prior.deepest_support.to_numpy(float),
        'm16c_deepest_weight':prior.deepest_weight.to_numpy(float),
        'm16c_fallback_count':prior.fallback_count.to_numpy(float),
        'm16d_neighbour_count':route.neighbour_count.to_numpy(float),
        'm16d_nearest_distance':route.nearest_distance.to_numpy(float),
        'm16d_median_neighbour_distance':route.median_neighbour_distance.to_numpy(float),
        'm16d_similarity_gap':route.similarity_gap.to_numpy(float),
        'm16d_gate_used':route.gate_used.astype(str).to_numpy(),
        'physics_eligible':valid.physics_eligible.astype(bool).to_numpy(),
        'resolution_confidence':pd.to_numeric(valid.resolution_confidence,errors='coerce').to_numpy(float),
        'physics_distance_gc_nm':pd.to_numeric(valid.physics_distance_gc_nm,errors='coerce').to_numpy(float),
        'physics_course_alignment_deg':pd.to_numeric(valid.physics_course_alignment_deg,errors='coerce').to_numpy(float),
        'physics_recent_speed_max_kn':pd.to_numeric(valid.physics_recent_speed_max_kn,errors='coerce').to_numpy(float),
        'm16e_effective_speed_kn':physics.effective_speed_kn.to_numpy(float),
        'm16e_distance_factor':physics.distance_factor.to_numpy(float),
        'm16e_route_distance_proxy_nm':physics.route_distance_proxy_nm.to_numpy(float),
    })


def _frozen_outer_tables(dev:pd.DataFrame):
    c=pd.read_csv(R/'m16c_oof_predictions.csv'); e=pd.read_csv(R/'m16e_oof_predictions.csv'); f=pd.read_csv(R/'m16f_oof_predictions.csv'); g=pd.read_csv(R/'m17g_oof_predictions.csv'); m16g=pd.read_csv(R/'m16g_oof_predictions.csv')
    details=pd.read_csv(R/'m17g_neighbour_details.csv'); details=details.loc[details.candidate.eq('moco_tcn64')].copy()
    # exactly one MoCo retrieval audit row per development MMSI
    if details.query_mmsi.nunique()!=386: raise AssertionError('unexpected M17G MoCo neighbour detail coverage')
    qroute=details[['query_mmsi','neighbour_count','nearest_distance','median_neighbour_distance','similarity_gap','gate_used']].rename(columns={'query_mmsi':'mmsi','neighbour_count':'m16d_neighbour_count','nearest_distance':'m16d_nearest_distance','median_neighbour_distance':'m16d_median_neighbour_distance','similarity_gap':'m16d_similarity_gap','gate_used':'m16d_gate_used'})
    base=(dev[['mmsi']]
          .merge(c[['mmsi','pred_m16c_hierarchical_prior_h']],on='mmsi',how='left')
          .merge(g[['mmsi','pred_m17g_moco_tcn64_h']],on='mmsi',how='left')
          .merge(e[['mmsi','pred_m16e_physics_h','physics_eligible']],on='mmsi',how='left')
          .merge(f[['mmsi','pred_m16f_hist_gb_h']],on='mmsi',how='left')
          .merge(m16g[['mmsi','pred_m16g_selected_h']],on='mmsi',how='left'))
    base['pred_m17h_physics_gate_h']=np.where(base.physics_eligible.astype(bool)&np.isfinite(base.pred_m16e_physics_h),base.pred_m16e_physics_h,base.pred_m17g_moco_tcn64_h)
    quality=(dev[['mmsi']]
             .merge(c[['mmsi','m16c_deepest_support','m16c_deepest_weight','m16c_fallback_count']],on='mmsi',how='left')
             .merge(qroute,on='mmsi',how='left')
             .merge(e[['mmsi','physics_eligible','resolution_confidence','physics_distance_gc_nm','physics_course_alignment_deg','physics_recent_speed_max_kn','m16e_effective_speed_kn','m16e_distance_factor','m16e_route_distance_proxy_nm']],on='mmsi',how='left'))
    return base,quality


def run_worker(outer:int,worker_dir:Path)->int:
    worker_dir.mkdir(parents=True,exist_ok=True); dev,final=prepare_development(); assert_development_only(dev,final.mmsi)
    states=pd.read_pickle(STATES); states=states.loc[states.mmsi.isin(dev.mmsi)].copy(); states['recorded_at']=pd.to_datetime(states.recorded_at)
    windows,owners,_=build_causal_pretraining_windows(states,dev)
    seq=np.asarray(np.load(R/'m16d_route_sequences.npz')['geo_kin_6h'],dtype=np.float32)
    y=dev.target_tte_h.to_numpy(float); folds=dev.m16a_outer_fold.to_numpy(int); mmsis=dev.mmsi.to_numpy(int); dest=dev.canonical_destination.fillna('UNKNOWN').astype(str).to_numpy()
    cc,ec,fc=_selected_configs(outer)
    outer_train=np.flatnonzero(folds!=outer); outer_valid=np.flatnonzero(folds==outer); outer_valid_mmsi=set(int(x) for x in mmsis[outer_valid])
    inner_map=balanced_hash_folds(mmsis[outer_train],n_splits=M16G_INNER_FOLDS,salt=f'{M16G_INNER_SALT}:outer={outer}')
    inner_fold=np.asarray([inner_map[int(x)] for x in mmsis[outer_train]],dtype=int)
    train_base=np.full((len(outer_train),4),np.nan,float); qparts=[]; qpos=[]; ssl_audit=[]
    for inner in range(M16G_INNER_FOLDS):
        va_local=np.flatnonzero(inner_fold==inner); tr_local=np.flatnonzero(inner_fold!=inner); va_idx=outer_train[va_local]; tr_idx=outer_train[tr_local]
        tr=dev.iloc[tr_idx].copy(); va=dev.iloc[va_idx].copy(); tr_m=set(int(x) for x in mmsis[tr_idx]); va_m=set(int(x) for x in mmsis[va_idx])
        prior=predict_hierarchical_prior(tr,va,cc); physics=predict_physics_expert(tr,va,ec); tab=fit_predict(tr,va,fc)
        corpus=np.asarray([int(x) in tr_m for x in owners],bool)
        if any(int(x) in va_m or int(x) in outer_valid_mmsi for x in owners[corpus]): raise AssertionError('M17H validation owner leaked into MoCo pretraining')
        tr_windows=windows[corpus]
        seed=M17G_SEED+10000+outer*100+inner
        ssl=train_candidate('moco_tcn64',tr_windows,seq,seed=seed)
        dist=cosine_distance_matrix(ssl.embeddings)
        route=predict_route_analogue(train_indices=tr_idx,valid_indices=va_idx,distance_matrix=dist,targets=y,destinations=dest,config=M17G_RETRIEVAL_CONFIG)
        route_pred=route.prediction_h.to_numpy(float); phys_pred=physics.prediction_h.to_numpy(float); elig=va.physics_eligible.astype(bool).to_numpy(); phys_gate=np.where(elig&np.isfinite(phys_pred),phys_pred,route_pred)
        train_base[va_local,:]=np.column_stack([prior.prediction_h.to_numpy(float),route_pred,phys_gate,tab])
        qparts.append(_quality(va,prior,route,physics)); qpos.append(va_local)
        ssl_audit.append({'outer_fold':outer,'inner_fold':inner,'pretraining_windows':len(tr_windows),'pretraining_owners':len(tr_m),'inner_valid_owners':len(va_m),'inner_valid_used':False,'outer_valid_used':False,'loss_start':ssl.loss_start,'loss_end':ssl.loss_end,'objective':ssl.objective})
    if not np.isfinite(train_base).all(): raise AssertionError('M17H inner OOF expert matrix contains non-finite values')
    tq=pd.DataFrame(index=np.arange(len(outer_train)))
    for pos,part in zip(qpos,qparts):
        part=part.reset_index(drop=True)
        for col in part.columns: tq.loc[pos,col]=part[col].to_numpy()
    train_features=build_gating_features(train_base,tq)

    frozen_base,frozen_quality=_frozen_outer_tables(dev); fb=frozen_base.iloc[outer_valid]
    valid_base=np.column_stack([fb.pred_m16c_hierarchical_prior_h.to_numpy(float),fb.pred_m17g_moco_tcn64_h.to_numpy(float),fb.pred_m17h_physics_gate_h.to_numpy(float),fb.pred_m16f_hist_gb_h.to_numpy(float)])
    valid_quality=frozen_quality.iloc[outer_valid].drop(columns=['mmsi']).reset_index(drop=True); valid_features=build_gating_features(valid_base,valid_quality)
    train_features,valid_features=train_features.align(valid_features,join='outer',axis=1,fill_value=0.0)

    locked=pd.read_csv(R/'m16g_outer_selected_meta.csv'); selected_meta=str(locked.loc[locked.outer_fold.eq(outer),'selected_meta_id'].iloc[0])
    if selected_meta not in M16G_META_CANDIDATES: raise AssertionError('unknown frozen M16G meta strategy')
    selected=fit_predict_meta(selected_meta,train_base,y[outer_train],train_features,valid_base,valid_features)
    pred=dev.iloc[outer_valid][['mmsi','target_tte_h','m16a_outer_fold']].copy().reset_index(drop=True)
    pred['pred_m17h_prior_h']=valid_base[:,0]; pred['pred_m17h_moco_route_h']=valid_base[:,1]; pred['pred_m17h_physics_gate_h']=valid_base[:,2]; pred['pred_m17h_tabular_h']=valid_base[:,3]; pred['pred_m17h_selected_h']=selected.prediction; pred['m17h_locked_meta_id']=selected_meta; pred['m17h_selected_expert']=selected.selected_expert; pred['pred_m16g_frozen_h']=fb.pred_m16g_selected_h.to_numpy(float)
    pred.to_csv(worker_dir/f'outer_{outer}_pred.csv',index=False,float_format='%.12g')
    pd.DataFrame([{'outer_fold':outer,'selected_meta_id':selected_meta,'meta_strategy_frozen_from_m16g':True,'outer_train_n':len(outer_train),'outer_valid_n':len(outer_valid),'inner_folds':M16G_INNER_FOLDS,'m17g_regenerated_inner_oof':True,'meta_candidate_reselected':False}]).to_csv(worker_dir/f'outer_{outer}_meta.csv',index=False)
    pd.DataFrame(ssl_audit).to_csv(worker_dir/f'outer_{outer}_ssl_audit.csv',index=False,float_format='%.12g')
    print(f'M17H outer={outer} meta={selected_meta} mae={mean_abs(y[outer_valid],selected.prediction):.6f} m16g={mean_abs(y[outer_valid],fb.pred_m16g_selected_h):.6f}',flush=True)
    return 0


def run(output_dir:Path,reuse_workers:Path|None=None):
    output_dir.mkdir(parents=True,exist_ok=True); dev,final=prepare_development(); tmp=Path(reuse_workers) if reuse_workers else output_dir/'.m17h_workers'
    if reuse_workers is None:
        if tmp.exists(): shutil.rmtree(tmp)
        tmp.mkdir(); env=dict(os.environ); env.update({'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
        for outer in range(5): subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker-fold',str(outer),'--worker-dir',str(tmp)],cwd=ROOT,env=env,check=True)
    pred=pd.concat([pd.read_csv(tmp/f'outer_{o}_pred.csv') for o in range(5)],ignore_index=True).sort_values('mmsi',kind='mergesort').reset_index(drop=True)
    meta=pd.concat([pd.read_csv(tmp/f'outer_{o}_meta.csv') for o in range(5)],ignore_index=True).sort_values('outer_fold')
    audit=pd.concat([pd.read_csv(tmp/f'outer_{o}_ssl_audit.csv') for o in range(5)],ignore_index=True)
    if reuse_workers is None: shutil.rmtree(tmp)
    if len(pred)!=386 or pred.mmsi.nunique()!=386: raise AssertionError('M17H ledger cardinality')
    if set(pred.mmsi.astype(int)) & set(final.mmsi.astype(int)): raise AssertionError('old-final leakage')
    y=pred.target_tte_h.to_numpy(float); p=pred.pred_m17h_selected_h.to_numpy(float); b=pred.pred_m16g_frozen_h.to_numpy(float); mm=extended_metrics(y,p); bm=extended_metrics(y,b)
    delta=np.abs(y-b)-np.abs(y-p); changed=np.abs(p-b)>1e-12; changed_win=float(np.mean(delta[changed]>0)) if changed.any() else 0.0
    foldrows=[]; wins=0; worst_reg=0.0
    for outer,g in pred.groupby('m16a_outer_fold'):
        ma=extended_metrics(g.target_tte_h,g.pred_m17h_selected_h); ba=extended_metrics(g.target_tte_h,g.pred_m16g_frozen_h); gain=ba['mae_h']-ma['mae_h']; wins+=int(gain>0); worst_reg=max(worst_reg,max(0.0,-gain)); foldrows.append({'outer_fold':int(outer),'m17h_mae_h':ma['mae_h'],'m16g_mae_h':ba['mae_h'],'gain_h':gain,'m17h_wins':gain>0})
    gain=bm['mae_h']-mm['mae_h']; p90r=mm['p90_ae_h']/bm['p90_ae_h']; gate=promotion_gate(gain_h=gain,fold_wins=wins,changed_row_win_share=changed_win,p90_ratio=p90r,worst_fold_regression_h=worst_reg)
    metrics=pd.DataFrame([
        {'model':'m17h_conservative_moco_integration','source':'nested OOF locked M16G meta strategy',**mm},
        {'model':'m16g_frozen','source':'frozen M16G OOF',**bm},
        {'model':'m17g_moco_route_component','source':'frozen M17G OOF',**extended_metrics(y,pred.pred_m17h_moco_route_h)},
        {'model':'m17h_physics_gate_moco_fallback','source':'M16E physics + M17G fallback',**extended_metrics(y,pred.pred_m17h_physics_gate_h)},
    ]).sort_values(['mae_h','p90_ae_h','model'],kind='mergesort')
    paired=pred[['mmsi','m16a_outer_fold','target_tte_h','m17h_locked_meta_id','m17h_selected_expert']].copy(); paired['ae_m17h_h']=np.abs(y-p); paired['ae_m16g_h']=np.abs(y-b); paired['delta_ae_h']=paired.ae_m16g_h-paired.ae_m17h_h; paired['changed']=changed; paired['m17h_wins']=paired.delta_ae_h>0
    pred.to_csv(output_dir/'m17h_oof_predictions.csv',index=False,float_format='%.12g'); metrics.to_csv(output_dir/'m17h_model_metrics.csv',index=False,float_format='%.12g'); pd.DataFrame(foldrows).to_csv(output_dir/'m17h_fold_comparison.csv',index=False,float_format='%.12g'); meta.to_csv(output_dir/'m17h_outer_meta.csv',index=False); audit.to_csv(output_dir/'m17h_ssl_inner_audit.csv',index=False,float_format='%.12g'); paired.to_csv(output_dir/'m17h_paired_audit.csv',index=False,float_format='%.12g')
    # plots
    fig,ax=plt.subplots(figsize=(8,5)); vis=metrics.sort_values('mae_h'); ax.bar(range(len(vis)),vis.mae_h); ax.set_xticks(range(len(vis))); ax.set_xticklabels(vis.model,rotation=18,ha='right'); ax.set_ylabel('OOF MAE (hours)'); ax.set_title('M17H conservative MoCo integration vs frozen M16G'); [ax.text(i,float(v)+.6,f'{v:.2f}',ha='center',fontsize=9) for i,v in enumerate(vis.mae_h)]; fig.tight_layout(); fig.savefig(output_dir/'m17h_integration_comparison.png',dpi=170); plt.close(fig)
    fr=pd.DataFrame(foldrows); fig,ax=plt.subplots(figsize=(8,5)); ax.bar(fr.outer_fold.astype(str),fr.gain_h); ax.axhline(0,linewidth=1); ax.set_xlabel('Outer fold'); ax.set_ylabel('MAE gain vs M16G (h; positive better)'); ax.set_title('M17H fold stability'); fig.tight_layout(); fig.savefig(output_dir/'m17h_fold_gain.png',dpi=170); plt.close(fig)
    summary={'milestone':'M17H','version':M17H_VERSION,'gate':gate,'development_rows':386,'blocked_old_final_rows':53,'final_test_used_for_selection':False,'expert_count':4,'route_replacement':'M16D route -> M17G MoCo-TCN64','physics_fallback_replacement':'M16D route fallback -> M17G MoCo-TCN64 fallback','meta_candidate_reselected':False,'meta_strategy_source':'frozen M16G outer-fold selected candidate; refit only on M17H inner-OOF base matrix','inner_folds':M16G_INNER_FOLDS,'m17g_regenerated_inner_oof':True,'outer_valid_used_for_ssl_pretraining':False,'target_used_for_ssl_pretraining':False,'m16g_mae_h':bm['mae_h'],'m17h_mae_h':mm['mae_h'],'gain_h_vs_m16g':gain,'fold_wins_vs_m16g':wins,'changed_row_win_share':changed_win,'p90_ratio_vs_m16g':p90r,'worst_fold_regression_h':worst_reg,'promotion_thresholds':{'min_gain_h':M17H_MIN_GAIN_H,'min_fold_wins':M17H_MIN_FOLD_WINS,'min_changed_row_win_share':M17H_MIN_CHANGED_ROW_WIN_SHARE,'max_p90_ratio':M17H_MAX_P90_RATIO,'max_worst_fold_regression_h':M17H_MAX_WORST_FOLD_REGRESSION_H},'claim':'Development-only conservative component substitution; M14/M16G freezes remain untouched and old-final stays blocked.'}
    (output_dir/'M17H_SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    report=f'''# M17H — Conservative M16G integration of M17G MoCo retrieval\n\n## Decision\n\n**{gate}**\n\nM17H does not enlarge the ensemble. It preserves the four-expert M16G architecture, replacing the M16D route expert one-for-one with the passing M17G MoCo-TCN64 retrieval component and using the same MoCo route as fallback when M16E physics is not eligible. The meta strategy itself is frozen per outer fold to the candidate selected by M16G before M17G existed; only that fixed strategy is refit on the new inner-OOF four-expert matrix.\n\n## Result\n\n- frozen M16G MAE: **{bm['mae_h']:.3f} h**\n- M17H MAE: **{mm['mae_h']:.3f} h**\n- pooled gain: **{gain:.3f} h**\n- fold wins: **{wins}/5**\n- M16G P90: **{bm['p90_ae_h']:.3f} h**\n- M17H P90: **{mm['p90_ae_h']:.3f} h**\n- changed-row win share: **{changed_win*100:.2f}%**\n- worst fold regression: **{worst_reg:.3f} h**\n\n## Leakage control\n\nThe 53 old-final MMSIs remain blocked. MoCo is retrained inside every inner fold using only causal AIS windows from inner-train owners. Inner-valid and outer-valid owners are excluded from SSL pretraining. No ETA labels are used by the encoder.\n\n## Interpretation\n\nThis is an intentionally conservative substitution test. A gain can be attributed to the stronger route representation more cleanly than in M17D because expert count and meta-strategy family are not expanded or re-selected.\n'''
    (output_dir/'M17H_REPORT.md').write_text(report)
    artifacts=['M17H_REPORT.md','M17H_SUMMARY.json','m17h_oof_predictions.csv','m17h_model_metrics.csv','m17h_fold_comparison.csv','m17h_outer_meta.csv','m17h_ssl_inner_audit.csv','m17h_paired_audit.csv','m17h_integration_comparison.png','m17h_fold_gain.png']
    freeze={'milestone':'M17H','version':M17H_VERSION,'gate':gate,'development_population':386,'blocked_old_final_population':53,'final_test_used_for_selection':False,'meta_candidate_reselected':False,'immutable_inputs':{'m16g_freeze':sha(R/'M16G_MIXTURE_FREEZE.json'),'m17g_freeze':sha(R/'M17G_STRONG_SSL_FREEZE.json'),'m16j_freeze':sha(R/'M16_FINAL_FREEZE.json'),'m14_submission':sha(ROOT/'dist/ais_eta_takehome_submission_final.zip')},'artifact_sha256':{n:sha(output_dir/n) for n in artifacts}}
    (output_dir/'M17H_CONSERVATIVE_INTEGRATION_FREEZE.json').write_text(json.dumps(freeze,indent=2)+'\n')
    return summary


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',type=Path,default=R); ap.add_argument('--worker-fold',type=int); ap.add_argument('--worker-dir',type=Path); ap.add_argument('--reuse-workers',type=Path); args=ap.parse_args()
    if args.worker_fold is not None:
        if args.worker_dir is None: raise SystemExit('--worker-dir required')
        return run_worker(int(args.worker_fold),args.worker_dir)
    print(json.dumps(run(args.output_dir,reuse_workers=args.reuse_workers),indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
