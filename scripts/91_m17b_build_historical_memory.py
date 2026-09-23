#!/usr/bin/env python3
"""Build M17B historical-memory/HNSW ETA expert with outer-fold workers."""
from __future__ import annotations
import argparse, hashlib, io, json, sys, zipfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from ais_eta.m16a import assert_development_only, extended_metrics
from ais_eta.m16d import predict_route_analogue
from ais_eta.m17a import M17A_RETRIEVAL_CONFIG, M17A_SEED, build_causal_pretraining_windows, cosine_distance_matrix, train_ssl_encoder
from ais_eta.m17b import CohortHNSWMemory, M17B_HNSW_MAX_MAE_DELTA_H, M17B_HNSW_MIN_OWNER_RECALL, M17B_K_OWNERS, M17B_PRETRAIN_WINDOWS_PER_MMSI, M17B_PROMOTION_MAX_P90_RATIO, M17B_PROMOTION_MIN_FOLD_WINS, M17B_SSL_EPOCHS, M17B_VERSION, build_historical_memory_windows, exact_memory_predict, memory_prediction_from_candidates
REF=ROOT/'data/derived/m10_reference_rows.pkl.gz'; STATES=ROOT/'data/derived/m0c_ship_states.pkl.gz'; R=ROOT/'reports'

def sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def save_npz_deterministic(path:Path, arrays:dict[str,np.ndarray])->None:
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as zf:
        for name in sorted(arrays):
            b=io.BytesIO(); np.save(b,np.asarray(arrays[name]),allow_pickle=False); info=zipfile.ZipInfo(name+'.npy',date_time=(1980,1,1,0,0,0)); info.compress_type=zipfile.ZIP_DEFLATED; info.external_attr=0o644<<16; zf.writestr(info,b.getvalue(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)

def prepare():
    raw=pd.read_pickle(REF); final=raw.loc[raw.split.eq('final_test')].copy(); dev=raw.loc[raw.split.isin(['train','calibration'])].copy(); assert len(dev)==386 and len(final)==53; assert_development_only(dev,final.mmsi)
    a=pd.read_csv(R/'m16a_oof_predictions.csv'); b=pd.read_csv(R/'m16b_destination_resolutions.csv'); aa=pd.read_csv(R/'m17a_oof_predictions.csv')
    dev=dev.merge(a[['mmsi','m16a_outer_fold']],on='mmsi',how='left').merge(b[['mmsi','canonical_destination']],on='mmsi',how='left').merge(aa[['mmsi','pred_m17a_ssl_h']],on='mmsi',how='left').sort_values('mmsi',kind='mergesort').reset_index(drop=True)
    return dev,final

def shared():
    dev,final=prepare(); states=pd.read_pickle(STATES); states=states.loc[states.mmsi.isin(dev.mmsi)].copy(); states.recorded_at=pd.to_datetime(states.recorded_at)
    pre_windows,pre_owners,pre_audit=build_causal_pretraining_windows(states,dev,max_windows_per_mmsi=M17B_PRETRAIN_WINDOWS_PER_MMSI); mem_windows,mem_meta=build_historical_memory_windows(states,dev)
    query=np.asarray(np.load(R/'m16d_route_sequences.npz')['geo_kin_6h'],dtype=np.float32)
    return dev,final,pre_windows,pre_owners,pre_audit,mem_windows,mem_meta,query

def run_fold(outer:int,out:Path)->dict:
    out.mkdir(parents=True,exist_ok=True); dev,final,pre_windows,pre_owners,_,mem_windows,mem_meta,query=shared()
    folds=dev.m16a_outer_fold.to_numpy(int); mmsis=dev.mmsi.to_numpy(int); dest=dev.canonical_destination.fillna('UNKNOWN').astype(str).to_numpy(); age_h=((pd.to_datetime(dev.last_update)-pd.to_datetime(dev.history_last_at)).dt.total_seconds()/3600).to_numpy(float)
    tr_idx=np.flatnonzero(folds!=outer); va_idx=np.flatnonzero(folds==outer); tr_owners=set(int(x) for x in mmsis[tr_idx]); va_owners=set(int(x) for x in mmsis[va_idx])
    pre_mask=np.asarray([int(x) in tr_owners for x in pre_owners],bool); tr_pre=pre_windows[pre_mask]; assert not (set(int(x) for x in pre_owners[pre_mask])&va_owners)
    mem_mask=mem_meta.mmsi.astype(int).isin(tr_owners).to_numpy(); tr_mem_seq=mem_windows[mem_mask]; tr_mem_meta=mem_meta.loc[mem_mask].reset_index(drop=True); assert not (set(tr_mem_meta.mmsi.astype(int))&va_owners)
    # M17B measures memory rather than representation capacity. Use one lightweight
    # outer-train-only encoder and evaluate owner-level and memory retrieval with
    # exactly the same embedding function.
    combined=np.concatenate([query,tr_mem_seq],axis=0); ssl=train_ssl_encoder(tr_pre,combined,seed=M17A_SEED+outer,epochs=M17B_SSL_EPOCHS); query_emb=ssl.embeddings[:len(query)]; mem_emb=ssl.embeddings[len(query):]; q_emb=query_emb[va_idx]
    owner_dist=cosine_distance_matrix(query_emb)
    owner_out=predict_route_analogue(train_indices=tr_idx,valid_indices=va_idx,distance_matrix=owner_dist,targets=dev.target_tte_h.to_numpy(float),destinations=dest,config=M17A_RETRIEVAL_CONFIG)
    owner_pred=dict(zip(owner_out.row_index.astype(int),owner_out.prediction_h.astype(float)))
    ann=CohortHNSWMemory(mem_emb,tr_mem_meta,seed=1702+outer)
    rows=[]; recall=[]; evex=[]; evh=[]
    for local,i in enumerate(va_idx):
        es=exact_memory_predict(memory_embeddings=mem_emb,memory_meta=tr_mem_meta,query_embedding=q_emb[local],query_destination=dest[i],query_age_h=age_h[i],label_mode='owner_target_static')
        ex=exact_memory_predict(memory_embeddings=mem_emb,memory_meta=tr_mem_meta,query_embedding=q_emb[local],query_destination=dest[i],query_age_h=age_h[i],label_mode='snapshot_aligned')
        hr=ann.query(q_emb[local],dest[i],k_owners=M17B_K_OWNERS); hp=memory_prediction_from_candidates(memory_meta=tr_mem_meta,memory_ids=hr['memory_ids'],distances=hr['distances'],query_age_h=age_h[i],label_mode='snapshot_aligned')
        eo=set(int(x) for x in str(ex['neighbour_owner_mmsi']).split(';') if x); ho=set(int(x) for x in hr['owner_mmsi']); rec=len(eo&ho)/max(len(eo),1); recall.append(rec); evex.append(ex['distance_evaluations']); evh.append(hr['distance_evaluations'])
        rows.append({'row_index':int(i),'mmsi':int(mmsis[i]),'outer_fold':int(outer),'canonical_destination':dest[i],'target_tte_h':float(dev.iloc[i].target_tte_h),'pred_m17a_ssl_h':float(dev.iloc[i].pred_m17a_ssl_h),'pred_m17b_owner_same_encoder_h':float(owner_pred[int(i)]),'pred_m17b_memory_static_exact_h':es['prediction_h'],'pred_m17b_memory_snapshot_exact_h':ex['prediction_h'],'pred_m17b_memory_snapshot_hnsw_h':hp['prediction_h'],'exact_gate_used':ex['gate_used'],'hnsw_gate_used':hr['gate_used'],'exact_owner_recall_at_k':rec,'exact_candidate_windows':ex['candidate_windows'],'hnsw_distance_evaluations':hr['distance_evaluations'],'exact_neighbour_owners':ex['neighbour_owner_mmsi'],'hnsw_neighbour_owners':';'.join(str(int(x)) for x in hr['owner_mmsi']),'exact_neighbour_memory_ids':ex['neighbour_memory_ids'],'hnsw_neighbour_memory_ids':';'.join(str(int(x)) for x in hr['memory_ids'])})
    pd.DataFrame(rows).to_csv(out/f'outer_{outer}_pred.csv',index=False,float_format='%.12g')
    pd.DataFrame([{'outer_fold':outer,'outer_train_mmsi':len(tr_owners),'outer_valid_mmsi':len(va_owners),'ssl_pretraining_windows':len(tr_pre),'memory_windows':len(tr_mem_meta),'memory_unique_mmsi':tr_mem_meta.mmsi.nunique(),'outer_valid_used_for_ssl_pretraining':False,'outer_valid_used_for_memory':False,'ssl_loss_start':ssl.train_loss_start,'ssl_loss_end':ssl.train_loss_end,'mean_hnsw_owner_recall_at_k':float(np.mean(recall)),'mean_exact_distance_evaluations':float(np.mean(evex)),'mean_hnsw_distance_evaluations':float(np.mean(evh))}]).to_csv(out/f'outer_{outer}_audit.csv',index=False,float_format='%.12g')
    save_npz_deterministic(out/f'outer_{outer}_embeddings.npz',{'memory_embeddings':mem_emb.astype(np.float32),'query_embeddings_all':query_emb.astype(np.float32)})
    result={'outer_fold':outer,'rows':len(rows),'memory_windows':len(tr_mem_meta),'mean_hnsw_owner_recall_at_k':float(np.mean(recall))}; (out/f'outer_{outer}_summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); return result

def aggregate(worker:Path,output_dir:Path)->dict:
    output_dir.mkdir(parents=True,exist_ok=True); dev,final,_,_,pre_audit,_,mem_meta,_=shared(); pre_audit.to_csv(output_dir/'m17b_ssl_pretraining_window_audit.csv',index=False); mem_meta.to_csv(output_dir/'m17b_memory_window_audit.csv',index=False,float_format='%.12g')
    preds=pd.concat([pd.read_csv(worker/f'outer_{i}_pred.csv') for i in range(5)],ignore_index=True).sort_values('row_index').reset_index(drop=True); audits=pd.concat([pd.read_csv(worker/f'outer_{i}_audit.csv') for i in range(5)],ignore_index=True).sort_values('outer_fold')
    assert len(preds)==386 and preds.mmsi.nunique()==386
    led=dev[['mmsi','split','reference_eta_status','canonical_destination','target_tte_h','m16a_outer_fold']].copy(); pmap=preds.set_index('mmsi');
    for c in ['pred_m17a_ssl_h','pred_m17b_owner_same_encoder_h','pred_m17b_memory_static_exact_h','pred_m17b_memory_snapshot_exact_h','pred_m17b_memory_snapshot_hnsw_h']: led[c]=led.mmsi.map(pmap[c]).astype(float)
    models={'m17a_ssl_owner_exact':'pred_m17a_ssl_h','m17b_owner_same_encoder':'pred_m17b_owner_same_encoder_h','m17b_memory_static_exact':'pred_m17b_memory_static_exact_h','m17b_memory_snapshot_exact':'pred_m17b_memory_snapshot_exact_h','m17b_memory_snapshot_hnsw':'pred_m17b_memory_snapshot_hnsw_h'}; y=led.target_tte_h.to_numpy(float)
    metrics=pd.DataFrame([{'model':n,**extended_metrics(y,led[c])} for n,c in models.items()]); fr=[]
    for fold,g in led.groupby('m16a_outer_fold'):
        for n,c in models.items(): fr.append({'outer_fold':int(fold),'model':n,**extended_metrics(g.target_tte_h,g[c])})
    fm=pd.DataFrame(fr); p=fm.pivot(index='outer_fold',columns='model',values='mae_h'); mm=metrics.set_index('model'); base=mm.loc['m17a_ssl_owner_exact']; own=mm.loc['m17b_owner_same_encoder']; ex=mm.loc['m17b_memory_snapshot_exact']; hn=mm.loc['m17b_memory_snapshot_hnsw']
    gain=float(base.mae_h-ex.mae_h); wins=int((p['m17b_memory_snapshot_exact']<p['m17a_ssl_owner_exact']).sum()); p90_ratio=float(ex.p90_ae_h/base.p90_ae_h)
    same_gain=float(own.mae_h-ex.mae_h); same_wins=int((p['m17b_memory_snapshot_exact']<p['m17b_owner_same_encoder']).sum()); same_p90_ratio=float(ex.p90_ae_h/own.p90_ae_h)
    memory_pass=bool(same_gain>0 and same_wins>=M17B_PROMOTION_MIN_FOLD_WINS and same_p90_ratio<=M17B_PROMOTION_MAX_P90_RATIO and gain>0)
    mean_recall=float(preds.exact_owner_recall_at_k.mean()); hdelta=float(abs(hn.mae_h-ex.mae_h)); heq=bool(mean_recall>=M17B_HNSW_MIN_OWNER_RECALL and hdelta<=M17B_HNSW_MAX_MAE_DELTA_H); gate='PASS_HISTORICAL_MEMORY_HNSW_EXPERT' if memory_pass and heq else ('PASS_HNSW_SCALING_NO_MEMORY_PROMOTION' if heq else 'NO_PROMOTION_M17B')
    led.to_csv(output_dir/'m17b_oof_predictions.csv',index=False,float_format='%.12g'); metrics.sort_values(['mae_h','model']).to_csv(output_dir/'m17b_model_metrics.csv',index=False,float_format='%.12g'); fm.to_csv(output_dir/'m17b_fold_metrics.csv',index=False,float_format='%.12g'); preds.to_csv(output_dir/'m17b_retrieval_audit.csv',index=False,float_format='%.12g'); audits.to_csv(output_dir/'m17b_fold_memory_audit.csv',index=False,float_format='%.12g')
    arrays={};
    for i in range(5):
        z=np.load(worker/f'outer_{i}_embeddings.npz'); arrays[f'fold{i}_memory_embeddings']=z['memory_embeddings']; arrays[f'fold{i}_query_embeddings_all']=z['query_embeddings_all']
    save_npz_deterministic(output_dir/'m17b_fold_memory_embeddings.npz',arrays)
    order=['m17a_ssl_owner_exact','m17b_owner_same_encoder','m17b_memory_static_exact','m17b_memory_snapshot_exact','m17b_memory_snapshot_hnsw']; vis=metrics.set_index('model').loc[order]
    fig,ax=plt.subplots(figsize=(10,5)); ax.bar(range(5),vis.mae_h); ax.set_xticks(range(5)); ax.set_xticklabels(['M17A\nfrozen','M17B owner\nsame encoder','Memory\nstatic','Memory\nsnapshot exact','Memory\nsnapshot HNSW']); ax.set_ylabel('OOF MAE (hours)'); ax.set_title('M17B historical memory / HNSW benchmark'); [ax.text(j,float(v)+1,f'{v:.2f}',ha='center',fontsize=9) for j,v in enumerate(vis.mae_h)]; fig.tight_layout(); fig.savefig(output_dir/'m17b_memory_comparison.png',dpi=170); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5)); ax.bar(audits.outer_fold,audits.mean_hnsw_owner_recall_at_k); ax.set_ylim(0,1.02); ax.axhline(M17B_HNSW_MIN_OWNER_RECALL,linestyle='--'); ax.set_xlabel('Outer fold'); ax.set_ylabel('Owner recall@K'); ax.set_title('M17B HNSW recall vs exact memory retrieval'); fig.tight_layout(); fig.savefig(output_dir/'m17b_hnsw_recall.png',dpi=170); plt.close(fig)
    summary={'milestone':'M17B','status':'HISTORICAL_MEMORY_HNSW_BENCHMARK_BUILT','version':M17B_VERSION,'gate':gate,'promoted_component':bool(memory_pass and heq),'development_rows':386,'blocked_old_final_rows':53,'final_test_used_for_selection':False,'outer_valid_used_for_ssl_pretraining':False,'outer_valid_used_for_memory':False,'historical_memory_total_windows':int(len(mem_meta)),'historical_memory_unique_mmsi':int(mem_meta.mmsi.nunique()),'memory_horizon_h':24,'memory_stride_h':1,'memory_window_h':6,'memory_label':'snapshot-aligned TTE = eta_reference_dt - memory_last_at; converted to original last_update target with known query age','owner_deduplication':True,'k_unique_owners':M17B_K_OWNERS,'hnsw_backend':'deterministic dependency-free HNSW-style hierarchy; cKDTree layer construction + greedy/ef graph search; exact cosine oracle audited','hnsw_M':16,'hnsw_ef_construction':96,'hnsw_ef_search':128,'m17a_ssl_oof':{k:(int(v) if k=='n' else float(v)) for k,v in base.to_dict().items()},'m17b_memory_exact_oof':{k:(int(v) if k=='n' else float(v)) for k,v in ex.to_dict().items()},'m17b_memory_hnsw_oof':{k:(int(v) if k=='n' else float(v)) for k,v in hn.to_dict().items()},'memory_exact_gain_h_vs_m17a':gain,'memory_exact_fold_wins_vs_m17a':wins,'memory_exact_p90_ratio_vs_m17a':p90_ratio,'memory_exact_gain_h_vs_same_encoder_owner':same_gain,'memory_exact_fold_wins_vs_same_encoder_owner':same_wins,'memory_exact_p90_ratio_vs_same_encoder_owner':same_p90_ratio,'m17b_ssl_epochs':M17B_SSL_EPOCHS,'m17b_pretrain_windows_per_mmsi':M17B_PRETRAIN_WINDOWS_PER_MMSI,'mean_hnsw_owner_recall_at_k':mean_recall,'hnsw_mae_delta_h_vs_exact':hdelta,'promotion_gate':{'memory_mae_better_than_m17a':True,'minimum_fold_wins':M17B_PROMOTION_MIN_FOLD_WINS,'max_p90_ratio':M17B_PROMOTION_MAX_P90_RATIO,'minimum_hnsw_owner_recall':M17B_HNSW_MIN_OWNER_RECALL,'max_hnsw_mae_delta_h':M17B_HNSW_MAX_MAE_DELTA_H},'memory_promotion_passed':memory_pass,'hnsw_equivalence_passed':heq}
    (output_dir/'M17B_SUMMARY.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    report=f'''# M17B — Historical Memory / HNSW ETA Expert\n\n## Decision\n\n**{gate}.** M17B is development-only; M16J/M17A remain frozen and the 53 old-final MMSIs stay blocked.\n\n## Method\n\nM17A used one owner-level query trajectory per training MMSI. M17B builds a recent historical memory of causal 6 h snapshots from the last 24 h, at 1 h stride. The same outer-train-only self-supervised encoder maps those snapshots and the validation query into latent space. Canonical-destination cohorts are searched first, with global fallback. Retrieved windows are deduplicated by MMSI before the unchanged K=9 weighted-median aggregation.\n\nMemory targets are phase-aligned: `eta_reference_dt - memory_last_at`; the known query age from `history_last_at` to `last_update` is subtracted after retrieval.\n\n## OOF metrics\n\n{metrics.sort_values('mae_h').to_markdown(index=False,floatfmt='.3f')}\n\nExact historical-memory gain vs frozen M17A: **{gain:.3f} h**; fold wins **{wins}/5**; P90 ratio **{p90_ratio:.3f}**.\n\nSame-encoder isolation: memory vs M17B owner control improves by **{same_gain:.3f} h**, with **{same_wins}/5** fold wins and P90 ratio **{same_p90_ratio:.3f}**.\n\nHNSW-style owner recall@K vs exact cosine: **{mean_recall:.4f}**; MAE delta vs exact memory **{hdelta:.4f} h**.\n\n## ANN note\n\nThe clean environment does not bundle the compiled `hnswlib` extension. M17B therefore uses a deterministic dependency-free HNSW-style reference backend: the standard random hierarchy, local M-neighbour layers built efficiently with cKDTree, greedy upper-layer descent, and ef best-first layer-0 graph search. Exact cosine is retained as the oracle, and owner recall is measured explicitly. A production `hnswlib` backend can replace this search layer without changing the memory, cohort-gating, owner-deduplication or ETA aggregation contract.\n\n## Promotion gates\n\nMemory promotion: beat the same-encoder owner control with at least {M17B_PROMOTION_MIN_FOLD_WINS}/5 fold wins and P90 <= {M17B_PROMOTION_MAX_P90_RATIO:.2f}x M17A. ANN equivalence: owner recall@K >= {M17B_HNSW_MIN_OWNER_RECALL:.2f} and |MAE_HNSW-MAE_exact| <= {M17B_HNSW_MAX_MAE_DELTA_H:.1f} h.\n'''; (output_dir/'M17B_REPORT.md').write_text(report)
    freeze={'milestone':'M17B','version':M17B_VERSION,'gate':gate,'immutable_inputs':{'m17a_freeze':sha(R/'M17A_SSL_RETRIEVAL_FREEZE.json'),'m17a_predictions':sha(R/'m17a_oof_predictions.csv'),'m16j_freeze':sha(R/'M16_FINAL_FREEZE.json')},'artifact_sha256':{}}
    for q in sorted(output_dir.glob('m17b_*'))+[output_dir/'M17B_SUMMARY.json',output_dir/'M17B_REPORT.md']:
        if q.exists() and q.is_file(): freeze['artifact_sha256'][q.name]=sha(q)
    (output_dir/'M17B_HISTORICAL_MEMORY_FREEZE.json').write_text(json.dumps(freeze,indent=2,sort_keys=True)+'\n'); return summary

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--outer-fold',type=int); ap.add_argument('--worker-dir',type=Path); ap.add_argument('--aggregate-from',type=Path); ap.add_argument('--output-dir',type=Path,default=R); args=ap.parse_args()
    if args.outer_fold is not None:
        if args.worker_dir is None:
            raise SystemExit('--worker-dir required')
        print(json.dumps(run_fold(args.outer_fold,args.worker_dir),indent=2,sort_keys=True))
        return 0
    if args.aggregate_from is not None:
        print(json.dumps(aggregate(args.aggregate_from,args.output_dir),indent=2,sort_keys=True))
        return 0
    raise SystemExit('use --outer-fold ... --worker-dir ... or --aggregate-from ...')
if __name__=='__main__': raise SystemExit(main())
