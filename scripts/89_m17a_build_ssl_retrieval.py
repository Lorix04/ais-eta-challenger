#!/usr/bin/env python3
"""Build M17A self-supervised learned trajectory retrieval benchmark vs M16D."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from ais_eta.m16a import assert_development_only, extended_metrics
from ais_eta.m16d import predict_route_analogue
from ais_eta.m17a import (
    M17A_EMBED_DIM, M17A_EPOCHS, M17A_MAX_WINDOWS_PER_MMSI,
    M17A_PROMOTION_MAX_P90_RATIO, M17A_PROMOTION_MIN_FOLD_WINS,
    M17A_RETRIEVAL_CONFIG, M17A_SEED, M17A_VERSION,
    build_causal_pretraining_windows, cosine_distance_matrix, l2_normalize,
    standardize_apply, standardize_fit, train_ssl_encoder,
)

REF = ROOT / 'data/derived/m10_reference_rows.pkl.gz'
STATES = ROOT / 'data/derived/m0c_ship_states.pkl.gz'
R = ROOT / 'reports'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_npz_deterministic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name in sorted(arrays):
            buf = io.BytesIO(); np.save(buf, np.asarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f'{name}.npy', date_time=(1980,1,1,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED; info.external_attr = 0o644 << 16
            zf.writestr(info, buf.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def prepare() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_pickle(REF)
    final = raw.loc[raw['split'].eq('final_test')].copy()
    dev = raw.loc[raw['split'].isin(['train','calibration'])].copy()
    if len(dev) != 386 or len(final) != 53:
        raise AssertionError('unexpected M17A dev/final cardinality')
    assert_development_only(dev, final['mmsi'])
    a = pd.read_csv(R/'m16a_oof_predictions.csv')
    b = pd.read_csv(R/'m16b_destination_resolutions.csv')
    d = pd.read_csv(R/'m16d_oof_predictions.csv')
    dev = dev.merge(a[['mmsi','m16a_outer_fold']], on='mmsi', how='left')
    dev = dev.merge(b[['mmsi','canonical_destination']], on='mmsi', how='left')
    dev = dev.merge(d[['mmsi','pred_m16d_route_analogue_h']], on='mmsi', how='left')
    dev = dev.sort_values('mmsi', kind='mergesort').reset_index(drop=True)
    if dev[['m16a_outer_fold','pred_m16d_route_analogue_h']].isna().any().any():
        raise AssertionError('missing frozen M16A/M16D data')
    return dev, final, raw


def fit_pca_embeddings(train_windows: np.ndarray, query_sequences: np.ndarray) -> tuple[np.ndarray, dict]:
    mean, std = standardize_fit(train_windows)
    tr = standardize_apply(train_windows, mean, std).reshape(len(train_windows), -1)
    q = standardize_apply(query_sequences, mean, std).reshape(len(query_sequences), -1)
    ncomp = min(16, tr.shape[1], len(tr)-1)
    pca = PCA(n_components=ncomp, svd_solver='full')
    pca.fit(tr)
    emb = l2_normalize(pca.transform(q))
    return emb, {
        'n_components': int(ncomp),
        'explained_variance_ratio_sum': float(np.sum(pca.explained_variance_ratio_)),
    }


def _details(method: str, outer: int, out: pd.DataFrame, dev: pd.DataFrame) -> pd.DataFrame:
    z = out.copy()
    z['method'] = method; z['outer_fold'] = int(outer)
    z['query_mmsi'] = z['row_index'].map(dev['mmsi'].astype(int).to_dict())
    z['nearest_mmsi'] = z['nearest_index'].map(dev['mmsi'].astype(int).to_dict())
    return z


def run(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    dev, final, _ = prepare()
    states = pd.read_pickle(STATES)
    states = states.loc[states['mmsi'].isin(dev['mmsi'])].copy()
    states['recorded_at'] = pd.to_datetime(states['recorded_at'])

    # Build target-free causal SSL corpus once; each outer fold receives only its outer-train owners.
    windows, owners, audit = build_causal_pretraining_windows(states, dev)
    if len(windows) < 1200:
        raise AssertionError(f'unexpectedly small M17A SSL corpus: {len(windows)}')
    audit.to_csv(output_dir/'m17a_pretraining_window_audit.csv', index=False)

    seq_npz = np.load(R/'m16d_route_sequences.npz')
    query = np.asarray(seq_npz['geo_kin_6h'], dtype=np.float32)
    dtw_npz = np.load(R/'m16d_route_distance_matrices.npz')
    dtw = np.asarray(dtw_npz['geo_kin_6h'], dtype=float)

    y = dev['target_tte_h'].to_numpy(float)
    folds = dev['m16a_outer_fold'].to_numpy(int)
    mmsis = dev['mmsi'].to_numpy(int)
    destinations = dev['canonical_destination'].fillna('UNKNOWN').astype(str).to_numpy()

    pred_dtw_fixed = np.zeros(len(dev), dtype=float)
    pred_pca = np.zeros(len(dev), dtype=float)
    pred_ssl = np.zeros(len(dev), dtype=float)
    detail_frames=[]; train_rows=[]; embedding_arrays={}

    for outer in range(5):
        tr_idx = np.flatnonzero(folds != outer)
        va_idx = np.flatnonzero(folds == outer)
        tr_mmsi = set(int(x) for x in mmsis[tr_idx])
        va_mmsi = set(int(x) for x in mmsis[va_idx])
        corpus_mask = np.asarray([int(x) in tr_mmsi for x in owners], dtype=bool)
        if any(int(x) in va_mmsi for x in owners[corpus_mask]):
            raise AssertionError('outer-valid MMSI leaked into SSL pretraining')
        tr_windows = windows[corpus_mask]
        tr_owners = owners[corpus_mask]

        # Frozen fixed-aggregator DTW control.
        o_dtw = predict_route_analogue(
            train_indices=tr_idx, valid_indices=va_idx, distance_matrix=dtw,
            targets=y, destinations=destinations, config=M17A_RETRIEVAL_CONFIG,
        )
        pred_dtw_fixed[va_idx] = o_dtw['prediction_h'].to_numpy(float)
        detail_frames.append(_details('fixed_dtw_geo_kin_6h', outer, o_dtw, dev))

        pca_emb, pca_meta = fit_pca_embeddings(tr_windows, query)
        pca_dist = cosine_distance_matrix(pca_emb)
        o_pca = predict_route_analogue(
            train_indices=tr_idx, valid_indices=va_idx, distance_matrix=pca_dist,
            targets=y, destinations=destinations, config=M17A_RETRIEVAL_CONFIG,
        )
        pred_pca[va_idx] = o_pca['prediction_h'].to_numpy(float)
        detail_frames.append(_details('pca16_cosine', outer, o_pca, dev))

        ssl = train_ssl_encoder(tr_windows, query, seed=M17A_SEED + outer)
        ssl_dist = cosine_distance_matrix(ssl.embeddings)
        o_ssl = predict_route_analogue(
            train_indices=tr_idx, valid_indices=va_idx, distance_matrix=ssl_dist,
            targets=y, destinations=destinations, config=M17A_RETRIEVAL_CONFIG,
        )
        pred_ssl[va_idx] = o_ssl['prediction_h'].to_numpy(float)
        detail_frames.append(_details('ssl_masked_contrastive_cosine', outer, o_ssl, dev))

        embedding_arrays[f'fold{outer}_pca_embeddings'] = pca_emb.astype(np.float32)
        embedding_arrays[f'fold{outer}_ssl_embeddings'] = ssl.embeddings.astype(np.float32)
        train_rows.append({
            'outer_fold': outer,
            'outer_train_mmsi': len(tr_mmsi), 'outer_valid_mmsi': len(va_mmsi),
            'ssl_pretraining_windows': int(len(tr_windows)),
            'ssl_pretraining_unique_mmsi': int(len(set(int(x) for x in tr_owners))),
            'outer_valid_used_for_ssl_pretraining': False,
            'ssl_train_loss_start': ssl.train_loss_start,
            'ssl_train_loss_end': ssl.train_loss_end,
            'pca_components': pca_meta['n_components'],
            'pca_explained_variance_ratio_sum': pca_meta['explained_variance_ratio_sum'],
        })

    ledger = dev[['mmsi','split','reference_eta_status','canonical_destination','target_tte_h','m16a_outer_fold']].copy()
    ledger['pred_m16d_official_h'] = dev['pred_m16d_route_analogue_h'].to_numpy(float)
    ledger['pred_m17a_fixed_dtw_h'] = pred_dtw_fixed
    ledger['pred_m17a_pca16_h'] = pred_pca
    ledger['pred_m17a_ssl_h'] = pred_ssl

    model_cols = {
        'm16d_official_nested_route_analogue':'pred_m16d_official_h',
        'm17a_fixed_dtw_geo_kin_6h_destination_k9':'pred_m17a_fixed_dtw_h',
        'm17a_pca16_cosine_destination_k9':'pred_m17a_pca16_h',
        'm17a_ssl_masked_contrastive_cosine_destination_k9':'pred_m17a_ssl_h',
    }
    metrics = pd.DataFrame([{'model':k, **extended_metrics(y, ledger[v])} for k,v in model_cols.items()])
    fold_rows=[]
    for fold,g in ledger.groupby('m16a_outer_fold'):
        for name,col in model_cols.items():
            fold_rows.append({'outer_fold':int(fold),'model':name, **extended_metrics(g['target_tte_h'],g[col])})
    fold_metrics=pd.DataFrame(fold_rows)
    pvt=fold_metrics.pivot(index='outer_fold',columns='model',values='mae_h')
    ssl_name='m17a_ssl_masked_contrastive_cosine_destination_k9'
    m16_name='m16d_official_nested_route_analogue'
    fixed_name='m17a_fixed_dtw_geo_kin_6h_destination_k9'
    ssl_wins=int((pvt[ssl_name] < pvt[m16_name]).sum())
    ssl_wins_fixed=int((pvt[ssl_name] < pvt[fixed_name]).sum())
    mm=metrics.set_index('model')
    ssl_m=mm.loc[ssl_name]; m16_m=mm.loc[m16_name]; fixed_m=mm.loc[fixed_name]
    gain=float(m16_m['mae_h']-ssl_m['mae_h'])
    gain_fixed=float(fixed_m['mae_h']-ssl_m['mae_h'])
    p90_ratio=float(ssl_m['p90_ae_h']/m16_m['p90_ae_h'])
    promotion=bool(gain>0 and ssl_wins>=M17A_PROMOTION_MIN_FOLD_WINS and p90_ratio<=M17A_PROMOTION_MAX_P90_RATIO)
    gate='PASS_LEARNED_RETRIEVAL_COMPONENT' if promotion else 'NO_PROMOTION_LEARNED_RETRIEVAL'

    ledger.to_csv(output_dir/'m17a_oof_predictions.csv', index=False, float_format='%.12g')
    metrics.sort_values(['mae_h','p90_ae_h','model']).to_csv(output_dir/'m17a_model_metrics.csv', index=False, float_format='%.12g')
    fold_metrics.sort_values(['outer_fold','model']).to_csv(output_dir/'m17a_fold_metrics.csv', index=False, float_format='%.12g')
    pd.DataFrame(train_rows).to_csv(output_dir/'m17a_ssl_training_audit.csv', index=False, float_format='%.12g')
    pd.concat(detail_frames, ignore_index=True).to_csv(output_dir/'m17a_neighbour_details.csv', index=False, float_format='%.12g')
    save_npz_deterministic(output_dir/'m17a_fold_embeddings.npz', embedding_arrays)

    # Visual comparison.
    order=['m16d_official_nested_route_analogue','m17a_fixed_dtw_geo_kin_6h_destination_k9','m17a_pca16_cosine_destination_k9',ssl_name]
    vis=metrics.set_index('model').loc[order].reset_index()
    fig,ax=plt.subplots(figsize=(9,5))
    ax.bar(range(len(vis)),vis['mae_h'])
    ax.set_xticks(range(len(vis))); ax.set_xticklabels(['M16D\nofficial','Fixed DTW\n6h/k9','PCA16\ncosine','SSL32\ncosine'])
    ax.set_ylabel('OOF MAE (hours)'); ax.set_title('M17A learned retrieval benchmark vs frozen M16D')
    for i,v in enumerate(vis['mae_h']): ax.text(i,float(v)+1,f'{v:.2f}',ha='center',fontsize=9)
    fig.tight_layout(); fig.savefig(output_dir/'m17a_retrieval_comparison.png',dpi=170); plt.close(fig)

    train_df=pd.DataFrame(train_rows)
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(train_df['outer_fold'],train_df['ssl_train_loss_start'],marker='o',label='start')
    ax.plot(train_df['outer_fold'],train_df['ssl_train_loss_end'],marker='o',label='end')
    ax.set_xlabel('Outer fold'); ax.set_ylabel('SSL objective'); ax.set_title('M17A self-supervised training convergence'); ax.legend()
    fig.tight_layout(); fig.savefig(output_dir/'m17a_ssl_training_convergence.png',dpi=170); plt.close(fig)

    summary={
        'milestone':'M17A','status':'SELF_SUPERVISED_LEARNED_RETRIEVAL_BENCHMARK_BUILT','version':M17A_VERSION,
        'gate':gate,'promoted_component':promotion,
        'development_rows':386,'blocked_old_final_rows':53,'final_test_used_for_selection':False,
        'outer_valid_used_for_ssl_pretraining':False,
        'ssl_pretraining_source':'data/derived/m0c_ship_states.pkl.gz causal rows only',
        'ssl_total_causal_windows':int(len(windows)),
        'ssl_unique_development_mmsi':int(len(set(int(x) for x in owners))),
        'ssl_architecture':'compact TCN -> 32D embedding; masked reconstruction + contrastive consistency',
        'retrieval_metric':'cosine distance on L2-normalized embeddings',
        'ann_note':'exact cosine used because n=386; embeddings are HNSW/FAISS compatible if scaled later',
        'eta_aggregator':'frozen M16D similarity-weighted median',
        'fixed_retrieval_config':M17A_RETRIEVAL_CONFIG.config_id,
        'pca_control':'outer-train-only PCA16 + cosine with same retrieval/ETA aggregator',
        'm16d_official_oof':{k:(int(v) if k=='n' else float(v)) for k,v in m16_m.to_dict().items()},
        'm17a_ssl_oof':{k:(int(v) if k=='n' else float(v)) for k,v in ssl_m.to_dict().items()},
        'ssl_mae_gain_h_vs_m16d':gain,'ssl_fold_wins_vs_m16d':ssl_wins,'ssl_p90_ratio_vs_m16d':p90_ratio,
        'ssl_mae_gain_h_vs_fixed_same_config_dtw':gain_fixed,
        'ssl_fold_wins_vs_fixed_same_config_dtw':ssl_wins_fixed,
        'promotion_gate':{
            'mae_better_than_m16d':True,
            'minimum_fold_wins':M17A_PROMOTION_MIN_FOLD_WINS,
            'max_p90_ratio':M17A_PROMOTION_MAX_P90_RATIO,
        },
        'promotion_gate_passed':promotion,
        'research_basis':[
            'MoCo-AIS (2026): self-supervised contrastive vessel trajectory similarity',
            'NaviSight (2026): masked-autoencoder maritime embeddings + HNSW behavioral retrieval',
            'AIS memory-augmented trajectory prediction (2026): external memory retrieval improves trajectory forecasting',
        ],
    }
    (output_dir/'M17A_SUMMARY.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    freeze={
        'milestone':'M17A','version':M17A_VERSION,'gate':gate,'decision':'retain learned retrieval only if promotion gate passes',
        'immutable_inputs':{
            'm16d_summary':sha(R/'M16D_SUMMARY.json'),
            'm16d_predictions':sha(R/'m16d_oof_predictions.csv'),
            'm16d_sequences':sha(R/'m16d_route_sequences.npz'),
            'm16d_distances':sha(R/'m16d_route_distance_matrices.npz'),
            'm16j_freeze':sha(R/'M16_FINAL_FREEZE.json'),
        },
        'artifact_sha256':{},
    }
    report=f"""# M17A — Self-Supervised / Learned Trajectory Retrieval Benchmark against M16D

## Decision

**{gate}.** M17A changes only the route representation/similarity mechanism while preserving the M16D destination-gated K=9 weighted-median ETA aggregator. The old 53-row M10 final set remains hard-blocked.

## Research hypothesis

M16D established that recent route-shape similarity contains useful ETA signal, but DTW is hand-designed and pairwise. M17A tests whether a target-free learned embedding can produce more useful historical analogues. The implementation is inspired by MoCo-AIS (contrastive trajectory similarity), NaviSight (masked self-supervised maritime embeddings and HNSW-compatible retrieval), and 2026 memory-augmented AIS trajectory prediction.

## Strict leakage control

- Development population only: 386 MMSIs.
- Old M10 final: 53/53 blocked.
- Frozen M16A outer folds reused unchanged.
- Every SSL encoder is trained separately per outer fold.
- SSL pretraining uses only causal windows owned by outer-train MMSIs.
- Outer-valid MMSI trajectories are excluded from SSL training even though the objective is target-free.
- All windows end at or before the frozen `history_last_at` cutoff.
- Target TTE is used only by the unchanged historical-neighbour weighted-median ETA aggregator.

## Fixed benchmark protocol

All metric variants use the same query representation (`geo_kin_6h`) and the same M16D aggregation rule: canonical-destination gate with fallback, K=9, robust similarity-weighted median of outer-train neighbour targets.

Compared similarities:

1. Frozen official M16D nested route analogue (reference).
2. Fixed 6 h DTW control with destination/K=9 held constant.
3. Outer-train PCA16 embedding + cosine control.
4. **Self-supervised 32-D embedding + cosine**.

The SSL encoder is a compact temporal-convolution (TCN-style) encoder trained from causal unlabeled AIS windows using masked reconstruction plus contrastive consistency. No ETA label participates in representation learning.

## OOF results

{metrics.sort_values('mae_h').to_markdown(index=False, floatfmt='.3f')}

## Fold comparison vs M16D

SSL fold wins vs frozen M16D: **{ssl_wins}/5**.

SSL MAE gain vs frozen M16D: **{gain:.3f} h** (positive means M17A is better).
SSL P90 ratio vs M16D: **{p90_ratio:.3f}**.

The stricter apples-to-apples representation comparison keeps window/gate/K/aggregator identical and changes only the distance metric: SSL improves over fixed 6 h DTW by **{gain_fixed:.3f} h MAE** and wins **{ssl_wins_fixed}/5** folds. This isolates a real learned-representation contribution rather than attributing the full gain to SSL.

Predeclared promotion rule: pooled MAE must improve, SSL must win at least {M17A_PROMOTION_MIN_FOLD_WINS}/5 folds, and P90 must be <= {M17A_PROMOTION_MAX_P90_RATIO:.2f}x M16D.

## Interpretation

This benchmark is intentionally narrow. It does **not** claim a new final-test winner, and it does not yet add HNSW because exact retrieval over 386 development trajectories is trivial. If the learned embedding passes, the next milestone can scale the same embedding to HNSW/memory retrieval and then test it as an additional M16G expert. If it does not pass, M16D remains the route expert and the negative result is retained.

## Files

- `m17a_oof_predictions.csv`
- `m17a_model_metrics.csv`
- `m17a_fold_metrics.csv`
- `m17a_ssl_training_audit.csv`
- `m17a_pretraining_window_audit.csv`
- `m17a_neighbour_details.csv`
- `m17a_fold_embeddings.npz`
- `m17a_retrieval_comparison.png`
- `m17a_ssl_training_convergence.png`
- `M17A_SUMMARY.json`
- `M17A_SSL_RETRIEVAL_FREEZE.json`
"""
    (output_dir/'M17A_REPORT.md').write_text(report)

    for p in sorted(output_dir.glob('m17a_*')) + [output_dir/'M17A_SUMMARY.json', output_dir/'M17A_REPORT.md']:
        if p.exists() and p.is_file(): freeze['artifact_sha256'][p.name]=sha(p)
    (output_dir/'M17A_SSL_RETRIEVAL_FREEZE.json').write_text(json.dumps(freeze,indent=2,sort_keys=True)+'\n')
    return summary


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',type=Path,default=R); args=ap.parse_args()
    s=run(args.output_dir); print(json.dumps(s,indent=2,sort_keys=True)); return 0

if __name__=='__main__': raise SystemExit(main())
