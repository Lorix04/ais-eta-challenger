#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'reports'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    s=json.loads((R/'M17B_SUMMARY.json').read_text()); f=json.loads((R/'M17B_HISTORICAL_MEMORY_FREEZE.json').read_text()); led=pd.read_csv(R/'m17b_oof_predictions.csv')
    manifest=json.loads((R/'M16A_DEVELOPMENT_MANIFEST.json').read_text())
    assert len(led)==386 and led.mmsi.nunique()==386
    assert set(led.mmsi.astype(int)).isdisjoint(manifest['blocked_old_final_mmsi'])
    assert s['final_test_used_for_selection'] is False and s['outer_valid_used_for_ssl_pretraining'] is False and s['outer_valid_used_for_memory'] is False
    assert f['immutable_inputs']['m17a_freeze']==sha(R/'M17A_SSL_RETRIEVAL_FREEZE.json')
    assert f['immutable_inputs']['m17a_predictions']==sha(R/'m17a_oof_predictions.csv')
    assert f['immutable_inputs']['m16j_freeze']==sha(R/'M16_FINAL_FREEZE.json')
    for name,h in f['artifact_sha256'].items(): assert sha(R/name)==h, name
    print(f"PASS M17B development-only 386; old-final {len(manifest['blocked_old_final_mmsi'])} hard-blocked")
    print(f"PASS historical memory windows {s['historical_memory_total_windows']} across {s['historical_memory_unique_mmsi']} MMSIs")
    print(f"PASS exact memory gain vs M17A {s['memory_exact_gain_h_vs_m17a']:.3f} h; folds {s['memory_exact_fold_wins_vs_m17a']}/5")
    print(f"PASS HNSW owner recall@K {s['mean_hnsw_owner_recall_at_k']:.4f}; MAE delta {s['hnsw_mae_delta_h_vs_exact']:.4f} h")
    print(f"PASS gate {s['gate']}")
    return 0
if __name__=='__main__': raise SystemExit(main())
