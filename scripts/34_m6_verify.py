#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ais_eta.m6 import file_sha256

R=ROOT/'reports'

def ok(c,msg):
    if not c: raise AssertionError(msg)
    print('PASS',msg)

freeze=json.loads((R/'M6_FREEZE.json').read_text())
opened=json.loads((R/'M6_FINAL_HOLDOUT_OPENED.json').read_text())
pred=pd.read_csv(R/'m6_final_predictions.csv')
splits=pd.read_csv(R/'m2_call_splits.csv')
final=set(splits.loc[splits.m2_split.eq('final_chronological_test'),'session_id'].astype(str))
dev=set(splits.loc[splits.m2_split.eq('development'),'session_id'].astype(str))

ok(opened['opened'] is True,'final holdout opening marker exists')
coverage=pd.read_csv(R/'m6_final_call_coverage.csv')
ok(len(final)==16,'final cohort remains 16 locked calls')
ok(set(coverage.session_id.astype(str))==final,'coverage ledger contains all 16 locked calls')
ok(set(pred.session_id.astype(str))==set(coverage.loc[coverage.scorable_under_frozen_scope.astype(bool),'session_id'].astype(str)),'predictions cover exactly the calls scorable under the frozen scope')
ok(pred.session_id.nunique()==13,'13 of 16 locked calls are scorable under frozen approach scope')
ok(not(set(pred.session_id.astype(str)) & dev),'no development call is mislabeled as final output')
ok(opened['freeze_sha256']==file_sha256(R/'M6_FREEZE.json'),'opening marker points to immutable freeze file')
for rel, expected in freeze['sha256'].items():
    ok(file_sha256(ROOT/rel)==expected,f'frozen artifact unchanged: {rel}')
models=pd.read_csv(R/'m6_final_model_manifest.csv')
ok(set(models.port)=={'CATANIA','AUGUSTA'},'one frozen route model per seen port')
ok(models.training_calls.sum()==67,'final route models train on 67 development calls only')
for c in ['pred_m2_geodesic_h','pred_m2_route_knn_h','pred_raw_h','pred_stabilized_h','reliability_tier']:
    ok(c in pred.columns,f'final output contains {c}')
ok(pred.loc[pred.reliability_tier.ne('HIGH'),'m6_pi_covered'].isna().all(),'interval withheld outside HIGH reliability')
ok(pred.loc[pred.reliability_tier.eq('HIGH'),'m6_pi_covered'].notna().all(),'interval emitted for every HIGH reliability row')
summary=json.loads((R/'m6_final_summary.json').read_text())
ok(summary['locked_final_calls_scored']==13,'raw final summary records 13 scorable calls')
ok(summary['no_post_holdout_tuning'] is True,'summary records no post-holdout tuning')
print('M6 verification complete')
