#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, zipfile
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'reports'; DIST=ROOT/'dist'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    s=json.loads((R/'M16_FINAL_SUMMARY.json').read_text())
    f=json.loads((R/'M16_FINAL_FREEZE.json').read_text())
    m=json.loads((R/'M16_CONTENT_MANIFEST.json').read_text())
    cmp=pd.read_csv(R/'m16j_challenger_comparison.csv')
    claims=pd.read_csv(R/'m16j_claim_register.csv')
    assert s['promotion_decision']=='PROMOTE_CHALLENGER_FOR_NEW_UNTOUCHED_HOLDOUT_ONLY'
    assert s['official_submission']=='M14' and s['m14_remains_official_submission'] is True
    assert s['development_rows']==386 and s['blocked_old_final_rows']==53 and s['old_final_used_for_m16_selection'] is False
    assert s['m16g_mae_h'] < s['best_single_m16e_mae_h'] < s['baseline_destination_median_mae_h']
    assert s['fold_wins_vs_best_single']>=4 and s['leave_one_fold_positive']==5
    assert s['bootstrap_gain_ci95_h'][0] <= 0 <= s['bootstrap_gain_ci95_h'][1]
    assert 0.78 <= s['m16h_empirical_coverage'] <= 0.86
    assert {'M16D','M16E','M16F','M16G'}.issubset(set(cmp.stage))
    assert 'NOT_ESTABLISHED' in set(claims.status)
    # M16J is a historical freeze. Later milestones legitimately modify the live
    # repository, so verify the M16 content manifest against the frozen M16 ZIP
    # rather than against today's working-tree copies.
    assert sha(R/'M16_CONTENT_MANIFEST.json')==f['content_manifest_sha256']
    z=DIST/'ais_eta_m16_challenger_freeze.zip'; assert sha(z)==f['m16_freeze_zip_sha256']
    with zipfile.ZipFile(z) as zz:
        assert zz.testzip() is None
        names=set(zz.namelist()); assert 'reports/M16_CONTENT_MANIFEST.json' in names
        for item in m['files']:
            assert item['path'] in names, item['path']
            data=zz.read(item['path'])
            assert len(data)==item['size'], item['path']
            assert hashlib.sha256(data).hexdigest()==item['sha256'], item['path']
        assert not any(n.startswith('evidence/') for n in names)
        assert not any(n.startswith('data/raw') for n in names)
    for item in f['immutable_pre_m16_sha256'].values(): assert sha(ROOT/item['path'])==item['sha256'],item['path']
    for name,d in f['prior_freeze_sha256'].items(): assert sha(R/name)==d,name
    state=json.loads((ROOT/'PROJECT_STATE.json').read_text())
    assert state['current_phase'].startswith(('M16J_','M17','M18','M19'))
    assert state['m16_plan']['status']=='M16J_DONE'
    assert state['m16_plan']['m14_submission_must_remain_immutable'] is True
    print('PASS M16J smart challenger final freeze')
    print(f"PASS promotion decision: {s['promotion_decision']}")
    print(f"PASS M16 development-only MAE: {s['m16g_mae_h']:.6f} h")
    print(f"PASS M16 content manifest: {m['file_count']} files")
    print('PASS old final blocked: 53/53')
    print('PASS M14 remains official frozen submission')
    print(f"PASS challenger freeze SHA256: {f['m16_freeze_zip_sha256']}")
    return 0
if __name__=='__main__': raise SystemExit(main())
