from __future__ import annotations
import hashlib, json, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ais_eta.m16j import PromotionEvidence, promotion_decision


def test_promotion_decision_preserves_heavy_tail_caveat():
    e=PromotionEvidence(7.04,4,5,9.66,4.07,.9218,-2.02,.5103)
    assert promotion_decision(e)=='PROMOTE_CHALLENGER_FOR_NEW_UNTOUCHED_HOLDOUT_ONLY'


def test_failed_core_robustness_is_not_promoted():
    e=PromotionEvidence(-1,2,3,-2,-1,.6,-10,.7)
    assert promotion_decision(e)=='DOCUMENT_NEGATIVE_OR_UNSTABLE_CHALLENGER'


def test_if_m16j_artifacts_exist_freeze_is_dev_only_and_m14_official():
    p=ROOT/'reports/M16_FINAL_FREEZE.json'
    if not p.exists(): return
    f=json.loads(p.read_text()); s=json.loads((ROOT/'reports/M16_FINAL_SUMMARY.json').read_text())
    assert f['official_submission']=='M14' and f['m14_replaced'] is False
    assert f['development_population']==386 and f['blocked_old_final_population']==53
    assert f['old_final_used_for_m16_selection'] is False and f['fresh_untouched_holdout_required'] is True
    assert s['promotion_decision']=='PROMOTE_CHALLENGER_FOR_NEW_UNTOUCHED_HOLDOUT_ONLY'
    assert hashlib.sha256((ROOT/'dist/ais_eta_takehome_submission_final.zip').read_bytes()).hexdigest()==f['immutable_pre_m16_sha256']['m14_submission']['sha256']


def test_company_note_uses_allowed_claim_language():
    p=ROOT/'docs/M16_COMPANY_NOTE_IT.md'
    if not p.exists(): return
    t=p.read_text()
    assert 'nuovo holdout mai visto' in t
    assert 'M14 resta la submission ufficiale congelata' in t
    assert 'prova definitiva' in t
