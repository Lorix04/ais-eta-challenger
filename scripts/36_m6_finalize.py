#!/usr/bin/env python3
"""Assemble M6 final reporting after the one-time locked holdout opening.

This script is reporting-only. It does not refit, rescore, tune or alter the frozen predictor.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ais_eta.m5 import paired_bootstrap_mean_gain  # noqa: E402

R=ROOT/'reports'


def _panel_coverage(final_ids:set[str]) -> pd.DataFrame:
    frames=[]
    cat=pd.read_csv(R/'catania_m1_decision_panel.csv'); cat['port']='CATANIA'
    aug=pd.read_csv(R/'augusta_m1x_decision_panel.csv'); aug['port']='AUGUSTA'
    for d in (cat,aug):
        d=d[d.session_id.astype(str).isin(final_ids)].copy()
        if d.empty: continue
        for sid,g in d.groupby('session_id'):
            frames.append({
                'session_id':str(sid),'port':str(g.port.iloc[0]),'mmsi':int(g.mmsi.iloc[0]),'name':str(g.name.iloc[0]),
                'ground_truth_time':str(g.ground_truth_time.iloc[0]),'decision_rows_total':int(len(g)),
                'frozen_scope_inbound_approach_rows':int(g.scope_inbound_approach.astype(bool).sum()),
                'any_input_quality_ok':bool(g.input_quality_ok.astype(bool).any()) if 'input_quality_ok' in g else None,
                'outside_gate_fraction':float(g['outside_research_gate'].astype(bool).mean()) if 'outside_research_gate' in g else float(g['outside_target_research_gate'].astype(bool).mean()),
            })
    return pd.DataFrame(frames)


def _reported_eta_alignment(pred:pd.DataFrame) -> dict:
    configured=os.environ.get('AIS_ETA_DATA_ZIP')
    data_zip=Path(configured) if configured else ROOT/'data.zip'
    if not data_zip.exists(): return {'status':'DATA_ZIP_NOT_AVAILABLE'}
    with zipfile.ZipFile(data_zip) as z:
        t=pd.read_csv(z.open('data/vessel_tracks.csv'),usecols=['mmsi','eta','last_update'])
    t['last_update']=pd.to_datetime(t['last_update'],errors='coerce')
    x=pred.copy(); x['source_observation_time']=pd.to_datetime(x['source_observation_time'],errors='raise')
    x=x.merge(t.dropna(subset=['eta','last_update']),on='mmsi',how='left')
    x['offset_min']=(x['source_observation_time']-x['last_update']).abs().dt.total_seconds()/60
    nearest=x.sort_values('offset_min').head(25)
    nearest.to_csv(R/'m6_reported_eta_nearest_snapshots.csv',index=False)
    return {
        'status':'NO_TEMPORALLY_ALIGNED_REPORTED_ETA' if int((x.offset_min<=5).sum())==0 else 'ALIGNED_ROWS_AVAILABLE',
        'aligned_rows_5min':int((x.offset_min<=5).sum()),
        'rows_within_60min':int((x.offset_min<=60).sum()),
        'closest_snapshot_offset_min':float(x.offset_min.min()) if x.offset_min.notna().any() else None,
    }


def main():
    freeze=json.loads((R/'M6_FREEZE.json').read_text())
    summary=json.loads((R/'m6_final_summary.json').read_text())
    pred=pd.read_csv(R/'m6_final_predictions.csv')
    splits=pd.read_csv(R/'m2_call_splits.csv')
    final=splits[splits.m2_split.eq('final_chronological_test')].copy()
    final_ids=set(final.session_id.astype(str))
    scored=set(pred.session_id.astype(str))

    cov=_panel_coverage(final_ids)
    cov['scorable_under_frozen_scope']=cov.session_id.isin(scored)
    cov['final_status']=np.where(cov.scorable_under_frozen_scope,'SCORED','UNSCORABLE_FROZEN_SCOPE')
    cov['reason']=np.where(cov.scorable_under_frozen_scope,'','NO_CAUSAL_INBOUND_APPROACH_ROWS_UNDER_PREDECLARED_SCOPE')
    cov.to_csv(R/'m6_final_call_coverage.csv',index=False)

    # Overall call-level paired final gain, reporting only.
    q=pred[pred.true_tta_h.astype(float).le(24)].copy()
    q['geo_ae']=(q.pred_m2_geodesic_h-q.true_tta_h).abs(); q['route_ae']=(q.pred_m2_route_knn_h-q.true_tta_h).abs()
    call=q.groupby('session_id',as_index=False).agg(port=('port','first'),mmsi=('mmsi','first'),geo_mae_h=('geo_ae','mean'),route_mae_h=('route_ae','mean'))
    call['paired_gain_h']=call.geo_mae_h-call.route_mae_h
    mean_gain,lo,hi=paired_bootstrap_mean_gain(call.paired_gain_h,draws=20000,seed=20260918)
    overall_boot=pd.DataFrame([{'calls':len(call),'mean_paired_gain_h':mean_gain,'ci95_low_h':lo,'ci95_high_h':hi,'ci_includes_zero':bool(lo<=0<=hi)}])
    overall_boot.to_csv(R/'m6_final_overall_bootstrap.csv',index=False)
    call.to_csv(R/'m6_final_call_errors.csv',index=False)

    reported=_reported_eta_alignment(pred)
    (R/'m6_reported_eta_alignment.json').write_text(json.dumps(reported,indent=2))

    port=pd.read_csv(R/'m6_final_port_metrics.csv')
    rel=pd.read_csv(R/'m6_final_reliability_metrics.csv')
    horizons=pd.read_csv(R/'m6_final_horizon_support.csv')
    aug=port[port.port.eq('AUGUSTA')].iloc[0]; cat=port[port.port.eq('CATANIA')].iloc[0]
    high=rel[rel.reliability_tier.eq('HIGH')].iloc[0]; med=rel[rel.reliability_tier.eq('MEDIUM')].iloc[0]
    iv=summary['interval_high_reliability']; st=summary['stability']; p=summary['primary_under24']

    claims=pd.DataFrame([
      {'claim':'Frozen route-kNN improves <=24h over geodesic on the locked final scorable set','status':'SMALL_POSITIVE_FINAL_EFFECT_NOT_STATISTICALLY_RESOLVED','evidence':f"{len(call)} scorable calls; {p['route_improvement_fraction']*100:.1f}% MAE gain; paired bootstrap 95% CI [{lo*60:.1f},{hi*60:.1f}] min includes 0"},
      {'claim':'Frozen route-kNN improves final Catania approach ETA','status':'POSITIVE_BUT_UNDERPOWERED','evidence':f"{int(cat.calls)} calls; {cat.route_improvement_fraction*100:.1f}% gain"},
      {'claim':'Frozen route-kNN improves final Augusta approach ETA','status':'NEUTRAL_FINAL_RESULT','evidence':f"{int(aug.calls)} calls; {aug.route_improvement_fraction*100:.1f}% gain; bootstrap CI spans 0"},
      {'claim':'Frozen model covers every locked final call','status':'FALSE_UNDER_FROZEN_OPERATIONAL_SCOPE','evidence':f"16 locked calls opened; {len(scored)} had causal inbound-approach rows; 3 Catania calls had none"},
      {'claim':'HIGH reliability identifies a lower-error regime','status':'SUPPORTED_SMALL_FINAL_SAMPLE','evidence':f"HIGH {int(high.calls)} calls, raw <=24h MAE {high.raw_mae_under24_h*60:.1f} min; MEDIUM {int(med.calls)} calls, {med.raw_mae_under24_h*60:.1f} min"},
      {'claim':'Frozen 90% simultaneous HIGH-reliability interval transfers to final holdout','status':'SUPPORTED_SMALL_FINAL_SAMPLE','evidence':f"{iv['calls']} calls; whole-call coverage {iv['trajectory_coverage']*100:.1f}%; row coverage {iv['row_coverage']*100:.1f}%"},
      {'claim':'Causal ETA stabilizer improves final point accuracy','status':'NOT_SUPPORTED_FINAL','evidence':f"raw <=24h {st['raw_under24_mae_h']*60:.1f} min vs stabilized {st['stabilized_under24_mae_h']*60:.1f} min; kept only as optional jitter-reduction layer"},
      {'claim':'Causal ETA stabilizer reduces final ETA jitter','status':'SUPPORTED_SMALL_FINAL_SAMPLE','evidence':f"P90 revision {st['raw']['p90_revision_min']:.1f} -> {st['stabilized']['p90_revision_min']:.1f} min"},
      {'claim':'Final test establishes >24h performance','status':'NOT_ESTABLISHED','evidence':f"{int(horizons.loc[horizons.horizon.eq('>24h'),'calls'].iloc[0])} final calls with >24h frozen-scope predictions"},
      {'claim':'Model beats reported AIS ETA','status':'NOT_ESTABLISHED','evidence':f"0 reported-ETA snapshots aligned within 60 min; closest {reported.get('closest_snapshot_offset_min',float('nan')):.1f} min"},
      {'claim':'Model generalizes to unseen ports / berth / all-fast','status':'NOT_ESTABLISHED','evidence':'M5 scope boundary remains unchanged'},
    ])
    claims.to_csv(R/'m6_final_claim_scope.csv',index=False)

    report=f'''# M6 — Final locked chronological evaluation and take-home conclusion\n\n## Freeze discipline\n\nThe final design was frozen before scoring. The first execution attempt failed during dataframe preparation before any holdout prediction was persisted; the failure is preserved in `M6_PREOPEN_FAILURE.md`. A second immutable freeze was created, then the chronological holdout was opened once. No point-model, route, reliability, uncertainty, stabilizer, target or scope parameter was changed after final results were visible.\n\n## What was actually tested\n\n- 16 locked chronological calls existed: 10 Augusta and 6 Catania.\n- Under the predeclared causal `scope_inbound_approach`, **{len(scored)} calls were scorable**: {pred[pred.port.eq('AUGUSTA')].session_id.nunique()} Augusta and {pred[pred.port.eq('CATANIA')].session_id.nunique()} Catania.\n- Three Catania calls had zero rows satisfying the frozen inbound-approach scope. They were not rescued post-hoc; they are deployment-coverage failures recorded in `m6_final_call_coverage.csv`.\n- All scorable final predictions are <= {pred.true_tta_h.max():.2f} h from the research-gate target; the final set contains no >24 h evidence.\n\n## Final point-forecast result\n\nPrimary metric: voyage-balanced MAE at <=24 h.\n\n| Scope | Calls | Geodesic | Frozen route-kNN | Relative change |\n|---|---:|---:|---:|---:|\n| All scorable final | {len(call)} | {p['geodesic_mae_h']*60:.1f} min | **{p['route_mae_h']*60:.1f} min** | {p['route_improvement_fraction']*100:+.1f}% |\n| Augusta | {int(aug.calls)} | {aug.geodesic_mae_h*60:.1f} min | {aug.route_mae_h*60:.1f} min | {aug.route_improvement_fraction*100:+.1f}% |\n| Catania | {int(cat.calls)} | {cat.geodesic_mae_h*60:.1f} min | {cat.route_mae_h*60:.1f} min | {cat.route_improvement_fraction*100:+.1f}% |\n\nThe overall route gain is small ({p['route_improvement_fraction']*100:.1f}%). The call-level paired bootstrap mean gain is {mean_gain*60:.1f} min with 95% interval [{lo*60:.1f}, {hi*60:.1f}] min, so the final sample **does not resolve a non-zero overall route advantage statistically**. Catania is strongly positive but only 3 scorable calls; Augusta is essentially neutral.\n\n## Reliability and uncertainty\n\nFrozen HIGH reliability has {int(high.calls)} final calls and <=24 h raw MAE {high.raw_mae_under24_h*60:.1f} min. MEDIUM has {int(med.calls)} calls and {med.raw_mae_under24_h*60:.1f} min. This supports reliability as a useful diagnostic regime separator, not as a probability of correctness.\n\nThe frozen M4 90% simultaneous call-level interval on HIGH rows achieved:\n- row coverage: {iv['row_coverage']*100:.1f}%\n- voyage-balanced point coverage: {iv['voyage_balanced_point_coverage']*100:.1f}%\n- whole-call coverage: **{iv['trajectory_coverage']*100:.1f}% ({int(round(iv['trajectory_coverage']*iv['calls']))}/{iv['calls']} calls)**\n- fixed nominal half-width: {json.loads((R/'M6_FREEZE.json').read_text())['design']['uncertainty']['simultaneous_call_level_half_width_h']*60:.1f} min.\n\nGiven only {iv['calls']} final HIGH calls, this is encouraging calibration evidence, not a universal coverage guarantee.\n\n## Stability\n\nThe frozen causal EWMA reduces final P90 absolute-ETA revision from {st['raw']['p90_revision_min']:.1f} to {st['stabilized']['p90_revision_min']:.1f} min, but worsens <=24 h point MAE from {st['raw_under24_mae_h']*60:.1f} to {st['stabilized_under24_mae_h']*60:.1f} min. Therefore the raw M2 ETA remains the primary accuracy output; stabilization is an optional presentation/operations layer when lower jitter is worth a modest accuracy trade-off.\n\n## Reported AIS ETA\n\nThe final holdout still cannot support a fair model-vs-reported-ETA comparison. There are 0 snapshot rows aligned within 60 min of final prediction observations; the closest same-MMSI reported-ETA snapshot is {reported.get('closest_snapshot_offset_min',float('nan')):.1f} min away. Backfilling it would be leakage.\n\n## What the final holdout changed\n\nIt did **not** justify changing the model. It changed the strength of our claims:\n1. route-kNN remains a defensible retained point model because development OOF was consistently positive and final overall MAE is not worse, but the locked final advantage is small and uncertain;\n2. the reliability layer transfers usefully;\n3. the simultaneous interval lands close to its nominal whole-call coverage on a small final sample;\n4. the stabilizer is a jitter-reduction trade-off, not a final-accuracy improvement;\n5. frozen operational coverage is incomplete: 3/16 calls had no eligible inbound-approach decision row;\n6. long-horizon, unseen-port, Scirocco, berth/all-fast and reported-ETA-superiority claims remain unsupported.\n\n## Recommended take-home message\n\nThe strongest contribution is not a claim that a complicated learner beats the company's ETA. It is a leakage-resistant maritime forecasting pipeline that reconstructs auditable arrival events, diagnoses provider snapshots, uses train-only historical route geometry, rejects ML complexity when it fails OOF, emits uncertainty only in a transparent reliability regime, quantifies forecast stability, and preserves failure/coverage cases rather than hiding them.\n\n## Data to request next\n\n1. exact internal ETA target and official ground-truth event;\n2. historical Message 5 ETA/destination with source timestamps;\n3. at least several months of AIS history;\n4. PCS / berth / pilotage / movement event records;\n5. provider timestamp semantics and raw message timestamp/type if available.\n\nWith those data, the next scientifically meaningful experiment is long-horizon and port-operations modelling—not a larger neural network on the same 13-day sample.\n'''
    (R/'M6_FINAL_REPORT.md').write_text(report)

    model_card=f'''# Model card — AIS ETA Challenger V1\n\n## Intended prediction\nTime remaining to a **validated research-gate entrance event** for Catania or Augusta Levante. This is not claimed to be official PBP, ATA berth or all-fast.\n\n## Point predictor\nTrain-only historical route-kNN remaining distance (k=3) divided by robust trailing 30-minute median SOG with 1 kn floor. No residual ML is used: M3 rejected it by cross-fitted validation.\n\n## Inputs\nCausal AIS position/state, robust recent speed, destination intent, provider-state freshness and train-only historical route geometry.\n\n## Reliability\n`HIGH` is a deterministic diagnostic tier requiring supported predicted horizon (<=24 h), destination support for target port, route cross-track <=5 km, source observation age <=90 s/no HIGH stale-risk, and robust speed >=1 kn. It is not a calibrated correctness probability.\n\n## Uncertainty\nFrozen 90% simultaneous call-level symmetric band, half-width 2.485 h, emitted only for HIGH reliability. Calibration unit is one maximum error per call/voyage.\n\n## Stability\nOptional causal EWMA on absolute predicted arrival timestamps, alpha 0.4. Final holdout shows lower jitter but worse point MAE, so raw ETA is the primary accuracy output.\n\n## Validated scope\nSeen ports Catania and Augusta Levante; approach states satisfying frozen scope. Development evidence includes cold-vessel stress.\n\n## Unsupported scope\nUnseen ports, Augusta Scirocco, Siracusa/Santa Panagia, berth/all-fast, safety-critical navigation, and superiority to reported AIS ETA. Long-horizon final performance is not established.\n\n## Final holdout\n16 chronological calls were locked. 13 were scorable under the frozen approach scope. On those, route-kNN MAE <=24 h was {p['route_mae_h']*60:.1f} min vs {p['geodesic_mae_h']*60:.1f} min geodesic. The 95% call-bootstrap interval for mean route gain includes zero.\n'''
    (ROOT/'docs'/'MODEL_CARD.md').write_text(model_card)

    # Update README with a compact immutable final result block if absent.
    readme=ROOT/'README.md'; txt=readme.read_text()
    marker='## Final M6 result'
    if marker not in txt:
        txt += f'''\n\n{marker}\n\nThe one-time chronological holdout is complete under `reports/M6_FREEZE.json`. Of 16 locked calls, 13 were scorable under the predeclared inbound-approach scope. Frozen route-kNN achieved {p['route_mae_h']*60:.1f} min voyage-balanced MAE <=24 h versus {p['geodesic_mae_h']*60:.1f} min for geodesic physics ({p['route_improvement_fraction']*100:.1f}% gain; call-bootstrap CI includes zero). HIGH-reliability whole-call interval coverage was {iv['trajectory_coverage']*100:.1f}% on {iv['calls']} calls. See `reports/M6_FINAL_REPORT.md` and `docs/MODEL_CARD.md`.\n'''
        readme.write_text(txt)

    # Update workflow/status without changing frozen model artifacts.
    wf=ROOT/'WORKFLOW.md'; wt=wf.read_text().replace('### M6 — Final take-home deliverable\n**Status: NEXT**','### M6 — Final take-home deliverable\n**Status: DONE — FINAL HOLDOUT OPENED ONCE; CLAIMS SCOPE-LIMITED**')
    wt=wt.replace('The **current task** is M6: freeze the retained M2 route-physics + M4 reliability/stability design, then open and score the 16 locked chronological final calls exactly once. No model, threshold, route family, reliability rule or stabilizer may be changed in response to the final-holdout result. Assemble the final technical deliverable around the frozen claim scope established in M5.', 'M6 is complete. The 16-call chronological holdout was opened once under the frozen design; 13 calls were scorable under the predeclared inbound-approach scope. No post-holdout predictor tuning is permitted. The next activity is company-call preparation / optional packaging, not model selection.')
    wf.write_text(wt)

    ps=ROOT/'PROJECT_STATE.json'; state=json.loads(ps.read_text()); state['state_version']=int(state.get('state_version',0))+1; state['current_phase']='M6_DONE'; state['current_task']='Final take-home package is complete; prepare company discussion and request target/ground-truth/history data. Do not tune the predictor on the final holdout.'; state['completed'].append('M6 immutable freeze, one-time chronological holdout evaluation, final scope/coverage audit, model card and take-home report'); state['m6_summary']={'locked_calls':16,'scorable_calls':len(scored),'unscorable_calls':16-len(scored),'scorable_unique_vessels':int(pred.mmsi.nunique()),'route_mae_under24_h':p['route_mae_h'],'geodesic_mae_under24_h':p['geodesic_mae_h'],'route_gain_fraction':p['route_improvement_fraction'],'paired_gain_ci95_h':[lo,hi],'high_interval_calls':iv['calls'],'high_whole_call_coverage':iv['trajectory_coverage'],'high_row_coverage':iv['row_coverage'],'raw_p90_revision_min':st['raw']['p90_revision_min'],'stabilized_p90_revision_min':st['stabilized']['p90_revision_min'],'reported_eta_alignment':reported,'post_holdout_tuning':False}; ps.write_text(json.dumps(state,indent=2))
    print('M6 reporting finalized')

if __name__=='__main__': main()
