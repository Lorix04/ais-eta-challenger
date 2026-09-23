#!/usr/bin/env python3
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'reports'

def final_metric_plot():
    x=pd.read_csv(R/'m6_final_port_metrics.csv').sort_values('port')
    pos=np.arange(len(x)); w=.36
    fig,ax=plt.subplots(figsize=(8.8,5.5))
    ax.bar(pos-w/2,x.geodesic_mae_h*60,w,label='Geodesic physics')
    ax.bar(pos+w/2,x.route_mae_h*60,w,label='Frozen route-kNN physics')
    ax.set_xticks(pos,x.port); ax.set_ylabel('Final voyage-balanced MAE <=24 h (min)')
    ax.set_title('M6 — one-time locked chronological holdout')
    ax.grid(axis='y',alpha=.25); ax.legend()
    for i,r in enumerate(x.itertuples()):
        ax.text(i,max(r.geodesic_mae_h,r.route_mae_h)*60+2,f"{r.route_improvement_fraction*100:.1f}%\n{int(r.calls)} calls",ha='center',fontsize=9)
    fig.tight_layout(); fig.savefig(R/'m6_final_port_comparison.png',dpi=160); plt.close(fig)

def reliability_plot():
    x=pd.read_csv(R/'m6_final_reliability_metrics.csv')
    x=x[x.calls>0].copy(); pos=np.arange(len(x)); w=.36
    fig,ax=plt.subplots(figsize=(8.8,5.5))
    ax.bar(pos-w/2,x.raw_mae_under24_h*60,w,label='Raw frozen ETA')
    ax.bar(pos+w/2,x.stabilized_mae_under24_h*60,w,label='Causal stabilized ETA')
    ax.set_xticks(pos,x.reliability_tier); ax.set_ylabel('Voyage-balanced MAE <=24 h (min)')
    ax.set_title('M6 — frozen reliability tiers on final holdout')
    ax.grid(axis='y',alpha=.25); ax.legend()
    for i,r in enumerate(x.itertuples()): ax.text(i,max(r.raw_mae_under24_h,r.stabilized_mae_under24_h)*60+2,f"{int(r.calls)} calls",ha='center',fontsize=9)
    fig.tight_layout(); fig.savefig(R/'m6_final_reliability.png',dpi=160); plt.close(fig)

def result_panel():
    s=json.loads((R/'m6_final_summary.json').read_text())
    p=s['primary_under24']; iv=s['interval_high_reliability']; st=s['stability']
    lines=[
      'M6 — FINAL LOCKED CHRONOLOGICAL EVALUATION','',
      f"Locked calls opened: 16; scorable under frozen scope: {s['locked_final_calls_scored']}",
      f"Scorable vessels: {s['final_unique_vessels']}",
      f"By port: {s['final_calls_by_port']}",f"Cold final calls: {s['cold_final_calls']}",'',
      f"<=24h geodesic MAE: {p['geodesic_mae_h']*60:.1f} min",
      f"<=24h route-kNN MAE: {p['route_mae_h']*60:.1f} min",
      f"route improvement: {p['route_improvement_fraction']*100:.1f}%",'',
      f"HIGH interval calls: {iv['calls']}",f"whole-call coverage: {iv['trajectory_coverage']*100:.1f}%",
      f"row coverage: {iv['row_coverage']*100:.1f}%",'',
      f"Raw P90 ETA revision: {st['raw'].get('p90_revision_min',float('nan')):.1f} min",
      f"Stabilized P90 revision: {st['stabilized'].get('p90_revision_min',float('nan')):.1f} min",'',
      'FINAL HOLDOUT OPENED ONCE. NO POST-HOLDOUT TUNING.',
    ]
    fig,ax=plt.subplots(figsize=(10.8,7)); ax.axis('off'); ax.text(.04,.96,'\n'.join(lines),va='top',family='monospace',fontsize=11.5)
    fig.tight_layout(); fig.savefig(R/'m6_result_panel.png',dpi=160); plt.close(fig)


def timeline_example():
    x=pd.read_csv(R/'m6_final_predictions.csv')
    x['decision_time']=pd.to_datetime(x['decision_time']); x['ground_truth_time']=pd.to_datetime(x['ground_truth_time'])
    # Predeclared reporting choice: most prediction points, not best/worst error.
    sid=x.groupby('session_id').size().sort_values(ascending=False).index[0]
    g=x[x.session_id.eq(sid)].sort_values('decision_time').copy()
    raw=g['decision_time']+pd.to_timedelta(g.pred_raw_h,unit='h')
    stable=pd.to_datetime(g.pred_stabilized_arrival)
    truth=g.ground_truth_time.iloc[0]
    fig,ax=plt.subplots(figsize=(10,5.6))
    ax.plot(g.decision_time,raw,label='Raw route-physics ETA')
    ax.plot(g.decision_time,stable,label='Causal stabilized ETA')
    hi=g.reliability_tier.eq('HIGH')
    ax.axhline(truth,label='Validated research-gate arrival',linewidth=1)
    ax.set_title(f'M6 final example: {sid} — ETA revisions over time')
    ax.set_xlabel('Prediction time'); ax.set_ylabel('Predicted arrival timestamp')
    ax.legend(); ax.grid(alpha=.25); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(R/'m6_final_timeline_example.png',dpi=160); plt.close(fig)

if __name__=='__main__':
    final_metric_plot(); reliability_plot(); result_panel(); timeline_example(); print('M6 visual audit complete')
