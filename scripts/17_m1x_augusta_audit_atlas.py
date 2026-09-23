from __future__ import annotations
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ais_eta.m1x import AugustaGeometryV1, label_augusta_geometry


def diagnostics(cands, states):
    geo=label_augusta_geometry(states)
    rows=[]
    for r in cands.itertuples(index=False):
        v=geo[geo.mmsi.eq(int(r.mmsi))].sort_values('recorded_at')
        start=pd.Timestamp(r.inbound_time)
        end=pd.Timestamp(r.outbound_time) if pd.notna(r.outbound_time) else pd.Timestamp(r.last_observed_at)
        sess=v[(v.recorded_at>=start)&(v.recorded_at<=end)]
        pre=v[(v.recorded_at>=start-pd.Timedelta(hours=6))&(v.recorded_at<start)]
        inner=sess[sess.augusta_inner_megarese_proxy]
        gaps=sess.recorded_at.diff().dt.total_seconds()
        dur=max((end-start).total_seconds(),0)
        exp=max(dur/61.0,1)
        confirmed=inner[inner.stop_duration_s>=20*60]
        rows.append({
            'session_id':r.session_id,
            'coverage':len(sess)/exp,
            'max_gap_min':gaps.max()/60 if len(gaps) else np.nan,
            'gaps_gt_10m':int(gaps.gt(600).sum()),
            'inner_fraction':float(sess.augusta_inner_megarese_proxy.mean()) if len(sess) else np.nan,
            'time_to_confirmed_stop_min':((confirmed.iloc[0].recorded_at-start).total_seconds()/60) if len(confirmed) else np.nan,
            'pre_obs_6h':len(pre),
            'pre_max_dist_km':float(pre.augusta_distance_to_nearest_gate_km.max()) if len(pre) else np.nan,
            'high_stale_fraction':float(sess.stale_risk_level.eq('HIGH').mean()) if len(sess) else np.nan,
        })
    return pd.DataFrame(rows)


def plot_one(ax,row,states,geom):
    v=states[states.mmsi.eq(int(row.mmsi))].sort_values('recorded_at')
    start=pd.Timestamp(row.inbound_time)
    end=pd.Timestamp(row.outbound_time) if pd.notna(row.outbound_time) else pd.Timestamp(row.last_observed_at)
    w=v[(v.recorded_at>=start-pd.Timedelta(hours=6))&(v.recorded_at<=end+pd.Timedelta(hours=2))]
    pre=w[w.recorded_at<start]; sess=w[(w.recorded_at>=start)&(w.recorded_at<=end)]; post=w[w.recorded_at>end]
    if len(pre): ax.plot(pre.lon_clean,pre.lat_clean,lw=.8,alpha=.8)
    if len(sess): ax.plot(sess.lon_clean,sess.lat_clean,lw=1.1,alpha=.9)
    if len(post): ax.plot(post.lon_clean,post.lat_clean,lw=.7,alpha=.6)
    for gate in geom.gates:
        ax.plot([gate.a[0],gate.b[0]],[gate.a[1],gate.b[1]],'k-',lw=1.5)
    vv=np.array(geom.inner_vertices)
    ax.plot(vv[:,0],vv[:,1],'k--',lw=.55)
    ax.scatter([row.inbound_lon],[row.inbound_lat],s=24,marker='>',color='black',zorder=5)
    title=(f"{row.session_id} {str(row.name)[:18]} [{row.inbound_entrance}]\n"
           f"{row.candidate_class.replace('_',' ')[:25]} Q={row.m1x_candidate_quality} "
           f"stop={row.max_inner_stop_duration_min:.0f}m dur={row.session_duration_min:.0f}m\n"
           f"dest={bool(row.destination_support_augusta)} closed={bool(row.session_closed)} "
           f"rc={bool(row.right_censored)} unr={bool(row.unresolved_missing_outbound)}")
    ax.set_title(title,fontsize=6.5)
    ax.set_xlim(15.18,15.255); ax.set_ylim(37.165,37.25); ax.grid(alpha=.18); ax.tick_params(labelsize=5.5)


def main():
    c=pd.read_csv(ROOT/'reports/augusta_m1x_candidate_calls.csv',parse_dates=['inbound_time','outbound_time','last_observed_at'])
    raw=pd.read_pickle(ROOT/'data/derived/m0c_ship_states.pkl.gz')
    states=label_augusta_geometry(raw)
    d=diagnostics(c,raw); d.to_csv(ROOT/'reports/augusta_m1x_session_diagnostics.csv',index=False)
    atlas=ROOT/'reports/m1x_augusta_audit_atlas'; atlas.mkdir(exist_ok=True)
    geom=AugustaGeometryV1()
    for p,start in enumerate(range(0,len(c),9),1):
        page=c.iloc[start:start+9]
        fig,axes=plt.subplots(3,3,figsize=(15,15),dpi=130)
        for ax in axes.flat: ax.axis('off')
        for ax,row in zip(axes.flat,page.itertuples(index=False)):
            ax.axis('on'); plot_one(ax,row,states,geom)
        fig.suptitle(f'M1X Augusta candidate audit atlas - page {p}',fontsize=14)
        fig.tight_layout(rect=(0,0,1,.97)); fig.savefig(atlas/f'page_{p:02d}.png',bbox_inches='tight'); plt.close(fig)
    print('atlas_pages',len(list(atlas.glob('page_*.png'))))
    print(d.describe(include='all').to_string())
if __name__=='__main__': main()
