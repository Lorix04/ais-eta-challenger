import pandas as pd, numpy as np, json, os, math
from pathlib import Path
files=['data/vessel_positions_part1.csv','data/vessel_positions_part2.csv']
usecols=['id','mmsi','name','category','nav_status','sog','cog','heading','lat','lon','destination','draught','recorded_at','created_at','updated_at']
dfs=[]
for f in files:
    df=pd.read_csv(f,usecols=usecols,low_memory=False)
    dfs.append(df)
df=pd.concat(dfs,ignore_index=True)
for c in ['recorded_at','created_at','updated_at']:
    df[c]=pd.to_datetime(df[c],errors='coerce')
summary={}
summary['rows']=len(df)
summary['time_min']=str(df.recorded_at.min())
summary['time_max']=str(df.recorded_at.max())
summary['unique_mmsi']=int(df.mmsi.nunique())
summary['categories']=df.groupby('category').agg(rows=('mmsi','size'),mmsi=('mmsi','nunique')).to_dict('index')
summary['recorded_eq_created_pct']=float((df.recorded_at==df.created_at).mean())
summary['recorded_eq_updated_pct']=float((df.recorded_at==df.updated_at).mean())
# batch timestamps
batch=df.groupby('recorded_at')['mmsi'].nunique().sort_values(ascending=False)
summary['unique_recorded_timestamps']=int(batch.size)
summary['batch_mmsi_quantiles']={str(k):float(v) for k,v in batch.quantile([0,.1,.25,.5,.75,.9,.95,.99,1]).items()}
summary['top_batch_sizes']={str(k):int(v) for k,v in batch.head(20).items()}
sec=df.recorded_at.dt.second.value_counts(normalize=True).sort_index()
summary['second_of_minute_top']= {int(k):float(v) for k,v in sec.sort_values(ascending=False).head(15).items()}
# batches interval
uts=pd.Series(batch.index.sort_values())
bdt=uts.diff().dt.total_seconds().dropna()
summary['global_batch_interval_counts_top']={str(k):int(v) for k,v in bdt.value_counts().head(10).items()}
# ships
ship=df[df.category.eq('ship')].copy().sort_values(['mmsi','recorded_at'])
summary['ship_rows']=len(ship); summary['ship_mmsi']=int(ship.mmsi.nunique())
# dt per mmsi
ship['dt']=ship.groupby('mmsi').recorded_at.diff().dt.total_seconds()
summary['ship_dt_top']={str(k):int(v) for k,v in ship['dt'].value_counts().head(15).items()}
summary['ship_dt_quantiles']={str(k):float(v) for k,v in ship['dt'].dropna().quantile([0,.01,.1,.25,.5,.75,.9,.95,.99,1]).items()}
# identical position / dynamic state
prev_lat=ship.groupby('mmsi').lat.shift(); prev_lon=ship.groupby('mmsi').lon.shift()
same_pos=(ship.lat.eq(prev_lat)&ship.lon.eq(prev_lon))
summary['same_position_consecutive_pct']=float(same_pos.mean())
# rounded state equality exact for core dynamics
for col in ['sog','cog','heading','nav_status','destination','draught']:
    ship['p_'+col]=ship.groupby('mmsi')[col].shift()
state_cols=['lat','lon','sog','cog','heading','nav_status','destination','draught']
# equality handling NaN==NaN
same_state=pd.Series(True,index=ship.index)
for c in state_cols:
    a=ship[c]; b=ship['p_'+c] if 'p_'+c in ship else ship.groupby('mmsi')[c].shift()
    same_state &= (a.eq(b) | (a.isna()&b.isna()))
# first rows should false
same_state &= ship.groupby('mmsi').cumcount().gt(0)
summary['same_core_state_consecutive_pct']=float(same_state.mean())
summary['same_pos_sog_positive_rows']=int((same_pos & ship.sog.gt(0.5)).sum())
summary['same_pos_sog_positive_pct_all_ship']=float((same_pos & ship.sog.gt(0.5)).mean())
summary['same_pos_sog_gt5_rows']=int((same_pos & ship.sog.gt(5)).sum())
# heading sentinel etc
summary['heading_511_pct']=float(ship.heading.eq(511).mean())
summary['cog_ge360_pct']=float(ship.cog.ge(360).mean())
summary['sog_missing_pct']=float(ship.sog.isna().mean())
summary['destination_missing_pct']=float(ship.destination.isna().mean())
summary['destination_unique']=int(ship.destination.nunique(dropna=True))
summary['destination_top']=ship.destination.value_counts(dropna=False).head(30).to_dict()
# per-vessel observed spans and rows
vg=ship.groupby('mmsi').agg(rows=('mmsi','size'),start=('recorded_at','min'),end=('recorded_at','max'),name=('name','last'))
vg['span_h']=(vg.end-vg.start).dt.total_seconds()/3600
summary['ship_rows_per_mmsi_quantiles']={str(k):float(v) for k,v in vg.rows.quantile([0,.1,.25,.5,.75,.9,.95,.99,1]).items()}
summary['ship_span_hours_quantiles']={str(k):float(v) for k,v in vg.span_h.quantile([0,.1,.25,.5,.75,.9,.95,.99,1]).items()}
# longest same-position runs
# compute run ids per mmsi based on position change
ship['pos_change']=~same_pos
ship['pos_run']=ship.groupby('mmsi')['pos_change'].cumsum()
runs=ship.groupby(['mmsi','pos_run']).agg(n=('mmsi','size'),start=('recorded_at','min'),end=('recorded_at','max'),lat=('lat','first'),lon=('lon','first'),sog_med=('sog','median'),name=('name','last'))
runs['duration_h']=(runs.end-runs.start).dt.total_seconds()/3600
summary['longest_same_position_runs']=runs.sort_values(['n','duration_h'],ascending=False).head(20).reset_index().to_dict('records')
# simultaneous batches composition quantiles for ships only
sb=ship.groupby('recorded_at')['mmsi'].nunique()
summary['ship_batch_median']=float(sb.median()); summary['ship_batch_p90']=float(sb.quantile(.9)); summary['ship_batch_max']=int(sb.max())
# change count per timestamp? how many ships core state changed at each poll
ship['state_changed']=~same_state
chg=ship.groupby('recorded_at').agg(ship_rows=('mmsi','size'),changes=('state_changed','sum'))
chg['change_fraction']=chg.changes/chg.ship_rows
summary['change_fraction_quantiles']={str(k):float(v) for k,v in chg.change_fraction.quantile([0,.1,.25,.5,.75,.9,.95,.99,1]).items()}
# save
Path('reports').mkdir(exist_ok=True)
with open('reports/audit_summary.json','w') as f: json.dump(summary,f,indent=2,default=str)
# save useful CSV tables
batch.rename('n_mmsi').reset_index().to_csv('reports/batch_sizes.csv',index=False)
vg.reset_index().to_csv('reports/vessel_coverage.csv',index=False)
runs.sort_values(['n','duration_h'],ascending=False).head(500).reset_index().to_csv('reports/longest_position_runs.csv',index=False)
chg.reset_index().to_csv('reports/batch_change_fraction.csv',index=False)
print(json.dumps(summary,indent=2,default=str)[:30000])
