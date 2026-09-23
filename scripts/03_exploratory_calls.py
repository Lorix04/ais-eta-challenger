import pandas as pd, numpy as np, math, json
from pathlib import Path
files=['data/vessel_positions_part1.csv','data/vessel_positions_part2.csv']
use=['mmsi','name','category','nav_status','sog','cog','lat','lon','destination','recorded_at']
df=pd.concat([pd.read_csv(f,usecols=use,low_memory=False) for f in files],ignore_index=True)
df=df[df.category.eq('ship')].copy(); df['recorded_at']=pd.to_datetime(df.recorded_at)
df=df.sort_values(['mmsi','recorded_at'])
R=6371.0088
def dist(lat,lon,lat0,lon0):
    la=np.radians(lat); lo=np.radians(lon); la0=math.radians(lat0); lo0=math.radians(lon0)
    a=np.sin((la-la0)/2)**2+np.cos(la)*math.cos(la0)*np.sin((lo-lo0)/2)**2
    return R*2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
ports={'Catania':(37.497,15.094,2.8),'Augusta':(37.211,15.205,5.0),'SantaPanagia':(37.125,15.264,4.0),'Siracusa':(37.053,15.283,3.0)}
allres={}
for port,(lat0,lon0,rad) in ports.items():
    d=dist(df.lat.values,df.lon.values,lat0,lon0)
    x=df.copy(); x['inside']=d<=rad
    # identify contiguous inside episodes per vessel; gap > 10 min breaks too
    prev_inside=x.groupby('mmsi').inside.shift(fill_value=False)
    dt=x.groupby('mmsi').recorded_at.diff().dt.total_seconds().fillna(1e9)
    start=(x.inside & (~prev_inside | dt.gt(600)))
    # episode id increments at every start; only meaningful while inside
    x['ep']=start.groupby(x.mmsi).cumsum()
    e=x[x.inside].groupby(['mmsi','ep']).agg(name=('name','last'),start=('recorded_at','min'),end=('recorded_at','max'),n=('mmsi','size'),sog_med=('sog','median'),sog_min=('sog','min'),frac_slow=('sog',lambda s:(s<=0.5).mean()),destination=('destination',lambda s:s.dropna().mode().iloc[0] if len(s.dropna()) else None),lat_med=('lat','median'),lon_med=('lon','median')).reset_index()
    e['duration_min']=(e.end-e.start).dt.total_seconds()/60
    # candidate call = inside >=30 min and at least 15 min approx slow fraction OR med sog <=1
    cand=e[(e.duration_min>=30)&((e.frac_slow>=0.25)|(e.sog_med<=1.0))].copy()
    allres[port]={'episodes':len(e),'candidate_calls':len(cand),'candidate_vessels':int(cand.mmsi.nunique()),'radius_km':rad}
    cand.sort_values('start').to_csv(f'reports/{port.lower()}_exploratory_calls.csv',index=False)
print(json.dumps(allres,indent=2,default=str))
with open('reports/exploratory_call_counts.json','w') as f: json.dump(allres,f,indent=2)
