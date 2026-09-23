import pandas as pd, numpy as np, math, json
from sklearn.cluster import DBSCAN
from pathlib import Path
files=['data/vessel_positions_part1.csv','data/vessel_positions_part2.csv']
use=['mmsi','name','category','nav_status','sog','cog','heading','lat','lon','destination','recorded_at']
df=pd.concat([pd.read_csv(f,usecols=use,low_memory=False) for f in files],ignore_index=True)
df=df[df.category.eq('ship')].copy()
df['recorded_at']=pd.to_datetime(df.recorded_at)
# haversine km vectorized
R=6371.0088
def dist_km(lat,lon,lat0,lon0):
    la=np.radians(lat); lo=np.radians(lon); la0=math.radians(lat0); lo0=math.radians(lon0)
    dlat=la-la0; dlon=lo-lo0
    a=np.sin(dlat/2)**2+np.cos(la)*math.cos(la0)*np.sin(dlon/2)**2
    return R*2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
ports={'Catania':(37.495,15.095),'Augusta':(37.225,15.220),'Santa_Panagia':(37.105,15.290)}
out={}
Path('reports').mkdir(exist_ok=True)
for name,(lat0,lon0) in ports.items():
    d=dist_km(df.lat.values,df.lon.values,lat0,lon0)
    g=df[d<=15].copy(); g['dist_center_km']=d[d<=15]
    stop=g[(g.sog<=0.5)].copy()
    # thin exact repeats: one point per mmsi+rounded pos to avoid poll dominance
    pts=stop.assign(lat_r=stop.lat.round(4),lon_r=stop.lon.round(4)).drop_duplicates(['mmsi','lat_r','lon_r'])
    # dbscan haversine eps 250 m, min 3 distinct vessel-pos samples
    if len(pts):
        X=np.radians(pts[['lat','lon']].values)
        labels=DBSCAN(eps=0.25/6371.0088,min_samples=3,metric='haversine').fit_predict(X)
        pts['cluster']=labels
        cl=pts[pts.cluster>=0].groupby('cluster').agg(n_points=('mmsi','size'),n_vessels=('mmsi','nunique'),lat=('lat','median'),lon=('lon','median'),dest_top=('destination',lambda x: x.value_counts().index[0] if x.notna().any() else None)).sort_values(['n_vessels','n_points'],ascending=False)
        cl.to_csv(f'reports/{name.lower()}_stop_clusters.csv')
        clusters=cl.head(20).reset_index().to_dict('records')
    else: clusters=[]
    out[name]={
      'rows_15km':len(g),'vessels_15km':int(g.mmsi.nunique()),'slow_rows':len(stop),'slow_vessels':int(stop.mmsi.nunique()),
      'clusters':clusters
    }
print(json.dumps(out,indent=2,default=str))
with open('reports/port_probe.json','w') as f: json.dump(out,f,indent=2,default=str)
