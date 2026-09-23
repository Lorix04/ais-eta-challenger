from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ais_eta.m1x import label_augusta_geometry, detect_augusta_crossings, build_augusta_sessions, geometry_manifest

def main():
    states=pd.read_pickle(ROOT/'data/derived/m0c_ship_states.pkl.gz')
    crossings=detect_augusta_crossings(states)
    sessions=build_augusta_sessions(states,crossings)
    reports=ROOT/'reports'; reports.mkdir(exist_ok=True)
    crossings.to_csv(reports/'augusta_m1x_gate_events.csv',index=False)
    sessions.to_csv(reports/'augusta_m1x_candidate_calls.csv',index=False)
    geo=label_augusta_geometry(states)
    local=geo[geo.augusta_analysis_envelope]
    summary={
      'geometry':geometry_manifest(),
      'rows_in_analysis_envelope':int(len(local)),
      'vessels_in_analysis_envelope':int(local.mmsi.nunique()),
      'gate_events':int(len(crossings)),
      'gate_event_vessels':int(crossings.mmsi.nunique()) if len(crossings) else 0,
      'events_by_entrance_direction':{f'{a}_{d}':int(n) for (a,d),n in crossings.groupby(['entrance_name','crossing_direction']).size().items()} if len(crossings) else {},
      'candidate_sessions':int(len(sessions)),
      'candidate_session_vessels':int(sessions.mmsi.nunique()) if len(sessions) else 0,
      'sessions_by_inbound_entrance':sessions.inbound_entrance.value_counts().to_dict() if len(sessions) else {},
      'candidate_class':sessions.candidate_class.value_counts().to_dict() if len(sessions) else {},
      'candidate_quality':sessions.m1x_candidate_quality.value_counts().to_dict() if len(sessions) else {},
      'right_censored':int(sessions.right_censored.sum()) if len(sessions) else 0,
      'unresolved_missing_outbound':int(sessions.unresolved_missing_outbound.sum()) if len(sessions) else 0,
      'destination_support_augusta':int(sessions.destination_support_augusta.sum()) if len(sessions) else 0,
      'warning':'Candidate research-gate sessions only; manual audit required before ground truth.'
    }
    (reports/'m1x_augusta_candidate_summary.json').write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str))
if __name__=='__main__': main()
