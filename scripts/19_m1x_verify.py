#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m1 import M1Config
from ais_eta.m1x_audit import build_augusta_decision_panel


def main():
    reports = ROOT / "reports"
    audited = pd.read_csv(reports / "augusta_m1x_audited_calls.csv")
    cohort = pd.read_csv(reports / "augusta_m1x_eta_primary_cohort.csv")
    panel = pd.read_csv(
        reports / "augusta_m1x_decision_panel.csv",
        parse_dates=["ground_truth_time", "decision_time", "voyage_start_time", "source_observation_time"],
    )
    events = pd.read_csv(reports / "augusta_m1x_gate_events.csv")
    gate = json.loads((reports / "m1x_gate_result.json").read_text())
    checks=[]
    checks.append(("all 79 Augusta candidates audited", len(audited)==79 and audited.session_id.nunique()==79))
    checks.append(("no unconfirmed-stop candidate retained", not audited[audited.candidate_class.ne("STOP_CONFIRMED_CALL_CANDIDATE")].final_ground_truth.astype(bool).any()))
    checks.append(("retained truth has A/B only", set(audited.loc[audited.final_ground_truth.astype(bool),"ground_truth_confidence"]).issubset({"A","B"})))
    sci=audited[audited.inbound_entrance.eq("SCIROCCO") & audited.final_ground_truth.astype(bool)]
    checks.append(("retained Scirocco capped at B", len(sci)>0 and sci.ground_truth_confidence.eq("B").all()))
    checks.append(("ETA cohort subset of truth", cohort.final_ground_truth.astype(bool).all()))
    checks.append(("ETA cohort named", cohort.name.notna().all()))
    checks.append(("ETA cohort has long-range pre-entry observation", cohort.pre_entry_max_distance_to_target_gate_km.ge(15.0).all()))
    checks.append(("ETA cohort has >=30 pre-entry observations", cohort.pre_entry_observations_6h.ge(30).all()))
    checks.append(("panel unique session/decision", not panel.duplicated(["session_id","decision_time"]).any()))
    checks.append(("source observation <= decision", bool((panel.source_observation_time<=panel.decision_time).all())))
    checks.append(("decision < ground truth", bool((panel.decision_time<panel.ground_truth_time).all())))
    checks.append(("provider age nonnegative", bool((panel.provider_observation_age_s>=0).all())))
    checks.append(("15-minute wall-clock grid", bool(panel.decision_time.dt.minute.mod(15).eq(0).all() and panel.decision_time.dt.second.eq(0).all())))
    checks.append(("pre-specified gate result exists", gate["status"] in {"GO_M2","HOLD_M2"}))
    checks.append(("combined gate passed", gate["status"]=="GO_M2"))

    # Prefix invariance on 10 distinct calls with causal approach rows.
    states=pd.read_pickle(ROOT/'data/derived/m0c_ship_states.pkl.gz')
    candidates=panel[panel.scope_inbound_approach.astype(bool)].drop_duplicates('session_id').head(10)
    invariant=0
    keys=["source_observation_time","provider_observation_age_s","distance_to_target_gate_km","sog_median_30m_kn","progress_to_gate_30m_kn","input_quality_ok","outside_target_research_gate","scope_inbound_approach","pred_geodesic_median30_sog_floor_h"]
    for r in candidates.itertuples(index=False):
        one=cohort[cohort.session_id.eq(r.session_id)].copy()
        prefix=states[(states.mmsi.astype(int).eq(int(r.mmsi))) & (pd.to_datetime(states.recorded_at)<=pd.Timestamp(r.decision_time))].copy()
        ev=events[~((events.mmsi.astype(int).eq(int(r.mmsi))) & (pd.to_datetime(events.crossing_time_proxy)>pd.Timestamp(r.decision_time)))].copy()
        rebuilt=build_augusta_decision_panel(prefix,one,ev,M1Config(decision_interval_min=15))
        match=rebuilt[rebuilt.decision_time.eq(pd.Timestamp(r.decision_time))]
        if len(match)!=1: continue
        got=match.iloc[0]; orig=panel[(panel.session_id.eq(r.session_id)) & (panel.decision_time.eq(pd.Timestamp(r.decision_time)))].iloc[0]
        same=True
        for key in keys:
            a,b=orig[key],got[key]
            if pd.isna(a) and pd.isna(b): continue
            if key.endswith('time'): same &= pd.Timestamp(a)==pd.Timestamp(b)
            elif isinstance(a,(bool,np.bool_)) or isinstance(b,(bool,np.bool_)): same &= bool(a)==bool(b)
            elif isinstance(a,str) or isinstance(b,str): same &= str(a)==str(b)
            else: same &= bool(np.isclose(float(a),float(b),rtol=1e-10,atol=1e-10))
        invariant += int(same)
    checks.append((f"prefix invariance {invariant}/{len(candidates)}", invariant==len(candidates)))

    failed=[n for n,ok in checks if not ok]
    for n,ok in checks: print(('PASS' if ok else 'FAIL'),n)
    if failed: raise SystemExit(f'M1X verification failed: {failed}')
    print(f'M1X verification PASS: {len(checks)} checks')

if __name__=='__main__': main()
