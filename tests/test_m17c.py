from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ais_eta.m17c import prepare_transition_events, predict_corridor_eta, M17C_GRID_DEG


def _states():
    t0 = pd.Timestamp('2026-04-10 00:00:00')
    return pd.DataFrame({
        'mmsi':[1,1,1,2,2,2],
        'recorded_at':[t0,t0+pd.Timedelta(minutes=5),t0+pd.Timedelta(minutes=10)]*2,
        'lat_clean':[37.01,37.21,37.41,37.01,37.21,37.41],
        'lon_clean':[15.01,15.21,15.41,15.01,15.21,15.41],
        'sog_clean':[10,10,10,12,12,12],
        'position_valid':[True]*6,
    })


def test_transition_events_are_directed_and_target_free():
    ev = prepare_transition_events(_states(), owner_ship_type={1:'Cargo',2:'Cargo'})
    assert len(ev) == 4
    assert {'src_ilat','src_ilon','dst_ilat','dst_ilon','speed_kn','time_bin','ship_type'} <= set(ev.columns)
    assert 'target_tte_h' not in ev.columns and 'eta_reference_dt' not in ev.columns
    assert (ev['src_ilat'] <= ev['dst_ilat']).all()


def test_query_cutoff_blocks_future_edge():
    ev = prepare_transition_events(_states(), owner_ship_type={1:'Cargo',2:'Cargo'})
    # At 00:06 only the first A->B transition exists, so C is not yet reachable.
    out = predict_corridor_eta(ev, query_mmsi=99, query_at=pd.Timestamp('2026-04-10 00:06:00'),
        query_lat=37.01, query_lon=15.01, destination_lat=37.41, destination_lon=15.41,
        query_ship_type='Cargo', recent_speed_kn=10.0, max_snap_nm=40)
    assert not out.supported or out.path_edges <= 1


def test_directed_graph_does_not_invent_reverse_corridor():
    ev = prepare_transition_events(_states(), owner_ship_type={1:'Cargo',2:'Cargo'})
    out = predict_corridor_eta(ev, query_mmsi=99, query_at=pd.Timestamp('2026-04-10 01:00:00'),
        query_lat=37.41, query_lon=15.41, destination_lat=37.01, destination_lon=15.01,
        query_ship_type='Cargo', recent_speed_kn=10.0, max_snap_nm=40)
    assert out.supported is False
    assert out.reason in {'no_directed_path','destination_outside_graph','source_outside_graph'}


def test_supported_forward_path_is_finite():
    ev = prepare_transition_events(_states(), owner_ship_type={1:'Cargo',2:'Cargo'})
    out = predict_corridor_eta(ev, query_mmsi=99, query_at=pd.Timestamp('2026-04-10 01:00:00'),
        query_lat=37.01, query_lon=15.01, destination_lat=37.41, destination_lon=15.41,
        query_ship_type='Cargo', recent_speed_kn=10.0, max_snap_nm=40)
    assert out.supported
    assert np.isfinite(out.prediction_h) and out.prediction_h > 0
    assert out.graph_edges >= 2


def test_m17c_artifacts_are_development_only_and_causal_if_present():
    s = ROOT/'reports/M17C_SUMMARY.json'; p = ROOT/'reports/m17c_oof_predictions.csv'
    if not s.exists() or not p.exists(): return
    summary = json.loads(s.read_text()); led = pd.read_csv(p)
    manifest = json.loads((ROOT/'reports/M16A_DEVELOPMENT_MANIFEST.json').read_text())
    assert len(led) == 386 and led.mmsi.nunique() == 386
    assert set(led.mmsi.astype(int)).isdisjoint(manifest['blocked_old_final_mmsi'])
    assert summary['final_test_used_for_selection'] is False
    assert summary['outer_valid_owners_used_for_graph'] is False
    assert summary['future_events_used_for_query_graph'] is False
    assert summary['target_used_for_graph_topology_or_edge_weight'] is False


def test_project_state_can_advance_to_m17c():
    state = json.loads((ROOT/'PROJECT_STATE.json').read_text())
    if 'm17c_summary' not in state: return
    assert state['current_phase'].startswith(('M17C_','M17D_','M17E_','M17F_','M17G_','M17H_','M18A_','M18B_','M18C_','M18D_','M18E_','M18F_','M18G_','M19_'))
    assert state['m17c_summary']['final_test_used_for_selection'] is False
    assert state['m17c_summary']['outer_valid_owners_used_for_graph'] is False
