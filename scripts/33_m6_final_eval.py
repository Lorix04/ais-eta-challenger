#!/usr/bin/env python3
"""Open the 16 locked chronological calls exactly once under the frozen M6 design."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from ais_eta.m2 import M2Config, add_horizon_band, fit_train_only_route_model, predict_route_distances, eta_metrics  # noqa: E402
from ais_eta.m4 import M4Config, add_symmetric_interval, apply_causal_arrival_ewma, enrich_with_source_state  # noqa: E402
from ais_eta.m6 import (  # noqa: E402
    file_sha256, grouped_final_metrics, final_reliability_metrics,
    final_interval_metrics, final_stability_metrics,
)

REPORTS = ROOT / 'reports'
STATES_PATH = ROOT / 'data' / 'derived' / 'm0c_ship_states.pkl.gz'
FREEZE = REPORTS / 'M6_FREEZE.json'
OPEN_MARKER = REPORTS / 'M6_FINAL_HOLDOUT_OPENED.json'


def _verify_freeze(freeze: dict) -> None:
    for rel, expected in freeze['sha256'].items():
        actual = file_sha256(ROOT / rel)
        if actual != expected:
            raise AssertionError(f'Frozen input changed after M6 freeze: {rel}')


def _voyage_start_lookup() -> dict[str, pd.Timestamp]:
    out = {}
    for f in ['catania_m1_decision_panel.csv', 'augusta_m1x_decision_panel.csv']:
        d = pd.read_csv(REPORTS / f)
        d['voyage_start_time'] = pd.to_datetime(d['voyage_start_time'], errors='raise')
        out.update(d.groupby('session_id')['voyage_start_time'].first().to_dict())
    return {str(k): pd.Timestamp(v) for k, v in out.items()}


def _normalized_panel(states: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    cat = pd.read_csv(REPORTS / 'catania_m1_decision_panel.csv')
    cat = cat.rename(columns={'distance_to_gate_mid_km':'geodesic_distance_km'})
    cat['port'] = 'CATANIA'; cat['inbound_entrance'] = 'CATANIA'
    aug = pd.read_csv(REPORTS / 'augusta_m1x_decision_panel.csv')
    aug = aug.rename(columns={'distance_to_target_gate_km':'geodesic_distance_km'})
    aug['port'] = 'AUGUSTA'
    cols = [
        'port','session_id','mmsi','name','inbound_entrance','ground_truth_time','ground_truth_confidence',
        'decision_time','source_observation_time','true_tta_h','geodesic_distance_km','sog_median_30m_kn','scope_inbound_approach'
    ]
    panel = pd.concat([cat[cols], aug[cols]], ignore_index=True)
    for c in ['ground_truth_time','decision_time','source_observation_time']:
        panel[c] = pd.to_datetime(panel[c], errors='raise')
    pos = states[['mmsi','recorded_at','lon_clean','lat_clean']].drop_duplicates(['mmsi','recorded_at']).copy()
    panel = panel.merge(pos, left_on=['mmsi','source_observation_time'], right_on=['mmsi','recorded_at'], how='left', validate='many_to_one').drop(columns='recorded_at')
    if panel[['lon_clean','lat_clean']].isna().any().any():
        raise AssertionError('Final panel source-position join failed')
    panel = panel.merge(splits[['session_id','m2_split']], on='session_id', how='left', validate='many_to_one')
    return add_horizon_band(panel)


def _jsonify(v):
    if isinstance(v, dict): return {str(k): _jsonify(x) for k,x in v.items()}
    if isinstance(v, (list,tuple)): return [_jsonify(x) for x in v]
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, (np.bool_,)): return bool(v)
    if isinstance(v, pd.Timestamp): return v.isoformat()
    try:
        if pd.isna(v): return None
    except Exception: pass
    return v


def main() -> None:
    if OPEN_MARKER.exists():
        raise SystemExit('Locked final holdout has already been opened; refusing to score it again')
    freeze = json.loads(FREEZE.read_text())
    _verify_freeze(freeze)
    cfg = M2Config(); m4cfg = M4Config()
    states = pd.read_pickle(STATES_PATH)
    splits = pd.read_csv(REPORTS / 'm2_call_splits.csv')
    splits['ground_truth_time'] = pd.to_datetime(splits['ground_truth_time'], errors='raise')
    final_calls = splits[splits.m2_split.eq('final_chronological_test')].copy()
    if final_calls.session_id.astype(str).tolist() != [x for x in splits[splits.m2_split.eq('final_chronological_test')].session_id.astype(str).tolist()]:
        raise AssertionError('Unexpected final call ordering')
    if set(final_calls.session_id.astype(str)) != set(freeze['locked_final_call_ids']):
        raise AssertionError('Final call IDs differ from freeze')
    panel = _normalized_panel(states, splits)
    final_panel = panel[panel.session_id.astype(str).isin(set(freeze['locked_final_call_ids'])) & panel.scope_inbound_approach.astype(bool)].copy()
    if final_panel.empty:
        raise AssertionError('No frozen final approach rows')
    starts = _voyage_start_lookup()

    rows = []
    model_manifest = []
    for port in ['CATANIA','AUGUSTA']:
        train_calls = splits[(splits.port.eq(port)) & splits.m2_split.eq('development')].copy()
        port_final = final_calls[final_calls.port.eq(port)].copy()
        model = fit_train_only_route_model(states, train_calls, starts, cfg)
        if set(model['training_call_ids']) & set(freeze['locked_final_call_ids']):
            raise AssertionError('Final call leaked into final route model training')
        train_vessels = set(train_calls.mmsi.astype(int))
        model_manifest.append({
            'port': port, 'training_calls': len(train_calls), 'training_unique_vessels': len(train_vessels),
            'training_max_arrival': str(train_calls.ground_truth_time.max()),
            'final_calls': len(port_final), 'family_counts': json.dumps(model['family_counts'], sort_keys=True),
        })
        q = final_panel[final_panel.port.eq(port)].copy()
        for obs in q.itertuples(index=False):
            route = predict_route_distances(model, float(obs.lon_clean), float(obs.lat_clean), cfg)
            if abs(route['geodesic_km'] - float(obs.geodesic_distance_km)) > 0.25:
                raise AssertionError(f'Final geodesic contract mismatch: {obs.session_id}')
            speed = max(float(obs.sog_median_30m_kn), cfg.speed_floor_kn)
            rows.append({
                'port':port,'session_id':str(obs.session_id),'mmsi':int(obs.mmsi),'name':obs.name,
                'ground_truth_time':pd.Timestamp(obs.ground_truth_time),'decision_time':pd.Timestamp(obs.decision_time),
                'source_observation_time':pd.Timestamp(obs.source_observation_time),'true_tta_h':float(obs.true_tta_h),
                'horizon_band':str(obs.horizon_band),'sog_median_30m_kn':float(obs.sog_median_30m_kn),
                'lon_clean':float(obs.lon_clean),'lat_clean':float(obs.lat_clean),'geodesic_distance_km':float(obs.geodesic_distance_km),
                'route_knn_distance_km':float(route['knn_route_km']),'knn_mean_cross_track_km':float(route['knn_mean_cross_track_km']),
                'route_family':route['current_route_family'],'knn_neighbor_sessions':route['knn_neighbor_sessions'],
                'knn_neighbor_mmsi':route['knn_neighbor_mmsi'],'cold_vessel_final':int(obs.mmsi) not in train_vessels,
                'pred_m2_geodesic_h':(float(obs.geodesic_distance_km)/1.852)/speed,
                'pred_m2_route_knn_h':(float(route['knn_route_km'])/1.852)/speed,
            })
    pred = pd.DataFrame(rows).sort_values(['ground_truth_time','session_id','decision_time'], kind='mergesort').reset_index(drop=True)
    pred.to_csv(REPORTS / 'm6_final_predictions_raw.csv', index=False)
    pd.DataFrame(model_manifest).to_csv(REPORTS / 'm6_final_model_manifest.csv', index=False)

    # Frozen M4 reliability, interval and stability.
    pred = enrich_with_source_state(pred, states, m4cfg)
    pred['pred_raw_h'] = pred['pred_route_physics_h'].astype(float)
    pred['pred_raw_arrival'] = pd.to_datetime(pred['decision_time']) + pd.to_timedelta(pred['pred_raw_h'], unit='h')
    alpha = float(freeze['design']['stability']['alpha'])
    pred = apply_causal_arrival_ewma(pred, 'pred_raw_h', alpha, m4cfg.smoothing_reset_gap_h, 'pred_stabilized_h', 'pred_stabilized_arrival')
    q = float(freeze['design']['uncertainty']['simultaneous_call_level_half_width_h'])
    high = add_symmetric_interval(pred[pred.reliability_tier.eq('HIGH')].copy(), 'pred_raw_h', q, prefix='m6_pi')
    pi_cols = ['session_id','decision_time','m6_pi_lower_tta_h','m6_pi_upper_tta_h','m6_pi_effective_width_h','m6_pi_covered','m6_pi_lower_arrival','m6_pi_upper_arrival']
    pred = pred.merge(high[pi_cols], on=['session_id','decision_time'], how='left', validate='one_to_one')
    pred['m6_interval_status'] = np.where(pred.reliability_tier.eq('HIGH'), 'FROZEN_M4_90PCT_SIMULTANEOUS', 'WITHHELD_NON_HIGH_RELIABILITY')
    pred.to_csv(REPORTS / 'm6_final_predictions.csv', index=False)

    # Descriptive final metrics; no selection occurs here.
    metric_rows = []
    scopes = [('under6',6.0),('under12',12.0),('under24',24.0),('all',None)]
    for label, maxh in scopes:
        qdf = pred if maxh is None else pred[pred.true_tta_h.le(maxh)]
        if qdf.empty: continue
        m = eta_metrics(qdf, ['pred_m2_geodesic_h','pred_m2_route_knn_h'])
        m['scope'] = label; m['port'] = 'ALL'; metric_rows.append(m)
        for port, g in qdf.groupby('port'):
            z = eta_metrics(g, ['pred_m2_geodesic_h','pred_m2_route_knn_h']); z['scope']=label; z['port']=port; metric_rows.append(z)
    metrics = pd.concat(metric_rows, ignore_index=True)
    metrics.to_csv(REPORTS / 'm6_final_eta_metrics.csv', index=False)

    grouped = grouped_final_metrics(pred, ['port'], 24.0)
    grouped.to_csv(REPORTS / 'm6_final_port_metrics.csv', index=False)
    grouped_cold = grouped_final_metrics(pred, ['port','cold_vessel_final'], 24.0)
    grouped_cold.to_csv(REPORTS / 'm6_final_cold_vessel_metrics.csv', index=False)
    rel = final_reliability_metrics(pred); rel.to_csv(REPORTS / 'm6_final_reliability_metrics.csv', index=False)
    interval = final_interval_metrics(pred)
    stability = final_stability_metrics(pred)

    # Horizon support by independent calls.
    horizons = []
    for label, maxh in [('<=6h',6),('<=12h',12),('<=24h',24),('>24h',None)]:
        g = pred[pred.true_tta_h.le(maxh)] if maxh is not None else pred[pred.true_tta_h.gt(24)]
        horizons.append({'horizon':label,'rows':len(g),'calls':g.session_id.nunique(),'unique_vessels':g.mmsi.nunique()})
    pd.DataFrame(horizons).to_csv(REPORTS / 'm6_final_horizon_support.csv', index=False)

    primary = metrics[(metrics.scope.eq('under24')) & metrics.port.eq('ALL')].set_index('baseline')
    geo_mae = float(primary.loc['pred_m2_geodesic_h','voyage_balanced_mae_h'])
    route_mae = float(primary.loc['pred_m2_route_knn_h','voyage_balanced_mae_h'])
    summary = {
        'status':'M6_FINAL_HOLDOUT_OPENED_ONCE',
        'locked_final_calls_scored':int(pred.session_id.nunique()),
        'final_unique_vessels':int(pred.mmsi.nunique()),
        'final_rows':int(len(pred)),
        'final_calls_by_port':pred.groupby('port').session_id.nunique().astype(int).to_dict(),
        'cold_final_calls':int(pred.groupby('session_id').cold_vessel_final.first().sum()),
        'primary_under24':{
            'geodesic_mae_h':geo_mae,'route_mae_h':route_mae,
            'route_improvement_fraction':float(1-route_mae/geo_mae) if geo_mae>0 else None,
            'calls':int(pred[pred.true_tta_h.le(24)].session_id.nunique()),
        },
        'reliability':rel.to_dict(orient='records'),
        'interval_high_reliability':interval,
        'stability':stability,
        'freeze_sha256':file_sha256(FREEZE),
        'no_post_holdout_tuning':True,
    }
    (REPORTS / 'm6_final_summary.json').write_text(json.dumps(_jsonify(summary), indent=2))
    OPEN_MARKER.write_text(json.dumps({
        'opened':True,'freeze_sha256':file_sha256(FREEZE),'scored_call_ids':sorted(pred.session_id.astype(str).unique().tolist()),
        'post_open_rule':freeze['post_open_rule'],
    }, indent=2))
    print(json.dumps(_jsonify(summary), indent=2))


if __name__ == '__main__':
    main()
