from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .m2 import M2Config, eta_metrics
from .m4 import M4Config, interval_metrics, revision_metrics, voyage_balanced_mae_h
from .m5 import paired_bootstrap_mean_gain


def file_sha256(path: Path) -> str:
    h = sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def frozen_design(m4_half_width_h: float, stabilizer_alpha: float) -> dict:
    m2 = M2Config()
    m4 = M4Config()
    return {
        'point_model': 'M2 route-kNN remaining distance / max(robust trailing 30m median SOG, 1 kn)',
        'geodesic_baseline': 'geodesic remaining distance / max(robust trailing 30m median SOG, 1 kn)',
        'm2_config': asdict(m2),
        'reliability': {
            'max_supported_pred_h': m4.max_supported_pred_h,
            'max_route_cross_track_km': m4.max_route_cross_track_km,
            'max_provider_age_s': m4.max_provider_age_s,
            'min_robust_speed_kn': m4.min_robust_speed_kn,
            'requires_destination_support': True,
        },
        'uncertainty': {
            'confidence_level': m4.confidence_level,
            'simultaneous_call_level_half_width_h': float(m4_half_width_h),
            'emit_only_for_reliability_tier': 'HIGH',
        },
        'stability': {
            'method': 'causal EWMA on absolute predicted arrival timestamp',
            'alpha': float(stabilizer_alpha),
            'reset_gap_h': m4.smoothing_reset_gap_h,
        },
        'primary_final_metric': 'voyage-balanced MAE <=24 h',
        'primary_final_scope': 'scope_inbound_approach rows from the precomputed 15-minute causal decision panels',
        'final_test_rule': 'score 16 locked chronological calls exactly once; no predictor/threshold/config change afterward',
    }


def call_level_errors(df: pd.DataFrame, pred_cols: Iterable[str], max_true_h: float | None = 24.0) -> pd.DataFrame:
    x = df.copy()
    if max_true_h is not None:
        x = x[x['true_tta_h'].astype(float).le(float(max_true_h))]
    rows = []
    for sid, g in x.groupby('session_id'):
        row = {
            'session_id': str(sid),
            'port': str(g['port'].iloc[0]),
            'mmsi': int(g['mmsi'].iloc[0]),
            'name': str(g['name'].iloc[0]),
            'points': int(len(g)),
        }
        for pred in pred_cols:
            row[f'{pred}_mae_h'] = float((g[pred].astype(float) - g['true_tta_h'].astype(float)).abs().mean())
        rows.append(row)
    return pd.DataFrame(rows)


def grouped_final_metrics(df: pd.DataFrame, group_cols: list[str], max_true_h: float = 24.0) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        q = g[g['true_tta_h'].astype(float).le(float(max_true_h))].copy()
        if q.empty:
            continue
        geo = eta_metrics(q, ['pred_m2_geodesic_h']).iloc[0]
        route = eta_metrics(q, ['pred_m2_route_knn_h']).iloc[0]
        call = call_level_errors(q, ['pred_m2_geodesic_h', 'pred_m2_route_knn_h'], max_true_h=None)
        gains = call['pred_m2_geodesic_h_mae_h'] - call['pred_m2_route_knn_h_mae_h']
        mean_gain, lo, hi = paired_bootstrap_mean_gain(gains, draws=10000, seed=20260918 + len(rows))
        row = {c: k for c, k in zip(group_cols, keys)}
        row.update({
            'calls': int(q['session_id'].nunique()),
            'unique_vessels': int(q['mmsi'].nunique()),
            'prediction_points': int(len(q)),
            'geodesic_mae_h': float(geo['voyage_balanced_mae_h']),
            'route_mae_h': float(route['voyage_balanced_mae_h']),
            'route_improvement_fraction': float(1.0 - route['voyage_balanced_mae_h'] / geo['voyage_balanced_mae_h']) if geo['voyage_balanced_mae_h'] > 0 else np.nan,
            'paired_gain_mean_h': float(mean_gain),
            'paired_gain_ci95_low_h': float(lo),
            'paired_gain_ci95_high_h': float(hi),
            'route_better_call_fraction': float((gains > 0).mean()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def final_reliability_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tier, g in df.groupby('reliability_tier'):
        rows.append({
            'reliability_tier': str(tier),
            'rows': int(len(g)),
            'calls': int(g['session_id'].nunique()),
            'raw_mae_under24_h': voyage_balanced_mae_h(g, 'pred_raw_h', 24.0),
            'stabilized_mae_under24_h': voyage_balanced_mae_h(g, 'pred_stabilized_h', 24.0),
        })
    return pd.DataFrame(rows)


def final_interval_metrics(df: pd.DataFrame) -> dict:
    high = df[df['reliability_tier'].eq('HIGH') & df['m6_pi_covered'].notna()].copy()
    if high.empty:
        return interval_metrics(high, 'm6_pi_covered', 'm6_pi_effective_width_h')
    return interval_metrics(high, 'm6_pi_covered', 'm6_pi_effective_width_h')


def final_stability_metrics(df: pd.DataFrame) -> dict:
    raw = revision_metrics(df, 'pred_raw_arrival')
    smooth = revision_metrics(df, 'pred_stabilized_arrival')
    return {
        'raw': raw,
        'stabilized': smooth,
        'raw_under24_mae_h': voyage_balanced_mae_h(df, 'pred_raw_h', 24.0),
        'stabilized_under24_mae_h': voyage_balanced_mae_h(df, 'pred_stabilized_h', 24.0),
    }
