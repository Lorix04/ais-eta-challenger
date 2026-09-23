#!/usr/bin/env python3
"""Full-data verification for the M0C causal vessel-state layer."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
summary = json.loads((ROOT / 'reports/m0c_state_summary.json').read_text())
baseline = json.loads((ROOT / 'reports/audit_summary.json').read_text())
path = ROOT / summary['derived_file']

assert path.exists(), f'Missing derived M0C layer: {path}'
assert summary['rows'] == baseline['ship_rows'] == 969084
assert summary['mmsi'] == baseline['ship_mmsi'] == 1129
assert summary['prefix_invariance_all_pass'] is True
assert len(summary['prefix_invariance_checks']) >= 10
assert summary['sentinels']['sentinel_counts']['heading_unavailable_511'] == 256354
assert summary['sentinels']['sentinel_counts']['cog_unavailable_ge_360'] == 97059
assert summary['sentinels']['sentinel_counts']['sog_unavailable_102_3'] == 48
assert summary['sentinels']['sentinel_counts']['sog_capped_102_2'] == 0

# Load the local binary layer to verify integrity/schema/order.  This intentionally
# happens outside pytest because the file is private/large and gitignored.
df = pd.read_pickle(path, compression='gzip')
required = {
    'sog_clean','cog_clean','heading_clean','draught_clean','nav_status_clean',
    'observation_dt_s','position_delta_m','implied_speed_kn','dynamic_state_hash',
    'dynamic_state_changed','dynamic_state_age_s','kinematic_conflict',
    'stop_candidate','slow_motion_candidate','turn_candidate',
    'observation_gap_candidate','stop_duration_s','stale_risk_level',
    'motion_state','semantic_events'
}
missing = sorted(required - set(df.columns))
assert not missing, f'Missing derived columns: {missing}'
assert len(df) == 969084
assert df['mmsi'].nunique() == 1129
assert df['recorded_at'].notna().all()
# Within each MMSI, time must be nondecreasing after the stable sort.
assert df.groupby('mmsi', sort=False)['recorded_at'].apply(lambda s: s.is_monotonic_increasing).all()
# Repeated provider snapshots are preserved, not removed.
assert df['dynamic_state_repeated'].mean() > 0.40
# Observation gaps must break stop-duration continuity whenever a stop is seen after the gap.
check = df[df['observation_gap_candidate'] & df['stop_candidate']]
assert (check['stop_duration_s'] == 0).all()
# No raw sentinel may survive in the corresponding normalized values.
assert not (df['heading_clean'] == 511).any()
assert not (df['cog_clean'] >= 360).any()
assert not (df['sog_clean'] > 102.2).any()
assert not (df['draught_clean'] <= 0).any()

print('PASS M0C full-data verification')
print(f"rows={len(df):,} mmsi={df['mmsi'].nunique():,}")
print(f"dynamic_state_repeated_pct={df['dynamic_state_repeated'].mean():.4%}")
print(f"kinematic_conflicts={int(df['kinematic_conflict'].sum()):,}")
print(f"observation_gaps={int(df['observation_gap_candidate'].sum()):,}")
print(f"stop_starts={int(df['stop_start'].sum()):,}")
print(f"turn_candidates={int(df['turn_candidate'].sum()):,}")
print('prefix_invariance=PASS')
