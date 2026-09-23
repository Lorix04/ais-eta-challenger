#!/usr/bin/env python3
"""Fast raw-data smoke test for M8 portability; does not rebuild the full 969k-row state layer."""
from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ais_eta.m0c import engineer_semantic_states, M0CConfig  # noqa: E402

USECOLS = ['id','mmsi','name','category','nav_status','sog','cog','heading','lat','lon','destination','draught','recorded_at','created_at','updated_at']


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, default=ROOT/'data')
    ap.add_argument('--rows-per-file', type=int, default=5000)
    args=ap.parse_args()
    parts=[]
    for name in ['vessel_positions_part1.csv','vessel_positions_part2.csv']:
        p=args.data_dir/name
        if not p.exists():
            raise SystemExit(f'FAIL missing raw file: {p}')
        d=pd.read_csv(p,usecols=USECOLS,nrows=args.rows_per_file,low_memory=False)
        parts.append(d[d.category.eq('ship')])
    raw=pd.concat(parts,ignore_index=True)
    if raw.empty: raise SystemExit('FAIL no ship rows in smoke sample')
    states=engineer_semantic_states(raw, M0CConfig())
    assert len(states)==len(raw)
    assert 'dynamic_state_age_s' in states
    assert 'motion_state' in states
    assert not (states['heading_clean']==511).any()
    assert not (states['cog_clean']>=360).any()
    print(f'PASS raw-data smoke rows={len(states):,} mmsi={states.mmsi.nunique():,}')
    print(f'PASS repeated_state_fraction={states.dynamic_state_repeated.mean():.3f}')
    print('PASS sentinel normalization and causal semantic-state construction')
    return 0

if __name__=='__main__': raise SystemExit(main())
