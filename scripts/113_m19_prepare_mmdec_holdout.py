#!/usr/bin/env python3
"""Prepare the target-free MMDEC external cohort for M18G1 blind scoring."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ais_eta.m18g import sha256_file
from ais_eta.m19 import M19Policy, M19_VERSION, M19_SOURCE_DOI, build_mmdec_target_free_holdout, verify_mmdec_source_files
from ais_eta.m19_mmdec_stream import build_target_free_from_parquet


def load_table(path: Path) -> pd.DataFrame:
    s = path.name.lower()
    if s.endswith('.csv') or s.endswith('.csv.gz'):
        return pd.read_csv(path)
    if s.endswith('.pkl') or s.endswith('.pkl.gz') or s.endswith('.pickle'):
        return pd.read_pickle(path)
    if s.endswith('.parquet'):
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise SystemExit("Parquet support requires pyarrow or fastparquet. Install pyarrow, or convert MMDEC Parquet to CSV without changing values.") from exc
    raise SystemExit(f"unsupported table: {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--positions', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--max-rows', type=int, default=500)
    ap.add_argument('--preselect-rows', type=int, default=2000)
    args = ap.parse_args()
    out = Path(args.output_dir)
    if out.exists():
        raise SystemExit(f"refusing to overwrite M19 cohort directory: {out}")
    out.mkdir(parents=True)
    spec_path, pos_path = Path(args.spec), Path(args.positions)
    source_integrity = verify_mmdec_source_files(pos_path, spec_path) if {pos_path.name, spec_path.name} == {'Dataset_AIS_POS.parquet','Dataset_AIS_SPEC.parquet'} else {
        pos_path.name: {'sha256': sha256_file(pos_path), 'bytes': pos_path.stat().st_size, 'published_md5_check': 'SKIPPED_NONCANONICAL_FILENAME'},
        spec_path.name: {'sha256': sha256_file(spec_path), 'bytes': spec_path.stat().st_size, 'published_md5_check': 'SKIPPED_NONCANONICAL_FILENAME'},
    }
    policy = M19Policy(max_rows=args.max_rows, preselect_rows=args.preselect_rows)
    if spec_path.name == 'Dataset_AIS_SPEC.parquet' and pos_path.name == 'Dataset_AIS_POS.parquet':
        # Canonical MMDEC: stream the two large Parquet files so the external
        # cohort can be prepared without loading ~19M position rows in memory.
        # This path also works in the clean-room environment where pyarrow is
        # intentionally unavailable.
        rows, states, locator, audit = build_target_free_from_parquet(spec_path, pos_path, policy)
        audit['parquet_reader'] = 'm19_parquet_lite_audited_streaming_v1'
    else:
        spec, pos = load_table(spec_path), load_table(pos_path)
        rows, states, locator, audit = build_mmdec_target_free_holdout(spec, pos, policy)
    rows_path = out/'m19_rows_unlabeled.pkl.gz'; states_path = out/'m19_states_unlabeled.pkl.gz'; loc_path = out/'m19_cohort_locator.csv'
    rows.to_pickle(rows_path, compression='gzip', protocol=5)
    states.to_pickle(states_path, compression='gzip', protocol=5)
    locator.to_csv(loc_path, index=False)
    manifest = {
        'milestone': 'M19', 'version': M19_VERSION, 'status': 'TARGET_FREE_COHORT_PREPARED',
        'source': {'name':'MMDEC v1','doi':M19_SOURCE_DOI,'integrity':source_integrity},
        'policy': policy.__dict__, 'audit': audit,
        'artifacts': {
            'rows': {'path': rows_path.name, 'sha256': sha256_file(rows_path)},
            'states': {'path': states_path.name, 'sha256': sha256_file(states_path)},
            'locator': {'path': loc_path.name, 'sha256': sha256_file(loc_path)},
        },
        'labels_materialized': False,
        'eta_values_exposed_to_scoring_rows': False,
    }
    mp = out/'M19_COHORT_PROVENANCE.json'; mp.write_text(json.dumps(manifest, indent=2)+'\n')
    (out/'M19_COHORT_PROVENANCE.json.sha256').write_text(f"{sha256_file(mp)}  {mp.name}\n")
    print(json.dumps({'rows':len(rows),'unique_mmsi':rows.mmsi.nunique(),'provenance_sha256':sha256_file(mp),'labels_materialized':False}, indent=2))
    return 0

if __name__ == '__main__': raise SystemExit(main())
