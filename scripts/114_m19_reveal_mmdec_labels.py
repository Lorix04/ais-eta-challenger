#!/usr/bin/env python3
"""Reveal MMDEC ETA labels only after the blind prediction ledger was sealed."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ais_eta.m18g import sha256_file, validate_label_ledger
from ais_eta.m19 import reveal_mmdec_label_ledger
from ais_eta.m19_mmdec_stream import reveal_labels_from_parquet


def load_table(path: Path) -> pd.DataFrame:
    s=path.name.lower()
    if s.endswith('.csv') or s.endswith('.csv.gz'): return pd.read_csv(path)
    if s.endswith('.pkl') or s.endswith('.pkl.gz'): return pd.read_pickle(path)
    if s.endswith('.parquet'):
        try: return pd.read_parquet(path)
        except ImportError as exc: raise SystemExit('Parquet support requires pyarrow or fastparquet.') from exc
    raise SystemExit(f'unsupported table: {path}')


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--spec', required=True)
    ap.add_argument('--cohort-locator', required=True)
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--opening-manifest', required=True)
    ap.add_argument('--output', required=True)
    args=ap.parse_args()
    out=Path(args.output)
    if out.exists(): raise SystemExit(f'refusing to overwrite revealed label ledger: {out}')
    opening=json.loads(Path(args.opening_manifest).read_text())
    if opening.get('status') != 'SEALED_BEFORE_LABEL_REVEAL': raise SystemExit('prediction ledger must be sealed before M19 label reveal')
    if opening.get('labels_observed_when_prediction_ledger_sealed') is not False: raise SystemExit('opening manifest indicates labels were observed')
    if opening.get('prediction_ledger_sha256') != sha256_file(args.predictions): raise SystemExit('prediction ledger hash mismatch before label reveal')
    locator=pd.read_csv(args.cohort_locator)
    pred=pd.read_csv(args.predictions)
    if set(locator.sample_key.astype(str)) != set(pred.sample_key.astype(str)): raise SystemExit('cohort locator and blind predictions must have identical sample keys')
    spec_path=Path(args.spec)
    if spec_path.name == 'Dataset_AIS_SPEC.parquet':
        labels=reveal_labels_from_parquet(spec_path, locator)
    else:
        labels=reveal_mmdec_label_ledger(load_table(spec_path), locator)
    validate_label_ledger(labels[['sample_key','target_tte_h']])
    out.parent.mkdir(parents=True, exist_ok=True)
    labels[['sample_key','target_tte_h']].to_csv(out,index=False,float_format='%.12g')
    Path(str(out)+'.sha256').write_text(f"{sha256_file(out)}  {out.name}\n")
    audit=out.with_suffix('.audit.json')
    audit.write_text(json.dumps({
        'status':'M19_LABELS_REVEALED_AFTER_BLIND_LEDGER_SEAL', 'rows':len(labels),
        'prediction_ledger_sha256':sha256_file(args.predictions), 'label_ledger_sha256':sha256_file(out),
        'reference_eta_status_counts':labels.reference_eta_status.value_counts().sort_index().to_dict(),
    }, indent=2)+'\n')
    print(f'PASS M19 labels revealed rows: {len(labels)}')
    print(f'PASS label ledger SHA-256: {sha256_file(out)}')
    return 0

if __name__=='__main__': raise SystemExit(main())
