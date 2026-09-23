#!/usr/bin/env python3
"""Freeze the M6 final-evaluation design before opening the locked holdout."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from ais_eta.m6 import file_sha256, frozen_design  # noqa: E402

REPORTS = ROOT / 'reports'
FREEZE = REPORTS / 'M6_FREEZE.json'

FROZEN_INPUTS = [
    'src/ais_eta/m2.py',
    'src/ais_eta/m4.py',
    'src/ais_eta/m6.py',
    'scripts/33_m6_final_eval.py',
    'reports/m2_call_splits.csv',
    'reports/m2_oof_route_predictions.csv',
    'reports/m4_summary.json',
    'reports/m4_calibration_call_scores.csv',
    'reports/m5_claim_scope.csv',
    'reports/catania_m1_decision_panel.csv',
    'reports/augusta_m1x_decision_panel.csv',
    'reports/catania_m0e_eta_primary_cohort.csv',
    'reports/augusta_m1x_eta_primary_cohort.csv',
    'data/derived/m0c_ship_states.pkl.gz',
]


def main() -> None:
    if FREEZE.exists():
        raise SystemExit('M6 freeze already exists; refusing to overwrite it')
    m4 = json.loads((REPORTS / 'm4_summary.json').read_text())
    half_width = float(m4['primary_interval']['half_width_h'])
    alpha = float(m4['stability']['selected_alpha_from_calibration'])
    splits = __import__('pandas').read_csv(REPORTS / 'm2_call_splits.csv')
    final = splits[splits.m2_split.eq('final_chronological_test')].copy()
    if len(final) != 16:
        raise AssertionError(f'Expected 16 final calls, found {len(final)}')

    missing = [p for p in FROZEN_INPUTS if not (ROOT / p).exists()]
    if missing:
        raise FileNotFoundError(missing)
    hashes = {p: file_sha256(ROOT / p) for p in FROZEN_INPUTS}
    head = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    payload = {
        'freeze_version': 2,
        'freeze_parent_git_commit': head,
        'frozen_before_final_scoring': True,
        'locked_final_call_count': int(len(final)),
        'locked_final_call_ids': final.sort_values(['ground_truth_time','port','session_id']).session_id.astype(str).tolist(),
        'locked_final_by_port': final.groupby('port').size().astype(int).to_dict(),
        'design': frozen_design(half_width, alpha),
        'claim_scope_before_final': __import__('pandas').read_csv(REPORTS/'m5_claim_scope.csv').to_dict(orient='records'),
        'sha256': hashes,
        'post_open_rule': 'No changes to M2/M4 predictor, route fitting, reliability thresholds, interval width, smoothing alpha, target, primary metric or final cohort may be made in response to final-holdout results.',
    }
    FREEZE.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
