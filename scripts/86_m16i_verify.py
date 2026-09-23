#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'reports'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 s=json.loads((R/'M16I_SUMMARY.json').read_text()); f=json.loads((R/'M16I_RED_TEAM_FREEZE.json').read_text())
 a=pd.read_csv(R/'m16i_row_audit.csv'); ab=pd.read_csv(R/'m16i_ablation_metrics.csv'); st=pd.read_csv(R/'m16i_subgroup_stress.csv')
 manifest=json.loads((R/'M16A_DEVELOPMENT_MANIFEST.json').read_text()); blocked=set(manifest['blocked_old_final_mmsi'])
 assert len(a)==386 and a.mmsi.nunique()==386 and set(a.mmsi.astype(int)).isdisjoint(blocked)
 assert s['final_test_used'] is False and f['red_team_only_no_tuning'] is True
 assert s['gate'] in {'PASS_RED_TEAM_STABLE','PASS_RED_TEAM_WITH_HEAVY_TAIL_CAVEAT'}
 assert s['fold_wins_vs_best_single']>=4 and s['leave_one_fold_positive']==5
 assert s['trimmed_5pct_abs_target_gain_h']>0 and s['winsorized_95_gain_h']>0
 assert s['gain_after_removing_largest_net_destination_h']>0
 assert s['bootstrap_positive_probability']>=.90
 assert .78<=s['m16h_coverage']<=.86 and s['confidence_ordered'] is True
 assert {'no_physics_structural','no_route_structural','no_canonical_destination_proxy','no_longitudinal_history_proxy'}.issubset(set(ab.scenario))
 assert {'destination_support','destination_semantics','vessel_category','reference_eta_status_DIAGNOSTIC_ONLY','route_support','confidence_tier'}.issubset(set(st.group))
 for n,d in f['artifact_sha256'].items(): assert sha(R/n)==d,n
 for n,d in f['prior_freeze_sha256'].items(): assert sha(R/n)==d,n
 state=json.loads((ROOT/'PROJECT_STATE.json').read_text()); assert state['current_phase'].startswith(('M16I_','M16J_','M17','M18','M19'))
 print('PASS M16I development-only red-team / ablation / stress audit')
 print(f"PASS gate: {s['gate']}")
 print(f"PASS pooled gain: {s['pooled_gain_h']:.6f} h")
 print(f"PASS leave-one-fold positive: {s['leave_one_fold_positive']}/5")
 print(f"PASS paired bootstrap P(gain>0): {s['bootstrap_positive_probability']:.6f}")
 print('PASS old final blocked: 53/53')
 print(f"PASS frozen M16I artifacts: {len(f['artifact_sha256'])}")
 return 0
if __name__=='__main__': raise SystemExit(main())
