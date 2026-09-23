#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'reports'

def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main() -> int:
    s = json.loads((R/'M17C_SUMMARY.json').read_text())
    f = json.loads((R/'M17C_CORRIDOR_GRAPH_FREEZE.json').read_text())
    led = pd.read_csv(R/'m17c_oof_predictions.csv')
    build = pd.read_csv(R/'m17c_graph_build_audit.csv')
    manifest = json.loads((R/'M16A_DEVELOPMENT_MANIFEST.json').read_text())
    assert len(led) == 386 and led.mmsi.nunique() == 386
    assert set(led.mmsi.astype(int)).isdisjoint(manifest['blocked_old_final_mmsi'])
    assert s['old_final_rows_hard_blocked'] == 53
    assert s['final_test_used_for_selection'] is False
    assert s['outer_valid_owners_used_for_graph'] is False
    assert s['future_events_used_for_query_graph'] is False
    assert s['target_used_for_graph_topology_or_edge_weight'] is False
    assert int(build['outer_valid_owner_overlap'].sum()) == 0
    assert int(build['old_final_owner_overlap'].sum()) == 0
    assert s['graph_supported_rows'] >= 20
    assert s['gate'] in {'PASS_CORRIDOR_SIGNAL_COMPONENT','NO_GRAPH_PROMOTION'}
    assert f['immutable_inputs']['m17b_freeze'] == sha(R/'M17B_HISTORICAL_MEMORY_FREEZE.json')
    assert f['immutable_inputs']['m17a_freeze'] == sha(R/'M17A_SSL_RETRIEVAL_FREEZE.json')
    assert f['immutable_inputs']['m16j_freeze'] == sha(R/'M16_FINAL_FREEZE.json')
    assert f['immutable_inputs']['m16e_predictions'] == sha(R/'m16e_oof_predictions.csv')
    for name, h in f['artifact_sha256'].items():
        assert sha(R/name) == h, name
    print(f"PASS M17C development-only {len(led)}; old-final {len(manifest['blocked_old_final_mmsi'])} hard-blocked")
    print(f"PASS graph support {s['graph_supported_rows']}/{s['physics_eligible_rows']}; outer-valid overlap 0; future events blocked")
    print(f"PASS supported MAE {s['graph_supported_mae_h']:.3f} h vs physics {s['physics_same_rows_mae_h']:.3f} h")
    print(f"PASS full graph-gate gain {s['graph_gate_gain_h_vs_m16e_hard_gate']:.4f} h; folds {s['graph_gate_fold_wins_vs_m16e_hard_gate']}/5")
    print(f"PASS gate {s['gate']}")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
