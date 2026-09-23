#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m18a import sha256_file  # noqa: E402
from ais_eta.m18c import validate_audit_dict  # noqa: E402


def main() -> int:
    audit = json.loads((R / "M18C_TARGET_LABEL_AUDIT.json").read_text())
    freeze = json.loads((R / "M18C_AUDIT_FREEZE.json").read_text())
    state = json.loads((ROOT / "PROJECT_STATE.json").read_text())
    validate_audit_dict(audit)

    assert freeze["model_change"] is False
    assert freeze["relabeling"] is False
    assert freeze["strict_rows_deleted"] == 0
    assert freeze["fresh_holdout_opened"] is False
    assert freeze["development_population"] == 386
    assert freeze["blocked_old_final_population"] == 53
    assert sha256_file(R / "M18A_BASELINE_FREEZE.json") == freeze["m18a_baseline_freeze_sha256"]
    assert sha256_file(R / "M18B_VALIDATION_FREEZE.json") == freeze["m18b_validation_freeze_sha256"]
    assert sha256_file(R / "m16g_oof_predictions.csv") == freeze["m16g_oof_sha256"]
    for rel, digest in freeze["audit_artifact_sha256"].items():
        assert sha256_file(ROOT / rel) == digest, rel

    f = pd.read_csv(R / "m18c_reference_forensics.csv")
    proxy = pd.read_csv(R / "m18c_historical_ais_proxy_alignment.csv")
    slices = pd.read_csv(R / "m18c_label_slice_diagnostics.csv")
    assert len(f) == 386 and f.mmsi.nunique() == 386
    assert int(f.reference_eta_status.eq("FUTURE_0_7D").sum()) == audit["findings"]["near_term_future_0_7d_rows"]
    assert len(proxy) == audit["findings"]["historical_ais_proxy_owner_overlap"]
    assert proxy.proxy_is_authoritative_ata.eq(False).all()
    assert proxy.proxy_alignment_is_posthoc_only.eq(True).all()
    assert set(slices["slice"]) == {"STRICT_ALL_386", "FUTURE_0_7D", "NOT_FUTURE_0_7D", "FUTURE_ONLY"}

    assert state["current_phase"].startswith(("M18C_", "M18D_", "M18E_", "M18F_", "M18G_", "M19_"))
    assert state["m18c_summary"]["model_changed"] is False
    assert state["m18c_summary"]["strict_rows_deleted"] == 0
    assert state["m18c_summary"]["fresh_holdout_opened"] is False

    print("PASS M18C machine-readable target/label audit")
    print("PASS 386 development labels retained; 53 old-final owners remain blocked")
    print(f"PASS near-term FUTURE_0_7D rows={audit['findings']['near_term_future_0_7d_rows']} MAE={audit['findings']['near_term_m16g_mae_h']:.2f}h")
    print(f"PASS non-near-term absolute-error share={100*audit['findings']['non_near_term_absolute_error_share']:.1f}%")
    print("PASS AIS research-gate alignment remains post-hoc diagnostic only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
