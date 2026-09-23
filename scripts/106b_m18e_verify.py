#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
sys.path.insert(0, str(ROOT / "src"))
from ais_eta.m18e import sha256_file, validate_ablation_audit, validate_source_registry  # noqa: E402


def main() -> int:
    audit = json.loads((R / "M18E_EXTERNAL_ABLATION.json").read_text())
    freeze = json.loads((R / "M18E_ABLATION_FREEZE.json").read_text())
    state = json.loads((ROOT / "PROJECT_STATE.json").read_text())
    validate_ablation_audit(audit)

    assert freeze["development_population"] == 386
    assert freeze["blocked_old_final_population"] == 53
    assert freeze["fresh_holdout_opened"] is False
    assert freeze["model_change"] is False
    assert freeze["new_external_model_promoted"] is False
    assert sha256_file(R / "M18A_BASELINE_FREEZE.json") == freeze["m18a_baseline_freeze_sha256"]
    assert sha256_file(R / "M18B_VALIDATION_FREEZE.json") == freeze["m18b_validation_freeze_sha256"]
    assert sha256_file(R / "M18C_AUDIT_FREEZE.json") == freeze["m18c_audit_freeze_sha256"]
    assert sha256_file(R / "M18D_ATTRIBUTION_FREEZE.json") == freeze["m18d_attribution_freeze_sha256"]
    assert sha256_file(R / "m16g_oof_predictions.csv") == freeze["m16g_oof_sha256"]
    for rel, digest in freeze["artifact_sha256"].items():
        assert sha256_file(ROOT / rel) == digest, rel

    registry = pd.read_csv(R / "m18e_external_source_registry.csv")
    validate_source_registry(registry)
    assert int(registry.status.eq("READY_NEW_DATA_PINNED").sum()) == 0
    assert set(registry.information_family).issuperset({"destination_port_intent", "navigable_routing", "weather_ocean", "port_operations"})

    abl = pd.read_csv(R / "m18e_ablation_results.csv")
    assert set(abl.ablation_id) == {"DEST_CANONICALIZATION", "DEST_GEOMETRY_PLUS_CAUSAL_PHYSICS"}
    assert {"STRICT_386", "NEAR_TERM_FUTURE_0_7D", "PHYSICS_ELIGIBLE", "PHYSICS_ELIGIBLE_NEAR_TERM"}.issubset(set(abl.scope))
    assert state["current_phase"].startswith(("M18E_", "M18F_", "M18G_", "M19_"))
    assert state["m18e_summary"]["fresh_holdout_opened"] is False
    assert state["m18e_summary"]["new_external_model_promoted"] is False

    print("PASS M18E external-information ablation lab and frozen hashes")
    print("PASS M16G/M16H + M18A-D frozen; 386 development / 53 old-final blocked")
    print("PASS retrospective destination/physics component ablations reproduced")
    print("PASS S-100/searoute/weather/port-ops blockers are explicit and unscored")
    print("PASS fresh holdout remains sealed; no new external model promoted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
