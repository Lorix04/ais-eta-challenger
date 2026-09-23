from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m0e import assemble_audited_calls, audit_summary, build_session_diagnostics


def main() -> int:
    candidates = pd.read_csv(ROOT / "reports" / "catania_m0d_candidate_calls.csv")
    audit = pd.read_csv(ROOT / "config" / "catania_m0e_manual_audit.csv")
    roles = pd.read_csv(ROOT / "config" / "catania_m0e_vessel_roles.csv")
    states = pd.read_pickle(ROOT / "data" / "derived" / "m0c_ship_states.pkl.gz")

    diagnostics = build_session_diagnostics(candidates, states)
    audited = assemble_audited_calls(candidates, audit, roles, diagnostics)
    summary = audit_summary(audited)

    reports = ROOT / "reports"
    diagnostics.to_csv(reports / "catania_m0e_session_diagnostics.csv", index=False)
    audited.to_csv(reports / "catania_m0e_audited_calls.csv", index=False)
    (reports / "m0e_catania_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    core = audited[audited["eta_primary_eligible"]].copy()
    core.to_csv(reports / "catania_m0e_eta_primary_cohort.csv", index=False)

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
