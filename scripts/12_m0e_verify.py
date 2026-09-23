from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def fail(message: str) -> None:
    raise SystemExit(f"FAIL {message}")


def main() -> int:
    audit = pd.read_csv(ROOT / "config" / "catania_m0e_manual_audit.csv")
    audited = pd.read_csv(ROOT / "reports" / "catania_m0e_audited_calls.csv")
    core = pd.read_csv(ROOT / "reports" / "catania_m0e_eta_primary_cohort.csv")
    summary = json.loads((ROOT / "reports" / "m0e_catania_summary.json").read_text())
    atlas_pages = sorted((ROOT / "reports" / "m0e_audit_atlas").glob("page_*.png"))

    if len(audit) != 61 or audit["session_id"].nunique() != 61:
        fail("manual audit must contain exactly 61 unique candidate sessions")
    if len(audited) != 61 or audited["session_id"].nunique() != 61:
        fail("audited ledger must contain exactly 61 unique sessions")

    kept = audited[audited["final_ground_truth"].astype(bool)]
    if len(kept) != 51:
        fail(f"expected 51 validated calls, found {len(kept)}")
    if len(audited) - len(kept) != 10:
        fail("expected 10 excluded/ambiguous sessions")

    conf = kept["ground_truth_confidence"].value_counts().to_dict()
    if conf != {"A": 37, "B": 14}:
        fail(f"unexpected retained confidence counts: {conf}")

    if len(core) != 28 or core["mmsi"].nunique() != 13:
        fail(f"primary ETA cohort expected 28 calls / 13 vessels, found {len(core)} / {core['mmsi'].nunique()}")
    if not core["final_ground_truth"].astype(bool).all():
        fail("primary ETA cohort contains rows outside final ground truth")
    if not core["eta_primary_eligible"].astype(bool).all():
        fail("primary ETA cohort contains rows not marked eta_primary_eligible")

    uncertainty = kept["ground_truth_uncertainty_s"].astype(float)
    if float(uncertainty.median()) != 61.0:
        fail(f"unexpected median crossing bracket: {uncertainty.median()}")
    if float(uncertainty.quantile(0.95)) > 62.0:
        fail(f"unexpected P95 crossing bracket: {uncertainty.quantile(0.95)}")

    if summary.get("validated_port_call_entries") != 51:
        fail("summary JSON disagrees with audited ledger")
    if summary.get("primary_eta_eligible_calls") != 28:
        fail("summary JSON disagrees with ETA cohort")
    if len(atlas_pages) != 7:
        fail(f"visual audit atlas expected 7 pages, found {len(atlas_pages)}")

    state = json.loads((ROOT / "PROJECT_STATE.json").read_text())
    if state.get("current_phase") != "M1":
        fail(f"project pointer should advance to M1 after M0E, found {state.get('current_phase')}")
    if state.get("go_no_go_A", {}).get("status") != "GO_LIMITED":
        fail("GO/NO-GO A must be GO_LIMITED after M0E")

    print("PASS M0E manual audit covers all 61 candidate sessions exactly once")
    print("PASS validated truth: 51 kept / 10 excluded")
    print("PASS retained confidence: A=37 / B=14")
    print("PASS primary ETA cohort: 28 calls / 13 unique vessels")
    print("PASS ETA cohort is a strict subset of final ground truth")
    print("PASS crossing-bracket uncertainty median=61s and P95<=62s")
    print("PASS seven-page visual audit atlas exists")
    print("PASS project pointer advanced to M1 with GO_LIMITED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
