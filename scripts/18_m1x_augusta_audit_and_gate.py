from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m1 import M1Config
from ais_eta.m1x_audit import (
    M1XGateCriteria,
    assemble_augusta_audited_calls,
    build_augusta_confirmation_diagnostics,
    build_augusta_decision_panel,
    call_horizon_support,
    evaluate_m1x_evidence_gate,
)


def _known_non_primary_mmsi() -> set[int]:
    roles = pd.read_csv(ROOT / "config/catania_m0e_vessel_roles.csv")
    return set(roles.loc[~roles["eta_primary_scope"].astype(bool), "mmsi"].astype(int))


def _combined_catania_support():
    panel = pd.read_csv(ROOT / "reports/catania_m1_decision_panel.csv")
    q = panel[panel["scope_inbound_approach"].astype(bool)].copy()
    q["port"] = "CATANIA"
    hs = (
        q.groupby(["port", "session_id", "mmsi"], as_index=False)["true_tta_h"]
        .max()
        .rename(columns={"true_tta_h": "max_supported_horizon_h"})
    )
    cohort = pd.read_csv(ROOT / "reports/catania_m0e_eta_primary_cohort.csv")[["session_id", "mmsi", "name", "ground_truth_time"]].copy()
    cohort["port"] = "CATANIA"
    return cohort, hs


def main():
    reports = ROOT / "reports"
    states = pd.read_pickle(ROOT / "data/derived/m0c_ship_states.pkl.gz")
    candidates = pd.read_csv(reports / "augusta_m1x_candidate_calls.csv")
    audit = pd.read_csv(ROOT / "config/augusta_m1x_manual_audit.csv")
    confdiag = build_augusta_confirmation_diagnostics(candidates, states)
    confdiag.to_csv(reports / "augusta_m1x_confirmation_diagnostics.csv", index=False)

    audited = assemble_augusta_audited_calls(
        candidates,
        audit,
        confdiag,
        known_non_primary_mmsi=_known_non_primary_mmsi(),
    )
    audited.to_csv(reports / "augusta_m1x_audited_calls.csv", index=False)
    cohort = audited[audited["eta_primary_eligible"]].copy()
    cohort.to_csv(reports / "augusta_m1x_eta_primary_cohort.csv", index=False)

    events = pd.read_csv(reports / "augusta_m1x_gate_events.csv")
    panel = build_augusta_decision_panel(
        states,
        cohort,
        events,
        config=M1Config(decision_interval_min=15),
    )
    panel.to_csv(reports / "augusta_m1x_decision_panel.csv", index=False)
    aug_hs = call_horizon_support(panel)
    aug_hs.to_csv(reports / "augusta_m1x_horizon_support.csv", index=False)

    cat_calls, cat_hs = _combined_catania_support()
    aug_calls = cohort[["session_id", "mmsi", "name", "ground_truth_time"]].copy()
    aug_calls["port"] = "AUGUSTA"
    combined_calls = pd.concat([cat_calls, aug_calls], ignore_index=True)
    combined_calls.to_csv(reports / "m1x_combined_eta_call_inventory.csv", index=False)
    combined_hs = pd.concat([cat_hs, aug_hs], ignore_index=True)
    combined_hs.to_csv(reports / "m1x_combined_horizon_support.csv", index=False)

    criteria = M1XGateCriteria()
    gate = evaluate_m1x_evidence_gate(combined_calls, combined_hs, criteria)
    (reports / "m1x_gate_result.json").write_text(json.dumps(gate, indent=2))

    horizon_rows = []
    for port, g in combined_hs.groupby("port"):
        horizon_rows.append({
            "port": port,
            "calls_with_any_approach": int(len(g)),
            "calls_over_6h": int(g["max_supported_horizon_h"].gt(6).sum()),
            "calls_over_12h": int(g["max_supported_horizon_h"].gt(12).sum()),
            "calls_over_24h": int(g["max_supported_horizon_h"].gt(24).sum()),
        })
    g = combined_hs
    horizon_rows.append({
        "port": "COMBINED",
        "calls_with_any_approach": int(len(g)),
        "calls_over_6h": int(g["max_supported_horizon_h"].gt(6).sum()),
        "calls_over_12h": int(g["max_supported_horizon_h"].gt(12).sum()),
        "calls_over_24h": int(g["max_supported_horizon_h"].gt(24).sum()),
    })
    htable = pd.DataFrame(horizon_rows)
    htable.to_csv(reports / "m1x_combined_horizon_counts.csv", index=False)

    # Plot combined evidence support.
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    plot = htable.set_index("port")[["calls_with_any_approach", "calls_over_6h", "calls_over_12h", "calls_over_24h"]]
    plot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Independent calls")
    ax.set_xlabel("")
    ax.set_title("M1X causal approach-horizon support")
    ax.grid(axis="y", alpha=.25)
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(reports / "m1x_combined_horizon_coverage.png")
    plt.close(fig)

    # Audit outcome figure.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=160)
    audited["final_ground_truth"].map({True: "retained", False: "excluded"}).value_counts().plot(kind="bar", ax=axes[0])
    axes[0].set_title("Augusta audit outcome"); axes[0].set_ylabel("Calls"); axes[0].tick_params(axis="x", rotation=0)
    audited.loc[audited.final_ground_truth, "ground_truth_confidence"].value_counts().sort_index().plot(kind="bar", ax=axes[1])
    axes[1].set_title("Retained confidence"); axes[1].tick_params(axis="x", rotation=0)
    pd.Series({"ETA eligible": int(audited.eta_primary_eligible.sum()), "Other/insufficient": int((~audited.eta_primary_eligible).sum())}).plot(kind="bar", ax=axes[2])
    axes[2].set_title("Augusta ETA scope"); axes[2].tick_params(axis="x", rotation=20)
    for ax in axes: ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(reports / "m1x_augusta_audit_summary.png")
    plt.close(fig)

    summary = {
        "augusta_candidates": int(len(audited)),
        "augusta_validated_entries": int(audited.final_ground_truth.sum()),
        "augusta_excluded": int((~audited.final_ground_truth).sum()),
        "augusta_confidence": audited.loc[audited.final_ground_truth, "ground_truth_confidence"].value_counts().sort_index().to_dict(),
        "augusta_eta_eligible_calls": int(audited.eta_primary_eligible.sum()),
        "augusta_eta_unique_vessels": int(cohort.mmsi.nunique()),
        "augusta_calls_with_causal_approach": int(len(aug_hs)),
        "augusta_calls_over_6h": int(aug_hs.max_supported_horizon_h.gt(6).sum()),
        "augusta_calls_over_12h": int(aug_hs.max_supported_horizon_h.gt(12).sum()),
        "augusta_calls_over_24h": int(aug_hs.max_supported_horizon_h.gt(24).sum()),
        "combined_eta_calls": int(len(combined_calls)),
        "combined_unique_vessels": int(combined_calls.mmsi.nunique()),
        "gate": gate,
        "target_warning": "Research-gate entry truth only; not official ATA/PBP/berth/all-fast.",
        "scirocco_warning": "Retained Scirocco entries are capped at confidence B because the cross-section is a research proxy.",
    }
    (reports / "m1x_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
