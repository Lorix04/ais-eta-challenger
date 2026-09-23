from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m0d import CataniaGeometryV1, label_catania_geometry


def choose_examples(calls: pd.DataFrame) -> pd.DataFrame:
    selected = []

    roadstead = calls[
        calls["roadstead_wait_before_entry"].astype(bool)
        & calls["candidate_class"].eq("STOP_CONFIRMED_CALL_CANDIDATE")
    ]
    if len(roadstead):
        selected.append(roadstead.iloc[0])

    normal = calls[
        calls["session_closed"].astype(bool)
        & calls["m0d_candidate_quality"].eq("A")
        & ~calls["roadstead_wait_before_entry"].astype(bool)
        & calls["candidate_class"].eq("STOP_CONFIRMED_CALL_CANDIDATE")
    ]
    if len(normal):
        selected.append(normal.iloc[0])

    short = calls[calls["candidate_class"].eq("ENTRY_WITHOUT_CONFIRMED_STOP")]
    if len(short):
        selected.append(short.sort_values("session_duration_min").iloc[0])

    unresolved = calls[calls["unresolved_missing_outbound"].astype(bool)]
    if len(unresolved):
        selected.append(unresolved.iloc[0])

    if not selected:
        return calls.head(4).copy()
    return pd.DataFrame(selected).drop_duplicates("session_id").head(4).reset_index(drop=True)


def main() -> int:
    df = pd.read_pickle(ROOT / "data" / "derived" / "m0c_ship_states.pkl.gz")
    geo = label_catania_geometry(df)
    calls = pd.read_csv(
        ROOT / "reports" / "catania_m0d_candidate_calls.csv",
        parse_dates=["inbound_time", "outbound_time", "last_observed_at"],
    )
    examples = choose_examples(calls)
    examples.to_csv(ROOT / "reports" / "m0d_visual_audit_selection.csv", index=False)

    geom = CataniaGeometryV1()
    fig, ax = plt.subplots(figsize=(10, 10), dpi=180)

    local = geo[
        geo["catania_analysis_envelope"]
        & geo["lat_clean"].between(37.472, 37.507)
        & geo["lon_clean"].between(15.086, 15.116)
    ]
    # Context cloud is deliberately faint; highlighted paths use matplotlib's default cycle.
    ax.scatter(local["lon_clean"], local["lat_clean"], s=0.2, alpha=0.025, label="all local AIS observations")

    px = [p[0] for p in geom.inner_harbour_vertices] + [geom.inner_harbour_vertices[0][0]]
    py = [p[1] for p in geom.inner_harbour_vertices] + [geom.inner_harbour_vertices[0][1]]
    ax.plot(px, py, linestyle="--", linewidth=1.2, label="inner-harbour proxy")
    ax.plot(
        [geom.gate_west[0], geom.gate_east[0]],
        [geom.gate_west[1], geom.gate_east[1]],
        linewidth=2.2,
        label="research gate v1",
    )

    for row in examples.itertuples(index=False):
        start = row.inbound_time - pd.Timedelta(hours=6)
        if pd.notna(row.outbound_time):
            end = row.outbound_time
        else:
            # Unclosed sessions can span days because the provider later loses the
            # vessel or never observes an outbound crossing.  For visual audit we
            # cap the post-entry window so a long missing-outbound tail does not
            # draw misleading straight lines across unrelated later observations.
            end = min(row.last_observed_at, row.inbound_time + pd.Timedelta(hours=12))
        path = geo[
            geo["mmsi"].eq(row.mmsi)
            & geo["recorded_at"].between(start, end)
            & geo["position_valid"]
        ].sort_values("recorded_at")
        label = f"{row.session_id} {row.name} — {row.candidate_class}"
        line = ax.plot(path["lon_clean"], path["lat_clean"], linewidth=1.4, label=label)[0]
        stop = path[path["stop_candidate"]]
        if len(stop):
            ax.scatter(stop["lon_clean"], stop["lat_clean"], s=6, alpha=0.45)
        ax.scatter([row.inbound_lon], [row.inbound_lat], marker="x", s=45)
        ax.annotate(row.session_id, (row.inbound_lon, row.inbound_lat), xytext=(4, 4), textcoords="offset points", fontsize=8)

    ax.set_xlim(15.086, 15.116)
    ax.set_ylim(37.472, 37.507)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.grid(True, alpha=0.2)
    ax.set_title(
        "M0D Catania visual audit — research gate, inner-port proxy and representative sessions\n"
        f"61 candidate sessions; 54 stop-confirmed; 5 right-censored; 12 unresolved missing-outbound"
    )
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "m0d_catania_event_map.png", bbox_inches="tight")
    print(examples[[
        "session_id", "mmsi", "name", "inbound_time", "outbound_time",
        "candidate_class", "m0d_candidate_quality", "roadstead_wait_before_entry",
        "right_censored", "unresolved_missing_outbound"
    ]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())