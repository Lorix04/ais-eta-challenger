from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ais_eta.m0d import CataniaGeometryV1, label_catania_geometry


def plot_session(ax, row, states, geom, compact: bool = False):
    v = states[states["mmsi"].eq(int(row.mmsi))].sort_values("recorded_at")
    start = pd.Timestamp(row.inbound_time)
    end = pd.Timestamp(row.outbound_time) if pd.notna(row.outbound_time) else pd.Timestamp(row.last_observed_at)
    w = v[(v["recorded_at"] >= start - pd.Timedelta(hours=6)) & (v["recorded_at"] <= end + pd.Timedelta(hours=2))]
    pre = w[w["recorded_at"] < start]
    session = w[(w["recorded_at"] >= start) & (w["recorded_at"] <= end)]
    post = w[w["recorded_at"] > end]
    if len(pre):
        ax.plot(pre["lon_clean"], pre["lat_clean"], lw=1.0, label="pre-entry")
    if len(session):
        ax.plot(session["lon_clean"], session["lat_clean"], lw=1.15, label="port-side session")
    if len(post):
        ax.plot(post["lon_clean"], post["lat_clean"], lw=0.8, label="post-session")
    ax.plot(
        [geom.gate_west[0], geom.gate_east[0]],
        [geom.gate_west[1], geom.gate_east[1]],
        "k-",
        lw=1.8,
        label="research gate",
    )
    vv = np.array(geom.inner_harbour_vertices + (geom.inner_harbour_vertices[0],))
    ax.plot(vv[:, 0], vv[:, 1], "k--", lw=0.6)
    ax.scatter([row.inbound_lon], [row.inbound_lat], s=24, marker=">", color="black", zorder=5)
    decision = "KEEP" if bool(row.final_ground_truth) else "EXCLUDE"
    subtitle = f"{decision}/{row.ground_truth_confidence} {row.audit_reason_code}"
    if compact:
        ax.set_title(f"{row.session_id} {row.name}\n{subtitle}", fontsize=7)
    else:
        ax.set_title(f"{row.session_id} {row.name}\n{decision} / {row.ground_truth_confidence} — {row.audit_reason_code}", fontsize=9)
    ax.set_xlim(15.06, 15.14)
    ax.set_ylim(37.44, 37.53)
    ax.grid(alpha=0.2)
    ax.tick_params(labelsize=6 if compact else 7)


def main() -> int:
    audited = pd.read_csv(
        ROOT / "reports" / "catania_m0e_audited_calls.csv",
        parse_dates=["inbound_time", "outbound_time", "last_observed_at"],
    )
    states = label_catania_geometry(pd.read_pickle(ROOT / "data" / "derived" / "m0c_ship_states.pkl.gz"))
    geom = CataniaGeometryV1()

    atlas_dir = ROOT / "reports" / "m0e_audit_atlas"
    atlas_dir.mkdir(parents=True, exist_ok=True)
    for page_no, start_idx in enumerate(range(0, len(audited), 9), 1):
        page = audited.iloc[start_idx : start_idx + 9]
        fig, axes = plt.subplots(3, 3, figsize=(15, 15), dpi=140)
        for ax in axes.flat:
            ax.axis("off")
        for ax, row in zip(axes.flat, page.itertuples(index=False)):
            ax.axis("on")
            plot_session(ax, row, states, geom, compact=True)
        fig.suptitle(f"M0E Catania manual-audit atlas — page {page_no}", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(atlas_dir / f"page_{page_no:02d}.png", bbox_inches="tight")
        plt.close(fig)

    # Representative result screenshot: clean keep, sparse keep, brief exclude,
    # missing-outbound keep, right-censored keep, discontinuous exclude.
    ids = ["CT-003", "CT-004", "CT-047", "CT-054", "CT-060", "CT-005"]
    sel = audited[audited["session_id"].isin(ids)].set_index("session_id").loc[ids].reset_index()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=160)
    for ax, row in zip(axes.flat, sel.itertuples(index=False)):
        plot_session(ax, row, states, geom, compact=False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle("M0E manual audit — representative Catania candidate decisions", fontsize=14)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    out = ROOT / "reports" / "m0e_catania_audit_examples.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)

    print(f"atlas_pages={len(list(atlas_dir.glob('page_*.png')))}")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
