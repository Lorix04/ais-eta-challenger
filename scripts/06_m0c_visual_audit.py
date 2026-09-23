#!/usr/bin/env python3
"""Create a representative M0C semantic-state timeline for manual audit."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_pickle(ROOT / 'data/derived/m0c_ship_states.pkl.gz', compression='gzip')

# IEVOLI STAR has a compact, interpretable sequence of slow/stop/moving states
# during the first hours of the dataset.  It is selected for visual validation,
# not because it is a training/test exemplar.
mmsi = 215872000
start = pd.Timestamp('2026-04-03 10:40:00')
end = pd.Timestamp('2026-04-03 13:20:00')
x = df[(df.mmsi == mmsi) & df.recorded_at.between(start, end)].copy()
assert len(x) > 30
name = x['name'].dropna().iloc[-1] if x['name'].notna().any() else str(mmsi)

fig, ax = plt.subplots(figsize=(13, 6))
ax.plot(x['recorded_at'], x['sog_clean'], label='Reported SOG (kn)', linewidth=1.4)
ax.plot(x['recorded_at'], x['implied_speed_kn'].clip(upper=20), label='Position-implied speed (kn, clipped 20)', linewidth=1.1)

for event, marker in [('STOP_START','v'), ('SLOW_START','o'), ('TURN','x'), ('KIN_CONFLICT','s')]:
    mask = x['semantic_events'].fillna('').str.contains(event, regex=False)
    if mask.any():
        y = x.loc[mask, 'sog_clean'].fillna(0)
        ax.scatter(x.loc[mask, 'recorded_at'], y, marker=marker, s=38, label=event)

ax.set_title(f'M0C semantic-state audit — {name} / MMSI {mmsi}')
ax.set_ylabel('knots')
ax.set_xlabel('provider observation time')
ax.grid(True, alpha=0.25)
ax.legend(ncol=3, fontsize=8)
fig.autofmt_xdate()
fig.tight_layout()
out = ROOT / 'reports/m0c_semantic_timeline.png'
fig.savefig(out, dpi=160)
plt.close(fig)

# Also persist the exact rows shown in the plot.
cols = ['recorded_at','lat_clean','lon_clean','sog_clean','implied_speed_kn','dynamic_state_age_s','stale_risk_level','motion_state','semantic_events']
x[cols].to_csv(ROOT / 'reports/m0c_visual_audit_rows.csv', index=False)
print(f'PASS wrote {out.relative_to(ROOT)} rows={len(x)}')
