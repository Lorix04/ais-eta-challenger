#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "dist/ais_eta_takehome_submission_final.zip": "65684d016775deea6b111db1a3ad76fe442cf84f1ceb1ac17f4327ea145c553c",
    "models/m10_strict_reference_catboost.cbm": "efbb86375751e901c48c2f5724514828a10f9eb8ff08ffaabf2d875e369ef86a",
    "reports/m10_final_predictions.csv": "fc6334d82f0b3c382e40805dc1a325163001a363ff6a6329da33160734011ee1",
    "reports/M6_FREEZE.json": "b551c12da6d05fbea22c365f0b2167879669d948c4d790d46ccd28a9dc8e17de",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    cov = json.loads((ROOT / "reports/M16B_CARDINALITY_COVERAGE.json").read_text())
    freeze = json.loads((ROOT / "reports/M16B_RESOLVER_FREEZE.json").read_text())
    df = pd.read_csv(ROOT / "reports/m16b_destination_resolutions.csv")
    m16a = json.loads((ROOT / "reports/M16A_DEVELOPMENT_MANIFEST.json").read_text())

    assert len(df) == 386 and df["mmsi"].nunique() == 386
    assert set(df["mmsi"].astype(int)).isdisjoint(m16a["blocked_old_final_mmsi"])
    assert cov["target_or_reference_eta_used_in_resolution"] is False
    assert cov["final_test_used_for_selection"] is False
    assert cov["canonical_destination_unique"] < cov["raw_destination_unique"]
    assert cov["resolved_port_rows"] > 0
    assert freeze["resolver_target_free"] is True

    for rel, expected in EXPECTED.items():
        got = sha(ROOT / rel)
        assert got == expected, f"immutable hash drift: {rel}: {got} != {expected}"
        print(f"PASS immutable {rel} {got}")

    print(f"PASS M16B dev-only rows={len(df)} blocked_old_final={len(m16a['blocked_old_final_mmsi'])}")
    print(f"PASS cardinality {cov['raw_destination_unique']} -> {cov['canonical_destination_unique']}")
    print(f"PASS resolved coverage {cov['resolved_port_coverage']:.1%}")
    print(f"PASS target-free resolver version={cov['resolver_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
