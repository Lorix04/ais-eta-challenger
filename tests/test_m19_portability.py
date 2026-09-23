from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_m19_parquet_module_import_does_not_probe_native_snappy(monkeypatch):
    # Regression for Windows GitHub-clean-room packaging: importing the module
    # must not require a Linux system libsnappy.
    import ais_eta.m19_parquet_lite as mod

    fake = types.SimpleNamespace(decompress=lambda data: b"decoded:" + data)
    monkeypatch.setitem(sys.modules, "snappy", fake)
    monkeypatch.setattr(
        mod,
        "_snappy_lib",
        lambda: (_ for _ in ()).throw(AssertionError("native backend should not be probed")),
    )
    assert mod._snappy(b"abc") == b"decoded:abc"
