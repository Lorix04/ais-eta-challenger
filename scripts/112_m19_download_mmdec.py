#!/usr/bin/env python3
"""Download the two MMDEC v1 AIS files required by M19 and verify published MD5s."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ais_eta.m19 import M19_SOURCE_FILES, M19_SOURCE_RECORD, md5_file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=str(ROOT / "data" / "external" / "mmdec_v1"))
    args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    for name, meta in M19_SOURCE_FILES.items():
        path = out / name
        url = f"{M19_SOURCE_RECORD}/files/{name}?download=1"
        if not path.exists():
            print(f"DOWNLOAD {url}")
            urllib.request.urlretrieve(url, path)
        got = md5_file(path)
        if got != meta["md5"]:
            raise SystemExit(f"MD5 mismatch for {name}: {got} != {meta['md5']}")
        print(f"PASS {name} md5={got} bytes={path.stat().st_size}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
