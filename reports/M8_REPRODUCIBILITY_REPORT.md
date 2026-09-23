# M8 — Reproducibility report

M8 distinguishes **package verification** from a **full data-backed analytical replay**.

## Data-free clean-room verification

The M8 company ZIP was extracted into a new empty directory, outside the working tree. No company AIS files or pre-existing derived state were present.

Final result:

```text
56 passed
compileall PASS
```

The first clean-room attempt failed 8 tests because the M7-style package omitted a compact M0D candidate table required by the manual-audit tests and still shipped an internal project-contract test whose files were intentionally excluded. This was treated as a packaging defect rather than hidden: M8 now includes the compact candidate table and excludes packaging/project-control tests from the company ZIP.

## Raw-data portability smoke test

The original company ZIP was unpacked into the clean-room `data/` directory and a bounded raw-data smoke test read both position CSVs, selected ship rows, normalized AIS sentinels and constructed the causal M0C semantic-state features.

Final smoke result:

```text
PASS raw-data smoke rows=5,639 mmsi=89
PASS repeated_state_fraction=0.502
PASS sentinel normalization and causal semantic-state construction
```

## Full replay status

M8 did **not** complete a second full 969,084-row clean-room M0C rebuild inside the interactive execution window: the full step exceeded 180 seconds in this environment. This is recorded as a runtime constraint, not silently presented as a pass.

The full data-backed run order remains documented in `docs/REPRODUCIBILITY.md`; the original project pipeline was already executed end-to-end to create the frozen M6 artefacts, and all M6 frozen SHA-256 hashes remain unchanged during M8.

## Portability hardening

- Python tested version is pinned in `PYTHON_VERSION.txt` as 3.13.5.
- Python source shipped to the company contains no absolute `/mnt/data` dependency.
- Optional Message-5 ZIP alignment is explicit (`--data-zip`) rather than container-specific.
- The company package excludes raw AIS, high-volume derived trajectories and internal evidence bundles.
- Compact manual-audit/config annotations remain included because exact label/cohort reproduction depends on them; the manifest explicitly marks the package as company-facing rather than automatically public-safe.
