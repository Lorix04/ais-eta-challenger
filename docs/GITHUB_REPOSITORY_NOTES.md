# GitHub repository notes

This repository is the cleaned M19 research snapshot intended for technical review.

## Recommended visibility

Use a **private GitHub repository** unless the owner of the original take-home AIS data explicitly authorizes public redistribution. Raw company inputs and MMDEC Parquet files are not included, but compact derived artifacts remain in this private reproducibility edition.

## Intentionally excluded

- internal `evidence/` screenshots, diffs and test transcripts;
- raw company AIS data;
- MMDEC Parquet files;
- personalized email/call preparation material;
- caches, virtual environments and temporary logs.

## Why some binary artifacts remain

`data/derived/`, `models/` and two small archives under `dist/` are retained because historical hash/verifier tests depend on them. No tracked file exceeds GitHub's 100 MiB hard per-file limit.

## Final scientific state

- frozen development champion: M16G Mixture of Experts;
- uncertainty/selective layer: M16H / M18F;
- external gate: M18G;
- full-development blind scorer: M18G1;
- one-shot external evaluation: M19 on MMDEC;
- MMDEC is spent and may not be used for retuning.
