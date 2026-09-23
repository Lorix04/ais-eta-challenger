# Company submission package guide

## Purpose

The M14 final submission is frozen around the **company-defined `Tracks.eta` reference benchmark**. M10–M12 are primary; M0–M9 are secondary maritime-domain research.

## Included

- company-facing `README.md`, executive summary and take-home report;
- primary M10 benchmark report/freeze and frozen CatBoost model;
- aggregate M11 reference-quality/error forensics;
- aggregate M12 frozen-model explainability/ablation outputs;
- model card, reproducibility guide and company-submission guide;
- analytical source code/tests needed to inspect and reproduce the benchmark with the supplied data;
- selected frozen M6 physical-arrival report/figure as a clearly labelled secondary branch.

## Excluded

- raw AIS CSV files;
- high-volume row-level derived trajectory/reference tables;
- internal evidence/screenshot bundles;
- personalized email/call-preparation documents;
- internal packaging/red-team process notes;
- row-level SHAP or top-error tables that are unnecessary for company review.

## Confidentiality boundary

The archive is **company-facing, not automatically public-safe**. It is built from analysis of company-supplied data even though raw data and row-level derived tables are excluded.

## Freeze boundary

M13 was report/package reconciliation only. M14 is the **final submission freeze** and may not refit the M10 model, change the M10 split, alter final predictions, promote a post-hoc subset to the primary benchmark, or rewrite the frozen M6 branch.

The final ZIP uses deterministic archive metadata, a per-file SHA-256 content manifest, a package-local integrity verifier and an external archive checksum. Any post-M14 change requires a new artifact/hash and must not be represented as the frozen submission.
