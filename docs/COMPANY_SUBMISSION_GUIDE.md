# Company submission guide

This package is organized around the **final company-defined exercise**: predict the ETA reference contained in `Tracks`.

## Suggested review path

**2 minutes**  
Read `EXECUTIVE_SUMMARY.md`.

**10 minutes**  
Read `TAKE_HOME_REPORT.md` sections 1–10 for task formulation, validation, final results, label-quality forensics and explainability.

**Technical audit**  
Review:
- `reports/M10_REPORT.md` — frozen benchmark;
- `reports/M11_REPORT.md` — reference/error forensics;
- `reports/M12_REPORT.md` — frozen-model attribution/ablation;
- `docs/MODEL_CARD.md` — intended use and limitations;
- `reports/M14_REPRODUCIBILITY_REPORT.md` — final reproducibility/submission freeze;
- `docs/REPRODUCIBILITY.md` — replay/test instructions.

## Primary vs secondary work

### Primary

M10–M12 answer the company-defined task using `Tracks.eta` as the exercise reference.

### Secondary

M0–M9 are retained as additional maritime-domain research on AIS-derived physical port entry. They are deliberately separated so the project does not silently substitute a different target for the requested exercise.

## Headline result policy

The headline strict final result is **312.0 h MAE / 25.3 h median absolute error on 53 final references**.

Post-hoc subsets such as the 0–7 day future-reference band are labelled diagnostic and do not replace the strict benchmark.

## Confidentiality

The submission excludes raw company AIS files and high-volume row-level derived company data. It is still a company-facing package, not automatically a public-release package.


## Final package integrity

M14 is the final submission freeze. Run `python VERIFY_SUBMISSION.py` from the extracted package root to verify every payload file against `SUBMISSION_CONTENT_MANIFEST.json`. The external archive checksum is recorded in `dist/ais_eta_takehome_submission_final_manifest.json` in the full audit bundle. Rebuilding or editing the package creates a different hash and is not the M14-frozen artifact.
