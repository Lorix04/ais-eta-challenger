# M19 — Fresh External Holdout

## Final status

**REAL MMDEC ONE-SHOT OPENING COMPLETED — NO RETUNING PERMITTED**

M19 originally froze MMDEC v1 (`10.5281/zenodo.17491518`) as the first public external-domain cohort. The canonical `Dataset_AIS_POS.parquet` and `Dataset_AIS_SPEC.parquet` were subsequently supplied, their published MD5 checks matched, a 500-MMSI target-free cohort was prepared, and the frozen M18G1 scorer produced a SHA-256-sealed prediction ledger before ETA labels were revealed.

The one-shot M18G evaluation is now complete. The frozen operational M16G baseline records **550.409 h full MAE** on all 500 external rows. M18F BALANCED_80 retains **76/500 = 15.2%**, with **342.545 h retained MAE**, **587.668 h deferred MAE**, **69.74% retained interval empirical coverage**, and **+0.326 Spearman** between predicted risk and absolute error. Because coverage and interval-coverage requirements fail, the frozen decision is **`DO_NOT_EXTERNALLY_VALIDATE_M18F_LAYER`**.

This is a negative but valid external-generalization result. MMDEC is now a spent test set and must not be used for iterative tuning, feature selection, threshold adjustment, or post-hoc promotion. See `M19_EXTERNAL_REPORT.md` and `M19_EXTERNAL_RESULT_FREEZE.json` for full provenance and diagnostics.
