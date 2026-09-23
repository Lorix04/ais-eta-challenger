# GitHub cleanup changelog

This package is derived from the frozen M19 external-validation repository. No model, score, split, threshold, prediction, or external-evaluation result was changed.

Changes are presentation/repository hygiene only:

- replaced the long historical README with a concise current-state landing page;
- preserved the previous README as `docs/RESEARCH_HISTORY.md`;
- appended the final M19 result to `EXECUTIVE_SUMMARY.md` and `TAKE_HOME_REPORT.md`;
- removed the internal `evidence/` history from the GitHub package;
- removed personalized email/call preparation documents;
- removed local caches/logs and an obsolete local-path output text file;
- hardened `.gitignore` so raw company/MMDEC inputs cannot be accidentally committed;
- retained the frozen compact derived/model/dist artifacts required for the private reproducibility suite;
- added `docs/GITHUB_REPOSITORY_NOTES.md` and `GITHUB_PACKAGE_MANIFEST.json`.

Validation after cleanup: **249/249 tests passed**, M18G/M18G1/M19 verifiers passed, and `compileall` passed.
## Windows portability fix

- Added `thrift==0.24.0` and `python-snappy==0.7.3` to `requirements.txt`.
- Removed import-time dependency on a Linux system `libsnappy`.
- MMDEC fallback decoding now prefers `python-snappy` on Windows/macOS/Linux and probes native `libsnappy` only lazily as a fallback.


## Windows UTF-8 pytest portability patch

- Added root `conftest.py` so historical pytest files that call `Path.read_text()` without an explicit encoding read repository text artifacts as UTF-8 on Windows.
- Historical frozen tests and scientific artifacts were left byte-for-byte unchanged.
- This is a non-scientific packaging/test portability patch only; M16G/M16H/M18/M19 models, predictions, labels, metrics, gates, and freezes are unchanged.
- Verified after the patch: 250/250 pytest PASS, compileall PASS, M18G verifier PASS, M18G1 verifier PASS, M19 verifier PASS.
