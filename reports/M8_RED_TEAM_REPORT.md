# M8 — Red-team report

M8 is a **non-modelling** hardening pass. It does not refit, rescore, retune or change the frozen M6 predictor.

## Outcome

- frozen artifact hashes: **14/14 PASS**;
- unsupported recruiter-facing overclaims: **0**;
- absolute `/mnt/data` code references after portability fixes: **0**;
- M7 packaging issue found: personalized internal call/email documents were being shipped; M8 removes them;
- confidentiality wording corrected: compact audit annotations are derived from company AIS and remain in the company package only because they are required for exact reproducibility.

## Findings

| ID | Severity | Status | Finding / disposition |
|---|---|---|---|
| M8-01 | HIGH | FIXED | Personalized internal call/email preparation documents were included in the M7 company ZIP. M8 company package explicitly excludes CALL_PREP_IT.md, EMAIL_DRAFT_IT.md, CHANGE_PROTOCOL.md and the internal company-response matrix. |
| M8-02 | MEDIUM | FIXED | M7 manifest said no company-derived data although compact human audit annotations derived from company AIS were included for reproducibility. M8 manifest states the exact boundary: raw AIS and high-volume derived trajectories are excluded; compact annotation/config files are included and should not be published externally without review. |
| M8-03 | MEDIUM | FIXED | Environment-specific /mnt/data paths reduce portability. Optional data-ZIP alignment is now CLI/environment configured; company package is scanned for absolute container paths. |
| M8-04 | HIGH | PASS | Post-holdout changes must not alter frozen M2/M4/M6 model artifacts or final-evaluation inputs. Verified 14/14 SHA-256 hashes against M6_FREEZE.json. |
| M8-05 | HIGH | PASS | Recruiter-facing text must not overclaim superiority, official ATA semantics or unseen-port transfer. Automated phrase/claim checks plus frozen claim-scope verification. |
| M8-06 | MEDIUM | DOCUMENTED | Full end-to-end replay requires the company AIS files and the versioned manual-audit annotations. Reproducibility guide now separates data-free verification from data-backed replay and pins the tested Python version. |
| M8-07 | MEDIUM | DOCUMENTED | The final 3.4% route gain is not statistically resolved; Catania final result is underpowered and Augusta is neutral. Executive summary foregrounds claim boundaries rather than the best-looking subgroup metric. |
| M8-08 | MEDIUM | DOCUMENTED | The 90% call-level interval has 91.7% empirical whole-call coverage on only 12 HIGH-reliability final calls. Documented as small-sample empirical evidence, not a universal conformal guarantee under temporal dependence. |
| M8-09 | MEDIUM | FIXED | The first clean-room ZIP test exposed package/test coupling: omitted M0D candidate data and internal project-control tests caused 8 failures. The final company ZIP includes the compact candidate table needed by model tests and excludes packaging/project-control tests; clean-room result is 56/56 PASS. |
| M8-10 | LOW | DOCUMENTED | A complete 969k-row clean-room M0C rebuild exceeded the 180 s interactive execution budget in this environment. Do not claim a full clean-room replay was completed in M8; data-free package tests and a raw-data semantic-state smoke test passed. The full replay runbook remains documented. |

## Claim posture after red-team

The strongest defensible statement remains: the project demonstrates a leakage-resistant AIS port-entry ETA pipeline with auditable ground truth, train-only route retrieval, explicit reliability/uncertainty and a one-time chronological holdout. It does **not** establish superiority to the company model, superiority to reported AIS ETA, unseen-port transfer, berth/all-fast ETA or final long-horizon performance.
