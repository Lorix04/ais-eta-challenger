"""M16J final challenger freeze helpers.

M16J is documentation/freeze only.  It must not train, tune, rescore, or use the
already-observed 53-row M10 final set for model selection.
"""
from __future__ import annotations

from dataclasses import dataclass

M16J_VERSION = "m16j-v1"
M16J_OFFICIAL_SUBMISSION = "M14"
M16J_PROMOTION_DECISION = "PROMOTE_CHALLENGER_FOR_NEW_UNTOUCHED_HOLDOUT_ONLY"


@dataclass(frozen=True)
class PromotionEvidence:
    pooled_gain_h: float
    fold_wins: int
    leave_one_fold_positive: int
    trimmed_gain_h: float
    winsorized_gain_h: float
    bootstrap_positive_probability: float
    bootstrap_ci95_low_h: float
    top5_positive_gain_share: float


def promotion_decision(e: PromotionEvidence) -> str:
    """Return the conservative M16J promotion class.

    A useful development challenger can be frozen for external validation even
    if its 95% bootstrap interval crosses zero.  It must never replace the M14
    official submission unless fresh untouched evidence is supplied.
    """
    core = (
        e.pooled_gain_h > 0
        and e.fold_wins >= 3
        and e.leave_one_fold_positive == 5
        and e.trimmed_gain_h > 0
        and e.winsorized_gain_h > 0
        and e.bootstrap_positive_probability >= 0.90
    )
    if not core:
        return "DOCUMENT_NEGATIVE_OR_UNSTABLE_CHALLENGER"
    if e.bootstrap_ci95_low_h > 0 and e.top5_positive_gain_share < 0.50:
        return "PROMOTE_CHALLENGER_STABLE_DEVELOPMENT_EVIDENCE"
    return M16J_PROMOTION_DECISION
