"""M17H conservative integration of the passing M17G MoCo retrieval component.

M17H does not enlarge the M16G ensemble. It replaces the frozen M16D route
expert one-for-one with M17G MoCo retrieval, and replaces the route fallback
inside the physics gate with that same MoCo prediction. The four-expert shape
and the M16G meta strategy selected for each outer fold remain fixed.
"""
from __future__ import annotations

M17H_VERSION = "m17h-conservative-moco-integration-v1-20260922"
M17H_MIN_GAIN_H = 1.0
M17H_MIN_FOLD_WINS = 3
M17H_MIN_CHANGED_ROW_WIN_SHARE = 0.50
M17H_MAX_P90_RATIO = 1.02
M17H_MAX_WORST_FOLD_REGRESSION_H = 15.0


def promotion_gate(*, gain_h: float, fold_wins: int, changed_row_win_share: float,
                   p90_ratio: float, worst_fold_regression_h: float) -> str:
    ok = (
        float(gain_h) >= M17H_MIN_GAIN_H
        and int(fold_wins) >= M17H_MIN_FOLD_WINS
        and float(changed_row_win_share) >= M17H_MIN_CHANGED_ROW_WIN_SHARE
        and float(p90_ratio) <= M17H_MAX_P90_RATIO
        and float(worst_fold_regression_h) <= M17H_MAX_WORST_FOLD_REGRESSION_H
    )
    return "PASS_CONSERVATIVE_MOCO_INTEGRATION" if ok else "NO_M17H_PROMOTION"
