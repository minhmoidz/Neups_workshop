"""Representative attacker selection (STEP 2B Part 6, Stage C).

Selection uses VALIDATION statistics only. The API takes a list of validation
summaries; there is no parameter that can receive test AUC/predictions/labels.
"""

from typing import Dict, List


def best_validation_auc_of(validation_summary: Dict) -> float:
    """Return the best (max) validation AUC of one restart's validation record."""
    auc = validation_summary.get('validation_auc_per_epoch')
    if not auc:
        raise ValueError('no validation AUC per-epoch series in record')
    return float(max(auc))


def select_representative(valid_validation_summaries: List[Dict]):
    """Select the representative restart.

    Among numerically valid completed restarts, take each restart's best validation
    AUC, then pick the restart whose best validation AUC is closest to the median of
    those best-validation-AUCs. Ties are broken by smaller attacker seed.

    :param valid_validation_summaries: list of dicts each carrying at least
        'attacker_seed' and 'validation_auc_per_epoch' (training/validation-only).
    :return: attacker_seed (int)
    :raises ValueError: on empty input.
    """
    if not valid_validation_summaries:
        raise ValueError('representative selection requires at least one valid restart')

    scored = []
    for rec in valid_validation_summaries:
        best_auc = best_validation_auc_of(rec)
        scored.append((rec['attacker_seed'], best_auc))

    best_aucs = sorted(best_auc for _, best_auc in scored)
    n = len(best_aucs)

    def _median(sorted_vals):
        mid = n // 2
        if n % 2 == 1:
            return sorted_vals[mid]
        return 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])

    median_auc = _median(best_aucs)

    def _dist2(seed, best_auc):
        # Rounded so float noise never decides near-ties; the deterministic seed
        # tie-break below still decides equal (rounded) distances.
        return round((best_auc - median_auc) ** 2, 9)

    # Distances first, then seed (ascending) as the deterministic tie-breaker.
    ranked = sorted(scored, key=lambda s: (_dist2(s[0], s[1]), s[0]))
    return int(ranked[0][0])