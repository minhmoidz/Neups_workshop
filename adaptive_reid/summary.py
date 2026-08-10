"""Final arm summary aggregation (STEP 2B Part 8).

Aggregates over EXACTLY the numerically valid completed attacks. Near-chance runs are
never dropped, and no "successful attacks only" mean is computed.
"""

import numpy as np

from . import constants as C


def summarize_arm(attempts):
    """Summarize an arm from a list of per-attempt outcome dicts.

    Each attempt dict must carry at least:
        {'attacker_seed': int, 'state': str, 'near_chance': bool, 'test_metrics': dict|None}
    where test_metrics carries the 'auc' key if present.

    :return: dict
    """
    valid = [a for a in attempts if a['state'] == C.VALID]

    test_aucs = []
    for a in valid:
        tm = a.get('test_metrics')
        if tm is not None and 'auc' in tm:
            test_aucs.append(float(tm['auc']))

    summary = {
        'n_attempted': len(attempts),
        'n_numerically_invalid': sum(1 for a in attempts if a['state'] == C.NUMERICALLY_INVALID),
        'n_valid': len(valid),
        'n_near_chance': sum(1 for a in valid if a['near_chance']),
        'test_auc_values': test_aucs,
    }

    if not test_aucs:
        summary.update({
            'mean_test_auc': None,
            'std_test_auc': None,
            'median_test_auc': None,
            'max_test_auc': None,
            'representative_attacker_seed': None,
        })
        return summary

    arr = np.asarray(test_aucs, dtype=np.float64)
    summary.update({
        'mean_test_auc': float(arr.mean()),
        'std_test_auc': float(arr.std(ddof=0)),
        'median_test_auc': float(np.median(arr)),
        'max_test_auc': float(arr.max()),
    })
    # representative seed from the arm provenance (set during Stage D)
    reps = [a['attacker_seed'] for a in valid if a.get('is_representative')]
    summary['representative_attacker_seed'] = reps[0] if reps else None
    return summary