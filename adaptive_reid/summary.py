"""Final arm summary aggregation (STEP 2B Part 8, STEP 2B.1 remediation).

Aggregates over EXACTLY the numerically valid completed attacks. Near-chance runs are
never dropped, and no "successful attacks only" mean is computed.

STEP 2B.1 changes:
- restart SD uses the SAMPLE standard deviation (``ddof=1``) per protocol §6.3/§12/R-10;
- stub/synthetic test metrics (``valid_for_scientific_reporting is False``) can never
  enter the scientific mean/median/max; if only synthetic metrics are present those
  fields stay None and the summary is flagged (R-11).
"""

import numpy as np

from . import constants as C


def _test_metrics_of(a):
    """Return (scientific_auc_or_None, synthetic_bool) for one attempt."""
    tm = a.get('test_metrics')
    if tm is None or 'auc' not in tm:
        return None, False
    if tm.get('valid_for_scientific_reporting', True):
        return float(tm['auc']), False
    return None, True


def summarize_arm(attempts):
    """Summarize an arm from a list of per-attempt outcome dicts.

    Each attempt dict must carry at least:
        {'attacker_seed': int, 'state': str, 'near_chance': bool, 'test_metrics': dict|None}
    where test_metrics carries the 'auc' key if present.

    :return: dict
    """
    valid = [a for a in attempts if a['state'] == C.VALID]

    test_aucs = []
    n_synthetic = 0
    for a in valid:
        auc, synthetic = _test_metrics_of(a)
        if auc is not None:
            test_aucs.append(auc)
        if synthetic:
            n_synthetic += 1

    summary = {
        'n_attempted': len(attempts),
        'n_numerically_invalid': sum(1 for a in attempts if a['state'] == C.NUMERICALLY_INVALID),
        'n_valid': len(valid),
        'n_near_chance': sum(1 for a in valid if a['near_chance']),
        'test_auc_values': test_aucs,
        'n_stub_or_synthetic_test_metrics': n_synthetic,
        'contains_stub_or_synthetic_metrics': n_synthetic > 0,
    }

    if not test_aucs:
        summary.update({
            'mean_test_auc': None,
            'std_test_auc': None,
            'median_test_auc': None,
            'max_test_auc': None,
            'representative_attacker_seed': None,
            'scientific_summary_available': False,
        })
        return summary

    arr = np.asarray(test_aucs, dtype=np.float64)
    summary.update({
        'mean_test_auc': float(arr.mean()),
        'std_test_auc': float(arr.std(ddof=1)) if arr.size >= 2 else None,
        'median_test_auc': float(np.median(arr)),
        'max_test_auc': float(arr.max()),
        'scientific_summary_available': True,
    })
    # representative seed from the arm provenance (set during Stage D)
    reps = [a['attacker_seed'] for a in valid if a.get('is_representative')]
    summary['representative_attacker_seed'] = reps[0] if reps else None
    return summary