"""Targeted CPU-only, synthetic-only tests for
reproduction/statistics/patient_graph_bootstrap.py.

Run: .venv/bin/python reproduction/tests/test_patient_graph_bootstrap.py
"""
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'reproduction', 'statistics'))

from patient_graph_bootstrap import (  # noqa: E402
    pair_weights_for_draw, weighted_auc, draw_patient_multiplicities,
    patient_graph_bootstrap_paired, effective_auc, OneClassResampleError, STATUS,
)

RESULTS = []


def check(name, condition):
    RESULTS.append((name, bool(condition)))
    print(('PASS' if condition else 'FAIL') + ' - ' + name)
    if not condition:
        raise AssertionError('Test failed: %s' % name)


def _synthetic_dataset(rng):
    patients = [0, 1, 2, 3, 4]
    # positives: same-patient pairs
    pos_p1 = [0, 1, 2, 3, 4]
    pos_p2 = [0, 1, 2, 3, 4]
    # negatives: distinct-patient pairs
    neg_p1 = [0, 0, 1, 2, 3]
    neg_p2 = [1, 2, 3, 4, 4]
    p1 = np.array(pos_p1 + neg_p1)
    p2 = np.array(pos_p2 + neg_p2)
    y_true = np.array([1] * 5 + [0] * 5)
    y_score_b = rng.uniform(0, 1, size=10)
    y_score_c = rng.uniform(0, 1, size=10)
    return p1, p2, y_true, y_score_b, y_score_c


def test_weighted_matches_explicit_row_expansion():
    """The weighted-AUC shortcut must equal AUC computed on an explicitly
    row-expanded (repeated-row) dataset for the same draw."""
    rng = np.random.default_rng(0)
    p1, p2, y_true, y_score_b, _ = _synthetic_dataset(rng)
    draw_counts = {0: 2, 1: 1, 2: 0, 3: 3, 4: 1}  # arbitrary multiplicities

    weights = pair_weights_for_draw(p1, p2, draw_counts)
    auc_weighted = weighted_auc(y_true, y_score_b, weights, raise_on_one_class=False)

    # Explicit expansion: repeat each row `weight` times (weights here are
    # small non-negative integers by construction of draw_counts).
    rows = []
    for i in range(len(p1)):
        w = int(weights[i])
        rows.extend([i] * w)
    if rows:
        yt_exp = y_true[rows]
        ys_exp = y_score_b[rows]
        auc_expanded = roc_auc_score(yt_exp, ys_exp) if len(set(yt_exp)) > 1 else None
    else:
        auc_expanded = None

    check('weighted AUC equals explicit integer-row-expansion AUC',
          (auc_weighted is None and auc_expanded is None) or
          (auc_weighted is not None and auc_expanded is not None and abs(auc_weighted - auc_expanded) < 1e-9))


def test_positive_pair_weight_is_cP():
    p1 = np.array([5])
    p2 = np.array([5])
    w = pair_weights_for_draw(p1, p2, {5: 3})
    check('positive pair weight equals c_P', w[0] == 3.0)


def test_negative_pair_weight_is_cP_times_cQ():
    p1 = np.array([5])
    p2 = np.array([9])
    w = pair_weights_for_draw(p1, p2, {5: 3, 9: 2})
    check('negative pair weight equals c_P * c_Q', w[0] == 6.0)


def test_zero_weight_for_undrawn_patient():
    p1 = np.array([5])
    p2 = np.array([9])
    w = pair_weights_for_draw(p1, p2, {5: 3})  # patient 9 not drawn -> count 0
    check('undrawn patient yields zero pair weight', w[0] == 0.0)


def test_one_class_resample_detected_not_silent():
    y_true = np.array([1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3])
    weights = np.array([1.0, 1.0, 1.0])
    raised = False
    try:
        weighted_auc(y_true, y_score, weights, raise_on_one_class=True)
    except OneClassResampleError:
        raised = True
    check('one-class resample raises by default (never silent)', raised)

    result = weighted_auc(y_true, y_score, weights, raise_on_one_class=False)
    check('one-class resample returns explicit None when not raising', result is None)


def test_effective_auc():
    check('effective_auc(0.9) == 0.9', abs(effective_auc(0.9) - 0.9) < 1e-12)
    check('effective_auc(0.2) == 0.8', abs(effective_auc(0.2) - 0.8) < 1e-12)
    check('effective_auc(0.5) == 0.5', abs(effective_auc(0.5) - 0.5) < 1e-12)


def test_paired_bootstrap_uses_same_draw_for_both_arms():
    rng = np.random.default_rng(1)
    p1, p2, y_true, y_score_b, y_score_c = _synthetic_dataset(rng)
    result = patient_graph_bootstrap_paired(p1, p2, y_true, y_score_b, y_score_c,
                                             n_bootstrap=20, bootstrap_seed=123)
    check('status flag reported as UNVALIDATED', result['status'] == STATUS == 'UNVALIDATED')
    check('n_valid + n_one_class_invalid == n_bootstrap_requested',
          result['n_valid'] + result['n_one_class_invalid'] == result['n_bootstrap_requested'])
    check('deltas length matches n_valid', len(result['deltas']) == result['n_valid'])

    # determinism: same seed -> identical result
    result2 = patient_graph_bootstrap_paired(p1, p2, y_true, y_score_b, y_score_c,
                                              n_bootstrap=20, bootstrap_seed=123)
    check('same bootstrap_seed reproduces identical deltas', result['deltas'] == result2['deltas'])

    # different bootstrap_seed but same "model" seeds -> generally different deltas
    result3 = patient_graph_bootstrap_paired(p1, p2, y_true, y_score_b, y_score_c,
                                              n_bootstrap=20, bootstrap_seed=456)
    check('different bootstrap_seed is independent of any model seed (typically differs)',
          result['deltas'] != result3['deltas'])


def main():
    test_weighted_matches_explicit_row_expansion()
    test_positive_pair_weight_is_cP()
    test_negative_pair_weight_is_cP_times_cQ()
    test_zero_weight_for_undrawn_patient()
    test_one_class_resample_detected_not_silent()
    test_effective_auc()
    test_paired_bootstrap_uses_same_draw_for_both_arms()
    n_pass = sum(1 for _, ok in RESULTS if ok)
    print('\n%d/%d checks passed' % (n_pass, len(RESULTS)))
    assert n_pass == len(RESULTS)
    print('ALL PATIENT_GRAPH_BOOTSTRAP TESTS PASS')


if __name__ == '__main__':
    main()
