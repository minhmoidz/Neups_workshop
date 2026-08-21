"""Candidate patient-graph-aware bootstrap for paired AUC comparison (G0.2 §8).

STATUS: UNVALIDATED — see reproduction/reports/G0_1_PROTOCOL_REPAIR_SPEC_2026-08-21.md
§4 and reproduction/statistics/simulate_bootstrap_coverage.py for the
predeclared acceptance gates this must pass before it may be treated as the
primary inferential method. Do not label this primary until simulation
evidence (recorded in the G0.2 report) supports it.

Weighting rule (as specified):
    positive pair (patient1 == patient2):  weight = c_P
    negative pair (patient1 != patient2):  weight = c_P * c_Q
where c_X is how many times patient X was drawn in one bootstrap resample of
the patient population (with replacement). Weighted AUC is used instead of
materializing repeated pair rows (equivalence is checked in
reproduction/tests/test_patient_graph_bootstrap.py against explicit
integer-row expansion).

UNRESOLVED (explicitly, not silently decided): whether inference should be
performed on raw AUC or on effective_auc() = max(AUC, 1-AUC) is a design
choice left open for G0.2B. `effective_auc()` is provided here as a utility
only; `patient_graph_bootstrap_paired()` returns raw AUCs/deltas and does not
itself pick an estimand — callers must not assume one was chosen for them.
"""
from collections import Counter

import numpy as np
from sklearn.metrics import roc_auc_score

STATUS = 'UNVALIDATED'


class OneClassResampleError(RuntimeError):
    """Raised (never silently swallowed) when a resample's weighted labels
    contain only one class, so AUC is undefined."""


class BootstrapInputError(ValueError):
    """Raised for any malformed input to patient_graph_bootstrap_paired.

    Fix 7 (G0.2A.2): replaces the previous bare `assert` statements, which
    are removable under `python -O` and therefore not a real contract for
    scientific code — every check here is an explicit, non-removable
    exception."""


def _validate_bootstrap_inputs(patient1_ids, patient2_ids, y_true,
                                y_score_baseline, y_score_candidate, n_bootstrap):
    n = len(y_true)
    if n == 0:
        raise BootstrapInputError('Inputs must be non-empty')
    for name, arr in (('patient1_ids', patient1_ids), ('patient2_ids', patient2_ids),
                       ('y_score_baseline', y_score_baseline), ('y_score_candidate', y_score_candidate)):
        if len(arr) != n:
            raise BootstrapInputError('%s has length %d, expected %d (len(y_true))' % (name, len(arr), n))

    if not isinstance(n_bootstrap, int) or isinstance(n_bootstrap, bool) or n_bootstrap <= 0:
        raise BootstrapInputError('n_bootstrap must be a positive int, got %r' % (n_bootstrap,))

    yt = np.asarray(y_true)
    if not np.isin(yt, (0, 1)).all():
        raise BootstrapInputError('y_true must be binary (0/1 only)')

    for name, scores in (('y_score_baseline', y_score_baseline), ('y_score_candidate', y_score_candidate)):
        if not np.isfinite(np.asarray(scores, dtype=np.float64)).all():
            raise BootstrapInputError('%s contains non-finite values' % name)

    for i, (p1, p2, y) in enumerate(zip(patient1_ids, patient2_ids, yt)):
        is_same_patient = (p1 == p2)
        expected_label = 1 if is_same_patient else 0
        if int(y) != expected_label:
            raise BootstrapInputError(
                'Pair label contract violated at row %d: patient1=%r patient2=%r y_true=%r '
                '(expected %d for %s-patient pair)'
                % (i, p1, p2, y, expected_label, 'same' if is_same_patient else 'different'))


def effective_auc(auc: float) -> float:
    return max(auc, 1.0 - auc)


def pair_weights_for_draw(patient1_ids, patient2_ids, draw_counts: dict) -> np.ndarray:
    """Vectorized weight for each pair given a patient multiplicity mapping.

    draw_counts: {patient_id: count_in_this_resample}. Patients not present
    in draw_counts (count 0) yield a pair weight of 0.
    """
    c1 = np.array([draw_counts.get(p, 0) for p in patient1_ids], dtype=np.float64)
    c2 = np.array([draw_counts.get(p, 0) for p in patient2_ids], dtype=np.float64)
    is_positive = np.array([p1 == p2 for p1, p2 in zip(patient1_ids, patient2_ids)])
    weights = np.where(is_positive, c1, c1 * c2)
    return weights


def weighted_auc(y_true, y_score, weights, raise_on_one_class: bool = True):
    """sklearn roc_auc_score with sample_weight, restricted to weight>0 rows.

    Returns None (never raises) only if raise_on_one_class=False and the
    weighted label set has a single class; callers must handle None
    explicitly — it is never silently treated as a valid AUC.
    """
    mask = weights > 0
    yt = np.asarray(y_true)[mask]
    ys = np.asarray(y_score)[mask]
    w = weights[mask]
    if len(np.unique(yt)) < 2:
        if raise_on_one_class:
            raise OneClassResampleError('Weighted resample has only one label class; AUC undefined')
        return None
    return float(roc_auc_score(yt, ys, sample_weight=w))


def draw_patient_multiplicities(patient_ids: list, rng: np.random.Generator) -> dict:
    """One bootstrap resample of the (unique) patient population, with
    replacement, size = number of unique patients. Returns {patient_id: count}."""
    unique_patients = sorted(set(patient_ids))
    n = len(unique_patients)
    draws = rng.choice(unique_patients, size=n, replace=True)
    return dict(Counter(draws))


def patient_graph_bootstrap_paired(patient1_ids, patient2_ids, y_true,
                                    y_score_baseline, y_score_candidate,
                                    n_bootstrap: int, bootstrap_seed: int):
    """Paired bootstrap of (candidate_AUC - baseline_AUC), same resample draw
    and weights applied to both arms at every replicate.

    Returns a dict with per-replicate deltas (only for replicates where BOTH
    arms yielded a valid AUC), plus explicit counts of invalid/one-class
    replicates. Invalid replicates are never silently dropped from the
    report — the count is always returned to the caller.
    """
    _validate_bootstrap_inputs(patient1_ids, patient2_ids, y_true, y_score_baseline, y_score_candidate, n_bootstrap)
    rng = np.random.default_rng(bootstrap_seed)  # independent of any model/attacker seed

    deltas = []
    baseline_aucs = []
    candidate_aucs = []
    one_class_count = 0

    for _ in range(n_bootstrap):
        draw_counts = draw_patient_multiplicities(list(patient1_ids) + list(patient2_ids), rng)
        weights = pair_weights_for_draw(patient1_ids, patient2_ids, draw_counts)

        auc_b = weighted_auc(y_true, y_score_baseline, weights, raise_on_one_class=False)
        auc_c = weighted_auc(y_true, y_score_candidate, weights, raise_on_one_class=False)
        if auc_b is None or auc_c is None:
            one_class_count += 1
            continue  # explicitly counted, never fabricated as 0 or skipped silently
        baseline_aucs.append(auc_b)
        candidate_aucs.append(auc_c)
        deltas.append(auc_c - auc_b)

    return {
        'status': STATUS,
        'n_bootstrap_requested': n_bootstrap,
        'n_valid': len(deltas),
        'n_one_class_invalid': one_class_count,
        'deltas': deltas,
        'baseline_aucs': baseline_aucs,
        'candidate_aucs': candidate_aucs,
        'bootstrap_seed': bootstrap_seed,
    }
