"""Synthetic coverage / type-I-error simulation for the candidate patient-graph
bootstrap (G0.2 §9). CPU-only, fully synthetic — no real project data.

Run: .venv/bin/python reproduction/statistics/simulate_bootstrap_coverage.py

SETTINGS below are predeclared BEFORE any scenario is run or inspected, per
the governance requirement in reproduction/reports/G0_HYPOTHESIS_GATE_AUDIT_2026-08-21.md
§9 ("do not tune thresholds or alter the method after observing results").
This is statistical method validation on synthetic data — it is not a
scientific claim about the real PriCheXy-Net/M2 method.
"""
import time

import numpy as np

from patient_graph_bootstrap import (
    patient_graph_bootstrap_paired, draw_patient_multiplicities, weighted_auc,
)

# ---------------------------------------------------------------------------
# PREDECLARED SETTINGS — frozen before any scenario is executed or inspected.
# ---------------------------------------------------------------------------
SETTINGS = {
    'monte_carlo_seed': 20260821,
    'n_simulated_datasets_per_scenario': 60,
    'n_bootstrap_replicates_per_dataset': 100,
    'coverage_target': 0.95,
    'coverage_mc_band': 0.06,          # allowed deviation given n=60 datasets (binomial MC noise)
    'type1_alpha': 0.05,
    'type1_mc_band': 0.05,             # allowed deviation given n=60 datasets
    'balanced_scenario_invalid_resample_rate_max': 0.01,
    'non_inferiority_margin': 0.03,
}


def _make_patient_population(n_patients: int, degree_kind: str, rng: np.random.Generator):
    """degree_kind: 'uniform' or 'high_degree' (a few patients appear far more)."""
    if degree_kind == 'uniform':
        weights = np.ones(n_patients)
    else:
        weights = np.ones(n_patients)
        n_hubs = max(1, n_patients // 20)
        hub_idx = rng.choice(n_patients, size=n_hubs, replace=False)
        weights[hub_idx] = 12.0  # a handful of patients ~12x more likely to be sampled into pairs
    return weights / weights.sum()


def _simulate_dataset(n_patients: int, n_pairs: int, degree_kind: str, label_balance: float,
                       patient_correlation: float, true_delta: float, sparse: bool,
                       rng: np.random.Generator):
    """Build one synthetic paired-comparison dataset with a KNOWN true delta
    (candidate_AUC_true - baseline_AUC_true, in expectation over the patient
    population), patient-level score correlation, and optional sparsity.

    Ground truth: baseline separability fixed at a moderate level; candidate
    shifted by `true_delta` via a patient-specific latent trait that both
    arms partially share (patient_correlation controls how much) plus an
    arm-specific component. Positive/negative pair status is generated from
    the same patient population that supplies the scores (positives share
    a patient; negatives draw two distinct patients).
    """
    patient_weights = _make_patient_population(n_patients, degree_kind, rng)
    patient_ids = np.arange(n_patients)

    # Latent per-patient trait, shared across arms (drives patient_correlation).
    patient_trait = rng.normal(0, 1, size=n_patients)

    n_pos = int(round(n_pairs * label_balance))
    n_neg = n_pairs - n_pos
    if sparse:
        # Restrict positives to a small subset of patients with enough occurrences
        # to form pairs (mimics a sparse pair graph with limited same-patient pairs).
        eligible = rng.choice(n_patients, size=max(2, n_patients // 4), replace=False,
                               p=patient_weights / patient_weights.sum())
        pos_patients = rng.choice(eligible, size=n_pos, replace=True)
    else:
        pos_patients = rng.choice(patient_ids, size=n_pos, replace=True, p=patient_weights)

    p1_pos = pos_patients
    p2_pos = pos_patients  # same patient -> positive pair by construction

    p1_neg = rng.choice(patient_ids, size=n_neg, replace=True, p=patient_weights)
    p2_neg = rng.choice(patient_ids, size=n_neg, replace=True, p=patient_weights)
    same_mask = p1_neg == p2_neg
    while same_mask.any():
        p2_neg[same_mask] = rng.choice(patient_ids, size=same_mask.sum(), replace=True, p=patient_weights)
        same_mask = p1_neg == p2_neg

    patient1 = np.concatenate([p1_pos, p1_neg])
    patient2 = np.concatenate([p2_pos, p2_neg])
    y_true = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

    def pair_score(p1, p2, arm_shift, arm_noise_scale):
        shared = patient_correlation * (patient_trait[p1] + patient_trait[p2]) / 2
        idio1 = (1 - patient_correlation) * rng.normal(0, 1, size=len(p1))
        signal = np.where(p1 == p2, 1.0, -1.0)  # positives score higher than negatives on average
        raw = signal + shared + idio1 * arm_noise_scale + arm_shift
        return 1 / (1 + np.exp(-raw))  # squash to (0,1) like a real score

    # Baseline fixed moderate separability. Candidate's arm_shift crafted so
    # that, in expectation over MANY simulated datasets at these settings,
    # candidate_AUC - baseline_AUC ~ true_delta (verified empirically in the
    # module smoke-check at the bottom, not assumed).
    y_score_baseline = pair_score(patient1, patient2, arm_shift=0.0, arm_noise_scale=1.0)
    y_score_candidate = pair_score(patient1, patient2, arm_shift=true_delta * 2.5, arm_noise_scale=1.0)

    return patient1, patient2, y_true, y_score_baseline, y_score_candidate


def _ordinary_pair_bootstrap_paired(y_true, y_score_baseline, y_score_candidate,
                                     n_bootstrap, bootstrap_seed):
    """Sensitivity-only comparator: resamples PAIR ROWS directly, ignoring
    patient identity entirely. Deliberately naive (per G0.1 §4.1 item 3)."""
    rng = np.random.default_rng(bootstrap_seed)
    n = len(y_true)
    deltas = []
    one_class = 0
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            one_class += 1
            continue
        from sklearn.metrics import roc_auc_score
        auc_b = roc_auc_score(yt, y_score_baseline[idx])
        auc_c = roc_auc_score(yt, y_score_candidate[idx])
        deltas.append(auc_c - auc_b)
    return {'deltas': deltas, 'n_one_class_invalid': one_class}


def run_scenario(name, n_patients, n_pairs, degree_kind, label_balance,
                  patient_correlation, true_delta, sparse, settings):
    rng = np.random.default_rng(settings['monte_carlo_seed'])
    n_cover_patient = 0
    n_cover_pair = 0
    n_reject_h0_patient = 0  # "reject H0: delta >= margin" i.e. one-sided upper CI bound < margin
    n_reject_h0_pair = 0
    widths_patient = []
    widths_pair = []
    invalid_rates_patient = []
    invalid_rates_pair = []

    for d in range(settings['n_simulated_datasets_per_scenario']):
        p1, p2, y_true, y_b, y_c = _simulate_dataset(
            n_patients, n_pairs, degree_kind, label_balance, patient_correlation,
            true_delta, sparse, rng)

        boot_seed = int(rng.integers(0, 2 ** 31 - 1))
        res_patient = patient_graph_bootstrap_paired(
            p1, p2, y_true, y_b, y_c,
            n_bootstrap=settings['n_bootstrap_replicates_per_dataset'], bootstrap_seed=boot_seed)
        deltas_p = np.array(res_patient['deltas'])
        invalid_rates_patient.append(
            res_patient['n_one_class_invalid'] / max(res_patient['n_bootstrap_requested'], 1))

        res_pair = _ordinary_pair_bootstrap_paired(
            y_true, y_b, y_c, n_bootstrap=settings['n_bootstrap_replicates_per_dataset'],
            bootstrap_seed=boot_seed)
        deltas_pair = np.array(res_pair['deltas'])
        invalid_rates_pair.append(
            res_pair['n_one_class_invalid'] / max(settings['n_bootstrap_replicates_per_dataset'], 1))

        if len(deltas_p) >= 2:
            lo_p, hi_p = np.percentile(deltas_p, [2.5, 97.5])
            widths_patient.append(hi_p - lo_p)
            if lo_p <= true_delta <= hi_p:
                n_cover_patient += 1
            upper95_p = np.percentile(deltas_p, 95)
            if upper95_p < settings['non_inferiority_margin']:
                n_reject_h0_patient += 1

        if len(deltas_pair) >= 2:
            lo_q, hi_q = np.percentile(deltas_pair, [2.5, 97.5])
            widths_pair.append(hi_q - lo_q)
            if lo_q <= true_delta <= hi_q:
                n_cover_pair += 1
            upper95_q = np.percentile(deltas_pair, 95)
            if upper95_q < settings['non_inferiority_margin']:
                n_reject_h0_pair += 1

    n = settings['n_simulated_datasets_per_scenario']
    is_null_scenario = abs(true_delta - settings['non_inferiority_margin']) < 1e-9 or true_delta == 0.0
    return {
        'scenario': name,
        'true_delta': true_delta,
        'patient_bootstrap': {
            'empirical_coverage_95': n_cover_patient / n,
            'mean_ci_width': float(np.mean(widths_patient)) if widths_patient else None,
            'mean_invalid_resample_rate': float(np.mean(invalid_rates_patient)),
            'type1_error_if_null_scenario': (n_reject_h0_patient / n) if is_null_scenario else None,
        },
        'pair_bootstrap_sensitivity_only': {
            'empirical_coverage_95': n_cover_pair / n,
            'mean_ci_width': float(np.mean(widths_pair)) if widths_pair else None,
            'mean_invalid_resample_rate': float(np.mean(invalid_rates_pair)),
            'type1_error_if_null_scenario': (n_reject_h0_pair / n) if is_null_scenario else None,
        },
    }


SCENARIOS = [
    dict(name='balanced_uniform_null', n_patients=80, n_pairs=300, degree_kind='uniform',
         label_balance=0.5, patient_correlation=0.3, true_delta=0.0, sparse=False),
    dict(name='balanced_uniform_margin_null', n_patients=80, n_pairs=300, degree_kind='uniform',
         label_balance=0.5, patient_correlation=0.3, true_delta=SETTINGS['non_inferiority_margin'], sparse=False),
    dict(name='balanced_high_degree_null', n_patients=80, n_pairs=300, degree_kind='high_degree',
         label_balance=0.5, patient_correlation=0.3, true_delta=0.0, sparse=False),
    dict(name='imbalanced_uniform_null', n_patients=80, n_pairs=300, degree_kind='uniform',
         label_balance=0.2, patient_correlation=0.3, true_delta=0.0, sparse=False),
    dict(name='sparse_correlated_null', n_patients=60, n_pairs=150, degree_kind='high_degree',
         label_balance=0.5, patient_correlation=0.7, true_delta=0.0, sparse=True),
    dict(name='alternative_improvement', n_patients=80, n_pairs=300, degree_kind='uniform',
         label_balance=0.5, patient_correlation=0.3, true_delta=-0.06, sparse=False),
]


def main():
    t0 = time.time()
    results = []
    for sc in SCENARIOS:
        r = run_scenario(settings=SETTINGS, **sc)
        results.append(r)
        print('%-30s patient_cov=%.3f pair_cov=%.3f patient_type1=%s pair_type1=%s' % (
            sc['name'], r['patient_bootstrap']['empirical_coverage_95'],
            r['pair_bootstrap_sensitivity_only']['empirical_coverage_95'],
            r['patient_bootstrap']['type1_error_if_null_scenario'],
            r['pair_bootstrap_sensitivity_only']['type1_error_if_null_scenario']))
    elapsed = time.time() - t0
    print('Total CPU runtime: %.1fs' % elapsed)
    return results, elapsed


if __name__ == '__main__':
    main()
