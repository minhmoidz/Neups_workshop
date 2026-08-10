"""STEP 2B regression suite for the frozen adaptive re-ID protocol.

Covers (cross-referenced to the task Parts):
  A  validation AUC/accuracy correctness            (Part 1, metrics)
  B  numerically invalid classification             (Part 3)
  C  near-chance is VALID, not excluded             (Part 3)
  D  weight-update detection                        (Part 4)
  E  confirmatory seed schedule                     (Part 5)
  F  screening seed schedule                        (Part 5)
  G  replacement only on NUMERICALLY_INVALID        (Part 5)
  H  representative selection on validation only    (Part 6)
  I  no test-derived inputs accepted                (Parts 3, 6, 13)
  J  mean/SD/median/max include near-chance attacks (Part 8)
  K  provenance contains required fields            (Part 12)
  L  determinism checker                            (Part 11)
  M  Top-k frozen-list construction                 (Part 10)
  N  patient-cluster bootstrap (R-9) status         (Part 9)
  O  legacy shared-operator regression              (Part 14)
  P  gradient-accumulation regression               (Part 14)
"""

import json
import os
import shutil
import tempfile

import numpy as np
import torch

import utils.utils as U

from adaptive_reid import diagnostics as diag
from adaptive_reid import health, metrics, weights, restarts, selection
from adaptive_reid import summary as summ
from adaptive_reid import determinism, provenance, bootstrap, topk

from adaptive_reid import constants as C


def _mini_mlp(seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))


# ----------------------------------------------------------------------------------
# A. validation metrics correctness
# ----------------------------------------------------------------------------------
def test_A_perfect_reversed_and_ties_auc():
    y = np.array([0, 0, 1, 1])
    perfect = np.array([0.1, 0.3, 0.6, 0.9])
    reversed_scores = np.array([0.9, 0.6, 0.3, 0.1])
    tied = np.array([0.5, 0.5, 0.5, 0.5])

    assert metrics.compute_auc(perfect, y) == 1.0
    assert metrics.compute_auc(reversed_scores, y) == 0.0
    assert metrics.compute_auc(tied, y) == 0.5
    print('A. AUC: perfect=1.0 reversed=0.0 ties=0.5 PASS')


def test_A_accuracy_correct():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.2, 0.6, 0.1, 0.9])
    acc = metrics.compute_accuracy(scores, y, threshold=0.5)
    # s>0.5 -> [0,1,0,1] => matches [0,0,1,1] at positions 0 and 3 = 0.5
    assert acc == 0.5
    print('A. Accuracy PASS')


def test_A_invalid_inputs_fail_loudly():
    for bad in (np.array([]), np.array([np.nan])):
        try:
            metrics.compute_auc(bad, np.array([0, 1]))
            raise AssertionError('should have raised')
        except ValueError:
            pass
    try:
        metrics.compute_auc(np.array([0.5, 0.5]), np.array([0, 0]))  # single class
        raise AssertionError('single-class AUC should raise')
    except ValueError:
        pass
    print('A. invalid inputs fail loudly PASS')


# ----------------------------------------------------------------------------------
# B / C. run health + near-chance
# ----------------------------------------------------------------------------------
def _valid_record(seed=0, best_loss=0.50, best_auc=0.80, near=False):
    rec = diag.build_training_diagnostics(
        attacker_seed=seed, transform_mode='legacy', mu=0.01, stochastic_lambda=0.0,
        generator_checkpoint_path='ck.pth', generator_checkpoint_hash='a' * 64,
        pair_train_path='t.txt', pair_validation_path='v.txt',
        pair_train_hash='b' * 64, pair_validation_hash='c' * 64,
        epochs_completed=12, termination_reason=C.TERMINATION_EARLY_STOPPING,
        training_loss_per_epoch=[0.5] * 12, validation_loss_per_epoch=[0.6] * 12,
        validation_auc_per_epoch=[0.7] * 12, validation_accuracy_per_epoch=[0.7] * 12,
        best_validation_loss=(C.NEAR_CHANCE_VAL_LOSS + 0.01) if near else best_loss,
        best_validation_loss_epoch=5,
        best_validation_auc=(C.NEAR_CHANCE_VAL_AUC - 0.01) if near else best_auc,
        best_validation_auc_epoch=7,
        any_nan_inf=False, checkpoint_exists=True, checkpoint_loadable=True,
        weights_changed_from_initialization=True,
        run_start_timestamp='2026-01-01T00:00:00+00:00', run_end_timestamp='2026-01-01T00:01:00+00:00')
    return rec


def test_B_numerically_invalid_conditions():
    base = _valid_record()

    cases = {
        'nan_inf': dict(base, any_nan_inf=True),
        'no_checkpoint': dict(base, checkpoint_exists=False),
        'corrupt_checkpoint': dict(base, checkpoint_loadable=False),
        'weights_unchanged': dict(base, weights_changed_from_initialization=False),
        'infra_termination': dict(base, termination_reason=C.TERMINATION_INFRASTRUCTURE),
        'illegal_termination': dict(base, termination_reason='crashed_weirdly'),
        'zero_epochs': dict(base, epochs_completed=0),
    }
    for name, rec in cases.items():
        state, near = health.classify_run_health(rec)
        assert state == C.NUMERICALLY_INVALID, name
        assert near is False, name
    # A *low-performing* but otherwise complete run must be VALID, never invalid.
    low = dict(base, best_validation_loss=0.9, best_validation_auc=0.51)
    state, near = health.classify_run_health(low)
    assert state == C.VALID and near is True
    print('B. numerically-invalid triggers + low-performance-not-excluded PASS')


def test_C_near_chance_is_valid_flag_not_exclusion():
    rec = _valid_record(near=True)
    state, near = health.classify_run_health(rec)
    assert state == C.VALID
    assert near is True
    print('C. near-chance flagged but remains VALID PASS')


# ----------------------------------------------------------------------------------
# D. weights-update detection
# ----------------------------------------------------------------------------------
def test_D_unchanged_vs_step():
    model = _mini_mlp(0)
    h0 = weights.snapshot_parameters(model)
    assert weights.weights_changed(h0, h0) is False

    x = torch.randn(8, 8)
    y = torch.rand(8).round()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.zero_grad()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x).squeeze(-1), y)
    loss.backward()
    opt.step()
    h1 = weights.snapshot_parameters(model)
    assert weights.weights_changed(h0, h1) is True
    print('D. unchanged=False, one-step=True PASS')


# ----------------------------------------------------------------------------------
# E / F / G. schedules + replacement
# ----------------------------------------------------------------------------------
def _fake_train(seed_to_state):
    """Return a train_and_report stub whose outcome depends only on the seed."""
    def worker(seed):
        state = seed_to_state.get(seed, C.VALID)
        return {'attacker_seed': seed, 'state': state, 'near_chance': False,
                'diagnostics': None, 'validation_record': None}
    return worker


def _results_for(schedule, worker):
    return restarts.run_schedule(schedule, worker)


def test_E_confirmatory_seeds_and_cap():
    good = {i: C.VALID for i in range(15)}
    attempts = _results_for(restarts.ConfirmatorySchedule(), _fake_train(good))
    seeds = [a['attacker_seed'] for a in attempts]
    assert seeds == list(range(10)), seeds  # exactly initial seeds, no extras needed
    assert len(attempts) == 10
    print('E. confirmatory: 10 valid on initial seeds 0..9 PASS')


def test_F_screening_seeds_and_cap():
    good = {i: C.VALID for i in range(5)}
    attempts = _results_for(restarts.ScreeningSchedule(), _fake_train(good))
    assert [a['attacker_seed'] for a in attempts] == [0, 1, 2]
    print('F. screening: 3 valid on seeds 0..2 PASS')


def test_G_replacement_only_on_invalid_and_ascending():
    # seed 2 invalid -> replaced with seed 3; seeds 0,1,3,4 valid keeps 3+... target 3
    lookup = {0: C.VALID, 1: C.VALID, 2: C.NUMERICALLY_INVALID, 3: C.VALID,
              4: C.VALID, 5: C.VALID}
    sched = restarts.ScreeningSchedule()
    attempts = _results_for(sched, _fake_train(lookup))
    seeds = [a['attacker_seed'] for a in attempts]
    assert seeds[:4] == [0, 1, 2, 3], seeds
    # it must stop at 3 valid runs (0,1,3) without sampling seed 4
    assert len(attempts) == 4, seeds
    print('G. replacement seed 2->3 ascending, stops at 3 valid PASS')


def test_G_replacement_never_for_near_chance():
    lookup = {0: C.VALID, 1: C.VALID, 2: C.VALID, 3: C.VALID, 4: C.VALID}
    # a low-performance completed run at seed 2 must remain VALID (never replaced)
    def worker(seed):
        return {'attacker_seed': seed,
                'state': C.VALID,
                'near_chance': (seed == 2),
                'diagnostics': None, 'validation_record': None}
    attempts = _results_for(restarts.ScreeningSchedule(), worker)
    assert len(attempts) == 3
    assert 2 in [a['attacker_seed'] for a in attempts]
    print('G. near-chance run never replaced PASS')


# ----------------------------------------------------------------------------------
# H / I. representative selection on validation only
# ----------------------------------------------------------------------------------
def _validation_record(seed, auc_series):
    return {'attacker_seed': seed, 'validation_auc_per_epoch': auc_series,
            'validation_loss_per_epoch': [0.5] * len(auc_series),
            'validation_accuracy_per_epoch': [0.7] * len(auc_series)}


def test_H_representative_closest_to_median():
    # best val AUCs: 0.50, 0.51, 0.60, 0.70, 0.72  -> median 0.60 (index 2)
    recs = [_validation_record(0, [0.50]), _validation_record(1, [0.51]),
            _validation_record(2, [0.60, 0.60]), _validation_record(3, [0.70]),
            _validation_record(4, [0.72])]
    rep = selection.select_representative(recs)
    assert rep == 2  # best val AUC 0.60 == median
    print('H. representative = closest-to-median best-val-AUC PASS')


def test_H_tie_break_smaller_seed():
    recs = [_validation_record(1, [0.50]), _validation_record(3, [0.50]),
            _validation_record(7, [0.60])]
    rep = selection.select_representative(recs)
    assert rep == 1
    print('H. tie broken by smaller seed PASS')


def test_H_even_count_median():
    # 0.50, 0.55, 0.63, 0.70 -> median (0.55+0.63)/2 = 0.59
    recs = [_validation_record(0, [0.50]), _validation_record(1, [0.55]),
            _validation_record(2, [0.63]), _validation_record(3, [0.70])]
    # closest to 0.59: seed2 (0.04) vs seed1 (0.04) tie -> smaller seed 1
    rep = selection.select_representative(recs)
    assert rep == 1
    print('H. even-count median + tie break PASS')


def test_I_selector_accepts_no_test_argument():
    sig = selection.select_representative
    import inspect
    params = inspect.signature(sig).parameters
    for forbidden in ('test_auc', 'test_predictions', 'test_labels', 'test_metrics'):
        assert forbidden not in params, forbidden
    print('I. selector signature has no test-derived parameter PASS')


def test_I_health_accepts_no_test_argument():
    import inspect
    params = inspect.signature(health.classify_run_health).parameters
    assert list(params) == ['training_diagnostics']
    print('I. health signature has no test-derived parameter PASS')


def test_I_diagnostics_file_has_no_test_field():
    rec = _valid_record()
    for key in ('test_auc', 'test_predictions', 'test_labels', 'test_metrics'):
        assert key not in rec, key
    print('I. validity JSON contains no test field PASS')


# ----------------------------------------------------------------------------------
# J. aggregation includes near-chance runs
# ----------------------------------------------------------------------------------
def _attempt(seed, state, near, test_auc):
    return {'attacker_seed': seed, 'state': state, 'near_chance': near,
            'test_metrics': None if test_auc is None else {'auc': test_auc},
            'is_representative': False}


def test_J_aggregation_includes_near_chance():
    attempts = [
        _attempt(0, C.VALID, near=True, test_auc=0.50),
        _attempt(1, C.VALID, near=True, test_auc=0.52),
        _attempt(2, C.VALID, near=False, test_auc=0.61),
        _attempt(3, C.VALID, near=False, test_auc=0.68),
        _attempt(4, C.NUMERICALLY_INVALID, near=False, test_auc=None),
    ]
    s = summ.summarize_arm(attempts)
    assert s['n_attempted'] == 5
    assert s['n_numerically_invalid'] == 1
    assert s['n_valid'] == 4
    assert s['n_near_chance'] == 2
    assert abs(s['mean_test_auc'] - (0.50 + 0.52 + 0.61 + 0.68) / 4) < 1e-12
    assert abs(s['median_test_auc'] - (0.52 + 0.61) / 2) < 1e-12
    assert abs(s['max_test_auc'] - 0.68) < 1e-12
    assert 0.50 in s['test_auc_values'] and 0.52 in s['test_auc_values']
    print('J. mean/median/max include near-chance attacks PASS')


# ----------------------------------------------------------------------------------
# K. provenance fields
# ----------------------------------------------------------------------------------
def test_K_provenance_required_fields():
    p = provenance.build_arm_provenance(
        arm_id='arm_corrected_mu0.01', git_commit='abc123', transform_mode='corrected',
        generator_checkpoint_path='ck.pth', generator_checkpoint_hash='a' * 64,
        mu=0.01, stochastic_lambda=0.0, attacker_architecture='ResNet-50 Siamese',
        attacker_hyperparameters={'lr': 1e-4, 'batch_size': 16},
        attacker_seeds_attempted=[0, 1, 2, 3], pair_train_path='t',
        pair_validation_path='v', pair_test_path='e', pair_train_hash='b',
        pair_validation_hash='c', pair_test_hash='d',
        representative_attacker_seed=1, representative_selection_criterion='median-best-val-auc',
        run_states={0: 'VALID'}, near_chance_flags={0: False},
        run_start_timestamp='t0', run_end_timestamp='t1', schedule_name='confirmatory')
    required = ['arm_id', 'git_commit', 'transform_mode', 'generator_checkpoint_path',
                'generator_checkpoint_hash', 'mu', 'stochastic_lambda', 'attacker_architecture',
                'attacker_hyperparameters', 'attacker_seeds_attempted', 'pair_train_path',
                'pair_validation_path', 'pair_test_path', 'pair_train_hash',
                'pair_validation_hash', 'pair_test_hash', 'representative_attacker_seed',
                'representative_selection_criterion', 'run_states', 'near_chance_flags',
                'run_start_timestamp', 'run_end_timestamp', 'schedule_name']
    for field in required:
        assert field in p, field
    # legacy vs corrected cannot be confused
    assert p['transform_mode'] == 'corrected'
    print('K. provenance required fields + mode granted PASS')


# ----------------------------------------------------------------------------------
# L. determinism checker
# ----------------------------------------------------------------------------------
def test_L_determinism_checker():
    rng = np.random.RandomState(7)
    fix = torch.from_numpy(rng.rand(2, 4)).float()

    def gen_det():
        return torch.nn.functional.relu(fix) + 1.0

    res = determinism.check_deterministic(gen_det)
    assert res['deterministic'] is True and res['max_abs_diff'] == 0.0

    def gen_stoch():
        return fix + torch.rand_like(fix)

    res2 = determinism.check_deterministic(gen_stoch)
    assert determinism.arm_is_stochastic(res2) is True
    print('L. deterministic vs stochastic detection PASS')


# ----------------------------------------------------------------------------------
# M. Top-k frozen list construction
# ----------------------------------------------------------------------------------
def test_M_topk_frozen_list_build(tmp_path):
    df = topk.build_frozen_topk_list(n_patients=10)
    assert len(df) == 10
    required_cols = {'patient_id', 'gallery_image', 'gallery_followup',
                     'probe_image', 'probe_followup'}
    assert required_cols.issubset(set(df.columns))
    # deterministic: same seed -> same list
    df2 = topk.build_frozen_topk_list(n_patients=10)
    assert df.equals(df2)
    path = os.path.join(_td_path(tmp_path), FROZEN := topk.FROZEN_LIST_FILENAME)
    topk.save_frozen_topk_list(path, df)
    assert os.path.exists(path)
    loaded = topk.load_frozen_topk_list(path)
    assert loaded.equals(df)
    print('M. frozen Top-k list construction + determinism PASS')


# ----------------------------------------------------------------------------------
# N. patient-cluster bootstrap (R-9)
# ----------------------------------------------------------------------------------
def _td_path(tmp_path):
    """Unwrap pytest tmp_path or tempfile.TemporaryDirectory into a str path."""
    if hasattr(tmp_path, 'name'):
        return str(tmp_path.name)
    return str(tmp_path)


def test_N_R9_ambiguity_detected(tmp_path):
    pairs = os.path.join(_td_path(tmp_path), 'pairs.txt')
    os.makedirs(os.path.dirname(pairs), exist_ok=True)
    with open(pairs, 'w') as f:
        # positive: same patient 0001; negative: two patients
        f.write('0001_000.png 0001_001.png 1.0\n')
        f.write('0002_000.png 0003_000.png 0.0\n')
    assert bootstrap.patient_cluster_bootstrap_is_ambiguous(pairs) is True
    rec = bootstrap.report_R9_ambiguity(pairs)
    assert rec['R9_status'] == 'BLOCKED_FOR_SCIENTIFIC_CLARIFICATION'
    assert rec['pair_statistics']['negative_pairs_span_two_patients'] == 1
    print('N. R-9 ambiguity documented, not guessed PASS')


def test_N_pair_schema_statistics(tmp_path):
    pairs = os.path.join(_td_path(tmp_path), 'pairs.txt')
    os.makedirs(os.path.dirname(pairs), exist_ok=True)
    with open(pairs, 'w') as f:
        f.write('0001_000.png 0001_001.png 1.0\n')
        f.write('0002_000.png 0003_000.png 0.0\n')
    stats = bootstrap.compute_patient_statistics(pairs)
    assert stats['positive_pairs_single_patient'] == 1
    assert stats['negative_pairs_span_two_patients'] == 1
    print('N. pair-schema statistics PASS')


# ----------------------------------------------------------------------------------
# O / P. backward compatibility
# ----------------------------------------------------------------------------------
def _has_fn(module, name):
    return hasattr(module, name)


def test_O_shared_operator_still_routes_via_build_sampling_grid():
    import inspect
    for fn in (U.deform, U.pretrain, U.preval):
        src = inspect.getsource(fn)
        assert 'build_sampling_grid(' in src, fn
    assert U.resolve_transform_mode(None) == 'legacy'
    print('O. shared operator regression PASS')


def test_P_gradient_accumulation_regression_imports_and_helpers():
    # quick smoke: recompute the accumulate vs doubled-batch identity torch-level
    from test_grad_accum import test_grad_accumulation_matches_doubled_batch as t
    t()
    print('P. gradient-accumulation regression PASS')


def test_P_diagnostics_json_hasno_test_mixin(tmp_path):
    rec = _valid_record()
    p = os.path.join(_td_path(tmp_path), diag.VALIDITY_FILENAME)
    diag.write_json(p, rec)
    with open(p) as f:
        raw = json.load(f)
    assert raw['transform_mode'] == 'legacy'
    assert 'test_auc' not in raw
    print('P. diagnostics JSON write/read PASS')


# ----------------------------------------------------------------------------------
# Stage-ordering anti-leakage (Part 13 #5): test eval must not alter representative
# ----------------------------------------------------------------------------------
def test_stage_order_test_never_selects(tmp_path):
    from adaptive_reid import pipeline as pl_mod
    from adaptive_reid import summary as summ_mod

    def train_v(seed):
        # validation series whose best-AUC median selection is forced to seed 1
        aucs = {0: [0.60], 1: [0.65], 2: [0.70]}
        return {
            'attacker_seed': seed,
            'state': C.VALID,
            'near_chance': False,
            'diagnostics': {'attacker_seed': seed, 'no_test': True},
            'validation_record': {'attacker_seed': seed,
                                  'validation_auc_per_epoch': aucs[seed],
                                  'validation_loss_per_epoch': [0.4],
                                  'validation_accuracy_per_epoch': [0.7]},
        }

    def test_eval(seed):
        # a "test metric" that would happily pick a different run if it were used
        return {'auc': 1.0 if seed == 2 else 0.0}

    def select_from_validation(records):
        # deliberately a broken selector ONLY IF test metrics were visible -- it uses
        # only the validation record, so returns 1 (median-ish), never 2
        bests = [(r['attacker_seed'], max(r['validation_auc_per_epoch'])) for r in records]
        bests_sorted = sorted(x[1] for x in bests)
        med = bests_sorted[len(bests_sorted) // 2]
        return [s for s, b in bests if b == med][0]

    attempts = [{'attacker_seed': s} for s in (0, 1, 2)]
    pl = pl_mod.ArmPipeline(train_validate_and_persist=train_v,
                            evaluate_test=test_eval,
                            select_representative_from_validation=select_from_validation)
    # run full pipeline; verify representative frozen at Stage D, test can't change it
    summary = pl.run(attempts)
    assert summary['representative_attacker_seed'] == 1
    assert attempt_has_rep(attempts, 1)
    # test metrics are present but didn't influence choice
    assert attempts[2]['test_metrics']['auc'] == 1.0
    print('X. test eval ran but could not alter representative selection PASS')


def attempt_has_rep(attempts, seed):
    for a in attempts:
        if a['attacker_seed'] == seed:
            return a.get('is_representative') is True
    return False


# ----------------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    import tempfile as _tf

    # tmp_path emulation
    TMP = tempfile.mkdtemp(prefix='t2b_')
    print('TMP:', TMP)

    test_A_perfect_reversed_and_ties_auc()
    test_A_accuracy_correct()
    test_A_invalid_inputs_fail_loudly()
    test_B_numerically_invalid_conditions()
    test_C_near_chance_is_valid_flag_not_exclusion()
    test_D_unchanged_vs_step()
    test_E_confirmatory_seeds_and_cap()
    test_F_screening_seeds_and_cap()
    test_G_replacement_only_on_invalid_and_ascending()
    test_G_replacement_never_for_near_chance()
    test_H_representative_closest_to_median()
    test_H_tie_break_smaller_seed()
    test_H_even_count_median()
    test_I_selector_accepts_no_test_argument()
    test_I_health_accepts_no_test_argument()
    test_I_diagnostics_file_has_no_test_field()
    test_J_aggregation_includes_near_chance()
    test_K_provenance_required_fields()
    test_L_determinism_checker()
    test_M_topk_frozen_list_build(_tf.TemporaryDirectory())
    test_N_R9_ambiguity_detected(_tf.TemporaryDirectory())
    test_N_pair_schema_statistics(_tf.TemporaryDirectory())
    test_O_shared_operator_still_routes_via_build_sampling_grid()
    test_P_gradient_accumulation_regression_imports_and_helpers()
    test_P_diagnostics_json_hasno_test_mixin(_tf.TemporaryDirectory())
    test_stage_order_test_never_selects(_tf.TemporaryDirectory())

    print('\nSTEP 2B PROTOCOL TEST SUITE: PASS')
    sys.exit(0)