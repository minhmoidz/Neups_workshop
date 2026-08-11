"""STEP 2B + STEP 2B.1 remediation regression suite.

Covers (cross-referenced to the task Parts):
  A  validation AUC/accuracy correctness            (Part 1, metrics)
  A2 accuracy boundary is sigmoid(logit)>=0.5      (BLOCKER 1; logits -3..2 case)
  B  numerically invalid classification             (Part 3)
  C  near-chance is VALID, not excluded             (Part 3)
  D  weight-update detection                        (Part 4)
  D2 exactly one training invocation per seed       (BLOCKER 2; no re-train in later stages)
  D3 idempotent reuse of completed runs             (BLOCKER 2B; run signature)
  E  confirmatory seed schedule                     (Part 5)
  E2 confirmatory ascending replacement on final    (BLOCKER 2A; seeds 0..9 then 10,11,...)
     invalid-only state
  E3 stub markers + non-stub test eval raises       (BLOCKER 3; R-11)
  F  screening seed schedule                        (Part 5)
  G  replacement only on NUMERICALLY_INVALID        (Part 5)
  H  representative selection on validation only    (Part 6)
  I  no test-derived inputs accepted                (Parts 3, 6, 13)
  J  mean/SD/median/max include near-chance attacks (Part 8)
  J2 sample SD uses ddof=1                          (AMENDMENT 1; [0.50,.52,.60,.70])
  J3 summary refuses synthetic/stub metrics         (BLOCKER 3; R-11)
  K  provenance contains required fields            (Part 12)
  K2 provenance carries protocol + frozen hashes    (AMENDMENT 2 / R-7 / R-12)
  L  determinism checker                            (Part 11)
  M  Top-k frozen-list construction                 (Part 10)
  N  R-9 final policy (patient-cluster withdrawn)   (Part 9 / amendment §6)
  O  legacy shared-operator regression              (Part 14)
  P  gradient-accumulation regression               (Part 14, CUDA skip-guarded)
  Z2 STEP 2B.2 real training-diagnostics wiring     (real per-epoch arrays, no test
     fields, best-epoch accounting, parameter-hash / checkpoint reality, health
     consumes the persisted file, loadable-checkpoint reuse, infra never reused)
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

import utils.utils as U

from adaptive_reid import diagnostics as diag
from adaptive_reid import health, metrics, weights, restarts, selection
from adaptive_reid import summary as summ
from adaptive_reid import determinism, provenance, bootstrap, topk
from adaptive_reid import pipeline as pl_mod

from adaptive_reid import constants as C

import run_adaptive_reid_arm as runner


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


def test_A2_accuracy_boundary_is_sigmoid_logit():
    # BLOCKER 1 regression: accuracy must use the 0.5 PROBABILITY boundary
    # (sigmoid(logit) >= 0.5, equivalently logit >= 0.0), never raw-logit 0.5.
    logits = np.array([-3.0, -0.2, 0.2, 0.3, 0.6, 2.0])
    labels = np.array([0, 0, 1, 1, 1, 1])
    acc = metrics.compute_accuracy_from_logits(logits, labels)
    assert acc == 1.0, acc
    # the old buggy rule (raw logits > 0.5) gives 4/6 on the same inputs
    buggy = float(np.mean((logits > 0.5).astype(np.int64) == labels))
    assert abs(buggy - 4 / 6) < 1e-12
    # exact boundary: sigmoid(0.0) == 0.5 counts as the positive class
    assert metrics.compute_accuracy_from_logits([0.0], [1]) == 1.0
    assert metrics.compute_accuracy_from_logits([-1e-9], [0]) == 1.0
    assert metrics.compute_accuracy_from_logits([0.0], [0]) == 0.0
    print('A2. accuracy boundary == sigmoid(logit)>=0.5 (BLOCKER 1) PASS')


def test_A2_validate_snn_routes_logits_accuracy():
    # The live validation path must route logits through the logits-aware estimator.
    import inspect
    src = inspect.getsource(U.validate_snn)
    assert 'validation_metrics_from_logits' in src
    print('A2. validate_snn uses logits-aware accuracy (BLOCKER 1) PASS')


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
# D2 / D3 / E2 / E3. STEP 2B.1 remediation: single training, idempotent reuse,
# invalid-only ascending replacement, and no fabricated test metrics.
# ----------------------------------------------------------------------------------
def test_D2_single_training_invocation_per_seed():
    calls = []

    def worker(seed):
        calls.append(seed)
        aucs = {0: [0.60], 1: [0.65], 2: [0.70]}
        return {'attacker_seed': seed, 'state': C.VALID, 'near_chance': False,
                'diagnostics': {'attacker_seed': seed},
                'validation_record': {'attacker_seed': seed,
                                      'validation_auc_per_epoch': aucs.get(seed, [0.60]),
                                      'validation_loss_per_epoch': [0.4],
                                      'validation_accuracy_per_epoch': [0.7]}}

    attempts = restarts.run_schedule(restarts.ScreeningSchedule(), worker)
    assert calls == [0, 1, 2]
    n_after_schedule = len(calls)
    assert n_after_schedule == 3
    # Later stages (B/C/D) must NOT invoke the training worker again (BLOCKER 2).
    pl = pl_mod.ArmPipeline(train_validate_and_persist=worker, evaluate_test=None)
    pl.stage_b_classify(attempts)
    rep = pl.stage_c_select_representative(attempts)
    pl.stage_d_persist_representative(attempts, rep, {})
    assert len(calls) == 3, 'a later stage re-trained a restart (BLOCKER 2): %s' % calls
    print('D2. exactly one training invocation per seed (BLOCKER 2) PASS')


def test_E2_confirmatory_ascending_replacement_on_invalid_only():
    lookup = {s: C.VALID for s in range(15)}
    lookup[5] = C.NUMERICALLY_INVALID
    lookup[8] = C.NUMERICALLY_INVALID

    def worker(seed):
        return {'attacker_seed': seed, 'state': lookup[seed], 'near_chance': False,
                'diagnostics': None, 'validation_record': None}

    attempts = restarts.run_schedule(restarts.ConfirmatorySchedule(), worker)
    seeds = [a['attacker_seed'] for a in attempts]
    assert seeds[:10] == list(range(10)), seeds
    # two NUMERICALLY_INVALID initial seeds are replaced strictly ascending: 10, 11
    assert seeds == list(range(10)) + [10, 11], seeds
    assert sum(1 for a in attempts if a['state'] == C.VALID) == 10
    print('E2. confirmatory ascending replacement, invalid-only, final-state (BLOCKER 2A) PASS')


def test_D3_idempotent_reuse_of_completed_run(tmp_path):
    import argparse
    td = _td_path(tmp_path)
    args = argparse.Namespace(arm_id='arm_x', transform_mode='corrected', mu=0.01,
                              stochastic_lambda=0.0, checkpoint=None, stub=True,
                              force=False)
    cfg = {'learning_rate': 1e-4, 'batch_size': 16, 'max_epochs': 100,
           'early_stopping': {'patience': 5}}
    pair_hashes = {runner.PAIR_TRAIN: 'a', runner.PAIR_VAL: 'b'}
    protocol_documents = {runner.PROTOCOL_01: 'c', runner.PROTOCOL_01B: 'd'}
    frozen_artifacts = {runner.FROZEN_TOPK_CSV: 'e'}
    sig = runner.run_signature(3, args, cfg, pair_hashes, protocol_documents, frozen_artifacts)

    run_dir = os.path.join(td, 'runs', 'retrain_snn_seed3')
    os.makedirs(run_dir, exist_ok=True)
    # nothing present -> must train
    assert runner.reuse_completed_run(run_dir, sig, 3, args) is None

    rec = diag.build_training_diagnostics(
        attacker_seed=3, transform_mode='corrected', mu=0.01, stochastic_lambda=0.0,
        generator_checkpoint_path='', generator_checkpoint_hash='',
        pair_train_path=runner.PAIR_TRAIN, pair_validation_path=runner.PAIR_VAL,
        pair_train_hash='a', pair_validation_hash='b',
        epochs_completed=8, termination_reason=C.TERMINATION_EARLY_STOPPING,
        training_loss_per_epoch=[0.6] * 8, validation_loss_per_epoch=[0.62] * 8,
        validation_auc_per_epoch=[0.55] * 8, validation_accuracy_per_epoch=[0.55] * 8,
        best_validation_loss=0.62, best_validation_loss_epoch=0,
        best_validation_auc=0.55, best_validation_auc_epoch=0,
        any_nan_inf=False, checkpoint_exists=True, checkpoint_loadable=True,
        weights_changed_from_initialization=True,
        run_start_timestamp='t0', run_end_timestamp='t1')
    diag.write_json(os.path.join(run_dir, diag.VALIDITY_FILENAME), rec)
    diag.write_json(os.path.join(run_dir, diag.RUNSTATE_FILENAME),
                    {'state': C.VALID, 'near_chance': False})
    diag.write_json(os.path.join(run_dir, runner.SIGNATURE_FILENAME), sig)

    loaded = runner.reuse_completed_run(run_dir, sig, 3, args)
    assert loaded is not None and loaded['attacker_seed'] == 3
    # changing any signed input (here: protocol doc hash) invalidates reuse
    sig2 = dict(sig)
    sig2['protocol_documents'] = {runner.PROTOCOL_01: 'CHANGED', runner.PROTOCOL_01B: 'd'}
    assert runner.reuse_completed_run(run_dir, sig2, 3, args) is None
    print('D3. idempotent reuse + signature invalidation (BLOCKER 2B) PASS')


def test_E3_stub_markers_and_nonstub_not_implemented():
    m = runner.stub_test_metrics(4)
    assert m['stub'] is True and m['synthetic'] is True
    assert m['valid_for_scientific_reporting'] is False
    try:
        runner.require_real_test_eval(False)
        raise AssertionError('non-stub Stage E must raise NotImplementedError')
    except NotImplementedError:
        pass
    runner.require_real_test_eval(True)  # stub path is allowed
    print('E3. stub markers + non-stub NotImplementedError (BLOCKER 3) PASS')


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


def test_J2_sample_sd_uses_ddof1():
    attempts = []
    for i, auc in enumerate([0.50, 0.52, 0.60, 0.70]):
        attempts.append({'attacker_seed': i, 'state': C.VALID, 'near_chance': False,
                         'test_metrics': {'auc': auc}, 'is_representative': False})
    s = summ.summarize_arm(attempts)
    arr = np.asarray([0.50, 0.52, 0.60, 0.70])
    assert abs(s['std_test_auc'] - arr.std(ddof=1)) < 1e-12
    assert abs(s['std_test_auc'] - arr.std(ddof=0)) > 1e-12
    # n == 1 -> sample SD undefined (None), never NaN emitted as a finished number
    s1 = summ.summarize_arm([{'attacker_seed': 0, 'state': C.VALID, 'near_chance': False,
                              'test_metrics': {'auc': 0.5}, 'is_representative': True}])
    assert s1['std_test_auc'] is None
    print('J2. restart SD uses ddof=1 sample SD (AMENDMENT 1) PASS')


def test_J3_summary_refuses_synthetic_metrics():
    attempts = [
        {'attacker_seed': 0, 'state': C.VALID, 'near_chance': False,
         'test_metrics': {'auc': 0.90, 'stub': True, 'synthetic': True,
                          'valid_for_scientific_reporting': False},
         'is_representative': True},
        {'attacker_seed': 1, 'state': C.VALID, 'near_chance': False,
         'test_metrics': {'auc': 0.91, 'valid_for_scientific_reporting': False},
         'is_representative': False},
    ]
    s = summ.summarize_arm(attempts)
    assert s['contains_stub_or_synthetic_metrics'] is True
    assert s['n_stub_or_synthetic_test_metrics'] == 2
    assert s['mean_test_auc'] is None and s['median_test_auc'] is None
    assert s['max_test_auc'] is None and s['std_test_auc'] is None
    assert s['test_auc_values'] == []
    assert s['scientific_summary_available'] is False
    # mixed: one real + one synthetic -> only the real one enters the aggregates
    attempts[1]['test_metrics'] = {'auc': 0.60}
    s2 = summ.summarize_arm(attempts)
    assert s2['mean_test_auc'] == 0.60
    assert s2['max_test_auc'] == 0.60
    assert s2['n_stub_or_synthetic_test_metrics'] == 1
    print('J3. summary refuses synthetic/stub metrics (BLOCKER 3 / R-11) PASS')


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
                'run_start_timestamp', 'run_end_timestamp', 'schedule_name',
                'protocol_documents', 'frozen_artifacts']
    for field in required:
        assert field in p, field
    # legacy vs corrected cannot be confused
    assert p['transform_mode'] == 'corrected'
    assert p['protocol_documents'] == {}
    assert p['frozen_artifacts'] == {}
    print('K. provenance required fields + mode granted PASS')


def test_K2_provenance_protocol_and_frozen_hashes():
    p = provenance.build_arm_provenance(
        arm_id='arm_corrected_mu0.01', git_commit='abc', transform_mode='corrected',
        generator_checkpoint_path='ck.pth', generator_checkpoint_hash='a' * 64,
        mu=0.01, stochastic_lambda=0.0, attacker_architecture='ResNet-50 Siamese',
        attacker_hyperparameters={'lr': 1e-4, 'batch_size': 16},
        attacker_seeds_attempted=[0, 1, 2], pair_train_path='t',
        pair_validation_path='v', pair_test_path='e', pair_train_hash='b',
        pair_validation_hash='c', pair_test_hash='d',
        representative_attacker_seed=1, representative_selection_criterion='median',
        run_states={0: 'VALID'}, near_chance_flags={0: False},
        run_start_timestamp='t0', run_end_timestamp='t1', schedule_name='confirmatory',
        protocol_documents={'01.md': 'h1', '01B.md': 'h2'},
        frozen_artifacts={'topk_frozen_list.csv': 'h3'})
    assert p['protocol_documents'] == {'01.md': 'h1', '01B.md': 'h2'}
    assert p['frozen_artifacts'] == {'topk_frozen_list.csv': 'h3'}
    print('K2. provenance carries protocol + frozen artifact hashes (AMENDMENT 2 / R-12) PASS')


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
# N. R-9 FINAL: patient-cluster bootstrap withdrawn; pair bootstrap label (amendment §6)
# ----------------------------------------------------------------------------------
def _td_path(tmp_path):
    """Unwrap pytest tmp_path or tempfile.TemporaryDirectory into a str path."""
    if hasattr(tmp_path, 'name'):
        return str(tmp_path.name)
    return str(tmp_path)


def test_N_R9_final_policy(tmp_path):
    pairs = os.path.join(_td_path(tmp_path), 'pairs.txt')
    os.makedirs(os.path.dirname(pairs), exist_ok=True)
    with open(pairs, 'w') as f:
        # positive: same patient 0001; negative: two patients
        f.write('0001_000.png 0001_001.png 1.0\n')
        f.write('0002_000.png 0003_000.png 0.0\n')
    assert bootstrap.patient_cluster_bootstrap_is_ambiguous(pairs) is True
    rec = bootstrap.report_R9_final_policy(pairs)
    assert rec['R9_status'] == bootstrap.R9_FINAL_STATUS
    assert 'WITHDRAWN' in rec['R9_status']
    assert rec['pair_statistics']['negative_pairs_span_two_patients'] == 1
    assert rec['restart_sd_ddof'] == 1
    assert rec['pair_bootstrap_label'] == bootstrap.PAIR_BOOTSTRAP_LABEL
    assert 'PAIR-SAMPLING DIAGNOSTIC' in bootstrap.PAIR_BOOTSTRAP_LABEL
    assert 'NOT PATIENT-LEVEL UNCERTAINTY' in bootstrap.PAIR_BOOTSTRAP_LABEL
    # no patient-cluster resampling implementation exists
    assert not hasattr(bootstrap, 'PatientClusterResampler')
    print('N. R-9 final policy: patient-cluster withdrawn, pair label fixed PASS')


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
    # CUDA-only regression; on a CPU-only machine it must PASS by skipping.
    if not torch.cuda.is_available():
        print('P. gradient-accumulation regression SKIPPED (no CUDA)')
        return
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
# Z. Runner end-to-end (stub): staging, provenance hashes, stub-marked test metrics
# ----------------------------------------------------------------------------------
def test_Z_runner_stub_end_to_end(tmp_path):
    out = os.path.join(_td_path(tmp_path), 'arm_s')
    cmd = [sys.executable, 'run_adaptive_reid_arm.py', '--mode', 'screening',
           '--arm_id', 'arm_stub', '--mu', '0.01', '--transform_mode', 'corrected',
           '--out_dir', out, '--stub', '--stage', 'a_e']
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
    assert res.returncode == 0, res.stdout + res.stderr
    prov = diag.read_json(os.path.join(out, 'arm_provenance.json'))
    # AMENDMENT 2: authoritative protocol docs + frozen Top-k CSV hashed into provenance
    assert runner.PROTOCOL_01 in prov['protocol_documents']
    assert prov['protocol_documents'][runner.PROTOCOL_01]
    assert runner.PROTOCOL_01B in prov['protocol_documents']
    assert prov['protocol_documents'][runner.PROTOCOL_01B]
    assert runner.FROZEN_TOPK_CSV in prov['frozen_artifacts']
    assert prov['frozen_artifacts'][runner.FROZEN_TOPK_CSV]
    # BLOCKER 3: synthetic stub metrics never reach scientific summary fields
    arm_summary = diag.read_json(os.path.join(out, 'arm_summary.json'))
    assert arm_summary['contains_stub_or_synthetic_metrics'] is True
    assert arm_summary['mean_test_auc'] is None
    assert arm_summary['max_test_auc'] is None
    assert arm_summary['scientific_summary_available'] is False
    # every attempt carries a run signature (idempotent-reuse groundwork)
    for seed in (0, 1, 2, 3):
        sig_path = os.path.join(out, 'runs', 'retrain_snn_seed%d' % seed,
                                runner.SIGNATURE_FILENAME)
        assert os.path.exists(sig_path), sig_path
    print('Z. runner stub end-to-end (a_e) PASS')


# ----------------------------------------------------------------------------------
# Z2. STEP 2B.2: real-run training-diagnostics wiring
# ----------------------------------------------------------------------------------
def _real_record(tmp_path, **overrides):
    """Build + persist a REAL-style completed training record (schema-valid)."""
    td = _td_path(tmp_path)
    rec = diag.build_training_diagnostics(
        attacker_seed=7, transform_mode='corrected', mu=0.01, stochastic_lambda=0.0,
        generator_checkpoint_path='gen.pth', generator_checkpoint_hash='ghash',
        pair_train_path=runner.PAIR_TRAIN, pair_validation_path=runner.PAIR_VAL,
        pair_train_hash='thash', pair_validation_hash='vhash',
        epochs_completed=6, termination_reason=C.TERMINATION_EARLY_STOPPING,
        training_loss_per_epoch=[0.71, 0.55, 0.44, 0.38, 0.34, 0.31],
        validation_loss_per_epoch=[0.70, 0.58, 0.50, 0.45, 0.43, 0.41],
        validation_auc_per_epoch=[0.60, 0.65, 0.70, 0.73, 0.74, 0.74],
        validation_accuracy_per_epoch=[0.61, 0.66, 0.71, 0.72, 0.73, 0.73],
        best_validation_loss=0.41, best_validation_loss_epoch=5,
        best_validation_auc=0.74, best_validation_auc_epoch=5,
        any_nan_inf=False, checkpoint_exists=True, checkpoint_loadable=True,
        weights_changed_from_initialization=True,
        initial_parameter_hash='i' * 64, final_parameter_hash='f' * 64,
        protocol_documents={runner.PROTOCOL_01: 'c'}, frozen_artifacts={runner.FROZEN_TOPK_CSV: 'e'},
        run_start_timestamp='t0', run_end_timestamp='t1')
    for k, v in overrides.items():
        rec[k] = v
    diag.persist_real_training_diagnostics(
        os.path.join(td, 'training_diagnostics.json'), rec)
    return rec


def test_Z2_persists_real_per_epoch_arrays(tmp_path):
    td = _td_path(tmp_path)
    _real_record(tmp_path)
    loaded = diag.read_json(os.path.join(td, 'training_diagnostics.json'))
    assert diag.validate_training_diagnostics_schema(loaded) == []
    assert loaded['epochs_completed'] == 6
    assert len(loaded['training_loss_per_epoch']) == 6
    assert len(loaded['validation_loss_per_epoch']) == 6
    assert len(loaded['validation_auc_per_epoch']) == 6
    assert len(loaded['validation_accuracy_per_epoch']) == 6
    # persisted real file must contain NO test-derived field
    assert not (diag.FORBIDDEN_DIAGNOSTICS_FIELDS & set(loaded))
    print('Z2. real per-epoch arrays persisted, no test fields PASS')


def test_Z2_health_classifies_persisted_real_record(tmp_path):
    td = _td_path(tmp_path)
    _real_record(tmp_path)
    state, near = health.classify_run_health(
        diag.read_json(os.path.join(td, 'training_diagnostics.json')))
    assert state == C.VALID and near is False
    print('Z2. health consumes persisted real training_diagnostics.json PASS')


def test_Z2_epoch_cap_is_valid(tmp_path):
    rec = _real_record(tmp_path, termination_reason=C.TERMINATION_EPOCH_CAP,
                       best_validation_loss_epoch=5, best_validation_auc_epoch=5)
    state, _ = health.classify_run_health(rec)
    assert state == C.VALID
    print('Z2. epoch-cap completed run is VALID PASS')


def test_Z2_early_stop_with_patience_valid(tmp_path):
    rec = _real_record(tmp_path, termination_reason=C.TERMINATION_EARLY_STOPPING)
    state, _ = health.classify_run_health(rec)
    assert state == C.VALID
    print('Z2. early-stopped run (with weights updated) is VALID PASS')


def test_Z2_nan_inf_real_run_is_invalid(tmp_path):
    rec = _real_record(tmp_path, any_nan_inf=True)
    state, _ = health.classify_run_health(rec)
    assert state == C.NUMERICALLY_INVALID
    print('Z2. NaN/Inf in real run -> NUMERICALLY_INVALID PASS')


def test_Z2_near_chance_real_run_valid_not_excluded(tmp_path):
    rec = _real_record(tmp_path, best_validation_loss=0.70, best_validation_loss_epoch=5,
                       best_validation_auc=0.52, best_validation_auc_epoch=5)
    state, near = health.classify_run_health(rec)
    assert state == C.VALID and near is True
    print('Z2. near-chance real attacker stays VALID + near_chance PASS')


def test_Z2_initial_final_parameter_hash_consistency(tmp_path):
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
    h0 = weights.parameters_hash(net)
    # unchanged parameters -> identical hash -> weights_changed False
    assert weights.parameters_hash(net) == h0
    assert weights.weights_changed(h0, h0) is False
    # a real optimizer step mutates the trainable weights
    opt = torch.optim.SGD(net.parameters(), lr=0.5)
    opt.zero_grad()
    x = torch.randn(2, 4, requires_grad=False)
    net(x).sum().backward()
    opt.step()
    h1 = weights.parameters_hash(net)
    assert weights.weights_changed(h0, h1) is True
    rec = _real_record(tmp_path, initial_parameter_hash=h0, final_parameter_hash=h1)
    assert rec['weights_changed_from_initialization'] is True
    # the diagnostics must reflect the real run: no step -> invalid (R-2)
    rec2 = _real_record(tmp_path, initial_parameter_hash=h0, final_parameter_hash=h0,
                        weights_changed_from_initialization=False)
    state, _ = health.classify_run_health(rec2)
    assert state == C.NUMERICALLY_INVALID
    print('Z2. initial vs final parameter hash consistency (R-2) PASS')


def test_Z2_checkpoint_loadability_reflects_reality(tmp_path):
    p = os.path.join(_td_path(tmp_path), 'net.pth')
    assert weights.checkpoint_loadable(p) is False  # does not exist
    torch.save({'w': torch.zeros(2, 2)}, p)
    assert weights.checkpoint_loadable(p) is True
    with open(p, 'w') as f:
        f.write('not a checkpoint')
    assert weights.checkpoint_loadable(p) is False
    print('Z2. checkpoint_loadable reflects reality (exists/loadable/corrupt) PASS')


def test_Z2_runner_no_fabrication_and_infra_not_reused(tmp_path):
    # rc==0 with no diagnostics file -> objective failure, NEVER invented metrics
    import argparse
    td = _td_path(tmp_path)
    args = argparse.Namespace(transform_mode='corrected', mu=0.01, stochastic_lambda=0.0,
                              checkpoint=None)
    rec = runner._infra_record(9, 't0', td, {}, args, {}, {}, {})
    state, _ = health.classify_run_health(rec)
    assert state == C.NUMERICALLY_INVALID
    assert rec['epochs_completed'] == 0
    assert rec['termination_reason'] == C.TERMINATION_INFRASTRUCTURE
    print('Z2. infra failure is NUMERICALLY_INVALID, not fabricated PASS')


def test_Z2_real_reuse_requires_loadable_checkpoint(tmp_path):
    import argparse
    td = _td_path(tmp_path)
    args = argparse.Namespace(arm_id='arm_x', transform_mode='corrected', mu=0.01,
                              stochastic_lambda=0.0, checkpoint=None, stub=False,
                              force=False)
    cfg = {'learning_rate': 1e-4, 'batch_size': 16, 'max_epochs': 100,
           'early_stopping': {'patience': 5}}
    pair_hashes = {runner.PAIR_TRAIN: 'a', runner.PAIR_VAL: 'b'}
    protocol_documents = {runner.PROTOCOL_01: 'c', runner.PROTOCOL_01B: 'd'}
    frozen_artifacts = {runner.FROZEN_TOPK_CSV: 'e'}
    sig = runner.run_signature(3, args, cfg, pair_hashes, protocol_documents, frozen_artifacts)

    run_dir = os.path.join(td, 'runs', 'retrain_snn_seed3')
    os.makedirs(run_dir, exist_ok=True)
    rec = _real_record(tmp_path, attacker_seed=3)
    diag.write_json(os.path.join(run_dir, diag.VALIDITY_FILENAME), rec)
    diag.write_json(os.path.join(run_dir, diag.RUNSTATE_FILENAME),
                    {'state': C.VALID, 'near_chance': False})
    diag.write_json(os.path.join(run_dir, runner.SIGNATURE_FILENAME), sig)

    # real mode requires the actual checkpoint to exist AND load (STEP 2B.2)
    old_root = runner.ARCHIVE_ROOT
    runner.ARCHIVE_ROOT = os.path.join(_td_path(tmp_path), 'archive')
    try:
        ck = runner.attacker_checkpoint_path(3)
        assert runner.reuse_completed_run(run_dir, sig, 3, args) is None  # no checkpoint yet

        archive_dir = os.path.dirname(ck)
        os.makedirs(archive_dir, exist_ok=True)
        torch.save({'w': torch.zeros(2, 2)}, ck)
        loaded = runner.reuse_completed_run(run_dir, sig, 3, args)
        assert loaded is not None and loaded['attacker_seed'] == 3

        # corrupt the checkpoint -> must NOT reuse, must re-train
        with open(ck, 'w') as f:
            f.write('corrupt')
        assert runner.reuse_completed_run(run_dir, sig, 3, args) is None

        # a real-mode infra record is never reused as a completed run
        rec_infra = runner._infra_record(3, 't0', td, {}, args, pair_hashes, {}, {})
        diag.write_json(os.path.join(run_dir, diag.VALIDITY_FILENAME), rec_infra)
        assert runner.reuse_completed_run(run_dir, sig, 3, args) is None
    finally:
        runner.ARCHIVE_ROOT = old_root
    print('Z2. real reuse requires loadable checkpoint + infra never reused PASS')


def test_Z2_partial_artifact_set_not_reused(tmp_path):
    import argparse
    td = _td_path(tmp_path)
    args = argparse.Namespace(arm_id='arm_x', transform_mode='corrected', mu=0.01,
                              stochastic_lambda=0.0, checkpoint=None, stub=True,
                              force=False)
    cfg = {'learning_rate': 1e-4, 'batch_size': 16, 'max_epochs': 100,
           'early_stopping': {'patience': 5}}
    pair_hashes = {runner.PAIR_TRAIN: 'a', runner.PAIR_VAL: 'b'}
    protocol_documents = {runner.PROTOCOL_01: 'c', runner.PROTOCOL_01B: 'd'}
    frozen_artifacts = {runner.FROZEN_TOPK_CSV: 'e'}
    sig = runner.run_signature(3, args, cfg, pair_hashes, protocol_documents, frozen_artifacts)
    run_dir = os.path.join(td, 'runs', 'retrain_snn_seed3')
    os.makedirs(run_dir, exist_ok=True)
    rec = _real_record(tmp_path)
    diag.write_json(os.path.join(run_dir, diag.VALIDITY_FILENAME), rec)
    diag.write_json(os.path.join(run_dir, runner.SIGNATURE_FILENAME), sig)
    # run_state.json missing -> incomplete artifact set -> must (re)train
    assert runner.reuse_completed_run(run_dir, sig, 3, args) is None
    diag.write_json(os.path.join(run_dir, diag.RUNSTATE_FILENAME),
                    {'state': C.VALID, 'near_chance': False})
    assert runner.reuse_completed_run(run_dir, sig, 3, args) is not None
    print('Z2. incomplete artifact set is NOT reused as a completed run PASS')


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
    test_A2_accuracy_boundary_is_sigmoid_logit()
    test_A2_validate_snn_routes_logits_accuracy()
    test_B_numerically_invalid_conditions()
    test_C_near_chance_is_valid_flag_not_exclusion()
    test_D_unchanged_vs_step()
    test_E_confirmatory_seeds_and_cap()
    test_F_screening_seeds_and_cap()
    test_G_replacement_only_on_invalid_and_ascending()
    test_G_replacement_never_for_near_chance()
    test_D2_single_training_invocation_per_seed()
    test_E2_confirmatory_ascending_replacement_on_invalid_only()
    test_D3_idempotent_reuse_of_completed_run(_tf.TemporaryDirectory())
    test_E3_stub_markers_and_nonstub_not_implemented()
    test_H_representative_closest_to_median()
    test_H_tie_break_smaller_seed()
    test_H_even_count_median()
    test_I_selector_accepts_no_test_argument()
    test_I_health_accepts_no_test_argument()
    test_I_diagnostics_file_has_no_test_field()
    test_J_aggregation_includes_near_chance()
    test_J2_sample_sd_uses_ddof1()
    test_J3_summary_refuses_synthetic_metrics()
    test_K_provenance_required_fields()
    test_K2_provenance_protocol_and_frozen_hashes()
    test_L_determinism_checker()
    test_M_topk_frozen_list_build(_tf.TemporaryDirectory())
    test_N_R9_final_policy(_tf.TemporaryDirectory())
    test_N_pair_schema_statistics(_tf.TemporaryDirectory())
    test_O_shared_operator_still_routes_via_build_sampling_grid()
    test_P_gradient_accumulation_regression_imports_and_helpers()
    test_P_diagnostics_json_hasno_test_mixin(_tf.TemporaryDirectory())
    test_stage_order_test_never_selects(_tf.TemporaryDirectory())
    test_Z_runner_stub_end_to_end(_tf.TemporaryDirectory())
    test_Z2_persists_real_per_epoch_arrays(_tf.TemporaryDirectory())
    test_Z2_health_classifies_persisted_real_record(_tf.TemporaryDirectory())
    test_Z2_epoch_cap_is_valid(_tf.TemporaryDirectory())
    test_Z2_early_stop_with_patience_valid(_tf.TemporaryDirectory())
    test_Z2_nan_inf_real_run_is_invalid(_tf.TemporaryDirectory())
    test_Z2_near_chance_real_run_valid_not_excluded(_tf.TemporaryDirectory())
    test_Z2_initial_final_parameter_hash_consistency(_tf.TemporaryDirectory())
    test_Z2_checkpoint_loadability_reflects_reality(_tf.TemporaryDirectory())
    test_Z2_runner_no_fabrication_and_infra_not_reused(_tf.TemporaryDirectory())
    test_Z2_real_reuse_requires_loadable_checkpoint(_tf.TemporaryDirectory())
    test_Z2_partial_artifact_set_not_reused(_tf.TemporaryDirectory())

    print('\nSTEP 2B + STEP 2B.1 REMEDIATION TEST SUITE: PASS')
    sys.exit(0)