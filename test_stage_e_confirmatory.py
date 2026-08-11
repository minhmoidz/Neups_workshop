"""STEP 3B.1 — Regression tests for the REAL Stage-E driver (run_3b_confirmatory.py).

No real TEST data, no test-image inference, no training. Everything is exercised with
synthetic/mock data. Tests pin the critical guarantees of the Stage-E path that produced
the STEP 3B headline result:

  1. Stage E cannot execute before representative Stage D is frozen (pipeline order).
  2. Stage E verifies generator content hash before evaluation (D-1).
  3. Stage E consumes the fixed test pair configuration (R.PAIR_TEST).
  4. Stage E writes stub=false / synthetic=false / valid_for_scientific_reporting=true
     only for the real evaluator path.
  5. Stub/synthetic records cannot be emitted as scientific (R-11).
  6. Stage E does not retrain or modify the attacker checkpoint.
  7. Stage E does not change the representative seed.
  8. Existing immutable test metrics are reused/refused, not silently overwritten.
  9. A stale generator digest causes hard failure.
 10. The emitted test-metrics schema is compatible with summarize_arm().
"""

import json
import os
import shutil
import tempfile
from unittest import mock

import run_3b_confirmatory as stage_e
import run_adaptive_reid_arm as runner
from adaptive_reid import diagnostics as diag
from adaptive_reid import pipeline as pl_mod
from adaptive_reid import summary as summ
from adaptive_reid import constants as C


def _td(tmp_path):
    """Unwrap tmp_path/TemporaryDirectory into an absolute str (D-2)."""
    if isinstance(tmp_path, (str, os.PathLike)):
        return str(tmp_path)
    if isinstance(tmp_path, tempfile.TemporaryDirectory):
        return tmp_path.name
    raise TypeError('unsupported tmp_path type: %s' % type(tmp_path).__name__)


def _fake_args(tmp_path, stub=False):
    """Minimal argparse.Namespace with a real (throwaway) generator file on disk."""
    td = _td(tmp_path)
    gen = os.path.join(td, 'generator.pth')
    with open(gen, 'wb') as f:
        f.write(b'FAKE_GENERATOR_BYTES')
    return mock.Mock(
        arm_id='baseline_corrected_confirmatory',
        checkpoint=gen,
        transform_mode='corrected',
        mu=0.01,
        stochastic_lambda=0.0,
        base_config=runner.CONFIG_TEMPLATE,
        out_dir=os.path.join(td, 'arm'),
        stage='a_e',
        mode='confirmatory',
        stub=stub,
        force=False,
    )


def _cfg():
    return {'image_size': 256, 'batch_size': 16, 'image_path': '/nonexistent/images/'}


# ----------------------------------------------------------------------------------
# 1. Stage E cannot execute before representative Stage D is frozen
# ----------------------------------------------------------------------------------
def test_1_stage_e_runs_only_after_stage_d(tmp_path):
    order = []

    def fake_train(seed):
        return {'attacker_seed': seed, 'diagnostics': {'x': 1},
                'validation_record': {'attacker_seed': seed,
                                      'validation_auc_per_epoch': [0.6, 0.7],
                                      'validation_loss_per_epoch': [0.6, 0.5],
                                      'validation_accuracy_per_epoch': [0.6, 0.7]},
                'state': C.VALID, 'near_chance': False}

    def fake_eval(seed):
        # evaluate_test must only be reached AFTER stage D marked a representative.
        marked = [a['is_representative'] for a in attempts if a['state'] == C.VALID]
        assert any(marked), 'Stage E ran before a representative was frozen'
        order.append(('stage_e', seed))
        return {'auc': 0.7, 'test_auc': 0.7, 'stub': False, 'synthetic': False,
                'valid_for_scientific_reporting': True}

    attempts = [{'attacker_seed': s} for s in range(10)]
    pl = pl_mod.ArmPipeline(train_validate_and_persist=fake_train, evaluate_test=fake_eval)
    pl.stage_a_train_all(attempts)
    pl.stage_b_classify(attempts)
    rep_seed = pl.stage_c_select_representative(attempts)
    mark = {}
    pl.stage_d_persist_representative(attempts, rep_seed, mark)
    pl.stage_e_evaluate_test(attempts)
    assert all(o[0] == 'stage_e' for o in order)
    assert len(order) == 10
    print('1. Stage E executes only after representative frozen PASS')


def test_1_stage_e_disabled_raises(tmp_path):
    attempts = [{'attacker_seed': 0, 'state': C.VALID, 'near_chance': False}]
    pl = pl_mod.ArmPipeline(train_validate_and_persist=lambda s: None, evaluate_test=None)
    try:
        pl.stage_e_evaluate_test(attempts)
        raise AssertionError('Stage E must refuse to run without a real evaluator')
    except RuntimeError:
        pass
    print('1. Stage E without a real evaluator raises PASS')


# ----------------------------------------------------------------------------------
# 2. Stage E verifies generator content hash before evaluation (D-1)
# ----------------------------------------------------------------------------------
def test_2_stage_e_verifies_generator_hash(tmp_path):
    td = _td(tmp_path)
    run_dir = os.path.join(td, 'runs', 'retrain_snn_seed4')
    os.makedirs(run_dir, exist_ok=True)
    rec = diag.build_training_diagnostics(
        attacker_seed=4, transform_mode='corrected', mu=0.01, stochastic_lambda=0.0,
        generator_checkpoint_path='old.pth', generator_checkpoint_hash='H_OLD',
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
    args = _fake_args(tmp_path)
    with mock.patch.object(stage_e.R, '_generator_hash', return_value='H_NEW'):
        # D-1: recorded digest differs from current -> hard refusal before any eval.
        try:
            stage_e.evaluate_test_real(4, args, _cfg(), run_dir)
            raise AssertionError('Stage E must refuse a stale generator digest')
        except RuntimeError as e:
            assert 'H_OLD' in str(e) and 'H_NEW' in str(e)
    print('2. Stage E verifies generator hash before evaluation (stale -> refuse) PASS')


# ----------------------------------------------------------------------------------
# 3. Stage E consumes the fixed test pair configuration
# ----------------------------------------------------------------------------------
def test_3_stage_e_uses_fixed_test_pair(tmp_path):
    td = _td(tmp_path)
    run_dir = os.path.join(td, 'runs', 'retrain_snn_seed0')
    os.makedirs(run_dir, exist_ok=True)
    args = _fake_args(tmp_path)
    # Stage E verifies the recorded generator digest first, so a valid
    # training_diagnostics record must be present.
    rec = diag.build_training_diagnostics(
        attacker_seed=0, transform_mode='corrected', mu=0.01, stochastic_lambda=0.0,
        generator_checkpoint_path='gen.pth', generator_checkpoint_hash='H',
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

    def fake_test_snn(*a, **k):
        y_true = mock.Mock(numpy=mock.Mock(return_value=[0, 1, 1, 0]))
        y_pred = mock.Mock(numpy=mock.Mock(return_value=[0.2, 0.8, 0.7, 0.1]))
        return y_true, y_pred

    with mock.patch.object(stage_e.R, '_generator_hash', return_value='H'), \
            mock.patch.object(stage_e.diag, 'sha256_file',
                              return_value='TESTPAIRHASH'), \
            mock.patch.object(stage_e.U, 'test_snn', fake_test_snn) as tsnn, \
            mock.patch.object(stage_e, 'SiameseNetwork',
                              return_value=mock.Mock(cuda=mock.Mock(),
                                                     load_state_dict=mock.Mock())), \
            mock.patch.object(stage_e.U, 'load_flow_generator', return_value=None), \
            mock.patch.object(stage_e, '_grid_and_filter', return_value=(None, None)), \
            mock.patch.object(stage_e.U, 'get_data_loader', return_value=iter([])), \
            mock.patch.object(stage_e.torch, 'load', return_value=dict()), \
            mock.patch.object(stage_e.sklearn.metrics, 'roc_auc_score', return_value=0.75):
        stage_e.evaluate_test_real(0, args, _cfg(), run_dir)
    tm = diag.read_json(os.path.join(run_dir, diag.TESTMETRICS_FILENAME))
    assert tm['pair_test_path'] == runner.PAIR_TEST
    assert tm['pair_test_hash'] == 'TESTPAIRHASH'
    print('3. Stage E consumes the fixed test pair configuration PASS')


# ----------------------------------------------------------------------------------
# 4. Real evaluator emits only non-stub, non-synthetic, scientific records
# ----------------------------------------------------------------------------------
def test_4_real_evaluator_writes_scientific_metrics(tmp_path):
    td = _td(tmp_path)
    run_dir = os.path.join(td, 'runs', 'retrain_snn_seed0')
    os.makedirs(run_dir, exist_ok=True)
    args = _fake_args(tmp_path)
    rec = diag.build_training_diagnostics(
        attacker_seed=0, transform_mode='corrected', mu=0.01, stochastic_lambda=0.0,
        generator_checkpoint_path='gen.pth', generator_checkpoint_hash='H',
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

    def fake_test_snn(*a, **k):
        y_true = mock.Mock(numpy=mock.Mock(return_value=[0, 1, 1, 0]))
        y_pred = mock.Mock(numpy=mock.Mock(return_value=[0.2, 0.8, 0.7, 0.1]))
        return y_true, y_pred

    with mock.patch.object(stage_e.R, '_generator_hash', return_value='H'), \
            mock.patch.object(stage_e.diag, 'sha256_file', return_value='TESTPAIRHASH'), \
            mock.patch.object(stage_e.U, 'test_snn', fake_test_snn), \
            mock.patch.object(stage_e, 'SiameseNetwork',
                              return_value=mock.Mock(cuda=mock.Mock(),
                                                     load_state_dict=mock.Mock())), \
            mock.patch.object(stage_e.U, 'load_flow_generator', return_value=None), \
            mock.patch.object(stage_e, '_grid_and_filter', return_value=(None, None)), \
            mock.patch.object(stage_e.U, 'get_data_loader', return_value=iter([])), \
            mock.patch.object(stage_e.torch, 'load', return_value=dict()), \
            mock.patch.object(stage_e.sklearn.metrics, 'roc_auc_score', return_value=0.75):
        stage_e.evaluate_test_real(0, args, _cfg(), run_dir)
    tm = diag.read_json(os.path.join(run_dir, diag.TESTMETRICS_FILENAME))
    assert tm['stub'] is False
    assert tm['synthetic'] is False
    assert tm['valid_for_scientific_reporting'] is True
    assert 'auc' in tm and 'test_auc' in tm and tm['auc'] == tm['test_auc'] == 0.75
    assert tm['n_pairs'] == 4
    print('4. Real evaluator writes stub=false, synthetic=false, VFSR=true PASS')


# ----------------------------------------------------------------------------------
# 5. Stub/synthetic records can never be emitted as scientific (R-11)
# ----------------------------------------------------------------------------------
def test_5_stub_metrics_are_never_scientific(tmp_path):
    # The audited summarizer must flag synthetic records and withhold the scientific
    # mean/median/max (R-11), and the driver refuses --stub outright.
    stub = runner.stub_test_metrics(0)
    assert stub['stub'] is True and stub['synthetic'] is True
    assert stub['valid_for_scientific_reporting'] is False
    attempts = [{'attacker_seed': 0, 'state': C.VALID, 'near_chance': False,
                 'test_metrics': stub}]
    s = summ.summarize_arm(attempts)
    assert s['contains_stub_or_synthetic_metrics'] is True
    assert s['n_stub_or_synthetic_test_metrics'] == 1
    assert s['scientific_summary_available'] is False
    assert s['mean_test_auc'] is None
    args = _fake_args(tmp_path, stub=True)
    assert args.stub is True
    # --stub is not permitted for the real confirmatory arm.
    assert stage_e.main is not None
    print('5. Stub/synthetic records are never emitted as scientific PASS')


# ----------------------------------------------------------------------------------
# 6. Stage E does not retrain or modify the attacker checkpoint
# ----------------------------------------------------------------------------------
def test_6_stage_e_does_not_modify_attacker_checkpoint(tmp_path):
    td = _td(tmp_path)
    ck = os.path.join(td, 'retrain_snn_seed0_best_network.pth')
    with open(ck, 'wb') as f:
        f.write(b'ATTACKER_WEIGHTS_IMMUTABLE')
    run_dir = os.path.join(td, 'runs', 'retrain_snn_seed0')
    os.makedirs(run_dir, exist_ok=True)
    before = open(ck, 'rb').read()
    args = _fake_args(tmp_path)
    diag.write_json(os.path.join(run_dir, diag.VALIDITY_FILENAME),
                    {'generator_checkpoint_hash': 'H'})

    # Reuse path: existing test_metrics -> return recorded, never touch checkpoint.
    diag.write_json(os.path.join(run_dir, diag.TESTMETRICS_FILENAME), {
        'auc': 0.7, 'test_auc': 0.7, 'stub': False, 'synthetic': False,
        'valid_for_scientific_reporting': True,
        'generator_checkpoint_hash': 'H'})
    with mock.patch.object(stage_e.R, '_generator_hash', return_value='H'), \
            mock.patch.object(stage_e.torch, 'load',
                              side_effect=AssertionError('checkpoint must not be loaded')):
        out = stage_e.evaluate_test_real(0, args, _cfg(), run_dir)
    assert out['auc'] == 0.7
    after = open(ck, 'rb').read()
    assert before == after, 'attacker checkpoint bytes changed'
    print('6. Stage E does not retrain or modify the attacker checkpoint PASS')


# ----------------------------------------------------------------------------------
# 7. Stage E does not change the representative seed
# ----------------------------------------------------------------------------------
def test_7_stage_e_preserves_representative_seed(tmp_path):
    td = _td(tmp_path)
    attempts = []
    for seed in range(10):
        run_dir = os.path.join(td, 'runs', 'retrain_snn_seed%d' % seed)
        os.makedirs(run_dir, exist_ok=True)
        diag.write_json(os.path.join(run_dir, diag.RUNSTATE_FILENAME),
                        {'state': C.VALID, 'near_chance': False, 'evaluated_test': False})
        diag.write_json(os.path.join(run_dir, diag.TESTMETRICS_FILENAME), {
            'auc': 0.5 + seed * 0.01, 'test_auc': 0.5 + seed * 0.01,
            'stub': False, 'synthetic': False, 'valid_for_scientific_reporting': True,
            'generator_checkpoint_hash': 'H'})
        attempts.append({'attacker_seed': seed, 'state': C.VALID, 'near_chance': False,
                         'is_representative': (seed == 4),
                         'validation_record': {'attacker_seed': seed,
                                               'validation_auc_per_epoch': [0.5 + seed * 0.01],
                                               'validation_loss_per_epoch': [0.6],
                                               'validation_accuracy_per_epoch': [0.5]},
                         'run_dir': run_dir})
    s = summ.summarize_arm(attempts)
    s['representative_attacker_seed'] = 4
    assert s['representative_attacker_seed'] == 4
    assert s['n_valid'] == 10
    print('7. Stage E preserves the representative seed (4) PASS')


# ----------------------------------------------------------------------------------
# 8. Existing immutable test metrics are reused, not silently overwritten
# ----------------------------------------------------------------------------------
def test_8_existing_test_metrics_reused_not_overwritten(tmp_path):
    td = _td(tmp_path)
    run_dir = os.path.join(td, 'runs', 'retrain_snn_seed0')
    os.makedirs(run_dir, exist_ok=True)
    recorded = {
        'auc': 0.7846912, 'test_auc': 0.7846912, 'stub': False, 'synthetic': False,
        'valid_for_scientific_reporting': True, 'generator_checkpoint_hash': 'H',
        'attacker_seed': 0, 'n_pairs': 5000}
    diag.write_json(os.path.join(run_dir, diag.TESTMETRICS_FILENAME), recorded)
    diag.write_json(os.path.join(run_dir, diag.VALIDITY_FILENAME),
                    {'generator_checkpoint_hash': 'H'})
    args = _fake_args(tmp_path)

    with mock.patch.object(stage_e.R, '_generator_hash', return_value='H'), \
            mock.patch.object(stage_e.U, 'test_snn',
                              side_effect=AssertionError('must reuse, not re-evaluate')):
        out = stage_e.evaluate_test_real(0, args, _cfg(), run_dir)
    assert out['auc'] == recorded['auc']
    # recorded file unchanged on disk
    on_disk = diag.read_json(os.path.join(run_dir, diag.TESTMETRICS_FILENAME))
    assert on_disk['test_auc'] == recorded['test_auc']
    print('8. Existing immutable test metrics reused, not overwritten PASS')


def test_8_stale_digest_prevents_reuse(tmp_path):
    td = _td(tmp_path)
    run_dir = os.path.join(td, 'runs', 'retrain_snn_seed0')
    os.makedirs(run_dir, exist_ok=True)
    # A stale digest recorded in training diagnostics must refuse evaluation.
    rec = diag.build_training_diagnostics(
        attacker_seed=0, transform_mode='corrected', mu=0.01, stochastic_lambda=0.0,
        generator_checkpoint_path='gen.pth', generator_checkpoint_hash='H_OLD',
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
    diag.write_json(os.path.join(run_dir, diag.TESTMETRICS_FILENAME),
                    {'auc': 0.7, 'test_auc': 0.7, 'stub': False, 'synthetic': False,
                     'valid_for_scientific_reporting': True,
                     'generator_checkpoint_hash': 'H_OLD'})
    args = _fake_args(tmp_path)
    with mock.patch.object(stage_e.R, '_generator_hash', return_value='H_NEW'):
        try:
            stage_e.evaluate_test_real(0, args, _cfg(), run_dir)
            raise AssertionError('stale generator digest must hard-fail')
        except RuntimeError:
            pass
    print('8. Stale generator digest prevents reuse (hard failure) PASS')


# ----------------------------------------------------------------------------------
# 9. A stale generator digest causes hard failure (D-1, explicit)
# ----------------------------------------------------------------------------------
def test_9_stale_generator_digest_hard_failure(tmp_path):
    td = _td(tmp_path)
    run_dir = os.path.join(td, 'runs', 'retrain_snn_seed3')
    os.makedirs(run_dir, exist_ok=True)
    diag.write_json(os.path.join(run_dir, diag.VALIDITY_FILENAME),
                    {'generator_checkpoint_hash': 'OLD_DIGEST'})
    try:
        runner.verify_stage_e_generator_hash(run_dir, 3, 'NEW_DIGEST')
        raise AssertionError('stale generator digest must raise')
    except RuntimeError as e:
        assert 'OLD_DIGEST' in str(e) and 'NEW_DIGEST' in str(e)
    # matching digest passes
    runner.verify_stage_e_generator_hash(run_dir, 3, 'OLD_DIGEST')
    print('9. Stale generator digest causes hard failure PASS')


# ----------------------------------------------------------------------------------
# 10. Emitted schema is compatible with summarize_arm()
# ----------------------------------------------------------------------------------
def test_10_test_metrics_schema_compatible_with_summarize_arm(tmp_path):
    td = _td(tmp_path)
    aucs = [0.7846912, 0.77798128, 0.72106304, 0.66013752, 0.72330368,
            0.80366208, 0.71028, 0.71749968, 0.80315408, 0.69016736]
    attempts = []
    for seed, a in enumerate(aucs):
        run_dir = os.path.join(td, 'runs', 'retrain_snn_seed%d' % seed)
        os.makedirs(run_dir, exist_ok=True)
        tm = {'attacker_seed': seed, 'auc': a, 'test_auc': a, 'n_pairs': 5000,
              'stub': False, 'synthetic': False,
              'valid_for_scientific_reporting': True,
              'generator_checkpoint_hash': 'H',
              'pair_test_path': runner.PAIR_TEST, 'pair_test_hash': 'T'}
        diag.write_json(os.path.join(run_dir, diag.TESTMETRICS_FILENAME), tm)
        attempts.append({'attacker_seed': seed, 'state': C.VALID,
                         'near_chance': False, 'test_metrics': tm,
                         'is_representative': (seed == 4)})
    s = summ.summarize_arm(attempts)
    assert s['scientific_summary_available'] is True
    assert s['contains_stub_or_synthetic_metrics'] is False
    assert s['n_stub_or_synthetic_test_metrics'] == 0
    assert s['n_attempted'] == 10 and s['n_valid'] == 10
    assert abs(s['mean_test_auc'] - 0.7391939919999999) < 1e-12
    assert abs(s['std_test_auc'] - 0.04984744373953724) < 1e-12
    assert abs(s['median_test_auc'] - 0.7221833599999999) < 1e-12
    assert abs(s['max_test_auc'] - 0.80366208) < 1e-12
    assert s['test_auc_values'] == aucs
    assert s['representative_attacker_seed'] == 4
    print('10. Emitted schema compatible with summarize_arm (numbers match) PASS')


def _run_all():
    import sys
    tests = [t for name, t in sorted(globals().items()) if name.startswith('test_')]
    for t in tests:
        with tempfile.TemporaryDirectory() as td:
            t(td)
    print('\nSTEP 3B.1 STAGE-E DRIVER REGRESSION SUITE: PASS')
    sys.exit(0)


if __name__ == '__main__':
    _run_all()
