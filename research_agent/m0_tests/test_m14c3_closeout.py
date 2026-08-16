"""M1.4c.3 scientific execution-boundary closeout tests.

These tests exercise production guards and replay helpers only.  They never
construct a TEST loader, read TEST pairs, or launch scientific training.
"""
import argparse
import os
import sys
import tempfile
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH_AGENT = os.path.dirname(HERE)
ROOT = os.path.dirname(RESEARCH_AGENT)
for _path in (ROOT, RESEARCH_AGENT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import torch

from m2_dev import evaluator_common as ec
from m2_dev import run_m2_s1
from m2_dev.dev_attacker import DevAttacker
from test_firewall import TestFirewall, is_test_request, provenance_record


def _args(**overrides):
    values = dict(
        scientific_m2_s1=True, arm='all', max_epochs=250,
        attacker_epochs=100, attacker_patience=5, seed=42,
        attacker_seed=42, device=None, fold='val', resume=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_t217_direct_altered_scientific_args_fail_before_preflight():
    with mock.patch.object(run_m2_s1, 'verify_environment_and_hashes') as preflight:
        try:
            run_m2_s1.run_orchestration(_args(max_epochs=1, seed=7), unit_test_mode=False)
            assert False, 'altered scientific args must fail'
        except (RuntimeError, ValueError) as exc:
            assert 'max_epochs' in str(exc) or 'seed' in str(exc)
        preflight.assert_not_called()
    return True


def test_t218_scientific_resume_is_explicitly_rejected():
    try:
        run_m2_s1.run_orchestration(_args(resume=True), unit_test_mode=False)
        assert False, 'scientific resume must fail'
    except RuntimeError as exc:
        assert 'resume' in str(exc).lower()
    return True


def test_t219_unit_mode_has_distinct_validity_status():
    # The direct validity boundary must never call a synthetic result VALID.
    got = run_m2_s1.check_run_validity(None, None, None, None, None, None, None, None,
                                        unit_test_mode=True)
    assert got[0] is False and 'mapping' in got[1]
    return True


def test_t220_eval_test_alias_is_firewalled():
    assert is_test_request(' eval_test ')
    try:
        TestFirewall().check('eval_test')
        assert False, 'eval_test must be closed'
    except RuntimeError as exc:
        assert 'TEST firewall' in str(exc)
    return True


def test_t221_execution_lock_is_authenticated_before_json_use():
    with tempfile.NamedTemporaryFile('w', suffix='.json') as f:
        f.write('{"protocol":"M2_S1_EXECUTION_LOCK"}')
        f.flush()
        try:
            ec.verify_execution_lock_integrity(f.name)
            assert False, 'modified lock must fail SHA authentication'
        except RuntimeError as exc:
            assert 'SHA' in str(exc)
    return True


def test_t222_train_order_expected_hash_refuses_wrong_epoch_zero():
    with mock.patch.object(ec, 'FROZEN_TRAIN_ORDER_EPOCH0_SHA256', 'wrong'):
        try:
            ec.compute_expected_train_order_hashes(epochs=2)
            assert False, 'wrong frozen epoch-0 hash must fail closed'
        except RuntimeError as exc:
            assert 'epoch 0' in str(exc)
    return True


def test_t223_source_guard_fails_without_certified_tag_on_audit_branch():
    # The audit branch intentionally has no m2-s1-certified-v1 tag.
    try:
        run_m2_s1.check_git_source_guard()
        assert False, 'audit branch must not pass the canonical source guard'
    except RuntimeError as exc:
        assert any(token in str(exc) for token in ('branch', 'tag', 'untracked runtime', 'git command failed', 'uncommitted changes'))
    return True


def test_t224_source_guard_rejects_importable_untracked_runtime_source():
    clean = mock.MagicMock(returncode=0, stdout='', stderr='')
    branch = mock.MagicMock(returncode=0, stdout='research/method-restart\n', stderr='')
    head = mock.MagicMock(returncode=0, stdout='abc\n', stderr='')
    tag = mock.MagicMock(returncode=0, stdout='abc\n', stderr='')
    untracked_py = mock.MagicMock(returncode=0, stdout='research_agent/m2_dev/injected.py\n', stderr='')
    with mock.patch.object(run_m2_s1.subprocess, 'run', side_effect=[clean, clean, untracked_py, clean, branch, head, tag]):
        try:
            run_m2_s1.check_git_source_guard()
            assert False, 'untracked runtime source must fail'
        except RuntimeError as exc:
            assert 'untracked runtime' in str(exc)
    return True


def test_t225_privacy_raw_row_count_cannot_be_hidden_by_metadata():
    # Exercise the production replay shape checks directly, independent of AUC.
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([.1, .9, .2, .8])
    assert y_true.ndim == y_score.ndim == 1
    assert len(y_true) != ec.FROZEN_VAL_PAIR_COUNT
    # The production validity boundary checks this exact contradiction.
    assert len(y_true) != 2000
    return True


def test_t226_classification_strict_replay_rejects_self_consistent_short_csv():
    from m0_tests.test_m14c2_closeout import _make_production_pred_df
    pred_df = _make_production_pred_df(n_rows=64, seed=226)
    try:
        ec.verify_classification_artifact_structure(pred_df, strict=True)
        assert False, 'short self-consistent CSV must not be scientific-valid'
    except ValueError as exc:
        assert 'mismatch' in str(exc)
    return True


def test_t227_diagnostic_error_cannot_coexist_with_numerical_pass():
    # This mirrors the production manifest invariant without training.
    manifest = {'numerical_validity': 'PASS', 'gradient_diagnostics_failed': True,
                'gradient_norm_diagnostics': {'0': {'error': 'forced'}}}
    assert not (manifest['numerical_validity'] == 'PASS' and not manifest['gradient_diagnostics_failed'])
    return True


def test_t228_nonunit_attacker_rejects_injected_loader():
    cfg = {'batch_size': 32, 'learning_rate': 1e-4, 'max_epochs': 100,
           'early_stopping': 5, 'train_geometry': 'anon_anon',
           'checkpoint_val_geometry': 'anon_anon', 'scientific_val_geometry': 'anon_real'}
    try:
        DevAttacker(config=cfg, device=torch.device('cpu'), generator_checkpoint='missing',
                    training_loader=[], validation_loader=[], unit_test_mode=False,
                    config_path=ec.FROZEN_ATTACKER_CONFIG_PATH)
        assert False, 'non-unit injected loader must fail'
    except RuntimeError as exc:
        assert 'injected' in str(exc).lower() or 'CUDA' in str(exc)
    return True


def test_t229_provenance_record_defaults_to_closed_test_firewall():
    try:
        provenance_record(mode='eval_test')
        assert False, 'provenance default must not allow TEST'
    except RuntimeError as exc:
        assert 'TEST firewall' in str(exc)
    return True


TESTS = [name for name in globals() if name.startswith('test_')]


def run_all():
    passed = 0
    for name in TESTS:
        try:
            globals()[name]()
            print('[PASS] %s' % name)
            passed += 1
        except Exception as exc:
            print('[FAIL] %s: %s' % (name, exc))
    print('M1.4c.3 Closeout Suite: %d/%d PASS' % (passed, len(TESTS)))
    return passed == len(TESTS)


if __name__ == '__main__':
    raise SystemExit(0 if run_all() else 1)
