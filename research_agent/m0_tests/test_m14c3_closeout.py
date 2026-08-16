"""M1.4c.3 scientific execution-boundary closeout tests.

These tests exercise production guards and replay helpers only.  They never
construct a TEST loader, read TEST pairs, or launch scientific training.
"""
import argparse
import copy
import json
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
from m2_dev.eval_classifier_val import evaluate_classification_val
from m2_dev.eval_reid_val import evaluate_reid_val
from m0_tests.test_m14c2_closeout import (
    _make_production_pred_df,
    _auc_df_from_pred,
    _production_class_bundle,
    _valid_check,
)
from test_firewall import TestFirewall, is_test_request, provenance_record


def _args(**overrides):
    values = dict(
        scientific_m2_s1=True, arm='all', max_epochs=250,
        attacker_epochs=100, attacker_patience=5, seed=42,
        attacker_seed=42, device=None, fold='val', resume=False,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def _production_boundary_bundle(tmp, gradient_diagnostics_failed=False, c4_diag_error=False):
    """Build a production-mode (non-unit) bundle that is valid through the
    privacy replay boundary but carries a short 4-row privacy NPZ.

    Classification results are intentionally empty mappings: the production
    validity boundary returns INVALID during the manifest/privacy phase before
    classification is ever inspected.
    """
    gen_p = os.path.join(tmp, 'gen.pth')
    att_p = os.path.join(tmp, 'att.pth')
    torch.save({}, gen_p)
    torch.save({}, att_p)
    gen_sha = ec.file_sha256(gen_p)
    att_sha = ec.file_sha256(att_p)

    b_dev_m = {
        'epochs_completed': 250, 'requested_max_epochs': 250,
        'numerical_validity': 'PASS', 'nan_inf_detected': False,
        'gradient_diagnostics_failed': False, 'gradient_norm_diagnostics': {},
        'selected_generator_checkpoint': gen_p, 'selected_generator_sha256': gen_sha,
        'config_sha256': ec.FROZEN_B_DEV_CONFIG_SHA,
    }
    c4_m = copy.deepcopy(b_dev_m)
    c4_m['config_sha256'] = ec.FROZEN_C4_CONFIG_SHA
    if gradient_diagnostics_failed:
        c4_m['gradient_diagnostics_failed'] = True
    if c4_diag_error:
        c4_m['gradient_norm_diagnostics'] = {'0': {'error': 'forced'}}

    b_att_m = {'best_attacker_path': att_p, 'best_attacker_sha256': att_sha,
               'generator_checkpoint_sha256': gen_sha, 'numerical_validity': 'PASS',
               'nan_inf_detected': False}
    c4_att_m = copy.deepcopy(b_att_m)

    npz_p = os.path.join(tmp, 'privacy.npz')
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.2, 0.8])
    np.savez_compressed(npz_p, y_true=y_true, y_score=y_score)
    npz_sha = ec.file_sha256(npz_p)
    b_priv = {'roc_auc': 1.0, 'generator_checkpoint_sha256': gen_sha,
              'attacker_checkpoint_sha256': att_sha, 'n_pairs': 4,
              'predictions_file': npz_p, 'predictions_file_sha256': npz_sha}
    c4_priv = copy.deepcopy(b_priv)

    return b_dev_m, c4_m, b_att_m, c4_att_m, b_priv, c4_priv, {}, {}


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
    # A valid unit-mode validity result must be EXACTLY DEVELOPMENT_VALID and
    # must never accept scientific VALID as an allowed alternative.
    with tempfile.TemporaryDirectory() as tmp:
        pred_df = _make_production_pred_df(n_rows=64, seed=219)
        auc_df = _auc_df_from_pred(pred_df)
        macro_auc = float(auc_df['auc'].mean())
        b_class, c4_class = _production_class_bundle(tmp, pred_df, auc_df, macro_auc)
        valid, msg = _valid_check(tmp, b_class, c4_class)
        assert valid is True, 'unit-mode valid bundle must pass: %s' % msg
        assert msg == 'DEVELOPMENT_VALID', 'unit-mode validity must be EXACTLY DEVELOPMENT_VALID; got %r' % msg
        assert msg != 'VALID', 'unit-mode validity must never accept scientific VALID'
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
    # Construct the real short NPZ bundle and call the PRODUCTION validity
    # boundary: the 2000-row provenance contract must make it INVALID.
    with tempfile.TemporaryDirectory() as tmp:
        bundle = _production_boundary_bundle(tmp)
        valid, msg = run_m2_s1.check_run_validity(*bundle, expected_epochs=250, unit_test_mode=False)
        assert valid is False, 'short privacy NPZ bundle must be INVALID in production'
        assert 'must contain exactly 2000 rows' in msg
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
    # Exercise the production diagnostic/validity boundary: a diagnostic error
    # must make the scientific result INVALID even when every other field passes.
    for kwargs, needle in (
            ({'gradient_diagnostics_failed': True}, 'gradient diagnostics reported failure'),
            ({'c4_diag_error': True}, 'contains failed gradient diagnostics')):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = _production_boundary_bundle(tmp, **kwargs)
            valid, msg = run_m2_s1.check_run_validity(*bundle, expected_epochs=250, unit_test_mode=False)
            assert valid is False, 'diagnostic failure must invalidate the production run'
            assert needle in msg, 'expected %r in %r' % (needle, msg)
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


def test_t230_nonunit_anonymizer_arm_passes_canonical_config_path():
    # Regression: run_anonymizer_arm() must pass the canonical CONFIG FILE PATH
    # to M2AnonymizerRunner in non-unit scientific mode (never an in-memory dict,
    # which the non-unit runner intentionally rejects).
    with tempfile.TemporaryDirectory() as tmp:
        fake = mock.MagicMock()
        fake.run.return_value = {
            'best_epoch': 5, 'best_selection_total': 0.5,
            'selected_generator_sha256': 'abc', 'epochs_completed': 250,
        }
        with mock.patch.object(run_m2_s1, 'M2AnonymizerRunner', return_value=fake) as runner_cls:
            manifest = run_m2_s1.run_anonymizer_arm(
                'B_dev', ec.FROZEN_B_DEV_CONFIG_PATH, 250, 42, 'cpu',
                out_base_dir=tmp, unit_test_mode=False)
        assert manifest['best_epoch'] == 5
        call_kwargs = runner_cls.call_args.kwargs
        assert call_kwargs['config'] == ec.FROZEN_B_DEV_CONFIG_PATH, 'non-unit arm must pass the canonical config PATH'
        assert isinstance(call_kwargs['config'], str)
        assert not isinstance(call_kwargs['config'], dict), 'non-unit arm must never pass an in-memory dict'
        assert call_kwargs['config_path'] == ec.FROZEN_B_DEV_CONFIG_PATH
        assert call_kwargs['unit_test_mode'] is False
    return True


def test_t231_config_unit_flag_cannot_activate_unit_mode():
    # Scientific call + config["unit_test_mode"]=True + injected model/dataloader
    # MUST reject: unit mode may only be activated by the explicit argument.
    from networks.UNet_PriCheXyNet import UNet
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with tempfile.TemporaryDirectory() as tmp:
        gen_p = os.path.join(tmp, 'gen.pth')
        torch.save(UNet(1, 2, 32).state_dict(), gen_p)
        loader = torch.utils.data.DataLoader(
            [(torch.rand(1, 64, 64), torch.rand(1, 64, 64), torch.tensor(float(i % 2))) for i in range(8)],
            batch_size=4)
        if not torch.cuda.is_available():
            # Scientific CPU is rejected before any injection is consulted.
            try:
                evaluate_classification_val(
                    config={'image_path': ec.SCIENTIFIC_IMAGE_ROOT, 'unit_test_mode': True,
                            'dataloader': loader, 'image_size': 64},
                    fold='val', model=torch.nn.Identity(), device='cpu',
                    generator_checkpoint=gen_p, unit_test_mode=False)
                assert False, 'scientific CPU must be rejected'
            except RuntimeError as exc:
                assert 'CUDA' in str(exc)
            return True
        try:
            evaluate_classification_val(
                config={'image_path': ec.SCIENTIFIC_IMAGE_ROOT, 'unit_test_mode': True,
                        'dataloader': loader, 'image_size': 64},
                fold='val', model=torch.nn.Identity(), device=device,
                generator_checkpoint=gen_p, unit_test_mode=False)
            assert False, 'config unit flag must not permit an injected model in a scientific call'
        except RuntimeError as exc:
            assert 'injected model' in str(exc)
        try:
            evaluate_classification_val(
                config={'image_path': ec.SCIENTIFIC_IMAGE_ROOT, 'unit_test_mode': True,
                        'dataloader': loader, 'image_size': 64},
                fold='val', device=device, generator_checkpoint=gen_p, unit_test_mode=False)
            assert False, 'config unit flag must not permit an injected dataloader in a scientific call'
        except RuntimeError as exc:
            assert 'injected dataloader' in str(exc)
    return True


def test_t232_scientific_data_root_is_bound_for_direct_apis():
    # Non-unit classification/privacy APIs must bind image_path to the approved
    # scientific data root; an arbitrary direct-API image directory must reject.
    from networks.UNet_PriCheXyNet import UNet
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with tempfile.TemporaryDirectory() as tmp:
        gen_p = os.path.join(tmp, 'gen.pth')
        att_p = os.path.join(tmp, 'att.pth')
        torch.save(UNet(1, 2, 32).state_dict(), gen_p)
        torch.save({}, att_p)
        try:
            evaluate_classification_val(
                config={'image_path': tmp, 'image_size': 64}, fold='val',
                device=device, generator_checkpoint=gen_p, unit_test_mode=False)
            assert False, 'non-unit classification with arbitrary image_path must reject'
        except RuntimeError as exc:
            assert 'approved scientific data root' in str(exc)
        try:
            evaluate_reid_val(
                config={'image_path': tmp, 'image_size': 64},
                attacker_checkpoint=att_p, generator_checkpoint=gen_p,
                device=device, unit_test_mode=False)
            assert False, 'non-unit privacy with arbitrary image_path must reject'
        except RuntimeError as exc:
            assert 'approved scientific data root' in str(exc)
    return True


def test_t233_attacker_in_memory_config_must_match_frozen():
    # DevAttacker must reject an in-memory config that does not itself match the
    # canonical frozen attacker config (scientific fields incl. data root), even
    # when the canonical config_path SHA is presented.
    with open(ec.FROZEN_ATTACKER_CONFIG_PATH) as f:
        frozen = json.load(f)
    bad_root = dict(frozen)
    bad_root['image_path'] = '/some/other/root'
    try:
        DevAttacker(config=bad_root, device=torch.device('cpu'), attacker_seed=42,
                    generator_checkpoint='missing', unit_test_mode=False,
                    config_path=ec.FROZEN_ATTACKER_CONFIG_PATH)
        assert False, 'attacker config with non-frozen data root must be rejected'
    except RuntimeError as exc:
        assert 'image_path' in str(exc) and 'frozen' in str(exc)
    missing_field = {k: v for k, v in frozen.items() if k != 'perturbation_type'}
    try:
        DevAttacker(config=missing_field, device=torch.device('cpu'), attacker_seed=42,
                    generator_checkpoint='missing', unit_test_mode=False,
                    config_path=ec.FROZEN_ATTACKER_CONFIG_PATH)
        assert False, 'attacker config missing a frozen scientific field must be rejected'
    except RuntimeError as exc:
        assert 'missing frozen field' in str(exc)
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
