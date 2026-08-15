"""M1.4b: Execution Integrity, Frozen Config Hashes, Metadata & Strict Validity Tests (T113–T136)."""
import os
import sys
import json
import shutil
import tempfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
for _p in (ROOT, os.path.join(ROOT, 'research_agent')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from test_firewall import TestFirewall
from m2_dev.evaluator_common import (
    file_sha256,
    LazyPairDataset,
    verify_frozen_scientific_configs,
    verify_scientific_dependencies,
    FROZEN_METADATA_SHA,
    FROZEN_METADATA_PATH,
    FROZEN_B_DEV_CONFIG_PATH,
    FROZEN_B_DEV_CONFIG_SHA,
    FROZEN_C4_CONFIG_PATH,
    FROZEN_C4_CONFIG_SHA,
    FROZEN_ATTACKER_CONFIG_SHA,
    FROZEN_CLASSIFIER_SHA,
    INITIAL_GENERATOR_SHA,
)
from m2_dev.anonymizer_runner import M2AnonymizerRunner
from m2_dev.dev_attacker import DevAttacker, SiameseNetwork
from m2_dev.eval_reid_val import evaluate_reid_val
from m2_dev.eval_classifier_val import evaluate_classification_val
from m2_dev.run_m2_s1 import (
    parse_args,
    check_run_validity,
    run_orchestration,
    run_anonymizer_arm,
    train_s1_attacker_arm,
    evaluate_privacy_arm,
    evaluate_classification_arm,
    verify_environment_and_hashes,
)
from networks.UNet_PriCheXyNet import UNet
from m0_tests.test_m14a_execution_harness import (
    SyntheticPairDataset,
    SyntheticAttackerPairDataset,
    SyntheticClassificationDataset,
)


def test_t113_anonymizer_manifest_contains_epochs_completed():
    """T113: Anonymizer manifest contains epochs_completed."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ds = SyntheticPairDataset(4, image_size=64)
        loader = torch.utils.data.DataLoader(ds, batch_size=2)
        runner = M2AnonymizerRunner(
            arm='B_dev',
            config={'image_size': 64, 'learning_rate': 1e-4},
            output_dir=tmp_dir,
            device='cpu',
            training_loader=loader,
            validation_loader=loader,
            unit_test_mode=True
        )
        manifest = runner.run(max_epochs=2)
        assert 'epochs_completed' in manifest
        assert manifest['epochs_completed'] == 2
        assert manifest['requested_max_epochs'] == 2
        assert manifest['final_completed_epoch'] == 1
    return True


def test_t114_epochs_completed_equals_expected_valid():
    """T114: epochs_completed == expected satisfies validity."""
    b_m = {'epochs_completed': 250, 'requested_max_epochs': 250, 'numerical_validity': 'PASS', 'nan_inf_detected': False,
           'selected_generator_checkpoint': '/tmp/gen.pth', 'selected_generator_sha256': 'dummy', 'config_sha256': FROZEN_B_DEV_CONFIG_SHA}
    c_m = {'epochs_completed': 250, 'requested_max_epochs': 250, 'numerical_validity': 'PASS', 'nan_inf_detected': False,
           'selected_generator_checkpoint': '/tmp/gen.pth', 'selected_generator_sha256': 'dummy', 'config_sha256': FROZEN_C4_CONFIG_SHA}
    b_att = {'best_attacker_path': '/tmp/att.pth', 'best_attacker_sha256': 'dummy', 'generator_checkpoint_sha256': 'dummy', 'numerical_validity': 'PASS', 'nan_inf_detected': False}
    c_att = {'best_attacker_path': '/tmp/att.pth', 'best_attacker_sha256': 'dummy', 'generator_checkpoint_sha256': 'dummy', 'numerical_validity': 'PASS', 'nan_inf_detected': False}
    b_p = {'roc_auc': 0.70, 'n_pairs': 8, 'generator_checkpoint_sha256': 'dummy', 'attacker_checkpoint_sha256': 'dummy'}
    c_p = {'roc_auc': 0.71, 'n_pairs': 8, 'generator_checkpoint_sha256': 'dummy', 'attacker_checkpoint_sha256': 'dummy'}
    b_c = {'macro_auc': 0.80, 'n_classes_valid': 14, 'n_images': 28, 'generator_checkpoint_sha256': 'dummy', 'classifier_checkpoint_sha256': FROZEN_CLASSIFIER_SHA}
    c_c = {'macro_auc': 0.81, 'n_classes_valid': 14, 'n_images': 28, 'generator_checkpoint_sha256': 'dummy', 'classifier_checkpoint_sha256': FROZEN_CLASSIFIER_SHA}

    with tempfile.NamedTemporaryFile() as f_gen, tempfile.NamedTemporaryFile() as f_att:
        gen_sha = file_sha256(f_gen.name)
        att_sha = file_sha256(f_att.name)
        b_m['selected_generator_checkpoint'] = f_gen.name
        b_m['selected_generator_sha256'] = gen_sha
        c_m['selected_generator_checkpoint'] = f_gen.name
        c_m['selected_generator_sha256'] = gen_sha
        b_att['best_attacker_path'] = f_att.name
        b_att['best_attacker_sha256'] = att_sha
        b_att['generator_checkpoint_sha256'] = gen_sha
        c_att['best_attacker_path'] = f_att.name
        c_att['best_attacker_sha256'] = att_sha
        c_att['generator_checkpoint_sha256'] = gen_sha
        b_p['generator_checkpoint_sha256'] = gen_sha
        b_p['attacker_checkpoint_sha256'] = att_sha
        c_p['generator_checkpoint_sha256'] = gen_sha
        c_p['attacker_checkpoint_sha256'] = att_sha
        b_c['generator_checkpoint_sha256'] = gen_sha
        c_c['generator_checkpoint_sha256'] = gen_sha

        valid, reason = check_run_validity(b_m, c_m, b_att, c_att, b_p, c_p, b_c, c_c, expected_epochs=250, unit_test_mode=True)
        assert valid, "Expected validity PASS, got %s" % reason
    return True


def test_t115_epochs_completed_missing_or_249_fails():
    """T115: epochs_completed missing or 249/250 fails validity."""
    b_m = {'requested_max_epochs': 250, 'numerical_validity': 'PASS', 'nan_inf_detected': False}
    c_m = {'epochs_completed': 249, 'requested_max_epochs': 250, 'numerical_validity': 'PASS', 'nan_inf_detected': False}
    dummy_att = {'best_attacker_path': '/dummy', 'best_attacker_sha256': 'dummy'}
    valid, reason = check_run_validity(b_m, c_m, dummy_att, dummy_att, {}, {}, {}, {}, expected_epochs=250, unit_test_mode=True)
    assert not valid
    assert "missing epochs_completed" in reason or "epochs_completed" in reason
    return True


def test_t116_b_dev_config_sha_mismatch_hard_fails():
    """T116: B_dev config SHA mismatch raises RuntimeError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        lock_p = os.path.join(tmp_dir, 'lock.json')
        with open(lock_p, 'w') as f:
            json.dump({'artifact_provenance': {'b_dev_config_sha256': '0000000000000000000000000000000000000000000000000000000000000000'}}, f)
        try:
            verify_frozen_scientific_configs(lock_path=lock_p)
            assert False, "Should have failed on B_dev config SHA mismatch"
        except RuntimeError as e:
            assert "B_dev config SHA mismatch" in str(e)
    return True


def test_t117_c4_config_sha_mismatch_hard_fails():
    """T117: C4 config SHA mismatch raises RuntimeError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        lock_p = os.path.join(tmp_dir, 'lock.json')
        with open(lock_p, 'w') as f:
            json.dump({'artifact_provenance': {'c4_config_sha256': '0000000000000000000000000000000000000000000000000000000000000000'}}, f)
        try:
            verify_frozen_scientific_configs(lock_path=lock_p)
            assert False, "Should have failed on C4 config SHA mismatch"
        except RuntimeError as e:
            assert "C4 config SHA mismatch" in str(e)
    return True


def test_t118_attacker_config_sha_mismatch_hard_fails():
    """T118: Attacker config SHA mismatch raises RuntimeError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        lock_p = os.path.join(tmp_dir, 'lock.json')
        with open(lock_p, 'w') as f:
            json.dump({'artifact_provenance': {'attacker_config_sha256': '0000000000000000000000000000000000000000000000000000000000000000'}}, f)
        try:
            verify_frozen_scientific_configs(lock_path=lock_p)
            assert False, "Should have failed on attacker config SHA mismatch"
        except RuntimeError as e:
            assert "Attacker config SHA mismatch" in str(e)
    return True


def test_t119_anonymizer_manifest_records_non_null_config_sha():
    """T119: Anonymizer runner records non-null config_sha256 in manifest."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ds = SyntheticPairDataset(4, image_size=256)
        loader = torch.utils.data.DataLoader(ds, batch_size=2)
        cfg_p = os.path.join(ROOT, 'config_files', 'config_dev_restored_baseline.json')
        runner = M2AnonymizerRunner(
            arm='B_dev',
            config_path=cfg_p,
            output_dir=tmp_dir,
            device='cpu',
            training_loader=loader,
            validation_loader=loader,
            unit_test_mode=True
        )
        manifest = runner.run(max_epochs=1)
        assert manifest['config_sha256'] == FROZEN_B_DEV_CONFIG_SHA
    return True


def test_t120_nan_training_metric_hard_fails():
    """T120: NaN training metric raises FloatingPointError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ds = SyntheticPairDataset(4, image_size=64)
        loader = torch.utils.data.DataLoader(ds, batch_size=2)
        runner = M2AnonymizerRunner(
            arm='B_dev',
            config={'image_size': 64, 'learning_rate': 1e-4},
            output_dir=tmp_dir,
            device='cpu',
            training_loader=loader,
            validation_loader=loader,
            unit_test_mode=True
        )
        # Inject NaN into train_epoch
        runner.train_epoch = lambda ep: {'train_ac_bce': float('nan'), 'train_ver_loss': 0.5, 'train_privacy_term': 0.5,
                                         'train_feature_term': 0.0, 'train_optimization_total': float('nan'), 'train_selection_total': float('nan')}
        try:
            runner.run(max_epochs=1)
            assert False, "Should have raised FloatingPointError on NaN"
        except FloatingPointError:
            pass
        assert runner.nan_inf_detected is True
    return True


def test_t121_inf_validation_metric_hard_fails():
    """T121: Inf validation metric raises FloatingPointError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ds = SyntheticPairDataset(4, image_size=64)
        loader = torch.utils.data.DataLoader(ds, batch_size=2)
        runner = M2AnonymizerRunner(
            arm='B_dev',
            config={'image_size': 64, 'learning_rate': 1e-4},
            output_dir=tmp_dir,
            device='cpu',
            training_loader=loader,
            validation_loader=loader,
            unit_test_mode=True
        )
        # Inject Inf into validate_epoch
        runner.validate_epoch = lambda ep: {'val_ac_bce': float('inf'), 'val_ver_loss': 0.5, 'val_privacy_term': 0.5,
                                            'val_feature_term': 0.0, 'val_optimization_total': float('inf'), 'val_selection_total': float('inf')}
        try:
            runner.run(max_epochs=1)
            assert False, "Should have raised FloatingPointError on Inf"
        except FloatingPointError:
            pass
        assert runner.nan_inf_detected is True
    return True


def test_t122_manifest_numerical_validity_required():
    """T122: numerical_validity == PASS required by check_run_validity."""
    b_m = {'epochs_completed': 250, 'requested_max_epochs': 250, 'numerical_validity': 'FAIL', 'nan_inf_detected': True}
    c_m = {'epochs_completed': 250, 'requested_max_epochs': 250, 'numerical_validity': 'PASS', 'nan_inf_detected': False}
    dummy_att = {'best_attacker_path': '/dummy', 'best_attacker_sha256': 'dummy'}
    valid, reason = check_run_validity(b_m, c_m, dummy_att, dummy_att, {}, {}, {}, {}, expected_epochs=250, unit_test_mode=True)
    assert not valid
    assert "numerical_validity" in reason or "nan_inf_detected" in reason
    return True


def test_t123_missing_metadata_file_hard_fails():
    """T123: Missing Data_Entry_2017_v2020.csv raises FileNotFoundError in LazyPairDataset."""
    try:
        LazyPairDataset(
            phase='training',
            image_path='/tmp',
            metadata_path='/nonexistent/Data_Entry.csv'
        )
        assert False, "Should have raised FileNotFoundError on missing metadata"
    except FileNotFoundError:
        pass
    return True


def test_t124_missing_image1_metadata_key_hard_fails():
    """T124: Image filename absent from metadata raises RuntimeError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dummy_meta_p = os.path.join(tmp_dir, 'Data_Entry.csv')
        df = pd.DataFrame({'Image Index': ['unknown_image.png'], 'Finding Labels': ['No Finding']})
        df.to_csv(dummy_meta_p, index=False)

        try:
            LazyPairDataset(
                phase='training',
                image_path=tmp_dir,
                metadata_path=dummy_meta_p
            )
            assert False, "Should have raised RuntimeError when image not in metadata"
        except RuntimeError as e:
            assert "absent from metadata" in str(e)
    return True


def test_t125_pathology_label_parity_with_canonical_metadata():
    """T125: True LazyPairDataset pathology label parity with independent metadata parser.
    Tests 'No Finding', single pathology, and multi-label pathology cases.
    """
    df_meta = pd.read_csv(FROZEN_METADATA_PATH)
    meta_dict = dict(zip(df_meta['Image Index'], df_meta['Finding Labels']))

    pred_label = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
        'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
        'Pleural_Thickening', 'Hernia'
    ]

    # Instantiate real LazyPairDataset for training
    ds = LazyPairDataset(phase='training', image_path='/home/minhtt/datasets/nih/images/')

    # Test cases: find examples of No Finding, single, and multi-label in dataset
    tested_no_finding = False
    tested_single = False
    tested_multi = False

    for i in range(min(len(ds), 500)):
        fname = ds.image_pairs[i][0]
        finding_raw = meta_dict[fname]
        actual_vec = ds.ac_labels_1[i]

        # Independent manual calculation from raw CSV string
        expected_vec = np.zeros(14, dtype=np.float32)
        if finding_raw != 'No Finding' and isinstance(finding_raw, str):
            tokens = [t.strip() for t in finding_raw.split('|')]
            for tok in tokens:
                if tok in pred_label:
                    expected_vec[pred_label.index(tok)] = 1.0

        np.testing.assert_array_equal(actual_vec, expected_vec)

        if finding_raw == 'No Finding':
            assert np.sum(actual_vec) == 0.0
            tested_no_finding = True
        elif '|' not in finding_raw:
            assert np.sum(actual_vec) == 1.0
            tested_single = True
        else:
            assert np.sum(actual_vec) >= 2.0
            tested_multi = True

    assert tested_no_finding, "Did not test 'No Finding' pathology case"
    assert tested_single, "Did not test single pathology case"
    assert tested_multi, "Did not test multi-label pathology case"
    return True


def test_t126_scientific_dependency_preflight_records_metadata():
    """T126: Dependency preflight returns valid metadata SHA and coverage."""
    res = verify_scientific_dependencies('/home/minhtt/datasets/nih/images/')
    assert res['status'] == 'PASS'
    assert res['metadata_sha256'] == FROZEN_METADATA_SHA
    assert res['train_image1_metadata_missing'] == 0
    assert res['val_image1_metadata_missing'] == 0
    return True


def test_t127_generator_manifest_sha_mutation_rejected_privacy():
    """T127: Generator manifest SHA mutation rejected before privacy evaluation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        gen_p = os.path.join(tmp_dir, 'gen.pth')
        att_p = os.path.join(tmp_dir, 'att.pth')
        torch.save(UNet(1, 2, 32).state_dict(), gen_p)
        torch.save(SiameseNetwork().state_dict(), att_p)

        real_gen_sha = file_sha256(gen_p)
        real_att_sha = file_sha256(att_p)

        # Mutate generator expected SHA
        try:
            evaluate_reid_val(
                config={'batch_size': 4},
                attacker_checkpoint=att_p,
                generator_checkpoint=gen_p,
                device='cpu',
                unit_test_mode=True,
                expected_generator_sha='0000000000000000000000000000000000000000000000000000000000000000',
                expected_attacker_sha=real_att_sha
            )
            assert False, "Should have rejected generator SHA mismatch"
        except RuntimeError as e:
            assert "Generator checkpoint SHA mismatch" in str(e)
    return True


def test_t128_attacker_manifest_sha_mutation_rejected_privacy():
    """T128: Attacker manifest SHA mutation rejected before privacy evaluation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        gen_p = os.path.join(tmp_dir, 'gen.pth')
        att_p = os.path.join(tmp_dir, 'att.pth')
        torch.save(UNet(1, 2, 32).state_dict(), gen_p)
        torch.save(SiameseNetwork().state_dict(), att_p)

        real_gen_sha = file_sha256(gen_p)

        try:
            evaluate_reid_val(
                config={'batch_size': 4},
                attacker_checkpoint=att_p,
                generator_checkpoint=gen_p,
                device='cpu',
                unit_test_mode=True,
                expected_generator_sha=real_gen_sha,
                expected_attacker_sha='0000000000000000000000000000000000000000000000000000000000000000'
            )
            assert False, "Should have rejected attacker SHA mismatch"
        except RuntimeError as e:
            assert "Attacker checkpoint SHA mismatch" in str(e)
    return True


def test_t129_generator_manifest_sha_mutation_rejected_classification():
    """T129: Generator manifest SHA mutation rejected before classification evaluation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        gen_p = os.path.join(tmp_dir, 'gen.pth')
        torch.save(UNet(1, 2, 32).state_dict(), gen_p)

        try:
            evaluate_classification_val(
                config={'unit_test_mode': True},
                fold='val',
                generator_checkpoint=gen_p,
                device='cpu',
                expected_generator_sha='0000000000000000000000000000000000000000000000000000000000000000'
            )
            assert False, "Should have rejected generator SHA mismatch"
        except RuntimeError as e:
            assert "Selected generator SHA mismatch" in str(e)
    return True


def test_t130_device_flag_does_not_bypass_preflight():
    """T130: Supplying --device does not bypass preflight verification (actual assertion)."""
    import unittest.mock as mock
    from m2_dev import run_m2_s1

    class MockArgsDevice:
        def __init__(self):
            self.scientific_m2_s1 = False
            self.arm = 'eval_only'
            self.max_epochs = 1
            self.attacker_epochs = 1
            self.attacker_patience = 1
            self.seed = 42
            self.attacker_seed = 42
            self.device = 'cpu'

    # Mock verify_environment_and_hashes to track if it was called
    called = []
    def fake_verify():
        called.append(True)
        return torch.device('cpu')

    with mock.patch.object(run_m2_s1, 'verify_environment_and_hashes', side_effect=fake_verify):
        with tempfile.TemporaryDirectory() as tmp_base:
            try:
                # non-unit_test_mode must call verify_environment_and_hashes
                run_m2_s1.run_orchestration(MockArgsDevice(), out_base_dir=tmp_base, unit_test_mode=False)
            except Exception:
                pass  # may fail downstream due to missing files in tmp_base

    assert len(called) > 0, "verify_environment_and_hashes was NOT called when --device was supplied!"
    return True


def test_t131_scientific_mode_rejects_arm_not_all():
    """T131: Scientific mode rejects --arm != 'all'."""
    sys.argv = ['run_m2_s1.py', '--scientific-m2-s1', '--arm', 'B_dev']
    try:
        parse_args()
        assert False, "Should have rejected --arm B_dev in scientific mode"
    except ValueError as e:
        assert "requires --arm all" in str(e)
    return True


def test_t132_t109_reaches_exact_10816_count_guard():
    """T132: Classification evaluator reaches exact 10,816-image contract guard."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        fake_gen_p = os.path.join(tmp_dir, 'gen.pth')
        torch.save(UNet(1, 2, 32).state_dict(), fake_gen_p)
        ds = SyntheticClassificationDataset(size=50, image_size=64)
        loader = torch.utils.data.DataLoader(ds, batch_size=16)

        try:
            evaluate_classification_val(
                config={'image_path': tmp_dir, 'dataloader': loader, 'unit_test_mode': False},
                fold='val',
                generator_checkpoint=fake_gen_p,
                device='cpu'
            )
            assert False, "Should have raised RuntimeError on images count != 10816"
        except RuntimeError as e:
            assert "Classification scientific VAL requires exactly 10,816 images, got 50" in str(e)
    return True


def test_t133_strict_validity_verifies_checkpoint_files_and_shas():
    """T133: check_run_validity fails if checkpoint files are missing or mutated."""
    with tempfile.NamedTemporaryFile() as f_gen, tempfile.NamedTemporaryFile() as f_att:
        gen_sha = file_sha256(f_gen.name)
        att_sha = file_sha256(f_att.name)

        b_m = {'epochs_completed': 250, 'requested_max_epochs': 250, 'numerical_validity': 'PASS', 'nan_inf_detected': False,
               'selected_generator_checkpoint': f_gen.name, 'selected_generator_sha256': gen_sha, 'config_sha256': FROZEN_B_DEV_CONFIG_SHA}
        c_m = {'epochs_completed': 250, 'requested_max_epochs': 250, 'numerical_validity': 'PASS', 'nan_inf_detected': False,
               'selected_generator_checkpoint': f_gen.name, 'selected_generator_sha256': gen_sha, 'config_sha256': FROZEN_C4_CONFIG_SHA}
        b_att = {'best_attacker_path': f_att.name, 'best_attacker_sha256': att_sha, 'generator_checkpoint_sha256': gen_sha,
                 'numerical_validity': 'PASS', 'nan_inf_detected': False}
        c_att = {'best_attacker_path': f_att.name, 'best_attacker_sha256': att_sha, 'generator_checkpoint_sha256': gen_sha,
                 'numerical_validity': 'PASS', 'nan_inf_detected': False}
        b_p = {'roc_auc': 0.70, 'n_pairs': 8, 'generator_checkpoint_sha256': gen_sha, 'attacker_checkpoint_sha256': att_sha}
        c_p = {'roc_auc': 0.71, 'n_pairs': 8, 'generator_checkpoint_sha256': gen_sha, 'attacker_checkpoint_sha256': att_sha}
        b_c = {'macro_auc': 0.80, 'n_classes_valid': 14, 'n_images': 28, 'generator_checkpoint_sha256': gen_sha, 'classifier_checkpoint_sha256': FROZEN_CLASSIFIER_SHA}
        c_c = {'macro_auc': 0.81, 'n_classes_valid': 14, 'n_images': 28, 'generator_checkpoint_sha256': gen_sha, 'classifier_checkpoint_sha256': FROZEN_CLASSIFIER_SHA}

        # Mutate recorded generator SHA
        b_m_bad = dict(b_m)
        b_m_bad['selected_generator_sha256'] = 'bad_sha'
        valid, reason = check_run_validity(b_m_bad, c_m, b_att, c_att, b_p, c_p, b_c, c_c, expected_epochs=250, unit_test_mode=True)
        assert not valid
        assert "generator SHA mismatch" in reason
    return True


def test_t134_invalid_run_cannot_produce_promote_verdict():
    """T134: Invalid orchestration run produces 'C4 S1: INVALID — NO SCIENTIFIC VERDICT'."""
    import unittest.mock as mock
    from m2_dev import run_m2_s1

    class MockArgsInvalid:
        def __init__(self):
            self.scientific_m2_s1 = False
            self.arm = 'all'
            self.max_epochs = 1
            self.attacker_epochs = 1
            self.attacker_patience = 1
            self.seed = 42
            self.attacker_seed = 42
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with tempfile.TemporaryDirectory() as tmp_dir:
        args = MockArgsInvalid()
        # Mock check_run_validity to return False
        with mock.patch.object(run_m2_s1, 'check_run_validity', return_value=(False, 'Injected invalid run error')):
            summary = run_m2_s1.run_orchestration(args, out_base_dir=tmp_dir, unit_test_mode=True)
            assert summary['run_status'] == 'INVALID', "Expected run_status == INVALID"
            assert summary['verdict'] == 'C4 S1: INVALID — NO SCIENTIFIC VERDICT', "Expected INVALID verdict, got: %s" % summary['verdict']
            assert summary['gates']['privacy_gate_status'] == 'NOT_EVALUATED_DUE_TO_INVALID_RUN'
            assert summary['gates']['classification_gate_status'] == 'NOT_EVALUATED_DUE_TO_INVALID_RUN'
    return True


def test_t135_scientific_config_semantic_key_drift_hard_fails():
    """T135: Injected semantic config drift (e.g. mu=0.02) raises RuntimeError."""
    # Test that genuine configs pass
    cfg_audit = verify_frozen_scientific_configs()
    assert cfg_audit['status'] == 'PASS'

    # Test that injected drift hard-fails
    with tempfile.TemporaryDirectory() as tmp_dir:
        fake_b_path = os.path.join(tmp_dir, 'config_dev_restored_baseline.json')
        with open(FROZEN_B_DEV_CONFIG_PATH) as f:
            cfg = json.load(f)
        cfg['mu'] = 0.02  # Inject semantic drift

        with open(fake_b_path, 'w') as f:
            json.dump(cfg, f)

        # Mutate the constant path temporarily
        import m2_dev.evaluator_common as ec
        old_path = ec.FROZEN_B_DEV_CONFIG_PATH
        old_sha = ec.FROZEN_B_DEV_CONFIG_SHA
        try:
            ec.FROZEN_B_DEV_CONFIG_PATH = fake_b_path
            ec.FROZEN_B_DEV_CONFIG_SHA = file_sha256(fake_b_path)
            try:
                ec.verify_frozen_scientific_configs()
                assert False, "Should have raised RuntimeError on mu=0.02"
            except RuntimeError as e:
                assert "mu must be 0.01" in str(e) or "B_dev config SHA mismatch" in str(e)
        finally:
            ec.FROZEN_B_DEV_CONFIG_PATH = old_path
            ec.FROZEN_B_DEV_CONFIG_SHA = old_sha

    return True


def test_t136_miniature_scientific_validity_bundle_and_mutations():
    """T136: Miniature scientific validity bundle is VALID; independent mutations are INVALID."""
    with tempfile.NamedTemporaryFile() as f_gen_b, tempfile.NamedTemporaryFile() as f_gen_c4, \
         tempfile.NamedTemporaryFile() as f_att_b, tempfile.NamedTemporaryFile() as f_att_c4:

        gen_b_sha = file_sha256(f_gen_b.name)
        gen_c4_sha = file_sha256(f_gen_c4.name)
        att_b_sha = file_sha256(f_att_b.name)
        att_c4_sha = file_sha256(f_att_c4.name)

        b_m = {
            'epochs_completed': 250, 'requested_max_epochs': 250, 'numerical_validity': 'PASS', 'nan_inf_detected': False,
            'selected_generator_checkpoint': f_gen_b.name, 'selected_generator_sha256': gen_b_sha,
            'config_sha256': FROZEN_B_DEV_CONFIG_SHA
        }
        c_m = {
            'epochs_completed': 250, 'requested_max_epochs': 250, 'numerical_validity': 'PASS', 'nan_inf_detected': False,
            'selected_generator_checkpoint': f_gen_c4.name, 'selected_generator_sha256': gen_c4_sha,
            'config_sha256': FROZEN_C4_CONFIG_SHA
        }
        b_att = {
            'best_attacker_path': f_att_b.name, 'best_attacker_sha256': att_b_sha,
            'generator_checkpoint_sha256': gen_b_sha,
            'numerical_validity': 'PASS', 'nan_inf_detected': False
        }
        c_att = {
            'best_attacker_path': f_att_c4.name, 'best_attacker_sha256': att_c4_sha,
            'generator_checkpoint_sha256': gen_c4_sha,
            'numerical_validity': 'PASS', 'nan_inf_detected': False
        }
        b_p = {'roc_auc': 0.70, 'n_pairs': 8, 'generator_checkpoint_sha256': gen_b_sha, 'attacker_checkpoint_sha256': att_b_sha}
        c_p = {'roc_auc': 0.71, 'n_pairs': 8, 'generator_checkpoint_sha256': gen_c4_sha, 'attacker_checkpoint_sha256': att_c4_sha}
        b_c = {'macro_auc': 0.80, 'n_classes_valid': 14, 'n_images': 28, 'generator_checkpoint_sha256': gen_b_sha, 'classifier_checkpoint_sha256': FROZEN_CLASSIFIER_SHA}
        c_c = {'macro_auc': 0.81, 'n_classes_valid': 14, 'n_images': 28, 'generator_checkpoint_sha256': gen_c4_sha, 'classifier_checkpoint_sha256': FROZEN_CLASSIFIER_SHA}

        # 1. Base bundle -> VALID
        valid, reason = check_run_validity(b_m, c_m, b_att, c_att, b_p, c_p, b_c, c_c, expected_epochs=250, unit_test_mode=True)
        assert valid, "Base bundle must be VALID: %s" % reason

        # 2. Mutate generator bytes
        f_gen_b.write(b'corrupted')
        f_gen_b.flush()
        valid, reason = check_run_validity(b_m, c_m, b_att, c_att, b_p, c_p, b_c, c_c, expected_epochs=250, unit_test_mode=True)
        assert not valid, "Mutated generator bytes must fail"

        # Restore gen_b
        f_gen_b.seek(0)
        f_gen_b.truncate()
        f_gen_b.flush()
        b_m['selected_generator_sha256'] = file_sha256(f_gen_b.name)
        b_att['generator_checkpoint_sha256'] = b_m['selected_generator_sha256']
        b_p['generator_checkpoint_sha256'] = b_m['selected_generator_sha256']
        b_c['generator_checkpoint_sha256'] = b_m['selected_generator_sha256']

        # 3. Mutate attacker bytes
        f_att_b.write(b'corrupted')
        f_att_b.flush()
        valid, reason = check_run_validity(b_m, c_m, b_att, c_att, b_p, c_p, b_c, c_c, expected_epochs=250, unit_test_mode=True)
        assert not valid, "Mutated attacker bytes must fail"

        # Restore att_b
        f_att_b.seek(0)
        f_att_b.truncate()
        f_att_b.flush()
        b_att['best_attacker_sha256'] = file_sha256(f_att_b.name)
        b_p['attacker_checkpoint_sha256'] = b_att['best_attacker_sha256']

        # 4. Mutate config SHA
        b_m_bad_cfg = dict(b_m)
        b_m_bad_cfg['config_sha256'] = 'wrong_cfg_sha'
        valid, reason = check_run_validity(b_m_bad_cfg, c_m, b_att, c_att, b_p, c_p, b_c, c_c, expected_epochs=250, unit_test_mode=False)
        assert not valid, "Mutated config SHA must fail"

        # 5. Mutate numerical validity
        b_m_bad_num = dict(b_m)
        b_m_bad_num['numerical_validity'] = 'FAIL'
        valid, reason = check_run_validity(b_m_bad_num, c_m, b_att, c_att, b_p, c_p, b_c, c_c, expected_epochs=250, unit_test_mode=True)
        assert not valid, "Bad numerical validity must fail"

        # 6. Mutate epochs_completed
        b_m_bad_ep = dict(b_m)
        b_m_bad_ep['epochs_completed'] = 249
        valid, reason = check_run_validity(b_m_bad_ep, c_m, b_att, c_att, b_p, c_p, b_c, c_c, expected_epochs=250, unit_test_mode=True)
        assert not valid, "Incomplete epochs must fail"

    return True


if __name__ == '__main__':
    tests = [
        ('T113', 'anonymizer manifest contains epochs_completed', test_t113_anonymizer_manifest_contains_epochs_completed),
        ('T114', 'epochs_completed == expected -> valid', test_t114_epochs_completed_equals_expected_valid),
        ('T115', 'epochs_completed missing or 249/250 -> invalid', test_t115_epochs_completed_missing_or_249_fails),
        ('T116', 'B_dev frozen config SHA mismatch -> HARD FAIL', test_t116_b_dev_config_sha_mismatch_hard_fails),
        ('T117', 'C4 frozen config SHA mismatch -> HARD FAIL', test_t117_c4_config_sha_mismatch_hard_fails),
        ('T118', 'attacker frozen config SHA mismatch -> HARD FAIL', test_t118_attacker_config_sha_mismatch_hard_fails),
        ('T119', 'anonymizer scientific manifest records non-null config SHA', test_t119_anonymizer_manifest_records_non_null_config_sha),
        ('T120', 'NaN training metric -> HARD FAIL', test_t120_nan_training_metric_hard_fails),
        ('T121', 'Inf validation metric -> HARD FAIL', test_t121_inf_validation_metric_hard_fails),
        ('T122', 'manifest numerical_validity required by check_run_validity', test_t122_manifest_numerical_validity_required),
        ('T123', 'missing Data_Entry_2017_v2020.csv -> HARD FAIL', test_t123_missing_metadata_file_hard_fails),
        ('T124', 'missing image1 metadata key -> HARD FAIL', test_t124_missing_image1_metadata_key_hard_fails),
        ('T125', 'pathology-label vector parity with canonical metadata parser', test_t125_pathology_label_parity_with_canonical_metadata),
        ('T126', 'scientific dependency preflight records metadata SHA', test_t126_scientific_dependency_preflight_records_metadata),
        ('T127', 'generator manifest SHA mutation rejected before privacy eval', test_t127_generator_manifest_sha_mutation_rejected_privacy),
        ('T128', 'attacker manifest SHA mutation rejected before privacy eval', test_t128_attacker_manifest_sha_mutation_rejected_privacy),
        ('T129', 'generator manifest SHA mutation rejected before classification eval', test_t129_generator_manifest_sha_mutation_rejected_classification),
        ('T130', '--device cuda does NOT bypass scientific preflight', test_t130_device_flag_does_not_bypass_preflight),
        ('T131', 'scientific mode rejects arm != all', test_t131_scientific_mode_rejects_arm_not_all),
        ('T132', 'T109 replacement reaches exact 10816-count guard, not generic FileNotFoundError', test_t132_t109_reaches_exact_10816_count_guard),
        ('T133', 'strict check_run_validity verifies checkpoint files + SHAs', test_t133_strict_validity_verifies_checkpoint_files_and_shas),
        ('T134', 'invalid run cannot produce scientific PROMOTE verdict', test_t134_invalid_run_cannot_produce_promote_verdict),
        ('T135', 'scientific config semantic key drift -> HARD FAIL', test_t135_scientific_config_semantic_key_drift_hard_fails),
        ('T136', 'full miniature scientific-validity manifest bundle -> VALID', test_t136_miniature_scientific_validity_bundle_and_mutations),
    ]

    all_pass = True
    print("=" * 60)
    print("RUNNING M1.4b EXECUTION INTEGRITY & VALIDITY TESTS (T113–T136)")
    print("=" * 60)
    for tid, name, fn in tests:
        try:
            fn()
            print(f"PASS     {tid:<6} {name}")
        except Exception as e:
            print(f"FAIL     {tid:<6} {name} -- {e}")
            import traceback
            traceback.print_exc()
            all_pass = False

    print("=" * 60)
    print("M1.4b SUITE:", f"{len([t for t in tests if all_pass])}/{len(tests)} PASS" if all_pass else "FAILURES PRESENT")
    print("=" * 60)
    sys.exit(0 if all_pass else 1)
