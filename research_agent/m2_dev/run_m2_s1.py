"""M2-S1 Master Execution Orchestrator.

Executes the full paired scientific experiment:
  1. Train B_dev (seed 42, 250 epochs, feature_loss_weight=0.0)
  2. Train C4 (seed 42, 250 epochs, feature_loss_weight=1.0)
  3. Freeze selected checkpoints (generator_best_method_neutral.pth)
  4. Train S1 Adaptive Attacker for B_dev (seed 42, 100 epochs max, patience 5)
  5. Train S1 Adaptive Attacker for C4 (seed 42, 100 epochs max, patience 5)
  6. Evaluate Scientific Privacy VAL Re-ID AUC (anon(img1), real(img2))
  7. Evaluate Classification Utility VAL Macro AUC & 14-disease AUCs
  8. Apply Frozen S1 Decision Gates (Privacy <= +0.03, Classification >= 0.0)
  9. Produce M2_S1_summary.json and M2_S1_C4_RESULT.md report.

STRICTLY TRAIN / VALIDATION ONLY — NO TEST ACCESS.
"""
import os
import sys
import json
import time
import hashlib
import subprocess
import argparse
from collections.abc import Mapping
import numpy as np
import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
for _p in (ROOT, os.path.join(ROOT, 'research_agent')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from test_firewall import provenance_record
from m2_dev.evaluator_common import (
    firewall_check,
    file_sha256,
    INITIAL_GENERATOR_PATH,
    INITIAL_GENERATOR_SHA,
    FROZEN_CLASSIFIER_PATH,
    FROZEN_CLASSIFIER_SHA,
    FROZEN_VERIFIER_PATH,
    FROZEN_VERIFIER_SHA,
    REPAIRED_ACLOSS_PATH,
    REPAIRED_ACLOSS_SHA,
    FROZEN_METADATA_PATH,
    FROZEN_METADATA_SHA,
    FROZEN_B_DEV_CONFIG_PATH,
    FROZEN_B_DEV_CONFIG_SHA,
    FROZEN_C4_CONFIG_PATH,
    FROZEN_C4_CONFIG_SHA,
    FROZEN_ATTACKER_CONFIG_PATH,
    FROZEN_ATTACKER_CONFIG_SHA,
    METHOD_NEUTRAL_CKPT_NAME,
    NIH_PATHOLOGIES,
    verify_repaired_acloss,
    verify_frozen_scientific_configs,
    verify_scientific_dependencies,
    verify_classification_val_contract,
    validate_scientific_args,
    verify_execution_lock_integrity,
    verify_frozen_val_pair_provenance,
    verify_classification_artifact_structure,
    FROZEN_VAL_PAIR_COUNT,
    FROZEN_VAL_PAIR_SHA256,
    FROZEN_CLASSIFICATION_VAL_N_IMAGES,
    FROZEN_CLASSIFICATION_VAL_IMAGE_INDEX_SHA,
    FROZEN_CLASSIFICATION_VAL_PATIENT_SEQUENCE_SHA,
    FROZEN_CLASSIFICATION_VAL_LABEL_MATRIX_SHA,
    FROZEN_SOURCE_TAG,
    FROZEN_SOURCE_BRANCH,
)
from m2_dev.anonymizer_runner import M2AnonymizerRunner
from m2_dev.dev_attacker import DevAttacker, SiameseNetwork
from m2_dev.eval_reid_val import evaluate_reid_val
from m2_dev.eval_classifier_val import evaluate_classification_val


def parse_args():
    parser = argparse.ArgumentParser(description="M2-S1 Master Execution Runner")
    parser.add_argument('--scientific-m2-s1', action='store_true', default=False,
                        help="Run full frozen scientific M2-S1 protocol (locks all parameters)")
    parser.add_argument('--arm', choices=['B_dev', 'C4', 'all', 'eval_only'], default='all',
                        help="Which phase to execute (default: all)")
    parser.add_argument('--max_epochs', type=int, default=250,
                        help="Number of anonymizer epochs (default: 250)")
    parser.add_argument('--attacker_epochs', type=int, default=100,
                        help="Max attacker training epochs (default: 100)")
    parser.add_argument('--attacker_patience', type=int, default=5,
                        help="Attacker early stopping patience (default: 5)")
    parser.add_argument('--seed', type=int, default=42,
                        help="Anonymizer seed (default: 42)")
    parser.add_argument('--attacker_seed', type=int, default=42,
                        help="Attacker seed (default: 42)")
    parser.add_argument('--device', type=str, default=None,
                        help="Compute device (cuda / cpu)")
    parser.add_argument('--fold', type=str, default='val',
                        help="Classification fold; scientific mode requires val")
    parser.add_argument('--resume', action='store_true', default=False,
                        help="Resume a run (not allowed in scientific mode)")
    args = parser.parse_args()

    if args.scientific_m2_s1:
        validate_scientific_args(args, unit_test_mode=False)

    # F1 (M1.4c): Canonical M2-S1 output namespace may ONLY be used with --scientific-m2-s1
    if not args.scientific_m2_s1:
        # Allow unit_test_mode to bypass (set at orchestration level, not here)
        args._require_scientific_gate = True
    else:
        args._require_scientific_gate = False

    return args


def assert_m2_scientific_mode_ready(image_path='/home/minhtt/datasets/nih/images/'):
    """Fail closed unless all dependencies, checkpoints, SHAs, pair files, metadata, configs, and images pass."""
    firewall_check('dev')
    dep_res = verify_scientific_dependencies(image_path=image_path)
    if dep_res['status'] != 'PASS':
        raise RuntimeError("Scientific dependencies audit failed: %s" % dep_res)
    return dep_res


def verify_environment_and_hashes():
    """Verify all frozen artifact hashes and dataset image availability before execution."""
    print("=" * 70)
    print("M2-S1: PRE-EXECUTION PROVENANCE & ARTIFACT AUDIT")
    print("=" * 70)
    firewall_check('dev')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s (%s)" % (device, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'))
    if torch.cuda.is_available():
        print("CUDA Capability: %s, VRAM: %.2f GB" % (
            torch.cuda.get_device_capability(0),
            torch.cuda.get_device_properties(0).total_memory / (1024**3)
        ))

    # Comprehensive dependency preflight (checkpoints, configs, metadata, pairs, images)
    dep = assert_m2_scientific_mode_ready('/home/minhtt/datasets/nih/images/')
    print("[PASS] Initial Generator SHA: %s..." % dep['initial_generator_sha256'][:16])
    print("[PASS] Frozen Classifier SHA: %s..." % dep['classifier_sha256'][:16])
    print("[PASS] Frozen Verifier SHA:   %s..." % dep['verifier_sha256'][:16])
    print("[PASS] Repaired ACLoss SHA:   %s..." % dep['acloss_sha256'][:16])
    print("[PASS] B_dev Config SHA:      %s..." % dep['b_dev_config_sha256'][:16])
    print("[PASS] C4 Config SHA:         %s..." % dep['c4_config_sha256'][:16])
    print("[PASS] Attacker Config SHA:   %s..." % dep['attacker_config_sha256'][:16])
    print("[PASS] Metadata SHA:          %s..." % dep['metadata_sha256'][:16])
    print("[PASS] TRAIN Pairs SHA:       %s... (%d pairs verified)" % (dep['train_pairs_sha256'][:16], dep['train_pairs_count']))
    print("[PASS] VAL Pairs SHA:         %s... (%d pairs verified)" % (dep['val_pairs_sha256'][:16], dep['val_pairs_count']))
    print("[PASS] Dataset Image Availability: 100%% (0 missing)")
    print("[PASS] Metadata Image1 Coverage: 100%% (0 missing)")
    print("=" * 70)
    return device


def run_anonymizer_arm(arm, config_path, max_epochs, seed, device, out_base_dir=None, unit_test_mode=False):
    """Execute anonymizer training for one arm."""
    print("\n" + "#" * 70)
    print("STARTING ANONYMIZER TRAINING: ARM %s (seed=%d, epochs=%d)" % (arm, seed, max_epochs))
    print("#" * 70)
    with open(config_path) as f:
        cfg = json.load(f)

    base = out_base_dir or os.path.join(ROOT, 'research_runs', 'M2_S1')
    out_dir = os.path.join(base, arm, 'seed_%d' % seed)
    os.makedirs(out_dir, exist_ok=True)

    train_loader = None
    val_loader = None
    if unit_test_mode:
        from m0_tests.test_m14a_execution_harness import SyntheticPairDataset
        cfg['image_size'] = 64
        ds = SyntheticPairDataset(8, image_size=64)
        train_loader = torch.utils.data.DataLoader(ds, batch_size=4)
        val_loader = torch.utils.data.DataLoader(ds, batch_size=4)

    runner = M2AnonymizerRunner(
        arm=arm,
        config=cfg,
        config_path=config_path,
        output_dir=out_dir,
        device=device,
        seed=seed,
        training_loader=train_loader,
        validation_loader=val_loader,
        unit_test_mode=unit_test_mode
    )

    t0 = time.time()
    manifest = runner.run(max_epochs=max_epochs)
    total_time = time.time() - t0
    manifest['training_runtime_sec'] = round(total_time, 2)
    manifest['training_runtime_hours'] = round(total_time / 3600.0, 3)

    with open(os.path.join(out_dir, 'checkpoint_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 70)
    print("COMPLETED ARM %s in %.2f hours" % (arm, total_time / 3600.0))
    print("Best Epoch: %s, Best Selection Total: %.5f" % (manifest['best_epoch'], manifest['best_selection_total']))
    print("Best Checkpoint SHA: %s" % manifest['selected_generator_sha256'])
    print("=" * 70)
    return manifest


def train_s1_attacker_arm(arm, seed, attacker_seed, max_epochs, patience, device, out_base_dir=None, unit_test_mode=False):
    """Train S1 adaptive attacker for one arm using its selected generator."""
    print("\n" + "-" * 70)
    print("TRAINING S1 ADAPTIVE ATTACKER FOR %s (seed=%d, max_epochs=%d)" % (arm, attacker_seed, max_epochs))
    print("-" * 70)
    base = out_base_dir or os.path.join(ROOT, 'research_runs', 'M2_S1')
    arm_dir = os.path.join(base, arm, 'seed_%d' % seed)
    manifest_p = os.path.join(arm_dir, 'checkpoint_manifest.json')
    if not os.path.exists(manifest_p):
        raise FileNotFoundError("Generator checkpoint manifest not found: %s" % manifest_p)

    with open(manifest_p) as f:
        gen_manifest = json.load(f)

    gen_ckpt = gen_manifest['selected_generator_checkpoint']
    expected_gen_sha = gen_manifest['selected_generator_sha256']
    actual_gen_sha = file_sha256(gen_ckpt)
    if actual_gen_sha != expected_gen_sha:
        raise RuntimeError("Generator checkpoint SHA mismatch: %s != %s" % (actual_gen_sha, expected_gen_sha))

    attacker_out_dir = os.path.join(arm_dir, 'attacker_%d' % attacker_seed)
    os.makedirs(attacker_out_dir, exist_ok=True)

    att_cfg_path = os.path.join(ROOT, 'config_files', 'config_dev_attacker_s1.json')
    if not os.path.exists(att_cfg_path):
        raise FileNotFoundError("Attacker config not found: %s" % att_cfg_path)
    actual_att_cfg_sha = file_sha256(att_cfg_path)
    if actual_att_cfg_sha != FROZEN_ATTACKER_CONFIG_SHA and not unit_test_mode:
        raise RuntimeError("Attacker config SHA mismatch: %s != %s" % (actual_att_cfg_sha, FROZEN_ATTACKER_CONFIG_SHA))

    with open(att_cfg_path) as f:
        attacker_cfg = json.load(f)

    if not unit_test_mode:
        if max_epochs != attacker_cfg.get('max_epochs', 100):
            raise ValueError("Attacker max_epochs %d does not match frozen config %d" % (max_epochs, attacker_cfg.get('max_epochs', 100)))
        if patience != attacker_cfg.get('early_stopping', 5):
            raise ValueError("Attacker patience %d does not match frozen config %d" % (patience, attacker_cfg.get('early_stopping', 5)))
        if attacker_seed != attacker_cfg.get('attacker_seed', 42):
            raise ValueError("Attacker seed %d does not match frozen config %d" % (attacker_seed, attacker_cfg.get('attacker_seed', 42)))

    attacker_cfg['max_epochs'] = max_epochs
    attacker_cfg['early_stopping'] = patience
    attacker_cfg['attacker_output_dir'] = attacker_out_dir

    train_loader = None
    val_loader = None
    if unit_test_mode:
        from m0_tests.test_m14a_execution_harness import SyntheticAttackerPairDataset
        ds = SyntheticAttackerPairDataset(8, image_size=64)
        train_loader = torch.utils.data.DataLoader(ds, batch_size=4)
        val_loader = torch.utils.data.DataLoader(ds, batch_size=4)

    attacker = DevAttacker(
        config=attacker_cfg,
        device=device,
        attacker_seed=attacker_seed,
        generator_checkpoint=gen_ckpt,
        training_loader=train_loader,
        validation_loader=val_loader,
        image_size=64 if unit_test_mode else None,
        unit_test_mode=unit_test_mode,
        config_path=att_cfg_path
    )

    t0 = time.time()
    hist = attacker.run(output_dir=attacker_out_dir)
    elapsed = time.time() - t0

    best_attacker_path = hist['checkpoint_path']
    if not best_attacker_path or not os.path.exists(best_attacker_path):
        raise RuntimeError("Best attacker checkpoint file missing: %s" % best_attacker_path)

    attacker_manifest_path = os.path.join(attacker_out_dir, 'attacker_manifest.json')
    with open(attacker_manifest_path) as f:
        attacker_manifest = json.load(f)

    print("Attacker training finished in %.2fs. Best Val BCE: %.5f at Epoch %s" % (
        elapsed, hist['best_val_loss'], hist['best_epoch']
    ))
    return attacker_manifest


def evaluate_privacy_arm(arm, seed, attacker_seed, device, out_base_dir=None, unit_test_mode=False):
    """Evaluate scientific VAL Re-ID AUC (anon(img1), real(img2)) and save predictions NPZ."""
    base = out_base_dir or os.path.join(ROOT, 'research_runs', 'M2_S1')
    arm_dir = os.path.join(base, arm, 'seed_%d' % seed)
    att_dir = os.path.join(arm_dir, 'attacker_%d' % attacker_seed)

    gen_manifest_p = os.path.join(arm_dir, 'checkpoint_manifest.json')
    att_manifest_p = os.path.join(att_dir, 'attacker_manifest.json')

    with open(gen_manifest_p) as f:
        gen_manifest = json.load(f)
    with open(att_manifest_p) as f:
        att_manifest = json.load(f)

    gen_ckpt = gen_manifest['selected_generator_checkpoint']
    attacker_ckpt = att_manifest['best_attacker_path']
    expected_gen_sha = gen_manifest['selected_generator_sha256']
    expected_att_sha = att_manifest['best_attacker_sha256']

    # Explicit SHA revalidation before loading
    actual_gen_sha = file_sha256(gen_ckpt)
    actual_att_sha = file_sha256(attacker_ckpt)
    if actual_gen_sha != expected_gen_sha:
        raise RuntimeError("Generator checkpoint SHA mismatch before privacy eval: %s != expected %s" % (actual_gen_sha, expected_gen_sha))
    if actual_att_sha != expected_att_sha:
        raise RuntimeError("Attacker checkpoint SHA mismatch before privacy eval: %s != expected %s" % (actual_att_sha, expected_att_sha))

    cfg = {
        'batch_size': 32,
        'image_path': '/home/minhtt/datasets/nih/images/',
    }
    eval_res = evaluate_reid_val(
        config=cfg,
        attacker_checkpoint=attacker_ckpt,
        generator_checkpoint=gen_ckpt,
        device=device,
        unit_test_mode=unit_test_mode,
        image_size=64 if unit_test_mode else None,
        expected_generator_sha=expected_gen_sha,
        expected_attacker_sha=expected_att_sha
    )

    # Save raw predictions NPZ with explicit frozen pair provenance.
    pred_npz_path = os.path.join(att_dir, 'privacy_val_predictions.npz')
    pair_provenance = verify_frozen_val_pair_provenance() if not unit_test_mode else {}
    np.savez_compressed(
        pred_npz_path,
        y_true=eval_res['y_true'],
        y_score=eval_res['y_score'],
        validation_pair_file_sha256=pair_provenance.get('pair_file_sha256', ''),
        validation_pair_count=pair_provenance.get('pair_count', int(eval_res['n_pairs'])),
        validation_pair_order_sha256=pair_provenance.get('pair_order_sha256', ''),
    )
    pred_sha = file_sha256(pred_npz_path)

    # Return scalar summary dict with path references
    return {
        'roc_auc': float(eval_res['roc_auc']),
        'accuracy': float(eval_res['accuracy']),
        'precision': float(eval_res['precision']),
        'recall': float(eval_res['recall']),
        'f1': float(eval_res['f1']),
        'n_pairs': int(eval_res['n_pairs']),
        'generator_checkpoint_sha256': eval_res['generator_checkpoint_sha256'],
        'attacker_checkpoint_sha256': eval_res['attacker_checkpoint_sha256'],
        'prediction_file': pred_npz_path,
        'prediction_file_sha256': pred_sha,
        'predictions_file': pred_npz_path,
        'predictions_file_sha256': pred_sha,
        'validation_pair_file_sha256': pair_provenance.get('pair_file_sha256'),
        'validation_pair_count': pair_provenance.get('pair_count', int(eval_res['n_pairs'])),
        'validation_pair_order_sha256': pair_provenance.get('pair_order_sha256'),
    }


def evaluate_classification_arm(arm, seed, device, out_base_dir=None, unit_test_mode=False):
    """Evaluate clinical utility classification VAL Macro AUC & 14 disease AUCs."""
    base = out_base_dir or os.path.join(ROOT, 'research_runs', 'M2_S1')
    arm_dir = os.path.join(base, arm, 'seed_%d' % seed)
    gen_manifest_p = os.path.join(arm_dir, 'checkpoint_manifest.json')

    with open(gen_manifest_p) as f:
        gen_manifest = json.load(f)

    gen_ckpt = gen_manifest['selected_generator_checkpoint']
    expected_gen_sha = gen_manifest['selected_generator_sha256']

    # Explicit SHA revalidation before loading
    actual_gen_sha = file_sha256(gen_ckpt)
    if actual_gen_sha != expected_gen_sha:
        raise RuntimeError("Generator checkpoint SHA mismatch before classification eval: %s != expected %s" % (actual_gen_sha, expected_gen_sha))

    cfg = {
        'batch_size': 32,
        'image_path': '/home/minhtt/datasets/nih/images/',
        'unit_test_mode': unit_test_mode
    }
    clf_res = evaluate_classification_val(
        config=cfg,
        fold='val',
        generator_checkpoint=gen_ckpt,
        device=device,
        image_size=64 if unit_test_mode else None,
        expected_generator_sha=expected_gen_sha,
        unit_test_mode=unit_test_mode
    )
    if clf_res['n_classes_valid'] != 14:
        raise RuntimeError("Classification evaluation returned %d valid classes, expected 14" % clf_res['n_classes_valid'])

    # F13 (M1.4c): Persist raw predictions and per-pathology AUCs
    pred_csv_path = os.path.join(arm_dir, 'classification_val_predictions.csv')
    auc_csv_path = os.path.join(arm_dir, 'classification_val_aucs.csv')
    clf_res['pred_df'].to_csv(pred_csv_path, index=False)
    clf_res['auc_df'].to_csv(auc_csv_path, index=False)
    clf_res['predictions_file'] = pred_csv_path
    clf_res['predictions_file_sha256'] = file_sha256(pred_csv_path)
    clf_res['aucs_file'] = auc_csv_path
    clf_res['aucs_file_sha256'] = file_sha256(auc_csv_path)

    return clf_res


def _check_run_validity_m14c3(b_dev_manifest, c4_manifest, b_att_manifest, c4_att_manifest,
                              b_priv, c4_priv, b_class, c4_class,
                              expected_epochs=250, unit_test_mode=False):
    """Fail-closed structured validity check used by both production and tests."""
    bundles = [
        ('B_dev manifest', b_dev_manifest), ('C4 manifest', c4_manifest),
        ('B_dev attacker manifest', b_att_manifest), ('C4 attacker manifest', c4_att_manifest),
        ('B_dev privacy result', b_priv), ('C4 privacy result', c4_priv),
        ('B_dev classification result', b_class), ('C4 classification result', c4_class),
    ]
    for label, value in bundles:
        if not isinstance(value, Mapping):
            return False, '%s must be a mapping; got %r' % (label, type(value).__name__)

    for name, manifest, expected_cfg_sha in (
            ('B_dev', b_dev_manifest, FROZEN_B_DEV_CONFIG_SHA),
            ('C4', c4_manifest, FROZEN_C4_CONFIG_SHA)):
        if manifest.get('epochs_completed') != expected_epochs:
            return False, '%s epochs_completed (%s) != expected (%d)' % (name, manifest.get('epochs_completed'), expected_epochs)
        if manifest.get('requested_max_epochs') != expected_epochs:
            return False, '%s requested_max_epochs (%s) != expected (%d)' % (name, manifest.get('requested_max_epochs'), expected_epochs)
        if manifest.get('numerical_validity') != 'PASS':
            return False, '%s numerical_validity != PASS' % name
        if manifest.get('nan_inf_detected') is not False:
            return False, '%s nan_inf_detected is True' % name
        if manifest.get('gradient_diagnostics_failed') is True:
            return False, '%s gradient diagnostics reported failure' % name
        diagnostics = manifest.get('gradient_norm_diagnostics', {})
        if not isinstance(diagnostics, Mapping):
            return False, '%s gradient diagnostics must be a mapping' % name
        for epoch_key, entry in diagnostics.items():
            if not isinstance(entry, Mapping):
                return False, '%s gradient diagnostic %r is malformed' % (name, epoch_key)
            if 'error' in entry:
                return False, '%s contains failed gradient diagnostics' % name
            for metric_key in ('base_grad_norm', 'feature_grad_norm', 'feature_base_ratio'):
                if entry.get(metric_key) is None:
                    continue
                try:
                    if not np.isfinite(float(entry[metric_key])):
                        return False, '%s contains non-finite gradient diagnostics' % name
                except (TypeError, ValueError):
                    return False, '%s contains malformed gradient diagnostics' % name
        gen_path = manifest.get('selected_generator_checkpoint')
        if not gen_path or not os.path.exists(gen_path):
            return False, '%s selected generator checkpoint missing: %s' % (name, gen_path)
        actual_gen_sha = file_sha256(gen_path)
        if actual_gen_sha != manifest.get('selected_generator_sha256'):
            return False, '%s selected generator SHA mismatch' % name
        if not unit_test_mode and manifest.get('config_sha256') != expected_cfg_sha:
            return False, '%s config_sha256 mismatch' % name

    for name, att_manifest, gen_manifest in (
            ('B_dev', b_att_manifest, b_dev_manifest), ('C4', c4_att_manifest, c4_manifest)):
        att_path = att_manifest.get('best_attacker_path')
        if not att_path or not os.path.exists(att_path):
            return False, '%s best attacker checkpoint missing: %s' % (name, att_path)
        if file_sha256(att_path) != att_manifest.get('best_attacker_sha256'):
            return False, '%s attacker checkpoint SHA mismatch' % name
        if att_manifest.get('generator_checkpoint_sha256') != gen_manifest.get('selected_generator_sha256'):
            return False, '%s attacker generator SHA link mismatch' % name
        if att_manifest.get('numerical_validity') != 'PASS' or att_manifest.get('nan_inf_detected') is not False:
            return False, '%s attacker numerical validity failed' % name

    for name, privacy, gen_manifest, att_manifest in (
            ('B_dev', b_priv, b_dev_manifest, b_att_manifest),
            ('C4', c4_priv, c4_manifest, c4_att_manifest)):
        try:
            auc = float(privacy.get('roc_auc'))
        except (TypeError, ValueError):
            return False, '%s privacy ROC-AUC is malformed' % name
        if not np.isfinite(auc):
            return False, 'Non-finite %s privacy ROC-AUC' % name
        if privacy.get('generator_checkpoint_sha256') != gen_manifest.get('selected_generator_sha256'):
            return False, '%s privacy generator SHA link mismatch' % name
        if privacy.get('attacker_checkpoint_sha256') != att_manifest.get('best_attacker_sha256'):
            return False, '%s privacy attacker SHA link mismatch' % name
        npz_path = privacy.get('predictions_file') or privacy.get('prediction_file')
        if unit_test_mode and not npz_path:
            continue
        if not npz_path or not os.path.exists(npz_path):
            return False, '%s privacy predictions file missing: %s' % (name, npz_path)
        expected_npz_sha = privacy.get('predictions_file_sha256') or privacy.get('prediction_file_sha256')
        if file_sha256(npz_path) != expected_npz_sha:
            return False, '%s privacy predictions SHA mismatch' % name
        try:
            with np.load(npz_path, allow_pickle=False) as raw:
                if 'y_true' not in raw or 'y_score' not in raw:
                    return False, '%s privacy replay requires y_true and y_score' % name
                y_true = np.asarray(raw['y_true'])
                y_score = np.asarray(raw['y_score'])
                if y_true.ndim != 1 or y_score.ndim != 1 or len(y_true) != len(y_score):
                    return False, '%s privacy arrays must be 1-D and equal length' % name
                if len(y_true) != int(privacy.get('n_pairs', -1)):
                    return False, '%s privacy metadata n_pairs does not match raw arrays' % name
                if not unit_test_mode and len(y_true) != FROZEN_VAL_PAIR_COUNT:
                    return False, '%s privacy raw arrays must contain exactly 2000 rows' % name
                if not np.isfinite(y_true).all() or not np.isfinite(y_score).all():
                    return False, '%s privacy arrays contain NaN/Inf' % name
                if not np.all(np.isin(y_true, [0, 1])):
                    return False, '%s privacy labels are not binary' % name
                if len(np.unique(y_true)) != 2:
                    return False, '%s privacy labels do not contain both classes' % name
                if not unit_test_mode:
                    frozen_pair = verify_frozen_val_pair_provenance()
                    if privacy.get('validation_pair_file_sha256') != FROZEN_VAL_PAIR_SHA256:
                        return False, '%s privacy VAL pair provenance SHA mismatch' % name
                    if privacy.get('validation_pair_count') != FROZEN_VAL_PAIR_COUNT:
                        return False, '%s privacy VAL pair provenance count mismatch' % name
                    if privacy.get('validation_pair_order_sha256') != frozen_pair['pair_order_sha256']:
                        return False, '%s privacy VAL pair order provenance mismatch' % name
                    for key, expected in (
                            ('validation_pair_file_sha256', FROZEN_VAL_PAIR_SHA256),
                            ('validation_pair_count', FROZEN_VAL_PAIR_COUNT),
                            ('validation_pair_order_sha256', frozen_pair['pair_order_sha256'])):
                        if key not in raw:
                            return False, '%s privacy replay missing %s metadata' % (name, key)
                        raw_value = np.asarray(raw[key]).reshape(-1)[0].item()
                        if str(raw_value) != str(expected):
                            return False, '%s privacy replay %s metadata mismatch' % (name, key)
                replay_auc = float(__import__('sklearn.metrics', fromlist=['roc_auc_score']).roc_auc_score(y_true, y_score))
        except Exception as exc:
            return False, '%s privacy replay failed: %s' % (name, exc)
        if abs(replay_auc - auc) > 1e-5:
            return False, '%s privacy replayed AUC mismatch: %e != %e' % (name, replay_auc, auc)

    for name, classification, gen_manifest in (
            ('B_dev', b_class, b_dev_manifest), ('C4', c4_class, c4_manifest)):
        try:
            macro = float(classification.get('macro_auc'))
        except (TypeError, ValueError):
            return False, '%s classification Macro AUC is malformed' % name
        if not np.isfinite(macro):
            return False, 'Non-finite %s classification Macro AUC' % name
        if classification.get('n_classes_valid') != 14:
            return False, 'Invalid class count in %s classification' % name
        if classification.get('generator_checkpoint_sha256') != gen_manifest.get('selected_generator_sha256'):
            return False, '%s classification generator SHA link mismatch' % name
        if not unit_test_mode and classification.get('classifier_checkpoint_sha256') != FROZEN_CLASSIFIER_SHA:
            return False, '%s classification classifier SHA drift' % name
        pred_csv = classification.get('predictions_file')
        auc_csv = classification.get('aucs_file')
        if unit_test_mode and not pred_csv and not auc_csv:
            continue
        if not pred_csv or not os.path.exists(pred_csv) or not auc_csv or not os.path.exists(auc_csv):
            return False, '%s classification raw artifact missing' % name
        if file_sha256(pred_csv) != classification.get('predictions_file_sha256'):
            return False, '%s classification predictions SHA mismatch' % name
        if file_sha256(auc_csv) != classification.get('aucs_file_sha256'):
            return False, '%s classification AUCs SHA mismatch' % name
        try:
            pred_df = pd.read_csv(pred_csv)
            missing_gt = [p for p in NIH_PATHOLOGIES if p not in pred_df.columns]
            if missing_gt:
                return False, '%s classification replay: missing ground-truth column(s): %s' % (name, missing_gt)
            missing_prob = ['prob_' + p for p in NIH_PATHOLOGIES if 'prob_' + p not in pred_df.columns]
            if missing_prob:
                return False, '%s classification replay: missing probability column(s): %s' % (name, missing_prob)
            strict = not unit_test_mode
            fingerprints = verify_classification_artifact_structure(pred_df, strict=strict)
            if strict and fingerprints['n_images'] != FROZEN_CLASSIFICATION_VAL_N_IMAGES:
                return False, '%s classification rows != 10816' % name
            auc_df = pd.read_csv(auc_csv)
            if list(auc_df.columns)[:2] != ['label', 'auc']:
                return False, '%s classification AUCs CSV schema invalid' % name
            labels = auc_df['label'].astype(str).tolist()
            if len(labels) != 14:
                return False, '%s classification AUCs CSV has %d rows; exactly 14 rows required' % (name, len(labels))
            if len(set(labels)) != 14:
                return False, '%s classification AUCs CSV contains duplicate pathology rows' % name
            if set(labels) != set(NIH_PATHOLOGIES):
                return False, '%s classification AUCs CSV has unknown/missing pathologies: %s' % (name, sorted(set(labels) ^ set(NIH_PATHOLOGIES)))
            values = pd.to_numeric(auc_df['auc'], errors='coerce').to_numpy(dtype=float)
            if not np.isfinite(values).all():
                return False, '%s classification AUCs CSV contains NaN/Inf' % name
            replayed = {}
            from sklearn.metrics import roc_auc_score
            for pathology in NIH_PATHOLOGIES:
                y_true = pred_df[pathology].to_numpy(dtype=float)
                y_score = pred_df['prob_' + pathology].to_numpy(dtype=float)
                if not np.isfinite(y_true).all() or not np.isfinite(y_score).all() or not np.all(np.isin(y_true, [0, 1])):
                    return False, '%s classification contains non-finite/nonbinary values for %s' % (name, pathology)
                if len(np.unique(y_true)) < 2:
                    return False, '%s classification pathology %r has one class' % (name, pathology)
                replayed[pathology] = float(roc_auc_score(y_true.astype(int), y_score))
            csv_map = dict(zip(labels, values.tolist()))
            mem_df = classification.get('auc_df')
            if not isinstance(mem_df, pd.DataFrame) or set(mem_df.get('label', [] ).astype(str)) != set(NIH_PATHOLOGIES):
                return False, '%s classification in-memory auc_df is malformed' % name
            mem_map = dict(zip(mem_df['label'].astype(str), pd.to_numeric(mem_df['auc'], errors='coerce').astype(float)))
            for pathology in NIH_PATHOLOGIES:
                if not np.isfinite(mem_map[pathology]) or abs(replayed[pathology] - mem_map[pathology]) > 1e-7:
                    return False, '%s classification per-pathology replay mismatch vs auc_df for %r' % (name, pathology)
                if abs(replayed[pathology] - csv_map[pathology]) > 1e-7:
                    return False, '%s classification per-pathology replay mismatch vs AUC CSV for %r' % (name, pathology)
            replayed_macro = float(np.mean(list(replayed.values())))
        except Exception as exc:
            return False, '%s classification replay failed: %s' % (name, exc)
        if abs(replayed_macro - macro) > 1e-7:
            return False, '%s classification replayed macro AUC mismatch: %e != %e' % (name, replayed_macro, macro)

    return True, 'VALID' if not unit_test_mode else 'DEVELOPMENT_VALID'


def check_run_validity(b_dev_manifest, c4_manifest, b_att_manifest, c4_att_manifest,
                       b_priv, c4_priv, b_class, c4_class, expected_epochs=250, unit_test_mode=False):
    """Verify completion artifacts and replay integrity with structured INVALID results."""
    return _check_run_validity_m14c3(
        b_dev_manifest, c4_manifest, b_att_manifest, c4_att_manifest,
        b_priv, c4_priv, b_class, c4_class,
        expected_epochs=expected_epochs, unit_test_mode=unit_test_mode
    )
    if not (b_dev_manifest and c4_manifest and b_att_manifest and c4_att_manifest):
        return False, "Missing manifest"

    # 1. Anonymizer completion contract
    for name, m, exp_cfg_sha in [('B_dev', b_dev_manifest, FROZEN_B_DEV_CONFIG_SHA), ('C4', c4_manifest, FROZEN_C4_CONFIG_SHA)]:
        if m.get('epochs_completed') is None:
            return False, "%s manifest missing epochs_completed" % name
        if m.get('epochs_completed') != expected_epochs:
            return False, "%s epochs_completed (%s) != expected (%d)" % (name, m.get('epochs_completed'), expected_epochs)
        if m.get('requested_max_epochs') != expected_epochs:
            return False, "%s requested_max_epochs (%s) != expected (%d)" % (name, m.get('requested_max_epochs'), expected_epochs)
        if m.get('numerical_validity') != 'PASS':
            return False, "%s numerical_validity != PASS" % name
        if m.get('nan_inf_detected') is not False:
            return False, "%s nan_inf_detected is True" % name

        gen_p = m.get('selected_generator_checkpoint')
        if not gen_p or not os.path.exists(gen_p):
            return False, "%s selected generator checkpoint missing: %s" % (name, gen_p)
        actual_gen_sha = file_sha256(gen_p)
        if actual_gen_sha != m.get('selected_generator_sha256'):
            return False, "%s selected generator SHA mismatch: %s != manifest %s" % (name, actual_gen_sha, m.get('selected_generator_sha256'))
        if not unit_test_mode and m.get('config_sha256') != exp_cfg_sha:
            return False, "%s config_sha256 (%s) != frozen (%s)" % (name, m.get('config_sha256'), exp_cfg_sha)

    # 2. Attacker completion contract
    for name, att_m, gen_m in [('B_dev', b_att_manifest, b_dev_manifest), ('C4', c4_att_manifest, c4_manifest)]:
        att_p = att_m.get('best_attacker_path')
        if not att_p or not os.path.exists(att_p):
            return False, "%s best attacker checkpoint missing: %s" % (name, att_p)
        actual_att_sha = file_sha256(att_p)
        if actual_att_sha != att_m.get('best_attacker_sha256'):
            return False, "%s attacker SHA mismatch: %s != manifest %s" % (name, actual_att_sha, att_m.get('best_attacker_sha256'))
        if att_m.get('generator_checkpoint_sha256') != gen_m.get('selected_generator_sha256'):
            return False, "%s attacker generator SHA link mismatch" % name
        # §30 / F5 (M1.4c): Attacker numerical validity
        if att_m.get('numerical_validity') != 'PASS':
            return False, "%s attacker numerical_validity != PASS" % name
        if att_m.get('nan_inf_detected') is not False:
            return False, "%s attacker nan_inf_detected is True" % name

    # 3. Privacy Evaluation invariants
    for name, p_res, gen_m, att_m in [('B_dev', b_priv, b_dev_manifest, b_att_manifest), ('C4', c4_priv, c4_manifest, c4_att_manifest)]:
        if not np.isfinite(p_res.get('roc_auc', float('nan'))):
            return False, "Non-finite %s privacy ROC-AUC" % name
        if p_res.get('generator_checkpoint_sha256') != gen_m.get('selected_generator_sha256'):
            return False, "%s privacy generator SHA link mismatch" % name
        if p_res.get('attacker_checkpoint_sha256') != att_m.get('best_attacker_sha256'):
            return False, "%s privacy attacker SHA link mismatch" % name
        if not unit_test_mode and p_res.get('n_pairs') != 2000:
            return False, "%s privacy pairs count != 2000" % name

        # Raw outputs & replay validation
        npz_p = p_res.get('predictions_file') or p_res.get('prediction_file')
        if not unit_test_mode or npz_p is not None:
            if not npz_p or not os.path.exists(npz_p):
                return False, "%s privacy predictions file missing: %s" % (name, npz_p)
            actual_npz_sha = file_sha256(npz_p)
            expected_npz_sha = p_res.get('predictions_file_sha256') or p_res.get('prediction_file_sha256')
            if actual_npz_sha != expected_npz_sha:
                return False, "%s privacy predictions SHA mismatch: %s != %s" % (name, actual_npz_sha, expected_npz_sha)

            try:
                npz_data = np.load(npz_p)
                y_true_re = npz_data['y_true']
                y_score_re = npz_data['y_score']
                from sklearn.metrics import roc_auc_score
                replayed_priv_auc = float(roc_auc_score(y_true_re, y_score_re))
                if abs(replayed_priv_auc - p_res['roc_auc']) > 1e-5:
                    return False, "%s privacy replayed AUC mismatch: %e != %e" % (name, replayed_priv_auc, p_res['roc_auc'])
            except Exception as _priv_e:
                return False, "%s privacy replay failed: %s" % (name, _priv_e)

    # 4. Classification Evaluation invariants
    for name, c_res, gen_m in [('B_dev', b_class, b_dev_manifest), ('C4', c4_class, c4_manifest)]:
        if not np.isfinite(c_res.get('macro_auc', float('nan'))):
            return False, "Non-finite %s classification Macro AUC" % name
        if c_res.get('n_classes_valid') != 14:
            return False, "Invalid class count (%s) in %s classification" % (c_res.get('n_classes_valid'), name)
        if c_res.get('generator_checkpoint_sha256') != gen_m.get('selected_generator_sha256'):
            return False, "%s classification generator SHA link mismatch" % name
        if not unit_test_mode and c_res.get('classifier_checkpoint_sha256') != FROZEN_CLASSIFIER_SHA:
            return False, "%s classification classifier SHA drift" % name
        if not unit_test_mode and c_res.get('n_images') != 10816:
            return False, "%s classification images count != 10816" % name

        # Raw outputs & replay validation
        pred_csv = c_res.get('predictions_file')
        if not unit_test_mode or pred_csv is not None:
            if not pred_csv or not os.path.exists(pred_csv):
                return False, "%s classification predictions file missing: %s" % (name, pred_csv)
            actual_pred_sha = file_sha256(pred_csv)
            if actual_pred_sha != c_res.get('predictions_file_sha256'):
                return False, "%s classification predictions SHA mismatch: %s != %s" % (name, actual_pred_sha, c_res.get('predictions_file_sha256'))

            auc_csv = c_res.get('aucs_file')
            if not auc_csv or not os.path.exists(auc_csv):
                return False, "%s classification AUCs file missing: %s" % (name, auc_csv)
            actual_auc_sha = file_sha256(auc_csv)
            if actual_auc_sha != c_res.get('aucs_file_sha256'):
                return False, "%s classification AUCs CSV SHA mismatch: %s != %s" % (name, actual_auc_sha, c_res.get('aucs_file_sha256'))

            try:
                from sklearn.metrics import roc_auc_score
                pred_df_re = pd.read_csv(pred_csv)

                # A3/A4 (M1.4c.2): fail-closed replay on the EXACT production
                # writer schema: ground-truth column == <Pathology>, probability
                # column == 'prob_<Pathology>'. Exactly 14/14 pairs required.
                missing_gt = [p for p in NIH_PATHOLOGIES if p not in pred_df_re.columns]
                missing_prob = ['prob_' + p for p in NIH_PATHOLOGIES if ('prob_' + p) not in pred_df_re.columns]
                if missing_gt:
                    return False, "%s classification replay: missing ground-truth column(s): %s" % (name, missing_gt)
                if missing_prob:
                    return False, "%s classification replay: missing probability column(s): %s" % (name, missing_prob)

                replayed_aucs = []
                for p in NIH_PATHOLOGIES:
                    t_col = p
                    p_col = 'prob_' + p
                    y_true_re = pred_df_re[t_col].values.astype(int)
                    y_score_re = pred_df_re[p_col].values.astype(float)
                    if len(np.unique(y_true_re)) < 2:
                        return False, "%s classification replay: pathology %r has only one class (cannot replay ROC-AUC)" % (name, p)
                    auc_val = float(roc_auc_score(y_true_re, y_score_re))
                    if not np.isfinite(auc_val):
                        return False, "%s classification replay: non-finite AUC for %r" % (name, p)
                    replayed_aucs.append(auc_val)

                # A2 (M1.4c.2): replay must find exactly 14/14 pathologies.
                if len(replayed_aucs) != 14:
                    return False, "%s classification replay found %d pathologies; exactly 14 required" % (name, len(replayed_aucs))

                # A5 (M1.4c.2): AUC CSV itself must be a complete, exact 14-row
                # contract: one row per canonical pathology, no duplicates, no
                # unknown pathology, no NaN/Inf.
                auc_df_re = pd.read_csv(auc_csv)
                if 'label' not in auc_df_re.columns or 'auc' not in auc_df_re.columns:
                    return False, "%s classification AUCs CSV schema invalid: requires 'label' and 'auc' columns, got %s" % (name, list(auc_df_re.columns))
                auc_labels = list(auc_df_re['label'].astype(str))
                if len(auc_labels) != 14:
                    return False, "%s classification AUCs CSV has %d rows; exactly 14 rows required" % (name, len(auc_labels))
                if len(set(auc_labels)) != 14:
                    return False, "%s classification AUCs CSV contains duplicate pathology rows" % name
                if set(auc_labels) != set(NIH_PATHOLOGIES):
                    return False, "%s classification AUCs CSV has unknown/missing pathologies: %s" % (name, sorted(set(auc_labels) ^ set(NIH_PATHOLOGIES)))
                auc_vals_csv = pd.to_numeric(auc_df_re['auc'], errors='coerce').tolist()
                if any(pd.isna(auc_vals_csv)):
                    return False, "%s classification AUCs CSV contains NaN/Inf AUC values" % name
                auc_csv_map = dict(zip(auc_labels, [float(v) for v in auc_vals_csv]))

                # A3 (M1.4c.2): compare EVERY replayed per-pathology AUC against
                # both the in-memory auc_df and the serialized AUC CSV.
                rep_auc_map = dict(zip(NIH_PATHOLOGIES, replayed_aucs))
                rep_auc_df = c_res.get('auc_df')
                if rep_auc_df is None:
                    return False, "%s classification result missing in-memory auc_df for per-pathology replay" % name
                if 'label' not in rep_auc_df.columns or 'auc' not in rep_auc_df.columns:
                    return False, "%s classification in-memory auc_df schema invalid" % name
                rep_df_labels = list(rep_auc_df['label'].astype(str))
                if set(rep_df_labels) != set(NIH_PATHOLOGIES) or len(set(rep_df_labels)) != 14:
                    return False, "%s classification in-memory auc_df does not contain exactly 14 pathologies" % name
                rep_df_map = dict(zip(rep_df_labels, [float(v) for v in rep_auc_df['auc']]))

                for p in NIH_PATHOLOGIES:
                    repl = rep_auc_map[p]
                    if abs(repl - rep_df_map[p]) > 1e-7:
                        return False, "%s classification per-pathology replay mismatch vs auc_df for %r: %e != %e" % (name, p, repl, rep_df_map[p])
                    if abs(repl - auc_csv_map[p]) > 1e-7:
                        return False, "%s classification per-pathology replay mismatch vs AUCs CSV for %r: %e != %e" % (name, p, repl, auc_csv_map[p])

                # Macro replay: mean over the 14 replayed per-pathology AUCs.
                replayed_macro_auc = float(np.mean(replayed_aucs))
                if abs(replayed_macro_auc - c_res['macro_auc']) > 1e-7:
                    return False, "%s classification replayed macro AUC mismatch: %e != %e" % (name, replayed_macro_auc, c_res['macro_auc'])
            except Exception as _clf_e:
                return False, "%s classification replay failed: %s" % (name, _clf_e)

    return True, "VALID"


# F4 (M1.4c): Stale scientific artifacts that must not pre-exist
SCIENTIFIC_STALE_ARTIFACTS = [
    # Run summaries and manifests
    'M2_S1_summary.json', 'M2_S1_C4_RESULT.md', 'checkpoint_manifest.json',
    'attacker_manifest.json',
    # Checkpoints
    'generator_best_method_neutral.pth', 'checkpoint_latest.pth', 'best_attacker.pth',
    # Raw evaluation replays and serialized metrics
    'privacy_val_predictions.npz', 'classification_val_predictions.csv',
    'classification_val_aucs.csv',
    # Load-bearing training telemetry
    'epoch_metrics.csv', 'train_log.jsonl',
]


def check_scientific_output_freshness(base_dir):
    """F4 (M1.4c): Scientific output directory must be fresh — no stale artifacts."""
    if not os.path.exists(base_dir):
        return True
    for root, dirs, files in os.walk(base_dir):
        for fname in files:
            if fname in SCIENTIFIC_STALE_ARTIFACTS:
                raise RuntimeError(
                    "Scientific output directory is not fresh: found stale artifact '%s' in %s. "
                    "Archive or move the previous run directory, then launch into a fresh scientific output directory. Do not overwrite." % (fname, root)
                )
    return True


def check_git_source_guard(required_ancestor=None):
    """F9: require the exact certified tag/branch/HEAD, clean tree, and no runtime drift.

    ``required_ancestor`` is retained only for source compatibility with older callers;
    ancestry alone is never sufficient for scientific execution.
    """
    def _run(args):
        try:
            result = subprocess.run(['git'] + list(args), cwd=ROOT, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError('Scientific git source guard: git not found') from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or '').strip()
            raise RuntimeError('Scientific git source guard: git command failed: git %s%s' % (
                ' '.join(args), (': ' + detail) if detail else ''))
        return result.stdout or ''

    # Both tracked/index modifications and importable untracked runtime files are fatal.
    try:
        subprocess_result = subprocess.run(['git', 'diff', '--quiet'], cwd=ROOT, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError('Scientific git source guard: git not found') from exc
    if subprocess_result.returncode == 1:
        raise RuntimeError('Scientific git source guard: tracked tree has uncommitted changes')
    if subprocess_result.returncode != 0:
        raise RuntimeError('Scientific git source guard: git diff command failed')
    try:
        subprocess_result = subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=ROOT, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError('Scientific git source guard: git not found') from exc
    if subprocess_result.returncode == 1:
        raise RuntimeError('Scientific git source guard: index has staged changes')
    if subprocess_result.returncode != 0:
        raise RuntimeError('Scientific git source guard: git cached-diff command failed')
    untracked = _run(['ls-files', '--others', '--exclude-standard', '--', '*.py', '*.pyi']).splitlines()
    untracked += _run(['ls-files', '--others', '--exclude-standard', '--', 'config_files', 'research_agent/m2_dev']).splitlines()
    runtime_untracked = []
    for path in untracked:
        path = path.strip()
        if not path or path in runtime_untracked:
            continue
        # Certification/evidence documents are allowed; runtime Python and config are not.
        if (path.endswith(('.py', '.pyi'))
                or path.startswith('config_files/')
                or (path.startswith('research_agent/m2_dev/') and path.endswith(('.json', '.yaml', '.yml')))
                or path == 'research_agent/M2_S1_EXECUTION_LOCK.json'):
            runtime_untracked.append(path)
    if runtime_untracked:
        raise RuntimeError('Scientific git source guard: importable untracked runtime source/config: %s' % runtime_untracked)

    branch = _run(['symbolic-ref', '--quiet', '--short', 'HEAD']).strip()
    if branch != FROZEN_SOURCE_BRANCH:
        raise RuntimeError('Scientific git source guard: branch %r != certified %r' % (branch, FROZEN_SOURCE_BRANCH))
    head = _run(['rev-parse', 'HEAD']).strip()
    tag_head = _run(['rev-parse', '%s^{commit}' % FROZEN_SOURCE_TAG]).strip()
    if head != tag_head:
        raise RuntimeError('Scientific git source guard: HEAD %s != %s target %s' % (head, FROZEN_SOURCE_TAG, tag_head))
    return True


def compute_classification_val_fingerprints():
    """§5 (M1.4c): Return deterministic SHA256 fingerprints of classification VAL cohort from verified contract."""
    contract = verify_classification_val_contract()
    return {
        'classification_val_n_images': contract['classification_val_n_images'],
        'classification_val_n_patients': contract['classification_val_n_patients'],
        'classification_val_image_index_sha256': contract['classification_val_image_index_sha256'],
        'classification_val_patient_sequence_sha256': contract['classification_val_patient_sequence_sha256'],
        'classification_val_label_matrix_sha256': contract['classification_val_label_matrix_sha256'],
    }


def run_orchestration(args, out_base_dir=None, unit_test_mode=False):
    """Core orchestration pipeline reusable across full runs and integration smoke tests."""
    # Validate at the direct API boundary as well as parse_args().
    validate_scientific_args(args, unit_test_mode=unit_test_mode)

    # Authenticate the external execution lock before any scientific preflight or output.
    if not unit_test_mode:
        verify_execution_lock_integrity()
        preflight_device = verify_environment_and_hashes()
        device = torch.device(args.device) if args.device else preflight_device
    else:
        device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    base = out_base_dir or os.path.join(ROOT, 'research_runs', 'M2_S1')

    # F4 (M1.4c): Scientific output directory must be fresh
    if getattr(args, 'scientific_m2_s1', False) and not unit_test_mode:
        check_scientific_output_freshness(base)

    # F9 (M1.4c): Git source guard in scientific mode
    if getattr(args, 'scientific_m2_s1', False) and not unit_test_mode:
        check_git_source_guard()

    b_dev_config = os.path.join(ROOT, 'config_files', 'config_dev_restored_baseline.json')
    c4_config = os.path.join(ROOT, 'config_files', 'config_dev_c4.json')

    b_dev_manifest = None
    c4_manifest = None

    # Step 1: Run B_dev Anonymizer Training
    if args.arm in ('B_dev', 'all'):
        b_dev_manifest = run_anonymizer_arm('B_dev', b_dev_config, args.max_epochs, args.seed, device, out_base_dir=base, unit_test_mode=unit_test_mode)

    # Step 2: Run C4 Anonymizer Training
    if args.arm in ('C4', 'all'):
        c4_manifest = run_anonymizer_arm('C4', c4_config, args.max_epochs, args.seed, device, out_base_dir=base, unit_test_mode=unit_test_mode)

    # Step 3: Run S1 Evaluators
    if args.arm in ('all', 'eval_only'):
        print("\n" + "=" * 70)
        print("M2-S1: SCIENTIFIC EVALUATION SUITE (ATTACKER SEED %d)" % args.attacker_seed)
        print("=" * 70)

        if b_dev_manifest is None:
            b_manifest_p = os.path.join(base, 'B_dev', 'seed_%d' % args.seed, 'checkpoint_manifest.json')
            b_dev_manifest = json.load(open(b_manifest_p))
        if c4_manifest is None:
            c4_manifest_p = os.path.join(base, 'C4', 'seed_%d' % args.seed, 'checkpoint_manifest.json')
            c4_manifest = json.load(open(c4_manifest_p))

        # 3a. Train Adaptive Attackers
        b_att_manifest = train_s1_attacker_arm('B_dev', args.seed, args.attacker_seed,
                                               args.attacker_epochs, args.attacker_patience, device, out_base_dir=base, unit_test_mode=unit_test_mode)
        c4_att_manifest = train_s1_attacker_arm('C4', args.seed, args.attacker_seed,
                                                args.attacker_epochs, args.attacker_patience, device, out_base_dir=base, unit_test_mode=unit_test_mode)

        # 3b. Scientific Privacy VAL Evaluation
        b_priv = evaluate_privacy_arm('B_dev', args.seed, args.attacker_seed, device, out_base_dir=base, unit_test_mode=unit_test_mode)
        c4_priv = evaluate_privacy_arm('C4', args.seed, args.attacker_seed, device, out_base_dir=base, unit_test_mode=unit_test_mode)

        auc_b_priv = b_priv['roc_auc']
        auc_c4_priv = c4_priv['roc_auc']
        delta_priv = auc_c4_priv - auc_b_priv

        # 3c. Clinical Utility Classification VAL Evaluation
        b_class = evaluate_classification_arm('B_dev', args.seed, device, out_base_dir=base, unit_test_mode=unit_test_mode)
        c4_class = evaluate_classification_arm('C4', args.seed, device, out_base_dir=base, unit_test_mode=unit_test_mode)

        auc_b_class = b_class['macro_auc']
        auc_c4_class = c4_class['macro_auc']
        delta_class = auc_c4_class - auc_b_class

        # 3d. Check Run Validity
        run_valid, val_reason = check_run_validity(
            b_dev_manifest, c4_manifest, b_att_manifest, c4_att_manifest,
            b_priv, c4_priv, b_class, c4_class, expected_epochs=args.max_epochs, unit_test_mode=unit_test_mode
        )

        if run_valid:
            privacy_gate_pass = (delta_priv <= 0.03)
            class_gate_pass = (delta_class >= 0.0)
            # B8 / §13: Only genuine scientific execution can issue PROMOTE / DO NOT PROMOTE verdicts
            if not unit_test_mode and getattr(args, 'scientific_m2_s1', False):
                s1_verdict = "C4 S1: PROMOTE TO S2" if (privacy_gate_pass and class_gate_pass) else "C4 S1: DO NOT PROMOTE"
            else:
                s1_verdict = "DEVELOPMENT_ONLY — not a scientific verdict"
            privacy_status = 'PASS' if privacy_gate_pass else 'FAIL'
            class_status = 'PASS' if class_gate_pass else 'FAIL'
        else:
            privacy_gate_pass = False
            class_gate_pass = False
            s1_verdict = "C4 S1: INVALID — NO SCIENTIFIC VERDICT"
            privacy_status = 'NOT_EVALUATED_DUE_TO_INVALID_RUN'
            class_status = 'NOT_EVALUATED_DUE_TO_INVALID_RUN'

        # Read peak VRAM from telemetry
        b_df_p = os.path.join(base, 'B_dev', 'seed_%d' % args.seed, 'epoch_metrics.csv')
        c4_df_p = os.path.join(base, 'C4', 'seed_%d' % args.seed, 'epoch_metrics.csv')
        b_peak_vram = pd.read_csv(b_df_p)['peak_vram_mb'].max() if os.path.exists(b_df_p) else 0.0
        c4_peak_vram = pd.read_csv(c4_df_p)['peak_vram_mb'].max() if os.path.exists(c4_df_p) else 0.0

        b_dis_aucs = dict(zip(b_class['auc_df']['label'], b_class['auc_df']['auc']))
        c4_dis_aucs = dict(zip(c4_class['auc_df']['label'], c4_class['auc_df']['auc']))

        summary = {
            'protocol': 'M2-S1',
            'run_status': ('DEVELOPMENT_VALID' if unit_test_mode else 'VALID') if run_valid else ('DEVELOPMENT_INVALID' if unit_test_mode else 'INVALID'),
            'validity_reason': val_reason,
            'provenance': {
                'generator_optimizer': 'Adam(lr=1e-4)',
                'verifier_critic_optimizer': 'Adam(lr=1e-4)',
                'classifier_critic_optimizer': 'SGD(lr=1e-4, momentum=0.9, weight_decay=1e-4)',
                'classification_split_csv_path': 'chexnet/nih_labels.csv',
                'classification_split_csv_sha256': file_sha256(os.path.join(ROOT, 'chexnet', 'nih_labels.csv')) if os.path.exists(os.path.join(ROOT, 'chexnet', 'nih_labels.csv')) else None,
                'classification_val_structural_fingerprints': compute_classification_val_fingerprints(),
            },
            'b_dev': {
                'seed': args.seed,
                'best_epoch': b_dev_manifest['best_epoch'],
                'best_selection_total': b_dev_manifest['best_selection_total'],
                'selected_checkpoint_sha256': b_dev_manifest['selected_generator_sha256'],
                'training_runtime_hours': b_dev_manifest.get('training_runtime_hours'),
                'peak_vram_mb': float(b_peak_vram),
                'attacker_seed42_checkpoint_sha256': b_att_manifest['best_attacker_sha256'],
                'privacy_val_roc_auc': float(auc_b_priv),
                'privacy_val_metrics': b_priv,
                'classification_val_macro_auc': float(auc_b_class),
                'classification_val_disease_aucs': {k: float(v) for k, v in b_dis_aucs.items()},
                'classification_val_predictions_file': b_class.get('predictions_file'),
                'classification_val_predictions_sha256': b_class.get('predictions_file_sha256'),
                'classification_val_aucs_file': b_class.get('aucs_file'),
                'classification_val_aucs_sha256': b_class.get('aucs_file_sha256'),
            },
            'c4': {
                'seed': args.seed,
                'best_epoch': c4_manifest['best_epoch'],
                'best_selection_total': c4_manifest['best_selection_total'],
                'selected_checkpoint_sha256': c4_manifest['selected_generator_sha256'],
                'training_runtime_hours': c4_manifest.get('training_runtime_hours'),
                'peak_vram_mb': float(c4_peak_vram),
                'gradient_norm_diagnostics': c4_manifest.get('gradient_norm_diagnostics', {}),
                'attacker_seed42_checkpoint_sha256': c4_att_manifest['best_attacker_sha256'],
                'privacy_val_roc_auc': float(auc_c4_priv),
                'privacy_val_metrics': c4_priv,
                'classification_val_macro_auc': float(auc_c4_class),
                'classification_val_disease_aucs': {k: float(v) for k, v in c4_dis_aucs.items()},
                'classification_val_predictions_file': c4_class.get('predictions_file'),
                'classification_val_predictions_sha256': c4_class.get('predictions_file_sha256'),
                'classification_val_aucs_file': c4_class.get('aucs_file'),
                'classification_val_aucs_sha256': c4_class.get('aucs_file_sha256'),
            },
            'deltas': {
                'delta_privacy_val_auc': float(delta_priv),
                'delta_classification_val_macro_auc': float(delta_class),
            },
            'gates': {
                'privacy_gate_pass': bool(privacy_gate_pass),
                'privacy_gate_status': privacy_status,
                'privacy_gate_threshold': '<= +0.03',
                'classification_gate_pass': bool(class_gate_pass),
                'classification_gate_status': class_status,
                'classification_gate_threshold': '>= 0.0',
                'segmentation_status': 'NOT APPLICABLE — evaluator provenance not yet certified',
            },
            'verdict': s1_verdict,
            'test_touched': False,
        }

        # Save summary JSON
        summary_path = os.path.join(base, 'M2_S1_summary.json')
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Build Markdown Report
        report_path = os.path.join(ROOT, 'research_agent', 'M2_S1_C4_RESULT.md') if out_base_dir is None else os.path.join(base, 'M2_S1_C4_RESULT.md')
        write_markdown_report(report_path, summary)

        print("\n" + "=" * 70)
        print("M2-S1 EXECUTION COMPLETE")
        print("Summary JSON: %s" % summary_path)
        print("Markdown Report: %s" % report_path)
        print("Verdict: %s" % s1_verdict)
        print("=" * 70)
        return summary


def write_markdown_report(report_path, summary):
    """Generate comprehensive scientific M2-S1 Markdown report."""
    b = summary['b_dev']
    c4 = summary['c4']
    d = summary['deltas']
    g = summary['gates']

    lines = [
        "# M2-S1 Scientific Report — Paired Baseline Control vs C4 (Feature Preservation)",
        "",
        "## Executive Summary",
        "This report documents the definitive findings of the paired **M2-S1** experimental run comparing the restored baseline control (`B_dev`, $\\mu=0.01$, legacy operator) against the penultimate feature-preservation method (`C4`, feature loss weight=1.0) under identical training semantics and exact data-order pairing.",
        "",
        "---",
        "",
        "## 1. Key Results Summary",
        "",
        "| Metric | B_dev (Control) | C4 (Method) | Delta (C4 - B_dev) | Gate / Target | Gate Status |",
        "|---|---:|---:|---:|---|:---:|",
        "| **Selected Generator Epoch** | `%s` | `%s` | — | Method-neutral min total loss | diagnostic |" % (b['best_epoch'], c4['best_epoch']),
        "| **Val Selection Total Loss** | `%.5f` | `%.5f` | `%+.5f` | $L_{AC} + L_{priv}$ | diagnostic |" % (b['best_selection_total'], c4['best_selection_total'], c4['best_selection_total'] - b['best_selection_total']),
        "| **Adaptive Re-ID VAL AUC** | `%.4f` | `%.4f` | `%+.4f` | $\\Delta_{priv} \\le +0.03$ | **%s** |" % (b['privacy_val_roc_auc'], c4['privacy_val_roc_auc'], d['delta_privacy_val_auc'], g.get('privacy_gate_status', 'PASS' if g['privacy_gate_pass'] else 'FAIL')),
        "| **Classification Macro VAL AUC** | `%.4f` | `%.4f` | `%+.4f` | $\\Delta_{class} \\ge 0.0$ | **%s** |" % (b['classification_val_macro_auc'], c4['classification_val_macro_auc'], d['delta_classification_val_macro_auc'], g.get('classification_gate_status', 'PASS' if g['classification_gate_pass'] else 'FAIL')),
        "| **Segmentation Dice** | *BLOCKED* | *BLOCKED* | — | Certified evaluator | **NOT APPLICABLE** |",
        "| **Peak VRAM (MB)** | `%.1f` | `%.1f` | `%+.1f` | < 16,000 MB | diagnostic |" % (b['peak_vram_mb'], c4['peak_vram_mb'], c4['peak_vram_mb'] - b['peak_vram_mb']),
        "| **Training Runtime (Hours)** | `%.2f` | `%.2f` | `%+.2f` | 250 epochs | diagnostic |" % (b.get('training_runtime_hours', 0), c4.get('training_runtime_hours', 0), c4.get('training_runtime_hours', 0) - b.get('training_runtime_hours', 0)),
        "",
        "---",
        "",
        "## 2. 14 Pathology Classification AUCs (Validation Fold)",
        "",
        "| Pathology | B_dev AUC | C4 AUC | Delta (C4 - B_dev) |",
        "|---|---:|---:|---:|",
    ]

    all_diseases = sorted(b['classification_val_disease_aucs'].keys())
    for dis in all_diseases:
        b_auc = b['classification_val_disease_aucs'].get(dis, 0.0)
        c4_auc = c4['classification_val_disease_aucs'].get(dis, 0.0)
        lines.append("| %s | `%.4f` | `%.4f` | `%+.4f` |" % (dis, b_auc, c4_auc, c4_auc - b_auc))

    lines.extend([
        "",
        "---",
        "",
        "## 3. C4 Feature-Loss Gradient Norm Diagnostics",
        "",
        "| Epoch | Base Objective Grad Norm | Feature Loss Grad Norm | Ratio (Feature / Base) |",
        "|---|---:|---:|---:|",
    ])
    diag = c4.get('gradient_norm_diagnostics', {})
    for ep_str in sorted(diag.keys(), key=lambda x: int(x) if str(x).isdigit() else 0):
        entry = diag[ep_str]
        if 'base_grad_norm' in entry:
            lines.append("| %s | `%.5e` | `%.5e` | `%.4f` |" % (
                ep_str, entry['base_grad_norm'], entry['feature_grad_norm'], entry['feature_base_ratio']
            ))

    lines.extend([
        "",
        "---",
        "",
        "## 4. Scientific Provenance & Artifact Hashes",
        "- **Branch**: `research/method-restart`",
        "- **B_dev Selected Generator SHA256**: `%s`" % b['selected_checkpoint_sha256'],
        "- **C4 Selected Generator SHA256**: `%s`" % c4['selected_checkpoint_sha256'],
        "- **B_dev Attacker Checkpoint SHA256**: `%s`" % b['attacker_seed42_checkpoint_sha256'],
        "- **C4 Attacker Checkpoint SHA256**: `%s`" % c4['attacker_seed42_checkpoint_sha256'],
        "- **Test Firewall**: STRICTLY CLOSED (`test_touched: false`)",
        "",
        "---",
        "",
        "## 5. Frozen S1 Decision Gate Evaluation",
        "- **Run Validity**: `%s` (%s)" % (summary['run_status'], summary.get('validity_reason', '')),
        "- **Privacy Gate ($\\Delta_{priv} \\le +0.03$)**: `%s` ($\\Delta_{priv} = %+.4f$)" % (
            g.get('privacy_gate_status', 'PASS' if g['privacy_gate_pass'] else 'FAIL'), d['delta_privacy_val_auc']
        ),
        "- **Classification Gate ($\\Delta_{class} \\ge 0.0$)**: `%s` ($\\Delta_{class} = %+.4f$)" % (
            g.get('classification_gate_status', 'PASS' if g['classification_gate_pass'] else 'FAIL'), d['delta_classification_val_macro_auc']
        ),
        "- **Segmentation**: `%s`" % g['segmentation_status'],
        "",
        "### Final S1 Verdict: **%s**" % summary['verdict'],
    ])

    with open(report_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    args = parse_args()
    run_orchestration(args)


if __name__ == '__main__':
    main()
