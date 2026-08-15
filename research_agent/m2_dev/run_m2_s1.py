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
    verify_repaired_acloss,
    verify_frozen_scientific_configs,
    verify_scientific_dependencies,
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
    args = parser.parse_args()

    if args.scientific_m2_s1:
        # Strictly enforce frozen scientific hyperparameters and full paired execution
        if args.arm != 'all':
            raise ValueError("Scientific M2-S1 mode requires --arm all, got %r" % args.arm)
        if args.max_epochs != 250:
            raise ValueError("Scientific M2-S1 mode requires max_epochs == 250, got %d" % args.max_epochs)
        if args.attacker_epochs != 100:
            raise ValueError("Scientific M2-S1 mode requires attacker_epochs == 100, got %d" % args.attacker_epochs)
        if args.attacker_patience != 5:
            raise ValueError("Scientific M2-S1 mode requires attacker_patience == 5, got %d" % args.attacker_patience)
        if args.seed != 42:
            raise ValueError("Scientific M2-S1 mode requires seed == 42, got %d" % args.seed)
        if args.attacker_seed != 42:
            raise ValueError("Scientific M2-S1 mode requires attacker_seed == 42, got %d" % args.attacker_seed)

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
        image_size=64 if unit_test_mode else None
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

    # Save raw predictions NPZ
    pred_npz_path = os.path.join(att_dir, 'privacy_val_predictions.npz')
    np.savez_compressed(pred_npz_path, y_true=eval_res['y_true'], y_score=eval_res['y_score'])
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
        expected_generator_sha=expected_gen_sha
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


def check_run_validity(b_dev_manifest, c4_manifest, b_att_manifest, c4_att_manifest,
                       b_priv, c4_priv, b_class, c4_class, expected_epochs=250, unit_test_mode=False):
    """Verify that all completion artifacts, hashes, non-NaN values, and split invariants hold."""
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

    return True, "VALID"


# F4 (M1.4c): Stale scientific artifacts that must not pre-exist
SCIENTIFIC_STALE_ARTIFACTS = [
    'M2_S1_summary.json', 'M2_S1_C4_RESULT.md', 'checkpoint_manifest.json',
    'generator_best_method_neutral.pth', 'checkpoint_latest.pth',
    'best_attacker.pth', 'attacker_manifest.json',
    'privacy_val_predictions.npz', 'classification_val_predictions.csv',
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
                    "Remove old results before a new scientific run." % (fname, root)
                )
    return True


def check_git_source_guard(required_ancestor='851c3f1a6912255c97345a7f53ed138e7ae7981d'):
    """F9 (M1.4c): Verify tracked tree is clean and HEAD descends from certified execution code."""
    try:
        # Tracked tree must be clean
        result = subprocess.run(['git', 'diff', '--quiet'], cwd=ROOT, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError("Scientific git source guard: tracked tree has uncommitted changes")
        result = subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=ROOT, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError("Scientific git source guard: index has staged changes")
        # HEAD must descend from certified execution code commit
        result = subprocess.run(
            ['git', 'merge-base', '--is-ancestor', required_ancestor, 'HEAD'],
            cwd=ROOT, capture_output=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Scientific git source guard: HEAD does not descend from certified execution code %s" % required_ancestor
            )
    except FileNotFoundError:
        raise RuntimeError("Scientific git source guard: git not found")
    return True


def compute_classification_val_fingerprints():
    """§5 (M1.4c): Compute deterministic SHA256 fingerprints of classification VAL cohort."""
    csv_path = os.path.join(ROOT, 'chexnet', 'nih_labels.csv')
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    val_df = df[df['fold'] == 'val'].copy()
    val_df = val_df.sort_values('Image Index').reset_index(drop=True)

    # Image Index fingerprint
    h_img = hashlib.sha256()
    for idx in val_df['Image Index']:
        h_img.update((str(idx) + '\n').encode('utf-8'))

    # Patient ID fingerprint
    h_pat = hashlib.sha256()
    patient_ids = []
    for idx in val_df['Image Index']:
        pid = str(idx).split('_')[0]
        patient_ids.append(pid)
        h_pat.update((pid + '\n').encode('utf-8'))

    # Label matrix fingerprint
    PRED_LABEL = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
        'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
        'Pleural_Thickening', 'Hernia'
    ]
    h_lbl = hashlib.sha256()
    for _, row in val_df.iterrows():
        label_vec = [str(int(row.get(p, 0))) for p in PRED_LABEL]
        h_lbl.update((','.join(label_vec) + '\n').encode('utf-8'))

    return {
        'classification_val_n_images': len(val_df),
        'classification_val_n_patients': len(set(patient_ids)),
        'classification_val_image_index_sha256': h_img.hexdigest(),
        'classification_val_patient_sequence_sha256': h_pat.hexdigest(),
        'classification_val_label_matrix_sha256': h_lbl.hexdigest(),
    }


def run_orchestration(args, out_base_dir=None, unit_test_mode=False):
    """Core orchestration pipeline reusable across full runs and integration smoke tests."""
    # F1 (M1.4c): Canonical M2-S1 requires --scientific-m2-s1
    is_canonical_output = (out_base_dir is None or
                           os.path.abspath(out_base_dir) == os.path.abspath(os.path.join(ROOT, 'research_runs', 'M2_S1')))
    if not unit_test_mode and is_canonical_output and not getattr(args, 'scientific_m2_s1', False):
        raise RuntimeError(
            "Canonical M2-S1 execution requires --scientific-m2-s1. "
            "Without this flag, the canonical output namespace research_runs/M2_S1/ is protected."
        )

    # Preflight dependency & hash verification is UNCONDITIONAL in scientific mode / standard runs
    if not unit_test_mode:
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
            # F1 (M1.4c): Only scientific mode may issue PROMOTE / DO NOT PROMOTE verdicts
            if unit_test_mode:
                s1_verdict = "DEVELOPMENT_ONLY — not a scientific verdict"
            else:
                s1_verdict = "C4 S1: PROMOTE TO S2" if (privacy_gate_pass and class_gate_pass) else "C4 S1: DO NOT PROMOTE"
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
            'run_status': 'VALID' if run_valid else 'INVALID',
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
