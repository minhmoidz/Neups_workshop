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
    METHOD_NEUTRAL_CKPT_NAME,
    verify_repaired_acloss,
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
        # Strictly enforce frozen scientific hyperparameters
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

    return args


def assert_m2_scientific_mode_ready(image_path='/home/minhtt/datasets/nih/images/'):
    """Fail closed unless all dependencies, checkpoints, SHAs, pair files, and images pass."""
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

    # Comprehensive dependency preflight
    dep = assert_m2_scientific_mode_ready('/home/minhtt/datasets/nih/images/')
    print("[PASS] Initial Generator SHA: %s..." % dep['initial_generator_sha256'][:16])
    print("[PASS] Frozen Classifier SHA: %s..." % dep['classifier_sha256'][:16])
    print("[PASS] Frozen Verifier SHA:   %s..." % dep['verifier_sha256'][:16])
    print("[PASS] Repaired ACLoss SHA:   %s..." % dep['acloss_sha256'][:16])
    print("[PASS] TRAIN Pairs SHA:       %s... (%d pairs verified)" % (dep['train_pairs_sha256'][:16], dep['train_pairs_count']))
    print("[PASS] VAL Pairs SHA:         %s... (%d pairs verified)" % (dep['val_pairs_sha256'][:16], dep['val_pairs_count']))
    print("[PASS] Dataset Image Availability: 100%% (0 missing)")
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
    with open(att_cfg_path) as f:
        attacker_cfg = json.load(f)

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
        image_size=64 if unit_test_mode else None
    )

    # Save raw predictions NPZ
    pred_npz_path = os.path.join(att_dir, 'privacy_val_predictions.npz')
    np.savez_compressed(pred_npz_path, y_true=eval_res['y_true'], y_score=eval_res['y_score'])
    pred_sha = file_sha256(pred_npz_path)

    # Return scalar summary dict with path references (no float-casting ndarrays)
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
        image_size=64 if unit_test_mode else None
    )
    if clf_res['n_classes_valid'] != 14:
        raise RuntimeError("Classification evaluation returned %d valid classes, expected 14" % clf_res['n_classes_valid'])
    return clf_res


def check_run_validity(b_dev_manifest, c4_manifest, b_att_manifest, c4_att_manifest,
                       b_priv, c4_priv, b_class, c4_class, expected_epochs=250, unit_test_mode=False):
    """Verify that all completion artifacts, hashes, non-NaN values, and split invariants hold."""
    if not (b_dev_manifest and c4_manifest and b_att_manifest and c4_att_manifest):
        return False, "Missing manifest"
    if not unit_test_mode and (b_dev_manifest.get('epochs_completed') != expected_epochs or c4_manifest.get('epochs_completed') != expected_epochs):
        return False, "Anonymizer epochs incomplete"
    if not (np.isfinite(b_priv['roc_auc']) and np.isfinite(c4_priv['roc_auc'])):
        return False, "Non-finite privacy ROC-AUC"
    if not (np.isfinite(b_class['macro_auc']) and np.isfinite(c4_class['macro_auc'])):
        return False, "Non-finite classification Macro AUC"
    if b_class['n_classes_valid'] != 14 or c4_class['n_classes_valid'] != 14:
        return False, "Invalid class count in classification"
    if not unit_test_mode and (b_priv['n_pairs'] != 2000 or c4_priv['n_pairs'] != 2000):
        return False, "Privacy pairs count != 2000"
    if not unit_test_mode and (b_class.get('n_images') != 10816 or c4_class.get('n_images') != 10816):
        return False, "Classification images count != 10816"
    return True, "VALID"


def run_orchestration(args, out_base_dir=None, unit_test_mode=False):
    """Core orchestration pipeline reusable across full runs and integration smoke tests."""
    device = torch.device(args.device) if args.device else verify_environment_and_hashes()
    base = out_base_dir or os.path.join(ROOT, 'research_runs', 'M2_S1')

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

        privacy_gate_pass = (delta_priv <= 0.03)
        class_gate_pass = (delta_class >= 0.0)

        s1_verdict = "C4 S1: PROMOTE TO S2" if (run_valid and privacy_gate_pass and class_gate_pass) else "C4 S1: DO NOT PROMOTE"

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
            },
            'deltas': {
                'delta_privacy_val_auc': float(delta_priv),
                'delta_classification_val_macro_auc': float(delta_class),
            },
            'gates': {
                'privacy_gate_pass': bool(privacy_gate_pass),
                'privacy_gate_threshold': '<= +0.03',
                'classification_gate_pass': bool(class_gate_pass),
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
        "| **Adaptive Re-ID VAL AUC** | `%.4f` | `%.4f` | `%+.4f` | $\\Delta_{priv} \\le +0.03$ | **%s** |" % (b['privacy_val_roc_auc'], c4['privacy_val_roc_auc'], d['delta_privacy_val_auc'], 'PASS' if g['privacy_gate_pass'] else 'FAIL'),
        "| **Classification Macro VAL AUC** | `%.4f` | `%.4f` | `%+.4f` | $\\Delta_{class} \\ge 0.0$ | **%s** |" % (b['classification_val_macro_auc'], c4['classification_val_macro_auc'], d['delta_classification_val_macro_auc'], 'PASS' if g['classification_gate_pass'] else 'FAIL'),
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
            'PASS' if g['privacy_gate_pass'] else 'FAIL', d['delta_privacy_val_auc']
        ),
        "- **Classification Gate ($\\Delta_{class} \\ge 0.0$)**: `%s` ($\\Delta_{class} = %+.4f$)" % (
            'PASS' if g['classification_gate_pass'] else 'FAIL', d['delta_classification_val_macro_auc']
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
