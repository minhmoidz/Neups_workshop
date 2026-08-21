"""Stage A pilot — attacker-seed variance pilot for S2 seed-count design.

NOT the certified M2-S1 pipeline. Lives entirely outside research_agent/m2_dev/
and does not import, modify, or touch any frozen config, lock, checkpoint, or
result file. Purpose: estimate real attacker-seed variance + B_dev/C4
correlation ρ inside the actual M2 architecture/loss/geometry, to replace the
external-reproduction proxy used in S2_CONFIRMATORY_DESIGN_PROPOSAL.md §2.

Fidelity policy (per explicit instruction — must stay faithful to the
certified project, zero silent parameter drift):
  - All hyperparameters are loaded FROM the frozen
    config_files/config_dev_attacker_s1.json file at run time (read-only),
    with its SHA256 asserted against the same FROZEN_ATTACKER_CONFIG_SHA
    constant the certified pipeline uses. Only `attacker_seed` is overridden
    in the in-memory copy.
  - PilotAttacker.train_epoch / validate_selection / run below are a
    line-for-line copy of research_agent/m2_dev/dev_attacker.py's
    DevAttacker (same geometry, same optimizer, same criterion, same
    early-stopping/NaN-Inf fail-fast discipline). The ONLY removed piece is
    the `attacker_seed must be exactly 42` scientific-mode assertion —
    everything else that assertion also checks (config path, config SHA,
    batch_size/lr, max_epochs/patience, geometry fields) is still asserted
    here, unchanged.
  - The final privacy AUC is computed by calling the real, unmodified
    research_agent.m2_dev.eval_reid_val.evaluate_reid_val() — not a
    reimplementation — against the exact generator checkpoint SHA that S1
    selected (hard SHA-verified before use).
  - Output never lands under research_runs/M2_S1/ or touches any file whose
    hash is embedded in M2_S1_summary.json. Every output file is explicitly
    labeled "pilot_uncertified": true.

STRICTLY TRAIN / VALIDATION ONLY — never constructs a TEST loader.
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
for _p in (ROOT, os.path.join(ROOT, 'research_agent')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from networks.SiameseNetwork import SiameseNetwork  # noqa: E402
from utils import utils  # noqa: E402

from m2_dev.evaluator_common import (  # noqa: E402
    firewall_check,
    file_sha256,
    build_dev_loaders,
    snn_preprocess,
    FROZEN_ATTACKER_CONFIG_PATH,
    FROZEN_ATTACKER_CONFIG_SHA,
    SCIENTIFIC_IMAGE_ROOT,
)
from m2_dev.dev_attacker import load_frozen_anonymizer  # noqa: E402
from m2_dev.eval_reid_val import evaluate_reid_val  # noqa: E402

PILOT_OUT_ROOT = os.path.join(ROOT, 'reproduction', 's2_pilot', 'results')
M2_S1_BASE = os.path.join(ROOT, 'research_runs', 'M2_S1')


class PilotAttacker:
    """Line-for-line copy of DevAttacker's train/val/run logic (dev_attacker.py:67-290),
    minus the `attacker_seed == 42` scientific-mode lock. Everything else that the
    original scientific-mode assertion block checks is still asserted below."""

    def __init__(self, config, attacker_seed, device, generator_checkpoint, image_size=None):
        firewall_check('dev')

        # Same checks DevAttacker's scientific-mode branch performs, EXCEPT attacker_seed == 42.
        if config.get('batch_size') != 32 or config.get('learning_rate') != 1e-4:
            raise RuntimeError('Pilot attacker config does not match frozen optimizer contract')
        if config.get('max_epochs') != 100 or config.get('early_stopping') != 5:
            raise RuntimeError('Pilot attacker config does not match frozen training contract')
        if config.get('train_geometry') != 'anon_anon' or config.get('checkpoint_val_geometry') != 'anon_anon':
            raise RuntimeError('Pilot attacker geometry contract mismatch')
        if config.get('scientific_val_geometry') != 'anon_real':
            raise RuntimeError('Pilot attacker scientific VAL geometry mismatch')

        self.config = config
        self.attacker_seed = attacker_seed
        self.device = device
        if self.device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('Pilot attacker requires CUDA (same requirement as certified attacker)')
        self.generator_checkpoint = generator_checkpoint

        # Seed BEFORE net init, DataLoader shuffle, and optimizer — same order as DevAttacker.
        utils.seed_all(attacker_seed)

        _, self.anonymize_fn = load_frozen_anonymizer(
            config, self.device, checkpoint_path=generator_checkpoint,
            image_size=image_size if image_size is not None else config.get('image_size'))

        self.training_loader, self.validation_loader = build_dev_loaders(config, seed=attacker_seed)

        self.net = SiameseNetwork().to(self.device)
        self.best_net = None
        self.criterion = nn.BCEWithLogitsLoss().to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=config['learning_rate'])

        self.max_epochs = config['max_epochs']
        self.early_stopping = config['early_stopping']
        self.best_val_loss = float('inf')
        self.best_epoch = None
        self.patience = 0
        self.loss_dict = {'training': [], 'validation': []}

    def train_epoch(self):
        self.net.train()
        running = 0.0
        for inputs1, inputs2, labels in self.training_loader:
            inputs1, inputs2, labels = inputs1.to(self.device), inputs2.to(self.device), labels.to(self.device)
            inputs1 = self.anonymize_fn(inputs1)
            inputs2 = self.anonymize_fn(inputs2)
            inputs1, inputs2 = snn_preprocess(inputs1), snn_preprocess(inputs2)

            self.optimizer.zero_grad()
            outputs = self.net(inputs1, inputs2).squeeze()
            labels = labels.type_as(outputs)
            loss = self.criterion(outputs, labels)
            if not torch.isfinite(loss).all():
                raise FloatingPointError('Pilot attacker training loss is non-finite before backward')
            loss.backward()
            for name, param in self.net.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    raise FloatingPointError("Pilot attacker gradient '%s' contains NaN/Inf" % name)
            self.optimizer.step()
            running += loss.item()
        avg_loss = running / max(len(self.training_loader), 1)
        if not np.isfinite(avg_loss):
            raise FloatingPointError("Pilot attacker training loss is non-finite: %s" % avg_loss)
        for name, param in self.net.named_parameters():
            if not torch.isfinite(param).all():
                raise FloatingPointError("Pilot attacker parameter '%s' contains NaN/Inf after training step" % name)
        return avg_loss

    def validate_selection(self):
        self.net.eval()
        running = 0.0
        with torch.no_grad():
            for inputs1, inputs2, labels in self.validation_loader:
                inputs1, inputs2, labels = inputs1.to(self.device), inputs2.to(self.device), labels.to(self.device)
                inputs1 = self.anonymize_fn(inputs1)
                inputs2 = self.anonymize_fn(inputs2)
                inputs1, inputs2 = snn_preprocess(inputs1), snn_preprocess(inputs2)
                outputs = self.net(inputs1, inputs2).squeeze()
                labels = labels.type_as(outputs)
                loss = self.criterion(outputs, labels)
                running += loss.item()
        avg_loss = running / max(len(self.validation_loader), 1)
        if not np.isfinite(avg_loss):
            raise FloatingPointError("Pilot attacker validation loss is non-finite: %s" % avg_loss)
        return avg_loss

    def run(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = None
        checkpoint_sha256 = None
        termination_reason = 'max_epochs_completed'

        for epoch in range(self.max_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate_selection()
            self.loss_dict['training'].append(float(train_loss))
            self.loss_dict['validation'].append(float(val_loss))

            if val_loss < self.best_val_loss:
                self.best_val_loss = float(val_loss)
                self.best_epoch = epoch
                self.best_net = copy.deepcopy(self.net)
                self.patience = 0
                checkpoint_path = os.path.join(output_dir, 'best_attacker_pilot.pth')
                torch.save(self.best_net.state_dict(), checkpoint_path)
            else:
                self.patience += 1

            if self.early_stopping is not None and self.patience >= self.early_stopping:
                termination_reason = 'early_stopping'
                print('Early stopping at epoch %d (patience %d)' % (epoch, self.early_stopping))
                break

        if self.best_epoch is None or self.best_net is None:
            raise RuntimeError('No valid best pilot attacker checkpoint saved during training')

        checkpoint_sha256 = file_sha256(checkpoint_path)
        manifest = {
            'pilot_uncertified': True,
            'best_attacker_path': checkpoint_path,
            'best_attacker_sha256': checkpoint_sha256,
            'best_epoch': self.best_epoch,
            'best_val_loss': float(self.best_val_loss),
            'generator_checkpoint': self.generator_checkpoint,
            'generator_checkpoint_sha256': file_sha256(self.generator_checkpoint),
            'attacker_seed': self.attacker_seed,
            'epochs_completed': len(self.loss_dict['training']),
            'termination_reason': termination_reason,
        }
        with open(os.path.join(output_dir, 'attacker_manifest_pilot.json'), 'w') as f:
            json.dump(manifest, f, indent=2)

        print('Finished pilot attacker TRAIN/VALIDATION! Best Epoch: %s, Best Val BCE: %.5f' % (
            self.best_epoch, self.best_val_loss))
        return manifest


def run_one(arm, attacker_seed, device):
    print('=' * 70)
    print('STAGE A PILOT: arm=%s attacker_seed=%d' % (arm, attacker_seed))
    print('=' * 70)

    # Read-only integrity check: the frozen config file must be byte-identical
    # to what S1 used (same SHA256 the certified pipeline asserts).
    actual_sha = file_sha256(FROZEN_ATTACKER_CONFIG_PATH)
    if actual_sha != FROZEN_ATTACKER_CONFIG_SHA:
        raise RuntimeError('Frozen attacker config SHA mismatch: %s != %s' % (actual_sha, FROZEN_ATTACKER_CONFIG_SHA))
    with open(FROZEN_ATTACKER_CONFIG_PATH) as f:
        config = json.load(f)
    config = dict(config)  # in-memory copy; source file never written to
    config['attacker_seed'] = attacker_seed  # the only overridden field

    gen_manifest_p = os.path.join(M2_S1_BASE, arm, 'seed_42', 'checkpoint_manifest.json')
    with open(gen_manifest_p) as f:
        gen_manifest = json.load(f)
    gen_ckpt = gen_manifest['selected_generator_checkpoint']
    expected_gen_sha = gen_manifest['selected_generator_sha256']
    actual_gen_sha = file_sha256(gen_ckpt)
    if actual_gen_sha != expected_gen_sha:
        raise RuntimeError('S1 generator checkpoint SHA drift: %s != %s' % (actual_gen_sha, expected_gen_sha))

    out_dir = os.path.join(PILOT_OUT_ROOT, arm, 'attacker_seed_%d' % attacker_seed)
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    attacker = PilotAttacker(config, attacker_seed, device, gen_ckpt, image_size=config.get('image_size'))
    att_manifest = attacker.run(output_dir=out_dir)
    train_elapsed = time.time() - t0

    priv = evaluate_reid_val(
        config={'image_path': SCIENTIFIC_IMAGE_ROOT},
        attacker_checkpoint=att_manifest['best_attacker_path'],
        generator_checkpoint=gen_ckpt,
        device=device,
        unit_test_mode=False,
        image_size=256,
        expected_generator_sha=expected_gen_sha,
        expected_attacker_sha=att_manifest['best_attacker_sha256'],
    )
    priv = {k: v for k, v in priv.items() if k not in ('y_true', 'y_score')}

    result = {
        'pilot_uncertified': True,
        'arm': arm,
        'attacker_seed': attacker_seed,
        'generator_checkpoint_sha256': expected_gen_sha,
        'attacker_manifest': att_manifest,
        'privacy_val_metrics': priv,
        'train_elapsed_sec': round(train_elapsed, 2),
    }
    with open(os.path.join(out_dir, 'pilot_result.json'), 'w') as f:
        json.dump(result, f, indent=2)

    print('Arm %s seed %d: privacy_val_roc_auc=%.4f (%.1fs)' % (
        arm, attacker_seed, priv['roc_auc'], train_elapsed))
    return result


def main():
    parser = argparse.ArgumentParser(description='Stage A pilot runner (uncertified)')
    parser.add_argument('--arms', nargs='+', default=['B_dev', 'C4'], choices=['B_dev', 'C4'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[43, 44, 45, 46, 47])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('Stage A pilot requires CUDA')

    os.makedirs(PILOT_OUT_ROOT, exist_ok=True)
    all_results = []
    for arm in args.arms:
        for seed in args.seeds:
            all_results.append(run_one(arm, seed, device))

    with open(os.path.join(PILOT_OUT_ROOT, 'pilot_summary.json'), 'w') as f:
        json.dump({'pilot_uncertified': True, 'runs': all_results}, f, indent=2)
    print('Stage A pilot complete. Summary: %s' % os.path.join(PILOT_OUT_ROOT, 'pilot_summary.json'))


if __name__ == '__main__':
    main()
