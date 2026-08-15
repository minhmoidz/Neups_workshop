"""M1.3 / M2 executable anonymizer runner.

Implements the exact paired execution path for M2-S1 (B_dev and C4 on seed 42):
  - Arm configuration: B_dev (feature_loss_weight=0.0) vs C4 (feature_loss_weight=1.0)
  - Loads and verifies frozen initial generator, classifier, and verifier checkpoints
  - Uses the repaired m0_port/ACLoss with verified SHA (fails closed on stale ACLoss)
  - Three optimizers: Generator Adam (lr=1e-4), Critic SGD (lr=1e-4, momentum=0.9, wd=1e-4),
    Verifier Adam (lr=1e-4)
  - Canonical anonymizer DataLoader with FingerprintedRandomSampler (single source of truth for batch order)
  - Method-neutral checkpoint selection: minimum selection_total (ac_bce + privacy_term) across validation epochs,
    saving generator_best_method_neutral.pth
  - Full-state resumable checkpoint saving (checkpoint_latest.pth) with exact trajectory recovery
  - TEST firewall fail-closed check before dataset construction
"""
import os
import sys
import json
import time
import copy
import random
import hashlib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
for _p in (ROOT, os.path.join(ROOT, 'research_agent')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from networks.UNet_PriCheXyNet import UNet
from networks.SiameseNetwork import SiameseNetwork
from utils.VerificationLoss import VerificationLoss
from test_firewall import TestFirewall, provenance_record

from .evaluator_common import (
    MU,
    IMAGE_SIZE,
    INITIAL_GENERATOR_PATH,
    INITIAL_GENERATOR_SHA,
    FROZEN_CLASSIFIER_PATH,
    FROZEN_CLASSIFIER_SHA,
    FROZEN_VERIFIER_PATH,
    FROZEN_VERIFIER_SHA,
    REPAIRED_ACLOSS_SHA,
    METHOD_NEUTRAL_CKPT_NAME,
    file_sha256,
    firewall_check,
    make_flow_field_components,
    verify_repaired_acloss,
    build_dev_anonymizer_loaders,
    compute_epoch_totals,
    select_method_neutral_best,
)


class M2AnonymizerRunner:
    def __init__(self, arm='B_dev', config=None, output_dir=None, device=None,
                 seed=42, initial_generator_path=None, training_loader=None,
                 validation_loader=None, train_sampler=None,
                 ac_model=None, verification_model=None):
        """M2 Anonymizer Runner for B_dev and C4.

        :param arm: 'B_dev' (control) or 'C4' (feature loss method).
        :param config: configuration dictionary.
        :param output_dir: path where checkpoints, logs, and provenance will be written.
        :param device: torch.device (defaults to CUDA if available).
        :param seed: RNG seed for anonymizer (42 for S1).
        :param initial_generator_path: optional override for initial generator weights.
        :param training_loader / validation_loader: optional injected loaders (tests).
        :param train_sampler: optional injected sampler (tests).
        """
        # 1. Enforce TEST firewall
        firewall_check('dev')

        if arm not in ('B_dev', 'C4'):
            raise ValueError("arm must be 'B_dev' or 'C4', got %r" % arm)
        self.arm = arm
        self.seed = seed
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.config = config or {}
        self.output_dir = output_dir or os.path.join(ROOT, 'research_runs', 'M2_S1', arm, 'seed_%d' % seed)
        os.makedirs(self.output_dir, exist_ok=True)

        self.learning_rate = self.config.get('learning_rate', 1e-4)
        self.max_epochs = self.config.get('max_epochs', 250)
        self.batch_size = self.config.get('batch_size', 16)
        self.image_size = self.config.get('image_size', IMAGE_SIZE)
        self.mu = self.config.get('mu', MU)
        self.ac_loss_weight = self.config.get('ac_loss_weight', 1.0)
        self.ver_loss_weight = self.config.get('ver_loss_weight', 1.0)

        # C4 delta: feature_loss_weight = 1.0 for C4, 0.0 for B_dev
        self.feature_loss_weight = 1.0 if self.arm == 'C4' else 0.0

        # 2. Seed all RNGs before model creation
        self._seed_all(self.seed)

        # 3. Flow field components
        self.grid_identity, self.gauss_filter = make_flow_field_components(self.device, self.image_size)

        # 4. Generator (UNet, 1ch -> 2ch flow field)
        self.generator = UNet(1, 2, 32).to(self.device)
        init_gen_path = initial_generator_path or INITIAL_GENERATOR_PATH
        if os.path.exists(init_gen_path):
            actual_gen_sha = file_sha256(init_gen_path)
            if actual_gen_sha != INITIAL_GENERATOR_SHA:
                raise RuntimeError("Initial generator SHA drift: %s != %s" % (actual_gen_sha, INITIAL_GENERATOR_SHA))
            self.generator.load_state_dict(torch.load(init_gen_path, map_location=self.device, weights_only=False))
        self.initial_generator_sha = file_sha256(init_gen_path) if os.path.exists(init_gen_path) else 'mock_init'

        # 5. Classifier & Repaired ACLoss
        ACLossClass, self.acloss_sha, self.acloss_module_path = verify_repaired_acloss()
        if ac_model is not None:
            self.ac_model = ac_model.to(self.device)
        elif os.path.exists(FROZEN_CLASSIFIER_PATH):
            actual_clf_sha = file_sha256(FROZEN_CLASSIFIER_PATH)
            if actual_clf_sha != FROZEN_CLASSIFIER_SHA:
                raise RuntimeError("Classifier SHA drift: %s != %s" % (actual_clf_sha, FROZEN_CLASSIFIER_SHA))
            self.ac_model = torch.load(FROZEN_CLASSIFIER_PATH, map_location=self.device, weights_only=False)['model']
        else:
            from torchvision.models import densenet121
            self.ac_model = densenet121(num_classes=14)

        self.ac_loss = ACLossClass(
            ac_model=self.ac_model,
            reduction='mean',
            pos_weight=self.config.get('ac_pos_weight', None),
            feature_loss_weight=self.feature_loss_weight
        ).to(self.device)

        # 6. Verifier & VerificationLoss
        if verification_model is not None:
            self.verification_model = verification_model.to(self.device)
        else:
            self.verification_model = SiameseNetwork().to(self.device)
            if os.path.exists(FROZEN_VERIFIER_PATH):
                actual_ver_sha = file_sha256(FROZEN_VERIFIER_PATH)
                if actual_ver_sha != FROZEN_VERIFIER_SHA:
                    raise RuntimeError("Verifier SHA drift: %s != %s" % (actual_ver_sha, FROZEN_VERIFIER_SHA))
                self.verification_model.load_state_dict(torch.load(FROZEN_VERIFIER_PATH, map_location=self.device, weights_only=False))
        self.verification_loss = VerificationLoss(
            verification_model=self.verification_model,
            reduction='none'
        ).to(self.device)

        # 7. Criterion & Optimizers
        self.criterion_ac = nn.BCELoss().to(self.device)
        self.criterion_ver = nn.BCEWithLogitsLoss().to(self.device)

        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        self.optimizer_ver = optim.Adam(self.verification_loss.verification_model.parameters(), lr=self.learning_rate)
        self.optimizer_ac = optim.SGD(
            filter(lambda p: p.requires_grad, self.ac_loss.ac_model.parameters()),
            lr=self.learning_rate, momentum=0.9, weight_decay=1e-4
        )

        # 8. Loaders & Deterministic Sampler
        if training_loader is not None and validation_loader is not None:
            self.training_loader = training_loader
            self.validation_loader = validation_loader
            self.train_sampler = train_sampler
        else:
            self.training_loader, self.validation_loader, self.train_sampler = build_dev_anonymizer_loaders(
                self.config, seed=self.seed, num_workers=self.config.get('num_workers', 0)
            )

        # Preprocessing transforms for SNN and classifier updates
        self.resize_224 = transforms.Resize((224, 224))
        self.imagenet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Tracking structures
        self.start_epoch = 0
        self.epoch_metrics = []
        self.best_selection_total = float('inf')
        self.best_epoch = None
        self.best_checkpoint_path = os.path.join(self.output_dir, METHOD_NEUTRAL_CKPT_NAME)
        self.latest_checkpoint_path = os.path.join(self.output_dir, 'checkpoint_latest.pth')

    def _seed_all(self, seed):
        """Set all RNG seeds deterministically."""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False

    def anonymize_tensor(self, image):
        """Shared legacy operator anonymization."""
        grids = self.generator(image)
        grids = self.grid_identity - self.mu * grids
        grids = self.gauss_filter(grids)
        grids = grids.permute(0, 2, 3, 1)
        return F.grid_sample(image, grids, padding_mode='border', align_corners=True)

    def train_epoch(self, epoch):
        """Execute one training epoch with exact model and optimizer updates."""
        self.generator.train()
        running_ac_bce = 0.0
        running_ver_loss = 0.0
        running_privacy_term = 0.0
        running_feature_term = 0.0
        running_opt_total = 0.0
        running_sel_total = 0.0

        n_batches = 0
        for batch in self.training_loader:
            inputs1, inputs2, labels, labels_id = batch
            inputs1 = inputs1.to(self.device)
            inputs2 = inputs2.to(self.device)
            labels = labels.to(self.device)
            labels_id = labels_id.to(self.device)

            # Anonymize input1
            fakes_1 = self.anonymize_tensor(inputs1)

            # 1. Auxiliary classifier loss & feature loss
            if self.arm == 'C4':
                # C4: pass real_image=inputs1 to compute detached source penultimate MSE
                ac_total_loss = self.ac_loss(fakes_1, labels, real_image=inputs1)
                # Compute feature component separately for logging
                with torch.no_grad():
                    def_feat = self.ac_loss._features(self.ac_loss._preprocess(fakes_1))
                    real_feat = self.ac_loss._features(self.ac_loss._preprocess(inputs1)).detach()
                    feat_val = F.mse_loss(def_feat, real_feat).item()
                    ac_bce_val = (ac_total_loss.item() - self.feature_loss_weight * feat_val)
            else:
                ac_total_loss = self.ac_loss(fakes_1, labels)
                ac_bce_val = ac_total_loss.item()
                feat_val = 0.0

            # 2. Verification loss (privacy term)
            ver_loss = self.verification_loss(fakes_1, inputs2)
            log_likelihood_ver_loss = - torch.log(torch.clamp(1.0 - ver_loss, min=1e-7))
            ver_loss_mean = ver_loss.mean()
            privacy_term = log_likelihood_ver_loss.mean()

            # 3. Generator optimization total loss
            gen_loss = self.ac_loss_weight * ac_total_loss + self.ver_loss_weight * privacy_term

            # Selection total (feature term EXCLUDED for both arms)
            sel_loss_val = self.ac_loss_weight * ac_bce_val + self.ver_loss_weight * privacy_term.item()

            # Optimize Generator
            self.optimizer_g.zero_grad()
            gen_loss.backward()
            self.optimizer_g.step()

            # 4. Update Verification Critic
            self.verification_loss.verification_model.train()
            inputs1_snn = self.imagenet_normalize(fakes_1.detach().expand(-1, 3, -1, -1))
            inputs2_snn = self.imagenet_normalize(inputs2.expand(-1, 3, -1, -1))

            self.optimizer_ver.zero_grad()
            outputs_snn = self.verification_loss.verification_model(inputs1_snn, inputs2_snn).squeeze()
            labels_id_cast = labels_id.type_as(outputs_snn)
            loss_ver_critic = self.criterion_ver(outputs_snn, labels_id_cast)
            loss_ver_critic.backward()
            self.optimizer_ver.step()
            self.verification_loss.verification_model.eval()

            # 5. Update Auxiliary Classifier Critic
            self.ac_loss.ac_model.train()
            inputs_ac = self.imagenet_normalize(self.resize_224(fakes_1.detach().expand(-1, 3, -1, -1)))

            self.optimizer_ac.zero_grad()
            outputs_ac = self.ac_loss.ac_model(inputs_ac)
            loss_ac_critic = self.criterion_ac(outputs_ac, labels)
            loss_ac_critic.backward()
            self.optimizer_ac.step()
            self.ac_loss.ac_model.eval()

            # Accumulate logging metrics
            running_ac_bce += ac_bce_val
            running_ver_loss += ver_loss_mean.item()
            running_privacy_term += privacy_term.item()
            running_feature_term += feat_val
            running_opt_total += gen_loss.item()
            running_sel_total += sel_loss_val
            n_batches += 1

        denom = max(n_batches, 1)
        return {
            'train_ac_bce': running_ac_bce / denom,
            'train_ver_loss': running_ver_loss / denom,
            'train_privacy_term': running_privacy_term / denom,
            'train_feature_term': running_feature_term / denom,
            'train_optimization_total': running_opt_total / denom,
            'train_selection_total': running_sel_total / denom,
        }

    def validate_epoch(self, epoch):
        """Execute one validation epoch (no grad)."""
        self.generator.eval()
        running_ac_bce = 0.0
        running_ver_loss = 0.0
        running_privacy_term = 0.0
        running_feature_term = 0.0
        running_opt_total = 0.0
        running_sel_total = 0.0

        n_batches = 0
        with torch.no_grad():
            for batch in self.validation_loader:
                inputs1, inputs2, labels, _ = batch
                inputs1 = inputs1.to(self.device)
                inputs2 = inputs2.to(self.device)
                labels = labels.to(self.device)

                fakes_1 = self.anonymize_tensor(inputs1)

                if self.arm == 'C4':
                    ac_total_loss = self.ac_loss(fakes_1, labels, real_image=inputs1)
                    def_feat = self.ac_loss._features(self.ac_loss._preprocess(fakes_1))
                    real_feat = self.ac_loss._features(self.ac_loss._preprocess(inputs1)).detach()
                    feat_val = F.mse_loss(def_feat, real_feat).item()
                    ac_bce_val = ac_total_loss.item() - self.feature_loss_weight * feat_val
                else:
                    ac_total_loss = self.ac_loss(fakes_1, labels)
                    ac_bce_val = ac_total_loss.item()
                    feat_val = 0.0

                ver_loss = self.verification_loss(fakes_1, inputs2)
                log_likelihood_ver_loss = - torch.log(torch.clamp(1.0 - ver_loss, min=1e-7))
                ver_loss_mean = ver_loss.mean()
                privacy_term = log_likelihood_ver_loss.mean()

                opt_total = self.ac_loss_weight * ac_total_loss.item() + self.ver_loss_weight * privacy_term.item()
                sel_total = self.ac_loss_weight * ac_bce_val + self.ver_loss_weight * privacy_term.item()

                running_ac_bce += ac_bce_val
                running_ver_loss += ver_loss_mean.item()
                running_privacy_term += privacy_term.item()
                running_feature_term += feat_val
                running_opt_total += opt_total
                running_sel_total += sel_total
                n_batches += 1

        denom = max(n_batches, 1)
        return {
            'val_ac_bce': running_ac_bce / denom,
            'val_ver_loss': running_ver_loss / denom,
            'val_privacy_term': running_privacy_term / denom,
            'val_feature_term': running_feature_term / denom,
            'val_optimization_total': running_opt_total / denom,
            'val_selection_total': running_sel_total / denom,
        }

    def save_resumable_checkpoint(self, epoch, path=None):
        """Save full deterministic state for crash recovery."""
        path = path or self.latest_checkpoint_path
        sampler_rng = None
        if self.train_sampler is not None and hasattr(self.train_sampler, 'generator') and self.train_sampler.generator is not None:
            sampler_rng = self.train_sampler.generator.get_state()

        state = {
            'epoch': epoch,
            'arm': self.arm,
            'seed': self.seed,
            'generator_state': self.generator.state_dict(),
            'ac_model_state': self.ac_loss.ac_model.state_dict(),
            'verification_model_state': self.verification_loss.verification_model.state_dict(),
            'optimizer_g_state': self.optimizer_g.state_dict(),
            'optimizer_ac_state': self.optimizer_ac.state_dict(),
            'optimizer_ver_state': self.optimizer_ver.state_dict(),
            'rng_torch': torch.get_rng_state(),
            'rng_cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            'rng_numpy': np.random.get_state(),
            'rng_random': random.getstate(),
            'sampler_rng': sampler_rng,
            'epoch_metrics': self.epoch_metrics,
            'best_selection_total': self.best_selection_total,
            'best_epoch': self.best_epoch,
        }
        torch.save(state, path)
        return path

    def load_resumable_checkpoint(self, path=None):
        """Restore full deterministic state from a saved checkpoint."""
        path = path or self.latest_checkpoint_path
        if not os.path.exists(path):
            raise FileNotFoundError("Resumable checkpoint not found: %s" % path)
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.start_epoch = state['epoch'] + 1
        self.generator.load_state_dict(state['generator_state'])
        self.ac_loss.ac_model.load_state_dict(state['ac_model_state'])
        self.ac_loss.refresh()
        self.verification_loss.verification_model.load_state_dict(state['verification_model_state'])
        self.optimizer_g.load_state_dict(state['optimizer_g_state'])
        self.optimizer_ac.load_state_dict(state['optimizer_ac_state'])
        self.optimizer_ver.load_state_dict(state['optimizer_ver_state'])

        torch.set_rng_state(state['rng_torch'])
        if torch.cuda.is_available() and state.get('rng_cuda') is not None:
            torch.cuda.set_rng_state(state['rng_cuda'])
        np.random.set_state(state['rng_numpy'])
        random.setstate(state['rng_random'])

        if self.train_sampler is not None and hasattr(self.train_sampler, 'generator') and self.train_sampler.generator is not None:
            if state.get('sampler_rng') is not None:
                self.train_sampler.generator.set_state(state['sampler_rng'])

        self.epoch_metrics = state.get('epoch_metrics', [])
        self.best_selection_total = state.get('best_selection_total', float('inf'))
        self.best_epoch = state.get('best_epoch', None)
        return state

    def run(self, max_epochs=None):
        """Execute the full training and validation loop with method-neutral selection."""
        max_epochs = max_epochs or self.max_epochs

        for epoch in range(self.start_epoch, max_epochs):
            t0 = time.time()
            train_m = self.train_epoch(epoch)
            val_m = self.validate_epoch(epoch)
            elapsed = time.time() - t0

            combined = {'epoch': epoch, 'elapsed_sec': elapsed, **train_m, **val_m}
            self.epoch_metrics.append(combined)

            # Method-neutral checkpoint selection: minimum val_selection_total (ac_bce + privacy_term)
            val_sel = val_m['val_selection_total']
            if val_sel < self.best_selection_total - 1e-12:
                self.best_selection_total = val_sel
                self.best_epoch = epoch
                torch.save(self.generator.state_dict(), self.best_checkpoint_path)

            # Save latest checkpoint for crash recovery
            self.save_resumable_checkpoint(epoch)

            # Write epoch metrics to CSV
            df = pd.DataFrame(self.epoch_metrics)
            df.to_csv(os.path.join(self.output_dir, 'epoch_metrics.csv'), index=False)

        # Write manifest
        best_sha = file_sha256(self.best_checkpoint_path) if os.path.exists(self.best_checkpoint_path) else None
        manifest = {
            'arm': self.arm,
            'seed': self.seed,
            'best_epoch': self.best_epoch,
            'best_selection_total': self.best_selection_total,
            'best_checkpoint_path': self.best_checkpoint_path,
            'best_checkpoint_sha256': best_sha,
            'total_epochs': max_epochs,
            'initial_generator_sha256': self.initial_generator_sha,
            'acloss_sha256': self.acloss_sha,
        }
        with open(os.path.join(self.output_dir, 'checkpoint_manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=2)

        # Write provenance
        prov = provenance_record(mode='dev', extra=manifest)
        with open(os.path.join(self.output_dir, 'provenance.json'), 'w') as f:
            json.dump(prov, f, indent=2)

        return manifest


def run_preflight_smoke(arm='B_dev', device=None):
    """Execute a 2-batch train + 2-batch val preflight smoke test on real data."""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = {
        'batch_size': 16,
        'image_size': 256,
        'learning_rate': 1e-4,
        'max_epochs': 1,
        'image_path': './',
    }
    tmp_out = os.path.join(ROOT, 'research_runs', '_preflight_tmp', arm)
    runner = M2AnonymizerRunner(arm=arm, config=cfg, output_dir=tmp_out, device=device)

    # Re-slice loader to 2 batches for quick preflight verification
    class _SlicedLoader:
        def __init__(self, loader, n=2):
            self.loader = loader
            self.n = n

        def __len__(self):
            return self.n

        def __iter__(self):
            for i, batch in enumerate(self.loader):
                if i >= self.n:
                    break
                yield batch

    runner.training_loader = _SlicedLoader(runner.training_loader, 2)
    runner.validation_loader = _SlicedLoader(runner.validation_loader, 2)

    # Measure initial VRAM
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    train_m = runner.train_epoch(0)
    val_m = runner.validate_epoch(0)

    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

    # Verify losses are finite
    assert np.isfinite(train_m['train_optimization_total']), "train opt total non-finite"
    assert np.isfinite(val_m['val_selection_total']), "val sel total non-finite"

    # Save smoke checkpoint and verify
    runner.save_resumable_checkpoint(0)
    torch.save(runner.generator.state_dict(), runner.best_checkpoint_path)
    assert os.path.exists(runner.best_checkpoint_path), "checkpoint failed to save"

    # Clean up temporary preflight artifacts
    if os.path.exists(tmp_out):
        import shutil
        shutil.rmtree(tmp_out, ignore_errors=True)

    return {
        'arm': arm,
        'train_metrics': train_m,
        'val_metrics': val_m,
        'peak_vram_mb': peak_vram,
        'status': 'PASS'
    }
