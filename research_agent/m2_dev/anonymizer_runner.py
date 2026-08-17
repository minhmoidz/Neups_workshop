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
  - Full-state resumable checkpoint saving (checkpoint_latest.pth)
    WARNING (F7 M1.4c): Resume is NOT certified for scientific use. Sampler epoch_indices
    are not restored, so resumed runs cannot guarantee exact trajectory recovery.
    Scientific runs MUST restart from epoch 0 if interrupted.
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
    verify_execution_lock_integrity,
    compute_expected_train_order_hashes,
    FROZEN_B_DEV_CONFIG_PATH,
    FROZEN_C4_CONFIG_PATH,
    FROZEN_B_DEV_CONFIG_SHA,
    FROZEN_C4_CONFIG_SHA,
)


class M2AnonymizerRunner:
    def __init__(self, arm='B_dev', config=None, output_dir=None, device=None,
                 seed=42, initial_generator_path=None, ac_model=None,
                 verification_model=None, training_loader=None, validation_loader=None,
                 train_sampler=None, unit_test_mode=False, gradient_diagnostics_enabled=True,
                 config_path=None):
        """Paired M2 Anonymizer Runner for B_dev and C4.

        :param arm: 'B_dev' (control) or 'C4' (feature preservation).
        :param config: dict or path to json config.
        :param output_dir: directory to store telemetry and checkpoints.
        :param device: torch device.
        :param seed: RNG seed (M1 frozen: 42).
        :param initial_generator_path: explicit path to initial generator checkpoint.
        :param ac_model: optional pre-instantiated classifier (for testing).
        :param verification_model: optional pre-instantiated verifier (for testing).
        :param training_loader: optional pre-built loader (for testing).
        :param validation_loader: optional pre-built loader (for testing).
        :param train_sampler: optional pre-built sampler.
        :param unit_test_mode: bool, if False (default) enforces strict scientific fail-closed checks.
        :param config_path: optional explicit path to config file.
        """
        firewall_check('dev')
        if arm not in ('B_dev', 'C4'):
            raise ValueError("arm must be 'B_dev' or 'C4', got %r" % arm)
        if not unit_test_mode:
            if device is not None and getattr(device, 'type', str(device).split(':')[0]) == 'cpu':
                raise RuntimeError('Scientific anonymizer execution requires CUDA; CPU is unit-test only')
            if device is None and not torch.cuda.is_available():
                raise RuntimeError('Scientific anonymizer execution requires CUDA')
            if any(value is not None for value in (ac_model, verification_model, training_loader, validation_loader, train_sampler)):
                raise RuntimeError('Scientific anonymizer execution does not accept injected models/loaders/samplers')
            if seed != 42:
                raise RuntimeError('Scientific anonymizer seed must be exactly 42')
            if isinstance(config, dict):
                raise RuntimeError('Scientific anonymizer execution requires the canonical config file, not an in-memory dict')
            expected_path = FROZEN_C4_CONFIG_PATH if arm == 'C4' else FROZEN_B_DEV_CONFIG_PATH
            if config is not None and os.path.abspath(config) != os.path.abspath(expected_path):
                raise RuntimeError('Scientific anonymizer config path is not canonical: %s' % config)
            verify_execution_lock_integrity()

        self.arm = arm
        self.seed = seed
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.unit_test_mode = unit_test_mode
        self.gradient_diagnostics_enabled = gradient_diagnostics_enabled and arm == 'C4'
        self.nan_inf_detected = False

        # 1. Load config
        if config is None:
            cfg_file = 'config_dev_c4.json' if arm == 'C4' else 'config_dev_restored_baseline.json'
            c_path = os.path.join(ROOT, 'config_files', cfg_file)
            with open(c_path) as f:
                self.config = json.load(f)
            self.config_path = c_path
        elif isinstance(config, str):
            with open(config) as f:
                self.config = json.load(f)
            self.config_path = config
        else:
            self.config = config
            if config_path is not None:
                self.config_path = config_path
            else:
                cfg_file = 'config_dev_c4.json' if arm == 'C4' else 'config_dev_restored_baseline.json'
                default_p = os.path.join(ROOT, 'config_files', cfg_file)
                self.config_path = default_p if os.path.exists(default_p) else None

        if not unit_test_mode:
            expected_cfg_sha = FROZEN_C4_CONFIG_SHA if arm == 'C4' else FROZEN_B_DEV_CONFIG_SHA
            if not self.config_path or file_sha256(self.config_path) != expected_cfg_sha:
                raise RuntimeError('Scientific anonymizer config SHA mismatch')

        self.output_dir = output_dir or os.path.join(ROOT, 'research_runs', 'M2_S1', arm, 'seed_%d' % seed)
        os.makedirs(self.output_dir, exist_ok=True)

        self.mu = self.config.get('mu', 0.01)
        self.batch_size = self.config.get('batch_size', 16)
        self.learning_rate = self.config.get('learning_rate', 1e-4)
        self.max_epochs = self.config.get('max_epochs', 250)
        self.requested_max_epochs = self.max_epochs
        self.image_size = self.config.get('image_size', IMAGE_SIZE)
        self.ac_loss_weight = self.config.get('ac_loss_weight', 1.0)
        self.ver_loss_weight = self.config.get('ver_loss_weight', 1.0)
        self.feature_loss_weight = 1.0 if self.arm == 'C4' else 0.0

        # 2. Seed all RNGs before model creation
        self._seed_all(self.seed)

        # 3. Flow field components
        self.grid_identity, self.gauss_filter = make_flow_field_components(self.device, self.image_size)

        # 4. Generator (UNet, 1ch -> 2ch flow field)
        self.generator = UNet(1, 2, 32).to(self.device)
        init_gen_path = initial_generator_path or INITIAL_GENERATOR_PATH
        if not os.path.exists(init_gen_path):
            if not self.unit_test_mode:
                raise FileNotFoundError("Initial generator checkpoint missing: %s" % init_gen_path)
            self.initial_generator_sha = 'mock_init'
        else:
            actual_gen_sha = file_sha256(init_gen_path)
            if actual_gen_sha != INITIAL_GENERATOR_SHA and not self.unit_test_mode:
                raise RuntimeError("Initial generator SHA drift: %s != %s" % (actual_gen_sha, INITIAL_GENERATOR_SHA))
            self.generator.load_state_dict(torch.load(init_gen_path, map_location=self.device, weights_only=False))
            self.initial_generator_sha = actual_gen_sha

        # 5. Classifier & Repaired ACLoss
        ACLossClass, self.acloss_sha, self.acloss_module_path = verify_repaired_acloss()
        if ac_model is not None:
            self.ac_model = ac_model.to(self.device)
        elif os.path.exists(FROZEN_CLASSIFIER_PATH):
            actual_clf_sha = file_sha256(FROZEN_CLASSIFIER_PATH)
            if actual_clf_sha != FROZEN_CLASSIFIER_SHA and not self.unit_test_mode:
                raise RuntimeError("Classifier SHA drift: %s != %s" % (actual_clf_sha, FROZEN_CLASSIFIER_SHA))
            self.ac_model = torch.load(FROZEN_CLASSIFIER_PATH, map_location=self.device, weights_only=False)['model']
        else:
            if not self.unit_test_mode:
                raise FileNotFoundError("Frozen classifier checkpoint missing: %s" % FROZEN_CLASSIFIER_PATH)
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
        elif os.path.exists(FROZEN_VERIFIER_PATH):
            actual_ver_sha = file_sha256(FROZEN_VERIFIER_PATH)
            if actual_ver_sha != FROZEN_VERIFIER_SHA and not self.unit_test_mode:
                raise RuntimeError("Verifier SHA drift: %s != %s" % (actual_ver_sha, FROZEN_VERIFIER_SHA))
            self.verification_model = SiameseNetwork().to(self.device)
            self.verification_model.load_state_dict(torch.load(FROZEN_VERIFIER_PATH, map_location=self.device, weights_only=False))
        else:
            if not self.unit_test_mode:
                raise FileNotFoundError("Frozen verifier checkpoint missing: %s" % FROZEN_VERIFIER_PATH)
            self.verification_model = SiameseNetwork().to(self.device)

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

        self.expected_train_order_hashes = None
        if not self.unit_test_mode:
            pair_file = getattr(getattr(self.training_loader, 'dataset', None), 'image_pairs', None)
            if pair_file is None:
                raise RuntimeError('Scientific anonymizer loader lacks canonical pair rows')
            pair_path = os.path.join(ROOT, 'image_pairs', 'image_pairs_training_10000.txt')
            self.expected_train_order_hashes = compute_expected_train_order_hashes(pair_path, seed=self.seed, epochs=250)

        # Preprocessing transforms for SNN and classifier updates
        self.resize_224 = transforms.Resize((224, 224))
        self.imagenet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Tracking structures
        self.start_epoch = 0
        self.epoch_metrics = []
        self.best_selection_total = float('inf')
        self.best_epoch = None
        self.gradient_norm_diagnostics = {}
        self.gradient_diagnostics_failed = False
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
        torch.backends.cudnn.deterministic = True

    def anonymize_tensor(self, image):
        """Shared legacy operator anonymization."""
        grids = self.generator(image)
        grids = self.grid_identity - self.mu * grids
        grids = self.gauss_filter(grids)
        grids = grids.permute(0, 2, 3, 1)
        return F.grid_sample(image, grids, padding_mode='border', align_corners=True)

    @staticmethod
    def _assert_finite_tensor(value, label):
        if not torch.isfinite(value).all():
            raise FloatingPointError('%s contains NaN/Inf' % label)

    def _assert_optimizer_gradients(self, optimizer, label):
        for group in optimizer.param_groups:
            for parameter in group['params']:
                if parameter.grad is not None:
                    self._assert_finite_tensor(parameter.grad, '%s gradient' % label)

    def _assert_optimizer_parameters(self, optimizer, label):
        for group in optimizer.param_groups:
            for parameter in group['params']:
                self._assert_finite_tensor(parameter.data, '%s parameter' % label)

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

            # 1. Anonymize source image
            fakes_1 = self.anonymize_tensor(inputs1)

            # 2. Auxiliary classification loss
            self.ac_loss.refresh()
            deformed_features = self.ac_loss._features(self.ac_loss._preprocess(fakes_1))
            ac_predictions = self.ac_loss.loss_model.classifier(deformed_features)
            ac_bce_loss = self.ac_loss.bce_loss(ac_predictions, labels)

            if self.arm == 'C4' and self.feature_loss_weight > 0:
                real_features = self.ac_loss._features(self.ac_loss._preprocess(inputs1)).detach()
                feat_loss = self.feature_loss_weight * F.mse_loss(deformed_features, real_features)
                ac_total_loss = ac_bce_loss + feat_loss
                feat_val = feat_loss.item()
            else:
                ac_total_loss = ac_bce_loss
                feat_loss = None
                feat_val = 0.0

            ac_bce_val = ac_bce_loss.item()

            # Verifier privacy loss
            inputs1_snn_g = self.imagenet_normalize(fakes_1.expand(-1, 3, -1, -1))
            inputs2_snn_g = self.imagenet_normalize(inputs2.expand(-1, 3, -1, -1))
            raw_verifier_logits = self.verification_loss.verification_model(inputs1_snn_g, inputs2_snn_g).squeeze()
            privacy_term = F.softplus(raw_verifier_logits).mean()
            with torch.no_grad():
                ver_loss_mean = torch.sigmoid(raw_verifier_logits.to(dtype=torch.float64)).mean()

            # 3. Generator optimization total loss
            gen_loss = self.ac_loss_weight * ac_total_loss + self.ver_loss_weight * privacy_term

            # Selection total (feature term EXCLUDED for both arms)
            sel_loss_val = self.ac_loss_weight * ac_bce_val + self.ver_loss_weight * privacy_term.item()

            # Optional C4 diagnostic: gradient norm ratio at epoch 0, 1, and every 25 epochs (batch 0 only).
            # Gated by an explicit flag (NOT by arm) so the diagnostic can be disabled without
            # altering the scientific feature-loss objective (non-interference contract, §5).
            if self.gradient_diagnostics_enabled and self.arm == 'C4' and (
                    epoch in (0, 1) or epoch % 25 == 0) and n_batches == 0 and feat_loss is not None:
                try:
                    base_loss = self.ac_loss_weight * ac_bce_loss + self.ver_loss_weight * privacy_term
                    base_grads = torch.autograd.grad(base_loss, self.generator.parameters(), retain_graph=True, allow_unused=True)
                    base_grads_valid = [g for g in base_grads if g is not None]
                    norm_base = torch.norm(torch.stack([torch.norm(g) for g in base_grads_valid])).item() if base_grads_valid else 0.0

                    feat_grads = torch.autograd.grad(feat_loss, self.generator.parameters(), retain_graph=True, allow_unused=True)
                    feat_grads_valid = [g for g in feat_grads if g is not None]
                    norm_feat = torch.norm(torch.stack([torch.norm(g) for g in feat_grads_valid])).item() if feat_grads_valid else 0.0

                    ratio = norm_feat / max(norm_base, 1e-12)
                    self.gradient_norm_diagnostics[epoch] = {
                        'epoch': epoch,
                        'base_grad_norm': float(norm_base),
                        'feature_grad_norm': float(norm_feat),
                        'feature_base_ratio': float(ratio)
                    }
                except Exception as _diag_e:
                    self.gradient_diagnostics_failed = True
                    self.gradient_norm_diagnostics[epoch] = {'epoch': epoch, 'error': str(_diag_e)}
                    if not self.unit_test_mode:
                        raise FloatingPointError('Required C4 gradient diagnostic failed at epoch %d: %s' % (epoch, _diag_e)) from _diag_e

            # Optimize Generator. Check the load-bearing loss before backward,
            # gradients after backward, and parameters after the optimizer step.
            self._assert_finite_tensor(gen_loss, 'generator loss')
            self._assert_finite_tensor(ac_total_loss, 'auxiliary classifier loss')
            self._assert_finite_tensor(privacy_term, 'privacy loss')
            self.optimizer_g.zero_grad()
            gen_loss.backward()
            self._assert_optimizer_gradients(self.optimizer_g, 'generator')
            self.last_gen_grads = [p.grad.detach().clone() for p in self.generator.parameters() if p.grad is not None]
            self.optimizer_g.step()
            self._assert_optimizer_parameters(self.optimizer_g, 'generator')

            # 4. Update Verification Critic
            self.verification_loss.verification_model.train()
            inputs1_snn = self.imagenet_normalize(fakes_1.detach().expand(-1, 3, -1, -1))
            inputs2_snn = self.imagenet_normalize(inputs2.expand(-1, 3, -1, -1))

            self.optimizer_ver.zero_grad()
            outputs_snn = self.verification_loss.verification_model(inputs1_snn, inputs2_snn).squeeze()
            labels_id_cast = labels_id.type_as(outputs_snn)
            loss_ver_critic = self.criterion_ver(outputs_snn, labels_id_cast)
            self._assert_finite_tensor(loss_ver_critic, 'verifier critic loss')
            loss_ver_critic.backward()
            self._assert_optimizer_gradients(self.optimizer_ver, 'verifier critic')
            self.last_ver_grads = [p.grad.detach().clone() for p in self.verification_loss.verification_model.parameters() if p.grad is not None]
            self.optimizer_ver.step()
            self._assert_optimizer_parameters(self.optimizer_ver, 'verifier critic')
            self.verification_loss.verification_model.eval()

            # 5. Update Auxiliary Classifier Critic
            self.ac_loss.ac_model.train()
            inputs_ac = self.imagenet_normalize(self.resize_224(fakes_1.detach().expand(-1, 3, -1, -1)))

            self.optimizer_ac.zero_grad()
            outputs_ac = self.ac_loss.ac_model(inputs_ac)
            loss_ac_critic = self.criterion_ac(outputs_ac, labels)
            self._assert_finite_tensor(loss_ac_critic, 'auxiliary classifier critic loss')
            loss_ac_critic.backward()
            self._assert_optimizer_gradients(self.optimizer_ac, 'auxiliary classifier critic')
            self.last_clf_grads = [p.grad.detach().clone() for p in self.ac_loss.ac_model.parameters() if p.requires_grad and p.grad is not None]
            self.optimizer_ac.step()
            self._assert_optimizer_parameters(self.optimizer_ac, 'auxiliary classifier critic')
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

                inputs1_snn_g = self.imagenet_normalize(fakes_1.expand(-1, 3, -1, -1))
                inputs2_snn_g = self.imagenet_normalize(inputs2.expand(-1, 3, -1, -1))
                raw_verifier_logits = self.verification_loss.verification_model(inputs1_snn_g, inputs2_snn_g).squeeze()
                privacy_term = F.softplus(raw_verifier_logits).mean()
                ver_loss_mean = torch.sigmoid(raw_verifier_logits.to(dtype=torch.float64)).mean()

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
        """Restore state only for explicit unit tests; scientific resume is forbidden."""
        if not self.unit_test_mode:
            raise RuntimeError('Scientific resume is not certified; restart from epoch 0')
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
        if not self.unit_test_mode:
            if max_epochs is not None and max_epochs != 250:
                raise RuntimeError('Scientific anonymizer max_epochs must be exactly 250')
            if self.seed != 42:
                raise RuntimeError('Scientific anonymizer seed must be exactly 42')
        self.requested_max_epochs = max_epochs if max_epochs is not None else self.max_epochs

        for epoch in range(self.start_epoch, self.requested_max_epochs):
            t0 = time.time()
            train_m = self.train_epoch(epoch)
            val_m = self.validate_epoch(epoch)
            elapsed = time.time() - t0
            peak_vram = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024) if (torch.cuda.is_available() and getattr(self.device, 'type', '') == 'cuda') else 0.0
            order_sha = self.train_sampler.get_epoch_order_hash(epoch) if self.train_sampler is not None else None
            if not self.unit_test_mode:
                if order_sha is None or self.expected_train_order_hashes is None:
                    raise RuntimeError('Scientific run missing runtime TRAIN order hash')
                if epoch >= len(self.expected_train_order_hashes) or order_sha != self.expected_train_order_hashes[epoch]:
                    raise RuntimeError('Scientific TRAIN order hash mismatch at epoch %d' % epoch)

            # Check for NaN / Inf in all load-bearing and diagnostic scalars
            all_vals = list(train_m.values()) + list(val_m.values())
            if self.arm == 'C4' and epoch in self.gradient_norm_diagnostics:
                diag = self.gradient_norm_diagnostics[epoch]
                all_vals.extend([diag.get('base_grad_norm'), diag.get('feature_grad_norm'), diag.get('feature_base_ratio')])

            has_nan_inf = self.gradient_diagnostics_failed or any(
                not np.isfinite(v) for v in all_vals
                if (v is not None and isinstance(v, (int, float, np.floating, np.integer)))
            )

            combined = {
                'epoch': epoch,
                'learning_rate': self.optimizer_g.param_groups[0]['lr'],
                'elapsed_sec': round(elapsed, 2),
                'peak_vram_mb': round(peak_vram, 2),
                'order_sha256': order_sha,
                'is_nan_inf': has_nan_inf,
                **train_m,
                **val_m
            }
            if self.arm == 'C4' and epoch in self.gradient_norm_diagnostics:
                combined['feature_grad_ratio'] = self.gradient_norm_diagnostics[epoch].get('feature_base_ratio')

            # Scientific Fail-Closed on NaN/Inf
            if has_nan_inf:
                self.nan_inf_detected = True
                raise FloatingPointError("Non-finite loss/diagnostic detected in epoch %d (arm %s): %s" % (epoch, self.arm, combined))

            self.epoch_metrics.append(combined)

            # Append to train_log.jsonl
            log_path = os.path.join(self.output_dir, 'train_log.jsonl')
            with open(log_path, 'a') as f_log:
                f_log.write(json.dumps(combined) + '\n')

            # Method-neutral checkpoint selection: minimum val_selection_total (ac_bce + privacy_term)
            val_sel = val_m['val_selection_total']
            if val_sel < self.best_selection_total - 1e-12:
                self.best_selection_total = val_sel
                self.best_epoch = epoch
                torch.save(self.generator.state_dict(), self.best_checkpoint_path)

            # Scientific continuation is not certified; only unit tests may save resumable state.
            if self.unit_test_mode:
                self.save_resumable_checkpoint(epoch)

            # Write epoch metrics to CSV
            df = pd.DataFrame(self.epoch_metrics)
            df.to_csv(os.path.join(self.output_dir, 'epoch_metrics.csv'), index=False)

        # Write manifest (§5 explicit checkpoint manifest handoff)
        best_sha = file_sha256(self.best_checkpoint_path) if os.path.exists(self.best_checkpoint_path) else None
        config_sha = file_sha256(self.config_path) if (self.config_path and os.path.exists(self.config_path)) else None
        try:
            import subprocess
            git_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode('utf-8').strip()
        except Exception:
            git_head = None

        epochs_done = len(self.epoch_metrics)
        final_ep = self.epoch_metrics[-1]['epoch'] if self.epoch_metrics else None

        manifest = {
            'arm': self.arm,
            'selected_generator_checkpoint': os.path.abspath(self.best_checkpoint_path),
            'selected_generator_sha256': best_sha,
            'best_epoch': self.best_epoch,
            'best_selection_total': self.best_selection_total,
            'initial_generator_sha256': self.initial_generator_sha,
            'config_path': os.path.abspath(self.config_path) if self.config_path else None,
            'config_sha256': config_sha,
            'git_head': git_head,
            'requested_max_epochs': self.requested_max_epochs,
            'start_epoch': self.start_epoch,
            'epochs_completed': epochs_done,
            'final_completed_epoch': final_ep,
            'total_epochs': self.requested_max_epochs,
            'nan_inf_detected': self.nan_inf_detected,
            'numerical_validity': 'PASS' if (not self.nan_inf_detected and not self.gradient_diagnostics_failed and epochs_done > 0) else 'FAIL',
            'seed': self.seed,
            'acloss_sha256': self.acloss_sha,
            'gradient_norm_diagnostics': self.gradient_norm_diagnostics,
            'gradient_diagnostics_failed': self.gradient_diagnostics_failed,
        }
        with open(os.path.join(self.output_dir, 'checkpoint_manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=2)

        # Write provenance
        prov = provenance_record(mode='dev', extra=manifest)
        with open(os.path.join(self.output_dir, 'provenance.json'), 'w') as f:
            json.dump(prov, f, indent=2)

        return manifest


def run_preflight_smoke(arm='B_dev', device=None):
    """Execute a 2-batch train + 2-batch val preflight smoke test on real data.

    Non-unit scientific mode: passes the canonical config FILE PATH (never an
    in-memory dict) so the runner's scientific fail-closed contract holds.
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg_file = 'config_dev_c4.json' if arm == 'C4' else 'config_dev_restored_baseline.json'
    cfg_path = os.path.join(ROOT, 'config_files', cfg_file)
    tmp_out = os.path.join(ROOT, 'research_runs', '_preflight_tmp', arm)
    runner = M2AnonymizerRunner(arm=arm, config=cfg_path, config_path=cfg_path, output_dir=tmp_out, device=device)

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
