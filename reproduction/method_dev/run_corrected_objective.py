"""Phase B — separability-targeting privacy objective.

Implements exactly the intervention preregistered in
`reproduction/reports/PHASE_B_CORRECTED_OBJECTIVE_PREREGISTRATION_2026-08-28.md`
(committed 96aacbe, BEFORE this file existed).

WHY
---
The certified privacy term is

    L_priv = -log(1 - sigmoid(z)) = softplus(z),   z = verifier logit

which is minimized by pushing every logit toward -inf. ROC AUC depends only on
the ORDER of z between positive and negative pairs, and a uniform shift
preserves order exactly, so L_priv can be driven to 0 with AUC untouched. This
is not speculative: `privacy_objective_diagnosis.json` (commit 09046f7) shows
the logged metric varying 9x across real checkpoints whose true AUC stays pinned
near 0.90, and shows the co-adapted critic re-identifying at 0.9147 where fresh
attackers reach only 0.7258.

THE ONE CHANGE (prereg §2)
--------------------------
For a batch with logits z_i and identity labels y_i, P = {i: y_i=1},
N = {j: y_j=0}:

    zhat       = (z - mean(z)) / (std(z) + eps)               # scale-invariant
    s          = mean_{i in P, j in N} sigmoid((zhat_i - zhat_j) / tau)   # tau = 0.1
    L_priv_new = (s - 0.5)^2

    L_gen = ac_loss_weight * L_AC_BCE + ver_loss_weight * L_priv_new

`s` estimates P(z_pos > z_neg), which IS ROC AUC. Targeting 0.5 targets
chance-level re-identification; the squared penalty makes 0.5 a stationary
point, so the objective does not push past chance into anti-correlation (AUC 0
re-identifies as well as AUC 1 and is explicitly not sought).

The verifier critic update is UNCHANGED -- still BCEWithLogitsLoss against true
identity labels, in the same position and order. Only what the generator asks of
the critic changes. Everything else is pinned to the frozen B_dev config
(prereg §3), including batch_size 16: upstream's 64 is a separate question and
changing two variables would make the result uninterpretable.

INTEGRITY (prereg §4) -- all seven requirements are implemented here:
  1. SHA assertions on config, initial generator, classifier, verifier.
  2. Per-epoch order hash COMPARED against an independent oracle
     (evaluator_common.compute_epoch_order_hash), not merely logged. The oracle
     was validated against the real seed-42 certified-recipe run: epochs
     0,1,2,5,13,50 all match.
  3. NaN/Inf fail-closed on every load-bearing loss, on gradients, and on
     post-step parameters.
  4. cudnn.deterministic = True and benchmark = False.
  5. Output under reproduction/method_dev/, method_uncertified: true.
  6. The co-adapted verifier is PERSISTED at the selected epoch and at the last
     epoch. The certified B_dev run did not do this, which is why diagnosis
     D4/D5 could only be measured on V2 arms; this closes that gap on a
     certified-recipe arm.
  7. `s` logged for train and val every epoch, with the degenerate-batch counter.

STRICTLY TRAIN/VALIDATION ONLY. No TEST loader is ever constructed.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
for _p in (ROOT, os.path.join(ROOT, 'research_agent')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from networks.UNet_PriCheXyNet import UNet  # noqa: E402
from networks.SiameseNetwork import SiameseNetwork  # noqa: E402
from utils.VerificationLoss import VerificationLoss  # noqa: E402

from m2_dev.evaluator_common import (  # noqa: E402
    IMAGE_SIZE,
    INITIAL_GENERATOR_PATH,
    INITIAL_GENERATOR_SHA,
    FROZEN_CLASSIFIER_PATH,
    FROZEN_CLASSIFIER_SHA,
    FROZEN_VERIFIER_PATH,
    FROZEN_VERIFIER_SHA,
    FROZEN_B_DEV_CONFIG_PATH,
    FROZEN_B_DEV_CONFIG_SHA,
    file_sha256,
    firewall_check,
    make_flow_field_components,
    verify_repaired_acloss,
    build_dev_anonymizer_loaders,
    compute_epoch_order_hash,
)

METHOD_OUT_ROOT = os.path.join(ROOT, 'reproduction', 'method_dev')
TRAIN_PAIR_FILE = os.path.join(ROOT, 'image_pairs', 'image_pairs_training_10000.txt')
PREREG = 'reproduction/reports/PHASE_B_CORRECTED_OBJECTIVE_PREREGISTRATION_2026-08-28.md'

# Frozen invariants from prereg §3. Any mismatch aborts: a run that silently
# differs from the preregistered configuration is worse than no run.
FROZEN_EXPECTED = {
    'mu': 0.01,
    'image_size': 256,
    'batch_size': 16,
    'accumulation_steps': 1,
    'learning_rate': 1e-4,
    'max_epochs': 250,
    'ac_loss_weight': 1.0,
    'ver_loss_weight': 1.0,
    'feature_loss_weight': 0.0,
}

# Prereg §8 validity + futility constants.
MAX_DEGENERATE_BATCH_RATE = 0.05
FUTILITY_EPOCH = 50
FUTILITY_MIN_IMPROVEMENT = 0.05

# Surrogate parameters -- fixed by ADDENDUM B (2026-08-28), before any run.
# Pre-flight testing showed the raw surrogate of prereg §2 is exploitable and
# biased; standardization + temperature fixes both. See Addendum B for the
# evidence table. These are NOT tunable within this preregistration.
SURROGATE_TAU = 0.1
SURROGATE_EPS = 1e-6


class CorrectedObjectiveRunner:
    def __init__(self, seed, output_dir, device):
        firewall_check('dev')
        if device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('Phase B requires CUDA')

        self.seed = seed
        self.device = device
        self.arm = 'B_dev'
        self.nan_inf_detected = False
        self.degenerate_batches = 0
        self.total_train_batches = 0
        self.futility_stop = False
        self.s_val_epoch0 = None
        self.best_abs_dev_from_half = float('inf')

        # ---- prereg §4.1: config SHA before use ----
        cfg_path = FROZEN_B_DEV_CONFIG_PATH
        actual_cfg_sha = file_sha256(cfg_path)
        if actual_cfg_sha != FROZEN_B_DEV_CONFIG_SHA:
            raise RuntimeError('Frozen B_dev config SHA mismatch: %s != %s'
                               % (actual_cfg_sha, FROZEN_B_DEV_CONFIG_SHA))
        with open(cfg_path) as f:
            self.config = json.load(f)
        self.config_path = cfg_path
        self.config_sha = actual_cfg_sha

        # ---- prereg §3: assert every frozen hyperparameter ----
        for key, expected in FROZEN_EXPECTED.items():
            actual = self.config.get(key)
            if actual is None or abs(float(actual) - float(expected)) > 1e-12:
                raise RuntimeError(
                    'Frozen invariant violated (prereg §3): %s = %r, expected %r'
                    % (key, actual, expected))

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.mu = self.config['mu']
        self.batch_size = self.config['batch_size']
        self.learning_rate = self.config['learning_rate']
        self.max_epochs = self.config['max_epochs']
        self.image_size = self.config.get('image_size', IMAGE_SIZE)
        self.ac_loss_weight = self.config['ac_loss_weight']
        self.ver_loss_weight = self.config['ver_loss_weight']
        # B_dev only; the feature term belongs to C4 and is not part of Phase B.
        self.feature_loss_weight = 0.0

        self._seed_all(self.seed)

        self.grid_identity, self.gauss_filter = make_flow_field_components(
            self.device, self.image_size)

        # ---- prereg §4.1: model SHA assertions ----
        self.generator = UNet(1, 2, 32).to(self.device)
        actual_gen_sha = file_sha256(INITIAL_GENERATOR_PATH)
        if actual_gen_sha != INITIAL_GENERATOR_SHA:
            raise RuntimeError('Initial generator SHA drift: %s != %s'
                               % (actual_gen_sha, INITIAL_GENERATOR_SHA))
        self.generator.load_state_dict(torch.load(
            INITIAL_GENERATOR_PATH, map_location=self.device, weights_only=False))
        self.initial_generator_sha = actual_gen_sha

        ACLossClass, self.acloss_sha, self.acloss_module_path = verify_repaired_acloss()
        actual_clf_sha = file_sha256(FROZEN_CLASSIFIER_PATH)
        if actual_clf_sha != FROZEN_CLASSIFIER_SHA:
            raise RuntimeError('Classifier SHA drift: %s != %s'
                               % (actual_clf_sha, FROZEN_CLASSIFIER_SHA))
        self.ac_model = torch.load(FROZEN_CLASSIFIER_PATH, map_location=self.device,
                                   weights_only=False)['model']
        self.ac_loss = ACLossClass(
            ac_model=self.ac_model, reduction='mean',
            pos_weight=self.config.get('ac_pos_weight', None),
            feature_loss_weight=self.feature_loss_weight,
        ).to(self.device)

        actual_ver_sha = file_sha256(FROZEN_VERIFIER_PATH)
        if actual_ver_sha != FROZEN_VERIFIER_SHA:
            raise RuntimeError('Verifier SHA drift: %s != %s'
                               % (actual_ver_sha, FROZEN_VERIFIER_SHA))
        self.verification_model = SiameseNetwork().to(self.device)
        self.verification_model.load_state_dict(torch.load(
            FROZEN_VERIFIER_PATH, map_location=self.device, weights_only=False))
        self.verification_loss = VerificationLoss(
            verification_model=self.verification_model, reduction='none').to(self.device)

        self.criterion_ac = nn.BCELoss().to(self.device)
        self.criterion_ver = nn.BCEWithLogitsLoss().to(self.device)

        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        self.optimizer_ver = optim.Adam(
            self.verification_loss.verification_model.parameters(), lr=self.learning_rate)
        self.optimizer_ac = optim.SGD(
            filter(lambda p: p.requires_grad, self.ac_loss.ac_model.parameters()),
            lr=self.learning_rate, momentum=0.9, weight_decay=1e-4,
        )

        self.training_loader, self.validation_loader, self.train_sampler = \
            build_dev_anonymizer_loaders(self.config, seed=self.seed, num_workers=0)

        self.resize_224 = transforms.Resize((224, 224))
        self.imagenet_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.epoch_metrics = []
        self.best_selection_total = float('inf')
        self.best_epoch = None
        self.best_checkpoint_path = os.path.join(
            self.output_dir, 'generator_lowest_total_loss.pth')
        self.best_verifier_path = os.path.join(
            self.output_dir, 'ver_model_trained_lowest_total_loss.pth')
        self.latest_checkpoint_path = os.path.join(self.output_dir, 'generator_latest.pth')
        self.latest_verifier_path = os.path.join(
            self.output_dir, 'ver_model_trained_latest.pth')

    def _seed_all(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # prereg §4.4 -- the V2 path set only benchmark=False.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def anonymize_tensor(self, image):
        grids = self.generator(image)
        grids = self.grid_identity - self.mu * grids
        grids = self.gauss_filter(grids)
        grids = grids.permute(0, 2, 3, 1)
        return F.grid_sample(image, grids, padding_mode='border', align_corners=True)

    # ------------------------------------------------------------------ #
    # THE INTERVENTION (prereg §2)                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def auc_surrogate(logits, labels_id):
        """Scale-invariant, low-bias differentiable estimator of ROC AUC.

            zhat = (z - mean(z)) / (std(z) + eps)
            s    = mean_{i in P, j in N} sigmoid((zhat_i - zhat_j) / tau)

        Returns None if either class is absent from the batch (prereg §2
        degenerate-batch guard).

        This is the Wilcoxon-Mann-Whitney estimator of P(z_pos > z_neg) -- i.e.
        of ROC AUC itself -- with the indicator replaced by a sigmoid.

        WHY STANDARDIZE, WHY A TEMPERATURE (Addendum B, verified before any run):
        the plain form `sigmoid(z_i - z_j)` reproduces the very defect this
        experiment exists to fix. It is not scale-invariant, so it can be driven
        to 0.5 by SHRINKING the logit scale while the true AUC is untouched --
        measured: at scale 0.03 the plain surrogate reads 0.523 while true AUC
        is 0.981. Standardization removes that degree of freedom. It is also
        biased at tau=1 (reads 0.81 when AUC is 0.98), and driving a biased
        estimate to 0.5 overshoots -- measured: true AUC lands at 0.425, which
        re-identifies just as well as 0.575. At tau=0.1 the estimator tracks
        true AUC to ~0.003 and optimization lands at chance (0.492).
        """
        z = logits.reshape(-1)
        y = labels_id.reshape(-1).type_as(z)
        if z.numel() < 2:
            return None
        z = (z - z.mean()) / (z.std() + SURROGATE_EPS)
        pos = z[y > 0.5]
        neg = z[y <= 0.5]
        if pos.numel() == 0 or neg.numel() == 0:
            return None
        return torch.sigmoid(
            (pos.unsqueeze(1) - neg.unsqueeze(0)) / SURROGATE_TAU).mean()

    # ------------------------------------------------------------------ #
    # Fail-closed numerics (prereg §4.3)                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _assert_finite(value, label):
        if not torch.isfinite(value).all():
            raise FloatingPointError('%s contains NaN/Inf' % label)

    @staticmethod
    def _assert_finite_grads(params, label):
        """Full coverage at scalar cost: gradient norms are non-negative, so a
        NaN or Inf anywhere propagates into the sum and cannot cancel."""
        total = None
        for p in params:
            if p.grad is None:
                continue
            n = p.grad.detach().norm()
            total = n if total is None else total + n
        if total is not None and not torch.isfinite(total):
            raise FloatingPointError('%s has non-finite gradient' % label)

    @staticmethod
    def _assert_finite_params(params, label):
        total = None
        for p in params:
            n = p.detach().norm()
            total = n if total is None else total + n
        if total is not None and not torch.isfinite(total):
            raise FloatingPointError('%s has non-finite parameter after step' % label)

    # ------------------------------------------------------------------ #
    # Epoch loops                                                          #
    # ------------------------------------------------------------------ #

    def train_epoch(self, epoch):
        self.generator.train()
        running = {'ac_bce': 0.0, 'ver_loss': 0.0, 'priv_new': 0.0,
                   's_surrogate': 0.0, 'opt_total': 0.0, 'sel_total': 0.0,
                   'ver_critic': 0.0}
        n_batches = 0
        n_s_batches = 0

        gen_params = [p for p in self.generator.parameters() if p.requires_grad]
        ver_params = list(self.verification_loss.verification_model.parameters())
        ac_params = [p for p in self.ac_loss.ac_model.parameters() if p.requires_grad]

        for batch in self.training_loader:
            inputs1, inputs2, labels, labels_id = batch
            inputs1, inputs2 = inputs1.to(self.device), inputs2.to(self.device)
            labels, labels_id = labels.to(self.device), labels_id.to(self.device)

            fakes_1 = self.anonymize_tensor(inputs1)

            # --- AC term: unchanged from the certified path ---
            self.ac_loss.refresh()
            deformed_features = self.ac_loss._features(self.ac_loss._preprocess(fakes_1))
            ac_predictions = self.ac_loss.loss_model.classifier(deformed_features)
            ac_bce_loss = self.ac_loss.bce_loss(ac_predictions, labels)
            ac_bce_val = ac_bce_loss.item()

            # --- Privacy term: THE ONLY CHANGE ---
            inputs1_snn_g = self.imagenet_normalize(fakes_1.expand(-1, 3, -1, -1))
            inputs2_snn_g = self.imagenet_normalize(inputs2.expand(-1, 3, -1, -1))
            raw_verifier_logits = self.verification_loss.verification_model(
                inputs1_snn_g, inputs2_snn_g).squeeze()

            s = self.auc_surrogate(raw_verifier_logits, labels_id)
            if s is None:
                # prereg §2 degenerate-batch guard
                self.degenerate_batches += 1
                priv_loss = torch.zeros((), device=self.device,
                                        dtype=raw_verifier_logits.dtype)
                s_item = float('nan')
            else:
                priv_loss = (s - 0.5) ** 2
                s_item = s.item()
                running['s_surrogate'] += s_item
                n_s_batches += 1

            # Logged for continuity with the certified path; NOT optimized.
            with torch.no_grad():
                ver_loss_mean = torch.sigmoid(
                    raw_verifier_logits.to(dtype=torch.float64)).mean()

            gen_loss = self.ac_loss_weight * ac_bce_loss + self.ver_loss_weight * priv_loss
            sel_loss_val = self.ac_loss_weight * ac_bce_val \
                + self.ver_loss_weight * priv_loss.item()

            self._assert_finite(gen_loss, 'generator loss')
            self.optimizer_g.zero_grad()
            gen_loss.backward()
            self._assert_finite_grads(gen_params, 'generator')
            self.optimizer_g.step()
            self._assert_finite_params(gen_params, 'generator')

            # --- Verifier critic: UNCHANGED (still BCE against true labels) ---
            self.verification_loss.verification_model.train()
            inputs1_snn = self.imagenet_normalize(fakes_1.detach().expand(-1, 3, -1, -1))
            inputs2_snn = self.imagenet_normalize(inputs2.expand(-1, 3, -1, -1))
            self.optimizer_ver.zero_grad()
            outputs_snn = self.verification_loss.verification_model(
                inputs1_snn, inputs2_snn).squeeze()
            labels_id_cast = labels_id.type_as(outputs_snn)
            loss_ver_critic = self.criterion_ver(outputs_snn, labels_id_cast)
            self._assert_finite(loss_ver_critic, 'verifier critic loss')
            loss_ver_critic.backward()
            self._assert_finite_grads(ver_params, 'verifier critic')
            self.optimizer_ver.step()
            self._assert_finite_params(ver_params, 'verifier critic')
            self.verification_loss.verification_model.eval()

            # --- AC critic: unchanged ---
            self.ac_loss.ac_model.train()
            inputs_ac = self.imagenet_normalize(
                self.resize_224(fakes_1.detach().expand(-1, 3, -1, -1)))
            self.optimizer_ac.zero_grad()
            outputs_ac = self.ac_loss.ac_model(inputs_ac)
            loss_ac_critic = self.criterion_ac(outputs_ac, labels)
            self._assert_finite(loss_ac_critic, 'auxiliary classifier critic loss')
            loss_ac_critic.backward()
            self._assert_finite_grads(ac_params, 'AC critic')
            self.optimizer_ac.step()
            self._assert_finite_params(ac_params, 'AC critic')
            self.ac_loss.ac_model.eval()

            running['ac_bce'] += ac_bce_val
            running['ver_loss'] += ver_loss_mean.item()
            running['priv_new'] += priv_loss.item()
            running['opt_total'] += gen_loss.item()
            running['sel_total'] += sel_loss_val
            running['ver_critic'] += loss_ver_critic.item()
            n_batches += 1
            self.total_train_batches += 1

        d = max(n_batches, 1)
        ds = max(n_s_batches, 1)
        return {
            'train_ac_bce': running['ac_bce'] / d,
            'train_ver_loss': running['ver_loss'] / d,
            'train_priv_new': running['priv_new'] / d,
            'train_s_surrogate': running['s_surrogate'] / ds,
            'train_optimization_total': running['opt_total'] / d,
            'train_selection_total': running['sel_total'] / d,
            'train_ver_critic_bce': running['ver_critic'] / d,
        }

    def validate_epoch(self, epoch):
        self.generator.eval()
        running = {'ac_bce': 0.0, 'ver_loss': 0.0, 's_surrogate': 0.0}
        n_batches = 0
        n_s_batches = 0
        pooled_logits, pooled_labels = [], []

        with torch.no_grad():
            for batch in self.validation_loader:
                inputs1, inputs2, labels, labels_id = batch
                inputs1, inputs2 = inputs1.to(self.device), inputs2.to(self.device)
                labels, labels_id = labels.to(self.device), labels_id.to(self.device)

                fakes_1 = self.anonymize_tensor(inputs1)
                ac_total_loss = self.ac_loss(fakes_1, labels)
                ac_bce_val = ac_total_loss.item()

                inputs1_snn_g = self.imagenet_normalize(fakes_1.expand(-1, 3, -1, -1))
                inputs2_snn_g = self.imagenet_normalize(inputs2.expand(-1, 3, -1, -1))
                z = self.verification_loss.verification_model(
                    inputs1_snn_g, inputs2_snn_g).squeeze()

                # Per-batch surrogate is retained ONLY as a diagnostic. It is
                # NOT usable on this fold: image_pairs_validation_2000.txt is
                # sorted by class (1000 positives then 1000 negatives) and the
                # VAL loader is sequential (shuffle=False), so exactly 1 of 125
                # batch-16 windows contains both classes. Anything averaged
                # per-batch here is computed from that single window.
                s = self.auc_surrogate(z, labels_id)
                if s is not None:
                    running['s_surrogate'] += s.item()
                    n_s_batches += 1

                ver_loss_mean = torch.sigmoid(z.to(dtype=torch.float64)).mean()

                pooled_logits += [float(v) for v in z.reshape(-1).float().cpu()]
                pooled_labels += [float(v) for v in labels_id.reshape(-1).float().cpu()]

                running['ac_bce'] += ac_bce_val
                running['ver_loss'] += ver_loss_mean.item()
                n_batches += 1

        d = max(n_batches, 1)
        ds = max(n_s_batches, 1)

        # POOLED over the whole VAL fold -- this is `val_L_priv_new` of prereg
        # §5 and the `s_val` of the §8 futility rule. Pooling is required, not
        # preferred: see the per-batch note above. It is also the honest
        # estimator of the quantity the objective targets.
        zt = torch.tensor(pooled_logits)
        yt = torch.tensor(pooled_labels)
        s_pooled = self.auc_surrogate(zt, yt)
        s_pooled = float(s_pooled) if s_pooled is not None else float('nan')
        priv_new_pooled = (s_pooled - 0.5) ** 2 if np.isfinite(s_pooled) else float('nan')
        try:
            # DIAGNOSTIC ONLY, never decisive: the co-adapted critic's true ROC
            # AUC on VAL. Decisions use the P0 adaptive attacker (prereg §6).
            true_auc = float(roc_auc_score(pooled_labels, pooled_logits))
        except ValueError:
            true_auc = float('nan')

        return {
            'val_ac_bce': running['ac_bce'] / d,
            'val_ver_loss': running['ver_loss'] / d,
            # prereg §5 selection inputs (pooled, see above)
            'val_priv_new': priv_new_pooled,
            'val_s_pooled': s_pooled,
            'val_selection_total': running['ac_bce'] / d + priv_new_pooled,
            # diagnostics, never decisive
            'val_s_surrogate_perbatch_DIAGNOSTIC': running['s_surrogate'] / ds,
            'val_n_batches_with_both_classes': n_s_batches,
            'val_coadapted_true_auc_DIAGNOSTIC': true_auc,
        }

    # ------------------------------------------------------------------ #

    def run(self, max_epochs):
        expected_hashes = {}
        for epoch in range(max_epochs):
            t0 = time.time()
            train_m = self.train_epoch(epoch)
            val_m = self.validate_epoch(epoch)
            elapsed = time.time() - t0
            peak_vram = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)

            # ---- prereg §4.2 / §8.3: COMPARE, do not merely log ----
            observed = self.train_sampler.get_epoch_order_hash(epoch)
            expected = compute_epoch_order_hash(TRAIN_PAIR_FILE, self.seed, epoch=epoch)
            expected_hashes[epoch] = expected
            if observed != expected:
                raise RuntimeError(
                    'Epoch %d TRAIN order hash mismatch (run INVALID per prereg §8.3): '
                    'observed %s != expected %s' % (epoch, observed, expected))

            # ---- prereg §8.2: degenerate-batch rate ----
            deg_rate = self.degenerate_batches / max(self.total_train_batches, 1)
            if deg_rate > MAX_DEGENERATE_BATCH_RATE:
                raise RuntimeError(
                    'Degenerate-batch rate %.4f exceeds %.2f (run INVALID per prereg §8.2)'
                    % (deg_rate, MAX_DEGENERATE_BATCH_RATE))

            all_vals = [v for v in list(train_m.values()) + list(val_m.values())
                        if isinstance(v, (int, float, np.floating, np.integer))]
            has_nan_inf = any(not np.isfinite(v) for v in all_vals)

            combined = {
                'epoch': epoch,
                'objective': 'standardized_auc_surrogate_tau%.2f_squared_deviation' % SURROGATE_TAU,
                'learning_rate': self.optimizer_g.param_groups[0]['lr'],
                'elapsed_sec': round(elapsed, 2),
                'peak_vram_mb': round(peak_vram, 2),
                'order_sha256': observed,
                'order_sha256_expected': expected,
                'order_hash_match': True,
                'degenerate_batches_cumulative': self.degenerate_batches,
                'degenerate_batch_rate': deg_rate,
                'is_nan_inf': has_nan_inf,
                **train_m, **val_m,
            }
            if has_nan_inf:
                self.nan_inf_detected = True
                raise FloatingPointError(
                    'Non-finite metric in epoch %d (run INVALID per prereg §8.1): %s'
                    % (epoch, combined))

            self.epoch_metrics.append(combined)
            with open(os.path.join(self.output_dir, 'train_log.jsonl'), 'a') as f_log:
                f_log.write(json.dumps(combined) + '\n')

            # ---- selection (prereg §5) ----
            val_sel = val_m['val_selection_total']
            if val_sel < self.best_selection_total - 1e-12:
                self.best_selection_total = val_sel
                self.best_epoch = epoch
                torch.save(self.generator.state_dict(), self.best_checkpoint_path)
                # prereg §4.6 -- persist the co-adapted verifier too.
                torch.save(self.verification_loss.verification_model.state_dict(),
                           self.best_verifier_path)

            # Secondary checkpoint (prereg §5): last epoch, evaluated but never
            # substituted for the selected one.
            torch.save(self.generator.state_dict(), self.latest_checkpoint_path)
            torch.save(self.verification_loss.verification_model.state_dict(),
                       self.latest_verifier_path)

            pd.DataFrame(self.epoch_metrics).to_csv(
                os.path.join(self.output_dir, 'epoch_metrics.csv'), index=False)

            # ---- prereg §8 futility ----
            s_pooled = val_m['val_s_pooled']
            if np.isfinite(s_pooled):
                dev = abs(s_pooled - 0.5)
                self.best_abs_dev_from_half = min(self.best_abs_dev_from_half, dev)
                if epoch == 0:
                    self.s_val_epoch0 = s_pooled
            print('Epoch %d/%d (%.1fs) sel=%.5f best_ep=%s | s_val=%.4f '
                  'coadapted_auc=%.4f (diag) ver_loss=%.4f'
                  % (epoch, max_epochs, elapsed, val_sel, self.best_epoch,
                     s_pooled, val_m['val_coadapted_true_auc_DIAGNOSTIC'],
                     val_m['val_ver_loss']))

            if epoch == FUTILITY_EPOCH and self.s_val_epoch0 is not None:
                improvement = abs(self.s_val_epoch0 - 0.5) - self.best_abs_dev_from_half
                if improvement < FUTILITY_MIN_IMPROVEMENT:
                    self.futility_stop = True
                    print('FUTILITY STOP (prereg §8): |s_val-0.5| improved only %.4f '
                          '(< %.2f) by epoch %d. Classification: H-B-MECHANISM-FAILED. '
                          'This does NOT authorize retuning within this preregistration.'
                          % (improvement, FUTILITY_MIN_IMPROVEMENT, FUTILITY_EPOCH))
                    break

        manifest = {
            'schema': 'PHASE_B_RUN_MANIFEST_V1',
            'method_uncertified': True,
            'test_split_accessed': False,
            'preregistration': PREREG,
            'arm': self.arm,
            'objective': 'L_priv = (mean_{i in P, j in N} sigmoid((zhat_i - zhat_j)/tau) - 0.5)^2',
            'surrogate_standardized': True,
            'surrogate_tau': SURROGATE_TAU,
            'surrogate_amendment': 'reproduction/reports/PHASE_B_ADDENDUM_B_2026-08-28.md',
            'seed': self.seed,
            'selected_generator_checkpoint': os.path.abspath(self.best_checkpoint_path),
            'selected_generator_sha256': (file_sha256(self.best_checkpoint_path)
                                          if os.path.exists(self.best_checkpoint_path) else None),
            'selected_verifier_checkpoint': os.path.abspath(self.best_verifier_path),
            'selected_verifier_sha256': (file_sha256(self.best_verifier_path)
                                         if os.path.exists(self.best_verifier_path) else None),
            'latest_generator_sha256': (file_sha256(self.latest_checkpoint_path)
                                        if os.path.exists(self.latest_checkpoint_path) else None),
            'latest_verifier_sha256': (file_sha256(self.latest_verifier_path)
                                       if os.path.exists(self.latest_verifier_path) else None),
            'best_epoch': self.best_epoch,
            'best_selection_total': self.best_selection_total,
            'epochs_completed': len(self.epoch_metrics),
            'futility_stop': self.futility_stop,
            'preregistered_classification_if_futility': (
                'H-B-MECHANISM-FAILED' if self.futility_stop else None),
            'degenerate_batches': self.degenerate_batches,
            'total_train_batches': self.total_train_batches,
            'degenerate_batch_rate': self.degenerate_batches / max(self.total_train_batches, 1),
            'nan_inf_detected': self.nan_inf_detected,
            'order_hashes_verified_against_oracle': True,
            'config_path': os.path.abspath(self.config_path),
            'config_sha256': self.config_sha,
            'initial_generator_sha256': self.initial_generator_sha,
            'acloss_module_sha256': self.acloss_sha,
            'frozen_invariants_asserted': FROZEN_EXPECTED,
            'torch': torch.__version__,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
        }
        with open(os.path.join(self.output_dir, 'checkpoint_manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=2)
        print('Finished. Best epoch: %s, best_selection_total: %.5f, futility_stop: %s'
              % (self.best_epoch, self.best_selection_total, self.futility_stop))
        return manifest


def main():
    parser = argparse.ArgumentParser(
        description='Phase B: separability-targeting privacy objective (preregistered)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_epochs', type=int, default=250)
    parser.add_argument('--tag', type=str, default='phaseB')
    parser.add_argument('--smoke_batches', type=int, default=0,
                        help='If > 0, run a short mechanism smoke test instead of training.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = os.path.join(METHOD_OUT_ROOT, 'corrected_objective_%s' % args.tag,
                           'B_dev', 'seed_%d' % args.seed)
    runner = CorrectedObjectiveRunner(args.seed, out_dir, device)
    if args.smoke_batches > 0:
        raise SystemExit('Use test_corrected_objective.py for smoke testing.')
    runner.run(max_epochs=args.max_epochs)


if __name__ == '__main__':
    main()
