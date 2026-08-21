"""Direction B pilot — hardened verifier critic (k:1 verifier:generator update ratio).

NOT the certified M2-S1 pipeline. Lives entirely outside research_agent/m2_dev/ and does
not import, modify, or touch any frozen config, lock, checkpoint, or result file.

Motivation (established from real Stage A pilot data, n=26 seeds, 2026-08-20): the
train-time verifier critic in M2AnonymizerRunner is co-adaptively trained (1 update per
generator update) and never converges against a *fixed* generator, whereas the S1 scientific
privacy metric is measured by an attacker trained to convergence (100 epochs, patience 5)
against the *final, frozen* generator. This asymmetry plausibly explains why measured Re-ID
AUC (B_dev mean 0.8237, n=26) is far higher than what train-time privacy_term would suggest.
This experiment tests whether giving the verifier k=3 updates per 1 generator update (instead
of 1:1) narrows that gap, i.e. produces a generator that is robust to a *harder* train-time
adversary and therefore measures lower under the real (converged) S1 attacker.

Fidelity policy (explicit user instruction — stay faithful to the certified project):
  - HardenedVerifierRunner.__init__ / train_epoch / validate_epoch / run below are a
    line-for-line copy of research_agent/m2_dev/anonymizer_runner.py's M2AnonymizerRunner,
    MINUS the scientific-mode locks that are specific to the certified S1 config path/seed
    (config path must be exactly config_dev_c4.json / config_dev_restored_baseline.json,
    seed must be exactly 42, no injected loaders). Config is still loaded read-only from the
    same frozen file S1 used (config_dev_restored_baseline.json for B_dev), with its SHA256
    asserted against FROZEN_B_DEV_CONFIG_SHA before use.
  - The ONLY behavioral change vs. the certified train_epoch: after the existing combined
    generator+verifier+AC-critic update (steps 1-5, unchanged, same batch, same order —
    generator epoch-level data traversal and update count is IDENTICAL to certified B_dev),
    K_EXTRA_VERIFIER_STEPS additional verifier-only update steps are run on the SAME batch,
    using a fresh no_grad forward of the just-updated generator (generator is not touched by
    these extra steps).
  - Output never lands under research_runs/M2_S1/ or touches any file whose hash is embedded
    in M2_S1_summary.json. All outputs are labeled "method_uncertified": true and live under
    reproduction/method_dev/.

STRICTLY TRAIN / VALIDATION ONLY — never constructs a TEST loader.
"""
import argparse
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
import torchvision.transforms as transforms
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
for _p in (ROOT, os.path.join(ROOT, 'research_agent')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from networks.UNet_PriCheXyNet import UNet  # noqa: E402
from networks.SiameseNetwork import SiameseNetwork  # noqa: E402
from utils.VerificationLoss import VerificationLoss  # noqa: E402

from m2_dev.evaluator_common import (  # noqa: E402
    MU,
    IMAGE_SIZE,
    INITIAL_GENERATOR_PATH,
    INITIAL_GENERATOR_SHA,
    FROZEN_CLASSIFIER_PATH,
    FROZEN_CLASSIFIER_SHA,
    FROZEN_VERIFIER_PATH,
    FROZEN_VERIFIER_SHA,
    file_sha256,
    firewall_check,
    make_flow_field_components,
    verify_repaired_acloss,
    build_dev_anonymizer_loaders,
    FROZEN_B_DEV_CONFIG_PATH,
    FROZEN_C4_CONFIG_PATH,
    FROZEN_B_DEV_CONFIG_SHA,
    FROZEN_C4_CONFIG_SHA,
)

METHOD_OUT_ROOT = os.path.join(ROOT, 'reproduction', 'method_dev')


class HardenedVerifierRunner:
    def __init__(self, arm, k_extra_verifier_steps, seed, output_dir, device):
        firewall_check('dev')
        if arm not in ('B_dev', 'C4'):
            raise ValueError("arm must be 'B_dev' or 'C4'")
        if device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('Hardened-verifier experiment requires CUDA')

        self.arm = arm
        self.k_extra_verifier_steps = int(k_extra_verifier_steps)
        self.seed = seed
        self.device = device
        self.nan_inf_detected = False

        # Config: read-only from the same frozen file the certified pipeline uses for this arm.
        cfg_path = FROZEN_C4_CONFIG_PATH if arm == 'C4' else FROZEN_B_DEV_CONFIG_PATH
        expected_cfg_sha = FROZEN_C4_CONFIG_SHA if arm == 'C4' else FROZEN_B_DEV_CONFIG_SHA
        actual_cfg_sha = file_sha256(cfg_path)
        if actual_cfg_sha != expected_cfg_sha:
            raise RuntimeError('Frozen anonymizer config SHA mismatch: %s != %s' % (actual_cfg_sha, expected_cfg_sha))
        with open(cfg_path) as f:
            self.config = json.load(f)
        self.config_path = cfg_path

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.mu = self.config.get('mu', 0.01)
        self.batch_size = self.config.get('batch_size', 16)
        self.learning_rate = self.config.get('learning_rate', 1e-4)
        self.max_epochs = self.config.get('max_epochs', 250)
        self.image_size = self.config.get('image_size', IMAGE_SIZE)
        self.ac_loss_weight = self.config.get('ac_loss_weight', 1.0)
        self.ver_loss_weight = self.config.get('ver_loss_weight', 1.0)
        self.feature_loss_weight = 1.0 if self.arm == 'C4' else 0.0

        self._seed_all(self.seed)

        self.grid_identity, self.gauss_filter = make_flow_field_components(self.device, self.image_size)

        self.generator = UNet(1, 2, 32).to(self.device)
        actual_gen_sha = file_sha256(INITIAL_GENERATOR_PATH)
        if actual_gen_sha != INITIAL_GENERATOR_SHA:
            raise RuntimeError('Initial generator SHA drift: %s != %s' % (actual_gen_sha, INITIAL_GENERATOR_SHA))
        self.generator.load_state_dict(torch.load(INITIAL_GENERATOR_PATH, map_location=self.device, weights_only=False))
        self.initial_generator_sha = actual_gen_sha

        ACLossClass, self.acloss_sha, self.acloss_module_path = verify_repaired_acloss()
        actual_clf_sha = file_sha256(FROZEN_CLASSIFIER_PATH)
        if actual_clf_sha != FROZEN_CLASSIFIER_SHA:
            raise RuntimeError('Classifier SHA drift: %s != %s' % (actual_clf_sha, FROZEN_CLASSIFIER_SHA))
        self.ac_model = torch.load(FROZEN_CLASSIFIER_PATH, map_location=self.device, weights_only=False)['model']

        self.ac_loss = ACLossClass(
            ac_model=self.ac_model, reduction='mean',
            pos_weight=self.config.get('ac_pos_weight', None),
            feature_loss_weight=self.feature_loss_weight,
        ).to(self.device)

        actual_ver_sha = file_sha256(FROZEN_VERIFIER_PATH)
        if actual_ver_sha != FROZEN_VERIFIER_SHA:
            raise RuntimeError('Verifier SHA drift: %s != %s' % (actual_ver_sha, FROZEN_VERIFIER_SHA))
        self.verification_model = SiameseNetwork().to(self.device)
        self.verification_model.load_state_dict(torch.load(FROZEN_VERIFIER_PATH, map_location=self.device, weights_only=False))
        self.verification_loss = VerificationLoss(verification_model=self.verification_model, reduction='none').to(self.device)

        self.criterion_ac = nn.BCELoss().to(self.device)
        self.criterion_ver = nn.BCEWithLogitsLoss().to(self.device)

        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        self.optimizer_ver = optim.Adam(self.verification_loss.verification_model.parameters(), lr=self.learning_rate)
        self.optimizer_ac = optim.SGD(
            filter(lambda p: p.requires_grad, self.ac_loss.ac_model.parameters()),
            lr=self.learning_rate, momentum=0.9, weight_decay=1e-4,
        )

        self.training_loader, self.validation_loader, self.train_sampler = build_dev_anonymizer_loaders(
            self.config, seed=self.seed, num_workers=0
        )

        self.resize_224 = transforms.Resize((224, 224))
        self.imagenet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.epoch_metrics = []
        self.best_selection_total = float('inf')
        self.best_epoch = None
        self.best_checkpoint_path = os.path.join(self.output_dir, 'generator_best_method_neutral.pth')

    def _seed_all(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def anonymize_tensor(self, image):
        grids = self.generator(image)
        grids = self.grid_identity - self.mu * grids
        grids = self.gauss_filter(grids)
        grids = grids.permute(0, 2, 3, 1)
        return F.grid_sample(image, grids, padding_mode='border', align_corners=True)

    @staticmethod
    def _assert_finite(value, label):
        if not torch.isfinite(value).all():
            raise FloatingPointError('%s contains NaN/Inf' % label)

    def train_epoch(self, epoch):
        self.generator.train()
        running = {'ac_bce': 0.0, 'ver_loss': 0.0, 'privacy_term': 0.0, 'feature_term': 0.0,
                   'opt_total': 0.0, 'sel_total': 0.0, 'extra_ver_loss': 0.0}
        n_batches = 0

        for batch in self.training_loader:
            inputs1, inputs2, labels, labels_id = batch
            inputs1, inputs2 = inputs1.to(self.device), inputs2.to(self.device)
            labels, labels_id = labels.to(self.device), labels_id.to(self.device)

            # --- unchanged certified combined step (identical to M2AnonymizerRunner) ---
            fakes_1 = self.anonymize_tensor(inputs1)

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
                feat_val = 0.0
            ac_bce_val = ac_bce_loss.item()

            inputs1_snn_g = self.imagenet_normalize(fakes_1.expand(-1, 3, -1, -1))
            inputs2_snn_g = self.imagenet_normalize(inputs2.expand(-1, 3, -1, -1))
            raw_verifier_logits = self.verification_loss.verification_model(inputs1_snn_g, inputs2_snn_g).squeeze()
            privacy_term = F.softplus(raw_verifier_logits).mean()
            with torch.no_grad():
                ver_loss_mean = torch.sigmoid(raw_verifier_logits.to(dtype=torch.float64)).mean()

            gen_loss = self.ac_loss_weight * ac_total_loss + self.ver_loss_weight * privacy_term
            sel_loss_val = self.ac_loss_weight * ac_bce_val + self.ver_loss_weight * privacy_term.item()

            self._assert_finite(gen_loss, 'generator loss')
            self.optimizer_g.zero_grad()
            gen_loss.backward()
            self.optimizer_g.step()

            self.verification_loss.verification_model.train()
            inputs1_snn = self.imagenet_normalize(fakes_1.detach().expand(-1, 3, -1, -1))
            inputs2_snn = self.imagenet_normalize(inputs2.expand(-1, 3, -1, -1))
            self.optimizer_ver.zero_grad()
            outputs_snn = self.verification_loss.verification_model(inputs1_snn, inputs2_snn).squeeze()
            labels_id_cast = labels_id.type_as(outputs_snn)
            loss_ver_critic = self.criterion_ver(outputs_snn, labels_id_cast)
            self._assert_finite(loss_ver_critic, 'verifier critic loss')
            loss_ver_critic.backward()
            self.optimizer_ver.step()
            self.verification_loss.verification_model.eval()

            self.ac_loss.ac_model.train()
            inputs_ac = self.imagenet_normalize(self.resize_224(fakes_1.detach().expand(-1, 3, -1, -1)))
            self.optimizer_ac.zero_grad()
            outputs_ac = self.ac_loss.ac_model(inputs_ac)
            loss_ac_critic = self.criterion_ac(outputs_ac, labels)
            self._assert_finite(loss_ac_critic, 'auxiliary classifier critic loss')
            loss_ac_critic.backward()
            self.optimizer_ac.step()
            self.ac_loss.ac_model.eval()
            # --- end unchanged certified combined step ---

            # --- Direction B addition: k_extra verifier-only steps, same batch,
            #     fresh no_grad forward of the just-updated generator. ---
            extra_ver_loss_sum = 0.0
            for _ in range(self.k_extra_verifier_steps):
                with torch.no_grad():
                    fakes_1_extra = self.anonymize_tensor(inputs1)
                inputs1_snn_e = self.imagenet_normalize(fakes_1_extra.expand(-1, 3, -1, -1))
                self.verification_loss.verification_model.train()
                self.optimizer_ver.zero_grad()
                outputs_extra = self.verification_loss.verification_model(inputs1_snn_e, inputs2_snn).squeeze()
                loss_extra = self.criterion_ver(outputs_extra, labels_id_cast)
                self._assert_finite(loss_extra, 'extra verifier critic loss')
                loss_extra.backward()
                self.optimizer_ver.step()
                self.verification_loss.verification_model.eval()
                extra_ver_loss_sum += loss_extra.item()

            running['ac_bce'] += ac_bce_val
            running['ver_loss'] += ver_loss_mean.item()
            running['privacy_term'] += privacy_term.item()
            running['feature_term'] += feat_val
            running['opt_total'] += gen_loss.item()
            running['sel_total'] += sel_loss_val
            running['extra_ver_loss'] += extra_ver_loss_sum / max(self.k_extra_verifier_steps, 1)
            n_batches += 1

        denom = max(n_batches, 1)
        return {
            'train_ac_bce': running['ac_bce'] / denom,
            'train_ver_loss': running['ver_loss'] / denom,
            'train_privacy_term': running['privacy_term'] / denom,
            'train_feature_term': running['feature_term'] / denom,
            'train_optimization_total': running['opt_total'] / denom,
            'train_selection_total': running['sel_total'] / denom,
            'train_extra_ver_loss': running['extra_ver_loss'] / denom,
        }

    def validate_epoch(self, epoch):
        self.generator.eval()
        running = {'ac_bce': 0.0, 'ver_loss': 0.0, 'privacy_term': 0.0, 'feature_term': 0.0,
                   'opt_total': 0.0, 'sel_total': 0.0}
        n_batches = 0
        with torch.no_grad():
            for batch in self.validation_loader:
                inputs1, inputs2, labels, _ = batch
                inputs1, inputs2 = inputs1.to(self.device), inputs2.to(self.device)
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

                running['ac_bce'] += ac_bce_val
                running['ver_loss'] += ver_loss_mean.item()
                running['privacy_term'] += privacy_term.item()
                running['feature_term'] += feat_val
                running['opt_total'] += opt_total
                running['sel_total'] += sel_total
                n_batches += 1

        denom = max(n_batches, 1)
        return {
            'val_ac_bce': running['ac_bce'] / denom,
            'val_ver_loss': running['ver_loss'] / denom,
            'val_privacy_term': running['privacy_term'] / denom,
            'val_feature_term': running['feature_term'] / denom,
            'val_optimization_total': running['opt_total'] / denom,
            'val_selection_total': running['sel_total'] / denom,
        }

    def run(self, max_epochs):
        for epoch in range(max_epochs):
            t0 = time.time()
            train_m = self.train_epoch(epoch)
            val_m = self.validate_epoch(epoch)
            elapsed = time.time() - t0
            peak_vram = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
            order_sha = self.train_sampler.get_epoch_order_hash(epoch) if self.train_sampler is not None else None

            all_vals = list(train_m.values()) + list(val_m.values())
            has_nan_inf = any(not np.isfinite(v) for v in all_vals if isinstance(v, (int, float, np.floating, np.integer)))

            combined = {
                'epoch': epoch, 'k_extra_verifier_steps': self.k_extra_verifier_steps,
                'learning_rate': self.optimizer_g.param_groups[0]['lr'],
                'elapsed_sec': round(elapsed, 2), 'peak_vram_mb': round(peak_vram, 2),
                'order_sha256': order_sha, 'is_nan_inf': has_nan_inf,
                **train_m, **val_m,
            }
            if has_nan_inf:
                self.nan_inf_detected = True
                raise FloatingPointError('Non-finite loss detected in epoch %d (arm %s): %s' % (epoch, self.arm, combined))

            self.epoch_metrics.append(combined)
            log_path = os.path.join(self.output_dir, 'train_log.jsonl')
            with open(log_path, 'a') as f_log:
                f_log.write(json.dumps(combined) + '\n')

            val_sel = val_m['val_selection_total']
            if val_sel < self.best_selection_total - 1e-12:
                self.best_selection_total = val_sel
                self.best_epoch = epoch
                torch.save(self.generator.state_dict(), self.best_checkpoint_path)

            pd.DataFrame(self.epoch_metrics).to_csv(os.path.join(self.output_dir, 'epoch_metrics.csv'), index=False)
            print('Epoch %d/%d done (%.1fs) val_sel_total=%.5f best_epoch=%s' % (
                epoch, max_epochs, elapsed, val_sel, self.best_epoch))

        best_sha = file_sha256(self.best_checkpoint_path) if os.path.exists(self.best_checkpoint_path) else None
        manifest = {
            'method_uncertified': True,
            'arm': self.arm,
            'k_extra_verifier_steps': self.k_extra_verifier_steps,
            'selected_generator_checkpoint': os.path.abspath(self.best_checkpoint_path),
            'selected_generator_sha256': best_sha,
            'best_epoch': self.best_epoch,
            'best_selection_total': self.best_selection_total,
            'initial_generator_sha256': self.initial_generator_sha,
            'config_path': os.path.abspath(self.config_path),
            'seed': self.seed,
            'epochs_completed': len(self.epoch_metrics),
            'nan_inf_detected': self.nan_inf_detected,
        }
        with open(os.path.join(self.output_dir, 'checkpoint_manifest.json'), 'w') as f:
            json.dump(manifest, f, indent=2)
        print('Finished. Best epoch: %s, best_selection_total: %.5f' % (self.best_epoch, self.best_selection_total))
        return manifest


def main():
    parser = argparse.ArgumentParser(description='Direction B: hardened verifier critic pilot')
    parser.add_argument('--arm', choices=['B_dev', 'C4'], default='B_dev')
    parser.add_argument('--k_extra_verifier_steps', type=int, default=2, help='k-1 extra steps (k=3 total -> pass 2)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_epochs', type=int, default=250)
    parser.add_argument('--tag', type=str, default='k3')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError('Requires CUDA')

    out_dir = os.path.join(METHOD_OUT_ROOT, 'hardened_verifier_%s' % args.tag, args.arm, 'seed_%d' % args.seed)
    runner = HardenedVerifierRunner(args.arm, args.k_extra_verifier_steps, args.seed, out_dir, device)
    runner.run(max_epochs=args.max_epochs)


if __name__ == '__main__':
    main()
