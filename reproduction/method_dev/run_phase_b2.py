"""Phase B2 — separability-targeting objective at the upstream recipe.

Implements exactly what is preregistered in
`reproduction/reports/PHASE_B2_PREREGISTRATION_2026-08-29.md` (commit 3d2b7a1)
as clarified by `PHASE_B2_ADDENDUM_A_2026-08-29.md` (commit 0e2004f). Both were
locked before this file existed.

A NEW FILE, deliberately. `run_corrected_objective.py` is the Phase B artifact
referenced by Phase B Addendum C; mutating it would make that closed record
irreproducible.

THE OBJECTIVE (prereg §2), chosen by measurement, not argument:

    zhat   = (z - mean(z)) / (std(z) + 1e-6)
    L_priv = ( mean_pos(zhat) - mean_neg(zhat) )^2

At the near-perfect-separation state the generator actually starts in this has
||dL/dz|| = 3.38e-01, against 1.76e-06 for the Phase B surrogate that stalled --
9.5x the ORIGINAL softplus objective rather than a fraction of it. Standardized,
so the shrink-the-logit-scale loophole is closed; non-saturating, which is the
specific defect that killed Phase B; zero hyperparameters.

THREE DECLARED OVERRIDES of the frozen config (prereg §3), each logged into the
manifest with a pointer to the preregistration. No further override is permitted:

    accumulation_steps  1  -> 4        effective batch 64, matching the paper
    ver_loss_weight     1.0 -> CLI     arm A = 1.0, arm B = 3.0, both declared
    initial generator   10122689 -> 4d82dcdd   start FROM the released model

STRICTLY TRAIN/VALIDATION ONLY. No TEST loader is ever constructed.
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
    IMAGE_SIZE, FROZEN_CLASSIFIER_PATH, FROZEN_CLASSIFIER_SHA,
    FROZEN_VERIFIER_PATH, FROZEN_VERIFIER_SHA,
    FROZEN_B_DEV_CONFIG_PATH, FROZEN_B_DEV_CONFIG_SHA,
    file_sha256, firewall_check, make_flow_field_components,
    verify_repaired_acloss, build_dev_anonymizer_loaders, compute_epoch_order_hash,
)

METHOD_OUT_ROOT = os.path.join(ROOT, 'reproduction', 'method_dev')
TRAIN_PAIR_FILE = os.path.join(ROOT, 'image_pairs', 'image_pairs_training_10000.txt')
PREREG = 'reproduction/reports/PHASE_B2_PREREGISTRATION_2026-08-29.md'
ADDENDUM_A = 'reproduction/reports/PHASE_B2_ADDENDUM_A_2026-08-29.md'

# The released endpoint: anchor AND initialization (prereg §3).
U_PUBLISHED_PATH = os.path.join(ROOT, 'networks', 'generator_lowest_total_loss_mu_0.01.pth')
U_PUBLISHED_SHA = '4d82dcdd1c1b5856d6361fd08b7a6838b044ffc7db89e8bf953a6279cb3bf153'

ACCUMULATION_STEPS = 4          # override 1: effective batch 64
MAX_EPOCHS = 50                 # prereg §3
FUTILITY_EPOCH = 25             # prereg §7, per-arm (Addendum A §2)
FUTILITY_MIN_FRACTIONAL_DROP = 0.25
TRUNCATION_FLAG_EPOCH = 45      # prereg §3: argmin at/after this is flagged
SURROGATE_EPS = 1e-6
# Values that must match the frozen config exactly; the three overrides above
# are the ONLY permitted deviations.
FROZEN_EXPECTED = {'mu': 0.01, 'image_size': 256, 'batch_size': 16,
                   'learning_rate': 1e-4, 'ac_loss_weight': 1.0,
                   'feature_loss_weight': 0.0}


class PhaseB2Runner:
    def __init__(self, ver_loss_weight, seed, output_dir, device):
        firewall_check('dev')
        if device.type != 'cuda':
            raise RuntimeError('Phase B2 requires CUDA')
        self.device, self.seed = device, seed
        self.ver_loss_weight = float(ver_loss_weight)
        self.degenerate_batches = self.total_micro_batches = 0
        self.futility_stop = False
        self.gap_epoch0 = None
        self.best_gap = float('inf')

        cfg_sha = file_sha256(FROZEN_B_DEV_CONFIG_PATH)
        if cfg_sha != FROZEN_B_DEV_CONFIG_SHA:
            raise RuntimeError('Frozen config SHA mismatch: %s' % cfg_sha)
        self.config = json.load(open(FROZEN_B_DEV_CONFIG_PATH))
        self.config_sha = cfg_sha
        for k, v in FROZEN_EXPECTED.items():
            if abs(float(self.config.get(k, 1e9)) - float(v)) > 1e-12:
                raise RuntimeError('Frozen invariant violated: %s=%r != %r'
                                   % (k, self.config.get(k), v))

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.mu = self.config['mu']
        self.image_size = self.config.get('image_size', IMAGE_SIZE)
        self.learning_rate = self.config['learning_rate']
        self.ac_loss_weight = self.config['ac_loss_weight']

        self._seed_all(seed)
        self.grid_identity, self.gauss_filter = make_flow_field_components(
            device, self.image_size)

        # override 3: initialize FROM the released endpoint, SHA-verified.
        gen_sha = file_sha256(U_PUBLISHED_PATH)
        if gen_sha != U_PUBLISHED_SHA:
            raise RuntimeError('U_PUBLISHED SHA drift: %s' % gen_sha)
        self.generator = UNet(1, 2, 32).to(device)
        self.generator.load_state_dict(torch.load(U_PUBLISHED_PATH,
                                                  map_location=device,
                                                  weights_only=False))
        self.initial_generator_sha = gen_sha

        ACLossClass, self.acloss_sha, _ = verify_repaired_acloss()
        clf_sha = file_sha256(FROZEN_CLASSIFIER_PATH)
        if clf_sha != FROZEN_CLASSIFIER_SHA:
            raise RuntimeError('Classifier SHA drift: %s' % clf_sha)
        self.ac_model = torch.load(FROZEN_CLASSIFIER_PATH, map_location=device,
                                   weights_only=False)['model']
        self.ac_loss = ACLossClass(ac_model=self.ac_model, reduction='mean',
                                   pos_weight=None, feature_loss_weight=0.0).to(device)

        ver_sha = file_sha256(FROZEN_VERIFIER_PATH)
        if ver_sha != FROZEN_VERIFIER_SHA:
            raise RuntimeError('Verifier SHA drift: %s' % ver_sha)
        self.verification_model = SiameseNetwork().to(device)
        self.verification_model.load_state_dict(
            torch.load(FROZEN_VERIFIER_PATH, map_location=device, weights_only=False))
        self.verification_loss = VerificationLoss(
            verification_model=self.verification_model, reduction='none').to(device)

        self.criterion_ac = nn.BCELoss().to(device)
        self.criterion_ver = nn.BCEWithLogitsLoss().to(device)
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        self.optimizer_ver = optim.Adam(
            self.verification_loss.verification_model.parameters(), lr=self.learning_rate)
        self.optimizer_ac = optim.SGD(
            filter(lambda p: p.requires_grad, self.ac_loss.ac_model.parameters()),
            lr=self.learning_rate, momentum=0.9, weight_decay=1e-4)

        self.training_loader, self.validation_loader, self.train_sampler = \
            build_dev_anonymizer_loaders(self.config, seed=seed, num_workers=0)
        self.resize_224 = transforms.Resize((224, 224))
        self.imagenet_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.epoch_metrics = []
        self.best_selection_total = float('inf')
        self.best_epoch = None
        self.p_best_gen = os.path.join(output_dir, 'generator_lowest_total_loss.pth')
        self.p_best_ver = os.path.join(output_dir, 'ver_model_trained_lowest_total_loss.pth')
        self.p_last_gen = os.path.join(output_dir, 'generator_latest.pth')
        self.p_last_ver = os.path.join(output_dir, 'ver_model_trained_latest.pth')

    def _seed_all(self, s):
        torch.manual_seed(s); torch.cuda.manual_seed_all(s); torch.cuda.manual_seed(s)
        np.random.seed(s); random.seed(s)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def anonymize_tensor(self, image):
        g = self.generator(image)
        g = self.grid_identity - self.mu * g
        g = self.gauss_filter(g).permute(0, 2, 3, 1)
        return F.grid_sample(image, g, padding_mode='border', align_corners=True)

    # ---------------- the intervention (prereg §2) ----------------
    @staticmethod
    def standardized_gap(logits, labels_id):
        """(mean_pos(zhat) - mean_neg(zhat)); None if either class is absent.

        Standardization removes the scale degree of freedom, so the objective
        cannot be satisfied by shrinking the logits -- the calibration-not-
        separability failure this whole project indicts. The squared value is
        the loss; the signed gap is returned so its trajectory is loggable.
        """
        z = logits.reshape(-1)
        y = labels_id.reshape(-1).type_as(z)
        if z.numel() < 2:
            return None
        z = (z - z.mean()) / (z.std() + SURROGATE_EPS)
        pos, neg = z[y > 0.5], z[y <= 0.5]
        if pos.numel() == 0 or neg.numel() == 0:
            return None
        return pos.mean() - neg.mean()

    @staticmethod
    def _finite(v, label):
        if not torch.isfinite(v).all():
            raise FloatingPointError('%s contains NaN/Inf' % label)

    @staticmethod
    def _finite_over(params, label, grad=False):
        tot = None
        for p in params:
            t = p.grad if grad else p
            if t is None:
                continue
            n = t.detach().norm()
            tot = n if tot is None else tot + n
        if tot is not None and not torch.isfinite(tot):
            raise FloatingPointError('%s non-finite (%s)' % (label, 'grad' if grad else 'param'))

    def train_epoch(self, epoch):
        self.generator.train()
        run = {k: 0.0 for k in ('ac', 'gap', 'priv', 'ver_loss', 'total', 'ver_bce')}
        nb = ngap = 0
        gpar = [p for p in self.generator.parameters() if p.requires_grad]
        vpar = list(self.verification_loss.verification_model.parameters())
        apar = [p for p in self.ac_loss.ac_model.parameters() if p.requires_grad]
        n_batches = len(self.training_loader)
        A = ACCUMULATION_STEPS

        for i, batch in enumerate(self.training_loader):
            x1, x2, labels, lid = [t.to(self.device) for t in batch]
            window_start = (i % A == 0)
            boundary = ((i + 1) % A == 0) or (i + 1 == n_batches)

            # GATE (prereg §4): the AC critic must be in eval() whenever the
            # generator loss is computed. Defect 4 of commit 10d9212 only fires
            # when accumulation > 1, which is exactly this configuration.
            if self.ac_loss.ac_model.training:
                raise RuntimeError('AC critic in train() at generator-loss time '
                                   '(epoch %d, micro-batch %d)' % (epoch, i))

            fakes = self.anonymize_tensor(x1)
            self.ac_loss.refresh()
            feats = self.ac_loss._features(self.ac_loss._preprocess(fakes))
            ac_loss = self.ac_loss.bce_loss(self.ac_loss.loss_model.classifier(feats), labels)

            z = self.verification_loss.verification_model(
                self.imagenet_normalize(fakes.expand(-1, 3, -1, -1)),
                self.imagenet_normalize(x2.expand(-1, 3, -1, -1))).squeeze()
            gap = self.standardized_gap(z, lid)
            if gap is None:
                self.degenerate_batches += 1
                priv = torch.zeros((), device=self.device, dtype=z.dtype)
            else:
                priv = gap ** 2
                run['gap'] += float(gap); ngap += 1
            with torch.no_grad():
                run['ver_loss'] += float(torch.sigmoid(z.to(torch.float64)).mean())

            gen_loss = self.ac_loss_weight * ac_loss + self.ver_loss_weight * priv
            self._finite(gen_loss, 'generator loss')
            if window_start:
                self.optimizer_g.zero_grad()
            (gen_loss / A).backward()
            if boundary:
                self._finite_over(gpar, 'generator', grad=True)
                self.optimizer_g.step()
                self._finite_over(gpar, 'generator')

            # --- verifier critic: UNCHANGED objective, accumulated ---
            self.verification_loss.verification_model.train()
            if window_start:
                self.optimizer_ver.zero_grad()
            out_v = self.verification_loss.verification_model(
                self.imagenet_normalize(fakes.detach().expand(-1, 3, -1, -1)),
                self.imagenet_normalize(x2.expand(-1, 3, -1, -1))).squeeze()
            lv = self.criterion_ver(out_v, lid.type_as(out_v))
            self._finite(lv, 'verifier critic loss')
            (lv / A).backward()
            if boundary:
                self._finite_over(vpar, 'verifier critic', grad=True)
                self.optimizer_ver.step()
            self.verification_loss.verification_model.eval()

            # --- AC critic: eval() restored EVERY micro-batch (defect 4) ---
            self.ac_loss.ac_model.train()
            if window_start:
                self.optimizer_ac.zero_grad()
            la = self.criterion_ac(self.ac_loss.ac_model(self.imagenet_normalize(
                self.resize_224(fakes.detach().expand(-1, 3, -1, -1)))), labels)
            self._finite(la, 'AC critic loss')
            (la / A).backward()
            if boundary:
                self._finite_over(apar, 'AC critic', grad=True)
                self.optimizer_ac.step()
            self.ac_loss.ac_model.eval()

            run['ac'] += float(ac_loss); run['priv'] += float(priv)
            run['total'] += float(gen_loss); run['ver_bce'] += float(lv)
            nb += 1; self.total_micro_batches += 1

        d, dg = max(nb, 1), max(ngap, 1)
        return {'train_ac_bce': run['ac'] / d, 'train_gap': run['gap'] / dg,
                'train_priv': run['priv'] / d, 'train_ver_loss': run['ver_loss'] / d,
                'train_total': run['total'] / d, 'train_ver_critic_bce': run['ver_bce'] / d}

    def validate_epoch(self, epoch):
        self.generator.eval()
        ac_sum = vl_sum = 0.0; nb = 0
        zs, ys = [], []
        with torch.no_grad():
            for batch in self.validation_loader:
                x1, x2, labels, lid = [t.to(self.device) for t in batch]
                fakes = self.anonymize_tensor(x1)
                ac_sum += float(self.ac_loss(fakes, labels))
                z = self.verification_loss.verification_model(
                    self.imagenet_normalize(fakes.expand(-1, 3, -1, -1)),
                    self.imagenet_normalize(x2.expand(-1, 3, -1, -1))).squeeze()
                vl_sum += float(torch.sigmoid(z.to(torch.float64)).mean())
                zs += [float(v) for v in z.reshape(-1).float().cpu()]
                ys += [float(v) for v in lid.reshape(-1).float().cpu()]
                nb += 1
        d = max(nb, 1)
        # POOLED over the whole VAL fold. The VAL pair file is class-sorted and
        # the loader is sequential, so only 1 of 125 batch-16 windows holds both
        # classes -- a per-batch mean here would be computed from that one
        # window (Phase B Addendum B §3b).
        gap = self.standardized_gap(torch.tensor(zs), torch.tensor(ys))
        gap = float(gap) if gap is not None else float('nan')
        priv = gap ** 2 if np.isfinite(gap) else float('nan')
        try:
            auc = float(roc_auc_score(ys, zs))
        except ValueError:
            auc = float('nan')
        return {'val_ac_bce': ac_sum / d, 'val_ver_loss': vl_sum / d,
                'val_gap': gap, 'val_priv': priv,
                'val_selection_total': ac_sum / d + priv,
                'val_coadapted_true_auc_DIAGNOSTIC': auc}

    def run(self, max_epochs=MAX_EPOCHS):
        for epoch in range(max_epochs):
            t0 = time.time()
            tm = self.train_epoch(epoch); vm = self.validate_epoch(epoch)
            obs = self.train_sampler.get_epoch_order_hash(epoch)
            exp = compute_epoch_order_hash(TRAIN_PAIR_FILE, self.seed, epoch=epoch)
            if obs != exp:
                raise RuntimeError('epoch %d order hash mismatch (run INVALID)' % epoch)
            deg = self.degenerate_batches / max(self.total_micro_batches, 1)
            if deg > 0.05:
                raise RuntimeError('degenerate-batch rate %.4f > 0.05 (INVALID)' % deg)

            rec = {'epoch': epoch, 'ver_loss_weight': self.ver_loss_weight,
                   'accumulation_steps': ACCUMULATION_STEPS,
                   'elapsed_sec': round(time.time() - t0, 2),
                   'order_sha256': obs, 'order_hash_match': True,
                   'degenerate_batch_rate': deg, **tm, **vm}
            vals = [v for v in rec.values() if isinstance(v, (int, float, np.floating))]
            if any(not np.isfinite(v) for v in vals):
                raise FloatingPointError('non-finite metric epoch %d (INVALID)' % epoch)
            self.epoch_metrics.append(rec)
            with open(os.path.join(self.output_dir, 'train_log.jsonl'), 'a') as f:
                f.write(json.dumps(rec) + '\n')

            if vm['val_selection_total'] < self.best_selection_total - 1e-12:
                self.best_selection_total = vm['val_selection_total']
                self.best_epoch = epoch
                torch.save(self.generator.state_dict(), self.p_best_gen)
                torch.save(self.verification_loss.verification_model.state_dict(), self.p_best_ver)
            torch.save(self.generator.state_dict(), self.p_last_gen)
            torch.save(self.verification_loss.verification_model.state_dict(), self.p_last_ver)
            pd.DataFrame(self.epoch_metrics).to_csv(
                os.path.join(self.output_dir, 'epoch_metrics.csv'), index=False)

            g = abs(vm['val_gap'])
            if np.isfinite(g):
                if epoch == 0:
                    self.gap_epoch0 = g
                self.best_gap = min(self.best_gap, g)
            print('ep %d/%d (%.0fs) w=%.1f sel=%.5f best=%s | gap=%.4f coadapt_auc=%.4f'
                  % (epoch, max_epochs, rec['elapsed_sec'], self.ver_loss_weight,
                     vm['val_selection_total'], self.best_epoch, vm['val_gap'],
                     vm['val_coadapted_true_auc_DIAGNOSTIC']), flush=True)

            # prereg §7 futility, PER-ARM (Addendum A §2)
            if epoch == FUTILITY_EPOCH and self.gap_epoch0:
                drop = (self.gap_epoch0 - self.best_gap) / self.gap_epoch0
                if drop < FUTILITY_MIN_FRACTIONAL_DROP:
                    self.futility_stop = True
                    print('FUTILITY STOP (arm w=%.1f): |gap| fell %.1f%% (<%.0f%%) by epoch %d. '
                          'H-B2-MECHANISM-FAILED for THIS ARM ONLY; the other arm is unaffected '
                          'and no retuning is authorized.'
                          % (self.ver_loss_weight, drop * 100,
                             FUTILITY_MIN_FRACTIONAL_DROP * 100, FUTILITY_EPOCH), flush=True)
                    break

        truncated = self.best_epoch is not None and self.best_epoch >= TRUNCATION_FLAG_EPOCH
        man = {'schema': 'PHASE_B2_RUN_MANIFEST_V1', 'method_uncertified': True,
               'test_split_accessed': False,
               'preregistration': PREREG, 'addendum': ADDENDUM_A,
               'objective': 'L_priv = (mean_pos(zhat) - mean_neg(zhat))^2, zhat standardized',
               'declared_overrides': {
                   'accumulation_steps': ACCUMULATION_STEPS,
                   'ver_loss_weight': self.ver_loss_weight,
                   'initial_generator': 'U_PUBLISHED %s' % U_PUBLISHED_SHA[:16]},
               'effective_batch_size': self.config['batch_size'] * ACCUMULATION_STEPS,
               'seed': self.seed, 'best_epoch': self.best_epoch,
               'best_selection_total': self.best_selection_total,
               'epochs_completed': len(self.epoch_metrics),
               'futility_stop': self.futility_stop,
               'arm_classification_if_futility': 'H-B2-MECHANISM-FAILED (arm)' if self.futility_stop else None,
               'SELECTION_POSSIBLY_TRUNCATED': truncated,
               'degenerate_batch_rate': self.degenerate_batches / max(self.total_micro_batches, 1),
               'selected_generator_sha256': file_sha256(self.p_best_gen) if os.path.exists(self.p_best_gen) else None,
               'selected_verifier_sha256': file_sha256(self.p_best_ver) if os.path.exists(self.p_best_ver) else None,
               'initial_generator_sha256': self.initial_generator_sha,
               'config_sha256': self.config_sha, 'acloss_module_sha256': self.acloss_sha,
               'order_hashes_verified': True, 'torch': torch.__version__}
        json.dump(man, open(os.path.join(self.output_dir, 'checkpoint_manifest.json'), 'w'), indent=2)
        print('Finished. best_epoch=%s futility=%s truncated=%s'
              % (self.best_epoch, self.futility_stop, truncated), flush=True)
        return man


def main():
    ap = argparse.ArgumentParser(description='Phase B2 (preregistered)')
    ap.add_argument('--ver_loss_weight', type=float, required=True,
                    help='1.0 = arm A, 3.0 = arm B; both declared in prereg §5')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max_epochs', type=int, default=MAX_EPOCHS)
    a = ap.parse_args()
    if a.ver_loss_weight not in (1.0, 3.0):
        raise SystemExit('only the two preregistered arms (1.0, 3.0) are permitted')
    out = os.path.join(METHOD_OUT_ROOT, 'phase_b2', 'w%g' % a.ver_loss_weight,
                       'seed_%d' % a.seed)
    PhaseB2Runner(a.ver_loss_weight, a.seed, out,
                  torch.device('cuda')).run(max_epochs=a.max_epochs)


if __name__ == '__main__':
    main()
