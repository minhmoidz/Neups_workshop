import hashlib
import json
import os
import subprocess
import time
from statistics import mean

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from utils import utils
from utils.ACLoss import ACLoss
from utils.FeatureConsistencyLoss import FeatureConsistencyLoss
from utils.VerificationLoss import VerificationLoss
from utils.GaussianSmoothing import GaussianSmoothing

from networks.UNet_AttentionPriCheXyNet import (
    UNetAtt,
    AttentionGate,
    load_pretrained_into_unet_att,
)
from networks.UNet_PriCheXyNet import UNet
from networks.SiameseNetwork import SiameseNetwork


# Frozen invariant inherited from M2_S1_EXECUTION_LOCK.json. V2 changes ONLY
# the generator architecture and adds one auxiliary loss; everything the
# certification froze stays frozen here.
_FROZEN_MU = 0.01


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


class AgentV2:
    """Training agent for PriCheXy-Net V2: attention-gated flow field U-Net
    plus an optional frozen-backbone feature-consistency loss.

    Faithfulness contract with agents.Agent (baseline):
      * Identical seeding (utils.seed_all(42)), data loaders, identity-grid
        construction, Gaussian smoothing, optimizers and update ORDER.
      * Identical checkpoint-selection semantics: selection uses the
        method-neutral score S = L_AC_BCE + L_priv(log-likelihood) on the
        validation pairs. The feature-consistency term is EXCLUDED from
        selection (certification invariant T174 parity).
      * Differences are strictly additive:
          (a) attention-gated generator (near-identity init),
          (b) L_feat added to the TRAINING objective only,
          (c) optional gate-entropy regularizer added to TRAINING only,
          (d) provenance manifest written at startup.
    """

    def __init__(self, config):
        self.config = config
        self._validate_config(config)

        self.SAVINGS_PATH = './archive/' + config['experiment_description'] + '/'
        self.IMAGE_PATH = config['image_path']
        self.generator_type = config['generator_type']

        # Reproducibility: identical to baseline.
        utils.seed_all(42)

        # Hyperparameters.
        self.ac_loss_weight = float(config['ac_loss_weight'])
        self.ver_loss_weight = float(config['ver_loss_weight'])
        self.feature_loss_weight = float(config.get('feature_loss_weight', 0.0))
        self.gate_entropy_weight = float(config.get('gate_entropy_weight', 0.0))
        # 0.0 reproduces the exact baseline formula -log(1 - ver).
        self.ver_loglik_eps = float(config.get('ver_loglik_eps', 0.0))

        self.mu = float(config['mu'])
        self.image_size = int(config['image_size'])
        self.batch_size = int(config['batch_size'])
        self.learning_rate = float(config['learning_rate'])
        self.max_epochs = int(config['max_epochs'])
        # Effective-batch emulation for the batch ablation arm. Default 1
        # reproduces the baseline exactly (one optimizer step per batch).
        # When > 1, ALL THREE optimizers accumulate gradients over N consecutive
        # micro-batches and step once, with losses scaled by 1/N so the
        # per-sample gradient weight is unchanged.
        self.accumulation_steps = max(1, int(config.get('accumulation_steps', 1)))

        self.num_workers = 8
        self.pin_memory = True

        self.show_every_n_epochs = config['show_every_n_epochs']
        self.show_every_n_iterations = config['show_every_n_iterations']

        self.writer = SummaryWriter(self.SAVINGS_PATH + 'runs/')

        # ---- Identity grid: byte-identical construction to baseline ----
        d = torch.linspace(-1, 1, self.image_size)
        mesh_x, mesh_y = torch.meshgrid((d, d), indexing='ij')
        grid_identity = torch.stack((mesh_y, mesh_x), 2)
        self.grid_identity = grid_identity.unsqueeze(0).permute(0, 3, 1, 2).cuda()

        # ---- Gaussian filter: identical parameters to baseline ----
        self.gauss_filter = GaussianSmoothing(channels=2, kernel_size=9, sigma=2).cuda()

        # ---- Generator: attention variant (V2) or plain U-Net (control) ----
        if config['generator_type'] == 'flow_field_att':
            self.generator = UNetAtt(1, 2, 32).cuda()
            ckpt_path = config.get('pretrained_generator_file',
                                   './networks/pretrained_generator_prichexy_net.pth')
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError('Pretrained generator not found: %s' % ckpt_path)
            self.load_summary = load_pretrained_into_unet_att(self.generator, ckpt_path)
            self.has_gates = True
            self._initial_gate_sanity_check(ckpt_path)
        elif config['generator_type'] == 'flow_field':
            # Plain baseline generator, run through the SAME V2 training and
            # evaluation stack -- used as the control arm of the batch
            # ablation so that ONLY the effective batch size differs.
            ckpt_path = config.get('pretrained_generator_file',
                                   './networks/pretrained_generator_prichexy_net.pth')
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError('Pretrained generator not found: %s' % ckpt_path)
            self.generator = UNet(1, 2, 32).cuda()
            state = torch.load(ckpt_path, map_location='cpu')
            self.generator.load_state_dict(state, strict=True)
            self.load_summary = {'num_loaded_tensors': len(state),
                                 'missing_attention_keys': []}
            self.has_gates = False
        else:
            raise ValueError("generator_type must be 'flow_field_att' or "
                             "'flow_field', got '%s'." % config['generator_type'])

        self.start_epoch = 0
        self.lowest_ac_loss = 10000
        self.lowest_ver_loss = 10000
        self.lowest_total_loss = 10000
        self.loss_dict = {
            'training': {'ac_loss': [], 'ver_loss': [], 'log_likelihood_ver_loss': [],
                         'total_loss': [], 'feat_loss': []},
            'validation': {'ac_loss': [], 'ver_loss': [], 'log_likelihood_ver_loss': [],
                           'total_loss': [], 'feat_loss': []},
        }

        # ---- Critics: identical objects as baseline ----
        # weights_only=False is REQUIRED here: this checkpoint stores a full
        # pickled module (torchvision DenseNet), not a bare state dict
        # (PyTorch >= 2.6 defaults to weights_only=True). Local trusted file.
        self.ac_model = torch.load('./networks/pretrained_classifier.pth',
                                   map_location='cuda', weights_only=False)['model']
        self.ac_loss = ACLoss(ac_model=self.ac_model).cuda()

        self.feature_loss = FeatureConsistencyLoss(ac_model=self.ac_model).cuda()
        self.feature_loss_enabled = self.feature_loss_weight > 0.0

        self.verification_model = SiameseNetwork().cuda()
        self.verification_model.load_state_dict(
            torch.load('./networks/pretrained_verification_model.pth'))
        self.verification_loss = VerificationLoss(
            verification_model=self.verification_model, reduction='none').cuda()

        self.criterion_ac = nn.BCELoss().cuda()
        self.criterion_ver = nn.BCEWithLogitsLoss().cuda()

        # ImageNet normalization reused by both critic updates (identical to
        # baseline constants).
        self.normalize_imagenet = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])
        self.resize_224 = transforms.Resize((224, 224))

        # ---- Optimizers: identical to baseline ----
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        self.optimizer_ver = optim.Adam(
            self.verification_loss.verification_model.parameters(),
            lr=self.learning_rate)
        self.optimizer_ac = optim.SGD(
            filter(lambda p: p.requires_grad, self.ac_loss.ac_model.parameters()),
            lr=self.learning_rate, momentum=0.9, weight_decay=1e-4)

        # ---- Data loaders: identical to baseline ----
        self.training_loader = utils.get_data_loader(
            phase='training', experimental_step='anonymization',
            image_size=self.image_size, n_channels=1,
            batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            image_path=self.IMAGE_PATH)
        self.validation_loader = utils.get_data_loader(
            phase='validation', experimental_step='anonymization',
            image_size=self.image_size, n_channels=1,
            batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            image_path=self.IMAGE_PATH)

        self._write_provenance_manifest()

    # ------------------------------------------------------------------ #
    # Configuration validation (fail-closed)                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_config(config):
        required = ['experiment_description', 'image_path', 'ac_loss_weight',
                    'ver_loss_weight', 'generator_type', 'mu', 'image_size',
                    'batch_size', 'learning_rate', 'max_epochs',
                    'show_every_n_epochs', 'show_every_n_iterations']
        missing = [k for k in required if k not in config]
        if missing:
            raise ValueError('Missing required config keys: %s' % missing)
        if config['generator_type'] not in ('flow_field_att', 'flow_field'):
            raise ValueError("generator_type must be 'flow_field_att' (V2) or "
                             "'flow_field' (control arm), got '%s'."
                             % config['generator_type'])
        if abs(float(config['mu']) - _FROZEN_MU) > 1e-12:
            raise ValueError('mu=%s violates the frozen invariant mu=%s '
                             '(M2_S1_EXECUTION_LOCK.json); refusing to run.'
                             % (config['mu'], _FROZEN_MU))
        if int(config['image_size']) % 16 != 0:
            raise ValueError('image_size must be divisible by 16 (four 2x pools).')
        if float(config.get('ver_loglik_eps', 0.0)) < 0.0:
            raise ValueError('ver_loglik_eps must be >= 0.')

    def _initial_gate_sanity_check(self, ckpt_path):
        """Numerically verify the initialization contract BEFORE training.

        Check 1 -- near-identity gates: with zero-initialized W_g/W_x/psi
        weights and psi bias b, every gate output equals sigmoid(b) regardless
        of input.
        Check 2 -- structural parity with the plain U-Net: feeding the same
        random image through both networks (same overlapping weights) must
        yield nearly identical flow fields. This is the check that catches
        data-flow bugs such as concatenating the wrong tensors in the decoder
        path; a violation aborts the run before any GPU-hours are spent.
        """
        self.generator.eval()
        g = torch.Generator().manual_seed(12345)
        dummy = torch.randn(1, 1, self.image_size, self.image_size,
                            generator=g).cuda()

        with torch.no_grad():
            expected_gate = torch.sigmoid(
                torch.tensor(AttentionGate.GATE_INIT_BIAS)).item()
            _, gates = self.generator(dummy, return_gates=True)
            for name, gmap in gates.items():
                g_min, g_max = gmap.min().item(), gmap.max().item()
                if abs(g_min - expected_gate) > 1e-4 or abs(g_max - expected_gate) > 1e-4:
                    raise RuntimeError(
                        'Gate %s not near-identity at init: range [%f, %f], '
                        'expected ~%f everywhere.' % (name, g_min, g_max, expected_gate))

            # Structural parity against the plain U-Net baseline.
            reference = UNet(1, 2, 32).cuda()
            reference.load_state_dict(torch.load(ckpt_path, map_location='cuda'))
            reference.eval()

            flow_att = self.generator(dummy)
            flow_ref = reference(dummy)
            if not torch.isfinite(flow_att).all():
                raise RuntimeError('Non-finite flow field at initialization.')
            if flow_att.min().item() < -1.0 or flow_att.max().item() > 1.0:
                raise RuntimeError('Flow field outside tanh range at initialization.')

            max_dev = (flow_att - flow_ref).abs().max().item()
            # Bound implied by gates ~= sigmoid(6) = 0.99752 propagating
            # through the decoder; measured ~0.04 empirically. A data-flow
            # bug produces O(1) deviations, far above this threshold.
            tolerance = 0.10
            if max_dev > tolerance:
                raise RuntimeError(
                    'Structural parity violated at init: max|flow_att - '
                    'flow_plain| = %.4f > %.2f. The attention U-Net forward '
                    'path does NOT reduce to the plain U-Net when gates are '
                    'near identity -- refusing to train a corrupted graph.'
                    % (max_dev, tolerance))
            print('[V2] Init sanity OK: gates=%.6f, parity max|dflow|=%.4f'
                  % (expected_gate, max_dev))
        self.generator.train()

    # ------------------------------------------------------------------ #
    # Provenance                                                          #
    # ------------------------------------------------------------------ #

    def _write_provenance_manifest(self):
        def git_head():
            try:
                return subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD'], cwd='./').decode().strip()
            except Exception:
                return None

        def safe_sha(path):
            return _sha256_file(path) if os.path.exists(path) else None

        manifest = {
            'agent': 'AgentV2',
            'git_head': git_head(),
            'torch': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'config': json.loads(json.dumps(self.config)),
            'pretrained_generator': {
                'path': self.config.get('pretrained_generator_file',
                                        './networks/pretrained_generator_prichexy_net.pth'),
                'sha256': safe_sha('./networks/pretrained_generator_prichexy_net.pth'),
                'num_loaded_tensors': self.load_summary['num_loaded_tensors'],
                'missing_attention_keys': self.load_summary['missing_attention_keys'],
            },
            'source_files_sha256': {p: safe_sha(p) for p in [
                'agents/AgentV2.py',
                'networks/UNet_AttentionPriCheXyNet.py',
                'utils/FeatureConsistencyLoss.py']},
            'image_pairs_sha256': {p: safe_sha('./image_pairs/' + p) for p in [
                'image_pairs_training_10000.txt',
                'image_pairs_validation_2000.txt']},
            'selection_metric': 'L_AC_BCE + L_priv (feature loss excluded, T174 parity)',
        }
        with open(self.SAVINGS_PATH + 'v2_provenance_manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
        print('[V2] Provenance manifest written.')

    # ------------------------------------------------------------------ #
    # Loss helpers                                                        #
    # ------------------------------------------------------------------ #

    def _loglik(self, ver_loss_per_sample):
        """-log(1 - ver). With eps=0 this is byte-identical to the baseline
        expression; eps > 0 only guards the degenerate ver == 1.0 overflow."""
        if self.ver_loglik_eps > 0.0:
            return -torch.log((1.0 - ver_loss_per_sample).clamp_min(self.ver_loglik_eps))
        return -torch.log(1.0 - ver_loss_per_sample)

    def _gate_entropy(self, gates):
        """Mean binary entropy over all gate maps. Objective term is
        -weight * H: a POSITIVE weight therefore pushes gates toward uniform
        (anti-concentration prior); a NEGATIVE weight pushes toward
        concentrated gates. Default weight is 0.0 (term disabled)."""
        entropies = []
        for g in gates.values():
            p = g.clamp(1e-6, 1.0 - 1e-6)
            entropies.append(-(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)).mean())
        return torch.stack(entropies).mean()

    # ------------------------------------------------------------------ #
    # Epoch loop (mirrors utils.train / utils.validate step-for-step)     #
    # ------------------------------------------------------------------ #

    def _run_epoch(self, loader, training, epoch):
        self.generator.train(mode=training)
        lists = {k: [] for k in ['ac_loss', 'ver_loss', 'log_likelihood_ver_loss',
                                 'total_loss', 'feat_loss']}
        phase_tag = 'Training' if training else 'Validation'
        print(phase_tag + '----->')

        context = torch.enable_grad() if training else torch.no_grad()
        accum = self.accumulation_steps if training else 1
        n_batches = len(loader)
        with context:
            for i, batch in enumerate(loader):
                inputs1, inputs2, labels, labels_id = batch
                inputs1 = inputs1.cuda()
                inputs2 = inputs2.cuda()
                labels = labels.cuda()

                # Single forward produces both the flow field and the gate
                # maps (attention variant), so logged gates correspond EXACTLY
                # to the deformation actually applied. The control arm's plain
                # U-Net has no gates.
                if self.has_gates:
                    grids, gates = self.generator(inputs1, return_gates=True)
                else:
                    grids, gates = self.generator(inputs1), None
                grids = self.grid_identity - self.mu * grids
                grids = self.gauss_filter(grids)
                grids = grids.permute(0, 2, 3, 1)
                fakes_1 = torch.nn.functional.grid_sample(
                    inputs1, grids, padding_mode='border', align_corners=True)

                if epoch % self.show_every_n_epochs == 0 \
                        and i % self.show_every_n_iterations == 0 and gates is not None:
                    tag = 'training' if training else 'validation'
                    writer = SummaryWriter(
                        self.SAVINGS_PATH + 'runs/epoch' + str(epoch) + '/')
                    writer.add_image(tag + '/original_images',
                                     torchvision.utils.make_grid(inputs1), i)
                    writer.add_image(tag + '/deformed_images',
                                     torchvision.utils.make_grid(fakes_1), i)
                    for gname, gmap in gates.items():
                        g_up = torch.nn.functional.interpolate(
                            gmap, size=(self.image_size, self.image_size),
                            mode='bilinear', align_corners=False)
                        writer.add_image(tag + '/gate_' + gname,
                                         torchvision.utils.make_grid(g_up), i)
                    writer.close()

                # --- Losses (formulas identical to baseline) ---
                ac_loss_value = self.ac_loss(fakes_1, labels)

                ver_loss_raw = self.verification_loss(fakes_1, inputs2)
                log_likelihood_ver_loss = self._loglik(ver_loss_raw).mean()
                ver_loss_mean = ver_loss_raw.mean()

                # Method-neutral score: EXCLUDES the feature term (T174).
                selection_total = (self.ac_loss_weight * ac_loss_value
                                   + self.ver_loss_weight * log_likelihood_ver_loss)

                if training:
                    if self.feature_loss_enabled:
                        feat_loss_value = self.feature_loss(fakes_1, inputs1)
                    else:
                        feat_loss_value = torch.zeros((), device=fakes_1.device)

                    total_loss = selection_total \
                        + self.feature_loss_weight * feat_loss_value

                    if self.gate_entropy_weight != 0.0 and gates is not None:
                        total_loss = total_loss \
                            + -self.gate_entropy_weight * self._gate_entropy(gates)

                    lists['ac_loss'].append(ac_loss_value.item())
                    lists['ver_loss'].append(ver_loss_mean.item())
                    lists['log_likelihood_ver_loss'].append(log_likelihood_ver_loss.item())
                    lists['total_loss'].append(total_loss.item())
                    lists['feat_loss'].append(feat_loss_value.item())

                    # ---- Gradient-hygiene contract ----
                    # Every backward() targets ONLY its own optimizer's
                    # parameters via inputs=..., mirroring the baseline's
                    # zero_grad discipline exactly: the generator update sees
                    # gradients solely from total_loss, each critic solely
                    # from its own loss. With accumulation_steps=1 this is
                    # mathematically identical to the baseline schedule; with
                    # N>1 it emulates an effective batch of batch_size * N.
                    g_params = [p for p in self.generator.parameters()
                                if p.requires_grad]
                    ver_params = list(
                        self.verification_loss.verification_model.parameters())
                    ac_params = [p for p in self.ac_loss.ac_model.parameters()
                                 if p.requires_grad]

                    window_start = (i % accum == 0)
                    is_step_boundary = ((i + 1) % accum == 0) or (i + 1 == n_batches)

                    # Generator phase (first -- identical order to baseline).
                    if window_start:
                        self.optimizer_g.zero_grad()
                    (total_loss / accum).backward(inputs=g_params)
                    if is_step_boundary:
                        self.optimizer_g.step()

                    # ---- Critics: identical order / zero_grad placement ----
                    self.verification_loss.verification_model.train()
                    self.ac_loss.ac_model.train()

                    inputs1_snn = fakes_1.detach().expand(-1, 3, -1, -1)
                    inputs2_snn = inputs2.expand(-1, 3, -1, -1)
                    inputs1_snn = self.normalize_imagenet(inputs1_snn)
                    inputs2_snn = self.normalize_imagenet(inputs2_snn)

                    if window_start:
                        self.optimizer_ver.zero_grad()
                        self.optimizer_ac.zero_grad()

                    outputs_snn = self.verification_loss.verification_model(
                        inputs1_snn, inputs2_snn).squeeze()
                    labels_id_t = labels_id.cuda().type_as(outputs_snn)
                    loss_ver = self.criterion_ver(outputs_snn, labels_id_t)
                    (loss_ver / accum).backward(inputs=ver_params)
                    if is_step_boundary:
                        self.optimizer_ver.step()
                    self.verification_loss.verification_model.eval()

                    # AC critic per micro-batch (accumulate), step at boundary.
                    inputs_ac = fakes_1.detach().expand(-1, 3, -1, -1)
                    inputs_ac = self.normalize_imagenet(self.resize_224(inputs_ac))
                    outputs_ac = self.ac_loss.ac_model(inputs_ac)
                    loss_ac = self.criterion_ac(outputs_ac, labels)
                    (loss_ac / accum).backward(inputs=ac_params)
                    if is_step_boundary:
                        self.optimizer_ac.step()
                        self.ac_loss.ac_model.eval()

                    print('Epoch [%d/%d], Iteration [%d/%d], '
                          'Verification Loss (ver_loss): %.4f'
                          % (epoch + 1, self.max_epochs, i + 1, len(loader),
                             ver_loss_mean.item()))
                else:
                    if self.feature_loss_enabled:
                        feat_loss_value = self.feature_loss(fakes_1, inputs1)
                    else:
                        feat_loss_value = torch.zeros((), device=fakes_1.device)

                    lists['ac_loss'].append(ac_loss_value.item())
                    lists['ver_loss'].append(ver_loss_mean.item())
                    lists['log_likelihood_ver_loss'].append(log_likelihood_ver_loss.item())
                    # Validation 'total_loss' IS the selection score (feature
                    # term excluded), matching T174 checkpoint semantics.
                    lists['total_loss'].append(selection_total.item())
                    lists['feat_loss'].append(feat_loss_value.item())

                    print('Epoch [%d/%d], Iteration [%d/%d], '
                          'Verification Loss (ver_loss): %.4f'
                          % (epoch + 1, self.max_epochs, i + 1, len(loader),
                             ver_loss_mean.item()))

        return {k: (mean(v) if len(v) > 0 else 10000.0) for k, v in lists.items()}

    def training_validation(self):
        for epoch in range(self.start_epoch, self.max_epochs):
            start_time = time.time()

            train_metrics = self._run_epoch(self.training_loader, True, epoch)
            val_metrics = self._run_epoch(self.validation_loader, False, epoch)

            print('Time elapsed for epoch ' + str(epoch + 1) + ': ' +
                  str(round((time.time() - start_time) / 60, 2)) + ' minutes')

            for phase, metrics in (('training', train_metrics),
                                   ('validation', val_metrics)):
                for k, v in metrics.items():
                    self.loss_dict[phase][k].append(v)
            utils.save_loss_dict(self.loss_dict, self.SAVINGS_PATH)

            # Checkpoint policy mirrors baseline. val_metrics['total_loss'] is
            # the method-neutral selection score (feature term excluded).
            torch.save(self.generator.state_dict(),
                       self.SAVINGS_PATH + 'generator_latest.pth')
            torch.save(self.ac_loss.ac_model.state_dict(),
                       self.SAVINGS_PATH + 'ac_model_trained_latest.pth')
            torch.save(self.verification_loss.verification_model.state_dict(),
                       self.SAVINGS_PATH + 'ver_model_trained_latest.pth')

            if val_metrics['ac_loss'] < self.lowest_ac_loss:
                self.lowest_ac_loss = val_metrics['ac_loss']
                torch.save(self.generator.state_dict(),
                           self.SAVINGS_PATH + 'generator_lowest_ac_loss.pth')
                print('Current generator with lowest ac_loss: epoch ' + str(epoch))

            if val_metrics['ver_loss'] < self.lowest_ver_loss:
                self.lowest_ver_loss = val_metrics['ver_loss']
                torch.save(self.generator.state_dict(),
                           self.SAVINGS_PATH + 'generator_lowest_ver_loss.pth')
                print('Current generator with lowest ver_loss: epoch ' + str(epoch))

            if val_metrics['total_loss'] < self.lowest_total_loss:
                self.lowest_total_loss = val_metrics['total_loss']
                torch.save(self.generator.state_dict(),
                           self.SAVINGS_PATH + 'generator_lowest_total_loss.pth')
                print('Current generator with lowest total_loss: epoch ' + str(epoch))
                torch.save(self.ac_loss.ac_model.state_dict(),
                           self.SAVINGS_PATH + 'ac_model_trained_lowest_total_loss.pth')
                torch.save(self.verification_loss.verification_model.state_dict(),
                           self.SAVINGS_PATH + 'ver_model_trained_lowest_total_loss.pth')

            try:
                utils.show_loss_curves(self.loss_dict, pre_train=False,
                                       save_fig=True, show_fig=False,
                                       path=self.SAVINGS_PATH)
            except Exception as e:
                print('[V2] loss-curve plotting skipped: %r' % e)

        print('Finished Training!')

    def run(self):
        self.training_validation()
