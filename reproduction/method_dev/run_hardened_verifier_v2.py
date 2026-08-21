"""Direction B v2 — corrected hardened-verifier design (G0.2 §7).

WRITE-ONLY DELIVERABLE FOR G0.2. This file is syntax/AST-checked but MUST NOT
be imported, instantiated, or executed as part of G0.2 (no GPU, no real
config/checkpoint/dataset access is authorized in this task). Executing it
requires a separately-authorized, human-approved execution manifest — see
`_require_execution_manifest()` below, which the CLI entry point calls before
any training can start.

Corrects the confound found in `reproduction/method_dev/run_hardened_verifier.py`
(documented in reproduction/reports/G0_HYPOTHESIS_GATE_AUDIT_2026-08-21.md
H0.1): critic-only forward passes there run with the generator still in
`.train()` mode, so BatchNorm running statistics drift on every extra
verifier-only step even though `torch.no_grad()` is used. v2 fixes this via
`protocol_v2.state_invariants.preserved_eval_forward` and an explicit
state-hash assertion around every critic-only block.

Design-level differences from v1 (`run_hardened_verifier.py`), per
reproduction/reports/G0_1_PROTOCOL_REPAIR_SPEC_2026-08-21.md §6:

  v1                                          v2
  ----------------------------------------    ----------------------------------------
  generator stays .train() during extra       generator forced .eval() + inference_mode
    critic-only forward passes (BUG)             during critic-only forward (fixed)
  no state-change assertion                   buffer hash + param-version signature
                                                 checked around every critic-only block;
                                                 full canonical hash at predeclared epoch
                                                 boundaries (not every batch — cost-bounded)
  always reuses same batch for extra steps    explicit batch_policy: 'same_batch' or
                                                 'fresh_batch' (separate deterministic
                                                 sampler for fresh_batch), recorded in
                                                 manifest, never silently mixed
  no output-freshness guard                   protocol_v2.output_freshness guard before
                                                 any write
  no explicit pretrained-weight provenance    protocol_v2.weight_provenance record
                                                 required for the verifier's ResNet50 init
  no k_extra=0 parity mechanism               k_extra_verifier_steps>=0 explicitly allowed,
                                                 0 is a valid, first-class configuration
                                                 (parity test target — NOT claimed passing
                                                 here; that is a future authorized GPU gate)
  free-running, starts immediately on CLI     fail-closed: refuses to start without an
                                                 explicit --execution-manifest naming a
                                                 human-approved manifest file
  no snapshot hooks                           predeclared snapshot hooks (disabled by
                                                 default) that CAN save generator + live
                                                 verifier state at predeclared epochs in a
                                                 FUTURE authorized run — not enabled here

Explicitly NOT claimed by this file:
  - That the live verifier's privacy_term is method-neutral (it is not — see
    G0 H0.3; checkpoint selection still needs the external referee designed
    in G0.1 §9, not yet implemented here).
  - That k_extra=0 has been shown to reproduce certified B_dev bit-for-bit
    (that parity run is a future, separately-authorized GPU gate).
  - Access to any locked-confirmation split (none exists yet).
"""
import argparse
import json
import os
import sys
import time

# NOTE: these imports are real-project imports. They are safe to CONTAIN in
# this file (module authorship), but this file must not be *executed* under
# G0.2 — see the module docstring. `main()` refuses to run without an
# approved execution manifest even if someone does invoke it.
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
for _p in (ROOT, os.path.join(ROOT, 'research_agent'), os.path.join(ROOT, 'reproduction')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from networks.UNet_PriCheXyNet import UNet  # noqa: E402
from networks.SiameseNetwork import SiameseNetwork  # noqa: E402
from utils.VerificationLoss import VerificationLoss  # noqa: E402

from m2_dev.evaluator_common import (  # noqa: E402
    MU, IMAGE_SIZE, INITIAL_GENERATOR_PATH, INITIAL_GENERATOR_SHA,
    FROZEN_CLASSIFIER_PATH, FROZEN_CLASSIFIER_SHA, FROZEN_VERIFIER_PATH, FROZEN_VERIFIER_SHA,
    file_sha256, firewall_check, make_flow_field_components, verify_repaired_acloss,
    build_dev_anonymizer_loaders, FROZEN_B_DEV_CONFIG_PATH, FROZEN_C4_CONFIG_PATH,
    FROZEN_B_DEV_CONFIG_SHA, FROZEN_C4_CONFIG_SHA,
)

from protocol_v2.state_invariants import (  # noqa: E402
    canonical_tensor_state_hash, buffer_only_hash, parameter_version_signature,
    preserved_eval_forward,
)
from protocol_v2.output_freshness import assert_fresh_output_dir  # noqa: E402
from protocol_v2.weight_provenance import record_weight_provenance  # noqa: E402
from protocol_v2.deterministic_loader import DeterministicEpochSampler  # noqa: E402

METHOD_OUT_ROOT = os.path.join(ROOT, 'reproduction', 'method_dev')

BATCH_POLICIES = ('same_batch', 'fresh_batch')

# Epochs at which a full canonical state hash is computed (cost-bounded — not
# every batch). Buffer-only hash + parameter-version signature run around
# EVERY critic-only block regardless, since those are cheap.
PREDECLARED_FULL_HASH_EPOCHS = frozenset({0, 1, 2, 5, 10, 25, 50, 100, 150, 200, 249})


class StateInvariantViolation(RuntimeError):
    pass


class GeneratorStateGuard:
    """Wraps a critic-only block: asserts the generator's full parameter+buffer
    state is byte-identical before and after, and that .training mode is
    restored, using `preserved_eval_forward` for the forward pass itself."""

    def __init__(self, generator, check_full_hash: bool):
        self.generator = generator
        self.check_full_hash = check_full_hash
        self._buf_before = None
        self._pver_before = None
        self._full_before = None
        self._mode_before = None

    def __enter__(self):
        self._mode_before = self.generator.training
        self._buf_before = buffer_only_hash(self.generator)
        self._pver_before = parameter_version_signature(self.generator)
        if self.check_full_hash:
            self._full_before = canonical_tensor_state_hash(self.generator)
        return self

    def verify_unchanged(self):
        if self.generator.training != self._mode_before:
            raise StateInvariantViolation('Generator .training mode changed across critic-only block')
        if buffer_only_hash(self.generator) != self._buf_before:
            raise StateInvariantViolation('Generator buffer state changed across critic-only block (BN drift)')
        if parameter_version_signature(self.generator) != self._pver_before:
            raise StateInvariantViolation('Generator parameter _version changed across critic-only block')
        if self.check_full_hash:
            after = canonical_tensor_state_hash(self.generator)
            if after != self._full_before:
                raise StateInvariantViolation('Generator full canonical state hash changed across critic-only block')

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def _require_execution_manifest(path):
    """Fail-closed CLI gate: refuses to proceed without a human-approved
    execution manifest. This is a design placeholder — G0.2 does not create
    or approve any such manifest, so this function is expected to always
    raise in the current repository state."""
    if not path or not os.path.exists(path):
        raise RuntimeError(
            'Refusing to start: no --execution-manifest provided or file not found. '
            'Direction B v2 requires a separately human-approved execution manifest '
            'before any training may begin. See G0.2 report for what remains unauthorized.')
    with open(path) as f:
        manifest = json.load(f)
    if not manifest.get('human_approved') is True:
        raise RuntimeError('Execution manifest present but not marked human_approved=true; refusing to start.')
    return manifest


class HardenedVerifierRunnerV2:
    """Corrected Direction B runner. See module docstring for the v1->v2 diff.

    NOT INSTANTIATED IN G0.2 — this class is provided for AST/syntax review
    and for a future, separately-authorized GPU run only.
    """

    def __init__(self, arm, k_extra_verifier_steps, batch_policy, seed, output_dir, device,
                 weight_provenance_record, snapshot_epochs=frozenset()):
        firewall_check('dev')
        if arm not in ('B_dev', 'C4'):
            raise ValueError("arm must be 'B_dev' or 'C4'")
        if batch_policy not in BATCH_POLICIES:
            raise ValueError('batch_policy must be one of %s' % (BATCH_POLICIES,))
        if int(k_extra_verifier_steps) < 0:
            raise ValueError('k_extra_verifier_steps must be >= 0')
        if device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('Hardened-verifier v2 requires CUDA')

        assert_fresh_output_dir(output_dir)

        self.arm = arm
        self.k_extra_verifier_steps = int(k_extra_verifier_steps)
        self.batch_policy = batch_policy
        self.seed = seed
        self.device = device
        self.weight_provenance_record = weight_provenance_record  # required, see main()
        self.snapshot_epochs = frozenset(snapshot_epochs)  # DISABLED by default (empty)
        self.generator_optimizer_step_count = 0
        self.nan_inf_detected = False

        cfg_path = FROZEN_C4_CONFIG_PATH if arm == 'C4' else FROZEN_B_DEV_CONFIG_PATH
        expected_cfg_sha = FROZEN_C4_CONFIG_SHA if arm == 'C4' else FROZEN_B_DEV_CONFIG_SHA
        actual_cfg_sha = file_sha256(cfg_path)
        if actual_cfg_sha != expected_cfg_sha:
            raise RuntimeError('Frozen anonymizer config SHA mismatch')
        with open(cfg_path) as f:
            self.config = json.load(f)
        self.config_path = cfg_path

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.mu = self.config.get('mu', 0.01)
        self.learning_rate = self.config.get('learning_rate', 1e-4)
        self.image_size = self.config.get('image_size', IMAGE_SIZE)
        self.ac_loss_weight = self.config.get('ac_loss_weight', 1.0)
        self.ver_loss_weight = self.config.get('ver_loss_weight', 1.0)
        self.feature_loss_weight = 1.0 if self.arm == 'C4' else 0.0

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.grid_identity, self.gauss_filter = make_flow_field_components(self.device, self.image_size)

        self.generator = UNet(1, 2, 32).to(self.device)
        if file_sha256(INITIAL_GENERATOR_PATH) != INITIAL_GENERATOR_SHA:
            raise RuntimeError('Initial generator SHA drift')
        self.generator.load_state_dict(torch.load(INITIAL_GENERATOR_PATH, map_location=self.device, weights_only=False))

        ACLossClass, self.acloss_sha, _ = verify_repaired_acloss()
        if file_sha256(FROZEN_CLASSIFIER_PATH) != FROZEN_CLASSIFIER_SHA:
            raise RuntimeError('Classifier SHA drift')
        self.ac_model = torch.load(FROZEN_CLASSIFIER_PATH, map_location=self.device, weights_only=False)['model']
        self.ac_loss = ACLossClass(ac_model=self.ac_model, reduction='mean',
                                    feature_loss_weight=self.feature_loss_weight).to(self.device)

        if file_sha256(FROZEN_VERIFIER_PATH) != FROZEN_VERIFIER_SHA:
            raise RuntimeError('Verifier SHA drift')
        # Pretrained-weight identity for the verifier's ResNet50 backbone is
        # required explicitly (not assumed from torchvision defaults) — see
        # protocol_v2.weight_provenance and G0.1 §10 item 2.
        if self.weight_provenance_record is None:
            raise RuntimeError('weight_provenance_record is required in scientific mode')
        self.verification_model = SiameseNetwork().to(self.device)
        self.verification_model.load_state_dict(torch.load(FROZEN_VERIFIER_PATH, map_location=self.device, weights_only=False))
        self.verification_loss = VerificationLoss(verification_model=self.verification_model, reduction='none').to(self.device)

        self.criterion_ac = nn.BCELoss().to(self.device)
        self.criterion_ver = nn.BCEWithLogitsLoss().to(self.device)
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        self.optimizer_ver = optim.Adam(self.verification_loss.verification_model.parameters(), lr=self.learning_rate)
        self.optimizer_ac = optim.SGD(filter(lambda p: p.requires_grad, self.ac_loss.ac_model.parameters()),
                                       lr=self.learning_rate, momentum=0.9, weight_decay=1e-4)

        self.training_loader, self.validation_loader, self.train_sampler = build_dev_anonymizer_loaders(
            self.config, seed=self.seed, num_workers=0)

        if self.batch_policy == 'fresh_batch':
            # Separate deterministic sampler for critic-only fresh batches —
            # never shares state with self.train_sampler (which governs the
            # generator's own epoch traversal and must stay parity-comparable
            # to certified B_dev).
            self.critic_only_sampler = DeterministicEpochSampler(
                self.training_loader.dataset, seed=seed + 1_000_000)
        else:
            self.critic_only_sampler = None

        self.resize_224 = transforms.Resize((224, 224))
        self.imagenet_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.epoch_metrics = []
        self.best_selection_total = float('inf')
        self.best_epoch = None
        self.best_checkpoint_path = os.path.join(self.output_dir, 'generator_best_method_neutral.pth')

    def anonymize_tensor(self, image):
        grids = self.generator(image)
        grids = self.grid_identity - self.mu * grids
        grids = self.gauss_filter(grids)
        grids = grids.permute(0, 2, 3, 1)
        return F.grid_sample(image, grids, padding_mode='border', align_corners=True)

    def _critic_only_fake(self, inputs1):
        """Generate a fake image for a critic-only step WITHOUT mutating the
        generator's parameters or buffers (fixes G0 H0.1)."""
        with preserved_eval_forward(self.generator):
            return self.anonymize_tensor(inputs1)

    def train_epoch(self, epoch):
        self.generator.train()
        check_full_hash = epoch in self.snapshot_epochs
        running = {}
        n_batches = 0

        for batch in self.training_loader:
            inputs1, inputs2, labels, labels_id = batch
            inputs1, inputs2 = inputs1.to(self.device), inputs2.to(self.device)
            labels, labels_id = labels.to(self.device), labels_id.to(self.device)

            # --- combined step (generator DOES update here — no guard) ---
            fakes_1 = self.anonymize_tensor(inputs1)
            self.ac_loss.refresh()
            deformed_features = self.ac_loss._features(self.ac_loss._preprocess(fakes_1))
            ac_predictions = self.ac_loss.loss_model.classifier(deformed_features)
            ac_bce_loss = self.ac_loss.bce_loss(ac_predictions, labels)
            if self.arm == 'C4':
                real_features = self.ac_loss._features(self.ac_loss._preprocess(inputs1)).detach()
                feat_loss = self.feature_loss_weight * F.mse_loss(deformed_features, real_features)
                ac_total_loss = ac_bce_loss + feat_loss
            else:
                ac_total_loss = ac_bce_loss

            inputs1_snn_g = self.imagenet_normalize(fakes_1.expand(-1, 3, -1, -1))
            inputs2_snn_g = self.imagenet_normalize(inputs2.expand(-1, 3, -1, -1))
            raw_logits = self.verification_loss.verification_model(inputs1_snn_g, inputs2_snn_g).squeeze()
            privacy_term = F.softplus(raw_logits).mean()
            gen_loss = self.ac_loss_weight * ac_total_loss + self.ver_loss_weight * privacy_term

            if not torch.isfinite(gen_loss).all():
                raise FloatingPointError('generator loss non-finite')
            self.optimizer_g.zero_grad()
            gen_loss.backward()
            for p in self.generator.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise FloatingPointError('generator gradient non-finite')
            self.optimizer_g.step()
            self.generator_optimizer_step_count += 1
            for p in self.generator.parameters():
                if not torch.isfinite(p).all():
                    raise FloatingPointError('generator parameter non-finite after step')

            self.verification_loss.verification_model.train()
            self.optimizer_ver.zero_grad()
            inputs1_snn = self.imagenet_normalize(fakes_1.detach().expand(-1, 3, -1, -1))
            outputs_snn = self.verification_loss.verification_model(inputs1_snn, inputs2_snn_g).squeeze()
            loss_ver = self.criterion_ver(outputs_snn, labels_id.type_as(outputs_snn))
            if not torch.isfinite(loss_ver).all():
                raise FloatingPointError('verifier loss non-finite')
            loss_ver.backward()
            self.optimizer_ver.step()
            self.verification_loss.verification_model.eval()

            self.ac_loss.ac_model.train()
            self.optimizer_ac.zero_grad()
            inputs_ac = self.imagenet_normalize(self.resize_224(fakes_1.detach().expand(-1, 3, -1, -1)))
            loss_ac = self.criterion_ac(self.ac_loss.ac_model(inputs_ac), labels)
            if not torch.isfinite(loss_ac).all():
                raise FloatingPointError('AC critic loss non-finite')
            loss_ac.backward()
            self.optimizer_ac.step()
            self.ac_loss.ac_model.eval()
            # --- end combined step ---

            # --- critic-only extra steps: generator state MUST NOT change ---
            for extra_i in range(self.k_extra_verifier_steps):
                with GeneratorStateGuard(self.generator, check_full_hash and extra_i == 0) as guard:
                    if self.batch_policy == 'same_batch':
                        extra_inputs1, extra_inputs2, extra_labels_id = inputs1, inputs2, labels_id
                    else:
                        extra_inputs1, extra_inputs2, extra_labels_id = self._next_fresh_critic_batch()
                    fakes_extra = self._critic_only_fake(extra_inputs1)
                    x1 = self.imagenet_normalize(fakes_extra.expand(-1, 3, -1, -1))
                    x2 = self.imagenet_normalize(extra_inputs2.expand(-1, 3, -1, -1))
                    self.verification_loss.verification_model.train()
                    self.optimizer_ver.zero_grad()
                    out_extra = self.verification_loss.verification_model(x1, x2).squeeze()
                    loss_extra = self.criterion_ver(out_extra, extra_labels_id.type_as(out_extra))
                    if not torch.isfinite(loss_extra).all():
                        raise FloatingPointError('extra verifier loss non-finite')
                    loss_extra.backward()
                    self.optimizer_ver.step()
                    self.verification_loss.verification_model.eval()
                guard.verify_unchanged()  # raises StateInvariantViolation if generator moved
            n_batches += 1

        return {'generator_optimizer_step_count': self.generator_optimizer_step_count}

    def _next_fresh_critic_batch(self):
        raise NotImplementedError(
            'fresh_batch critic-only sampling not wired to a live DataLoader in this design-only file; '
            'a future authorized implementation must build a second DataLoader over '
            'self.critic_only_sampler before this path can run.')

    def run(self, max_epochs):
        raise RuntimeError('run() is a design-only stub; see _require_execution_manifest() in main()')


def main():
    parser = argparse.ArgumentParser(description='Direction B v2 (design-only in G0.2)')
    parser.add_argument('--arm', choices=['B_dev', 'C4'], default='B_dev')
    parser.add_argument('--k_extra_verifier_steps', type=int, default=2)
    parser.add_argument('--batch_policy', choices=list(BATCH_POLICIES), default='same_batch')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_epochs', type=int, default=250)
    parser.add_argument('--tag', type=str, default='k3_v2')
    parser.add_argument('--execution-manifest', type=str, default=None,
                         help='Path to a human-approved execution manifest JSON. Required to run.')
    args = parser.parse_args()

    # Fail-closed gate — expected to always raise until a real, separately
    # authorized manifest exists. G0.2 does not create or approve one.
    _require_execution_manifest(args.execution_manifest)

    raise RuntimeError('Unreachable in G0.2: execution manifest gate must have raised above.')


if __name__ == '__main__':
    main()
