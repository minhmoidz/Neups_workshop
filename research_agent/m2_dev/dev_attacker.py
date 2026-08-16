"""M1.2 development adaptive attacker (TRAIN/VALIDATION only).

Replaces agents/AgentSiameseNetwork for M2 development because the upstream
class constructs a TEST loader inside __init__ and its run() executes a TEST
evaluation (P0-A). This runner:

  - constructs ONLY a training loader and a validation loader
    (no test_loader, no testing phase, no testing pair path)
  - calls the TEST firewall before any loader is built
  - uses a CONFIGURABLE attacker seed (utils.seed_all before net init,
    DataLoader shuffle, and optimizer creation) instead of the hard-coded 42
  - attacker TRAIN geometry:           anon(x1), anon(x2)
  - attacker checkpoint VAL geometry:  anon(x1), anon(x2)  (BCE loss selection)
  - scientific VAL privacy geometry:   anon(x1), real(x2)   (in eval_reid_val)
"""
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from networks.SiameseNetwork import SiameseNetwork

from .evaluator_common import (
    build_anonymize_fn,
    build_dev_loaders,
    firewall_check,
    make_flow_field_components,
    snn_preprocess,
    MU,
    FROZEN_ATTACKER_CONFIG_PATH,
    FROZEN_ATTACKER_CONFIG_SHA,
    file_sha256,
)
from utils import utils


def load_frozen_anonymizer(config=None, device=None, checkpoint_path=None, image_size=None):
    """Load the frozen generator + legacy operator components for the attacker.
    Requires explicit checkpoint_path (no scientific fallback allowed).
    """
    from networks.UNet_PriCheXyNet import UNet

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = checkpoint_path
    if not ckpt_path:
        raise RuntimeError("selected generator checkpoint path must be explicitly provided (no scientific fallback allowed)")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("Selected generator checkpoint not found: %s" % ckpt_path)

    if image_size is None:
        image_size = (config or {}).get('image_size', 256)

    generator = UNet(1, 2, 32).to(device)
    generator.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    generator.eval()
    for p in generator.parameters():
        p.requires_grad = False
    grid_identity, gauss_filter = make_flow_field_components(device, image_size=image_size)
    anonymize_fn = build_anonymize_fn(generator, grid_identity, gauss_filter, MU)
    return generator, anonymize_fn


class DevAttacker:
    def __init__(self, config, attacker_seed=42, device=None,
                 anonymize_fn=None, training_loader=None, validation_loader=None,
                 net_factory=None, generator_checkpoint=None, image_size=None,
                 unit_test_mode=False, config_path=None):
        """Development attacker runner.

        :param config: dev config dict.
        :param attacker_seed: attacker RNG seed (42 for S1, 42/43/44 for S2).
        :param device: torch device; defaults to CUDA if available.
        :param anonymize_fn: optional injected anonymizer (tests); defaults to
            the frozen legacy flow_field operator loaded from generator_checkpoint.
        :param generator_checkpoint: explicit path to selected M2 generator checkpoint (required).
        :param training_loader / validation_loader: optional injected loaders
            (tests); defaults to TRAIN/VAL retrainSNN loaders. TEST loader is
            NEVER constructed here.
        :param net_factory: optional callable returning an attacker net (tests);
            defaults to the canonical fresh ImageNet SiameseNetwork.
        """
        # TEST firewall must pass BEFORE any loader is built
        firewall_check('dev')
        self.unit_test_mode = bool(unit_test_mode)
        if not self.unit_test_mode:
            if config_path != FROZEN_ATTACKER_CONFIG_PATH:
                raise RuntimeError('Scientific attacker requires the canonical attacker config path')
            if file_sha256(config_path) != FROZEN_ATTACKER_CONFIG_SHA:
                raise RuntimeError('Scientific attacker config SHA mismatch')
            if anonymize_fn is not None or training_loader is not None or validation_loader is not None or net_factory is not None:
                raise RuntimeError('Scientific attacker does not accept injected models, loaders, anonymizers, or factories')
            if attacker_seed != 42:
                raise RuntimeError('Scientific attacker seed must be exactly 42')
            if config.get('batch_size') != 32 or config.get('learning_rate') != 1e-4:
                raise RuntimeError('Scientific attacker config does not match frozen optimizer contract')
            if config.get('max_epochs') != 100 or config.get('early_stopping') != 5:
                raise RuntimeError('Scientific attacker config does not match frozen training contract')
            if config.get('train_geometry') != 'anon_anon' or config.get('checkpoint_val_geometry') != 'anon_anon':
                raise RuntimeError('Scientific attacker geometry contract mismatch')
            if config.get('scientific_val_geometry') != 'anon_real':
                raise RuntimeError('Scientific attacker scientific VAL geometry mismatch')

        self.config = config
        self.attacker_seed = attacker_seed
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not self.unit_test_mode:
            if self.device.type != 'cuda' or not torch.cuda.is_available():
                raise RuntimeError('Scientific attacker evaluation requires CUDA')
        self.generator_checkpoint = generator_checkpoint

        # Seed BEFORE net init, DataLoader shuffle, and optimizer.
        utils.seed_all(attacker_seed)

        if anonymize_fn is None:
            if not generator_checkpoint:
                raise RuntimeError("generator_checkpoint must be explicitly provided to DevAttacker")
            _, self.anonymize_fn = load_frozen_anonymizer(
                config, self.device, checkpoint_path=generator_checkpoint,
                image_size=image_size if image_size is not None else config.get('image_size'))
        else:
            self.anonymize_fn = anonymize_fn

        if training_loader is None or validation_loader is None:
            t, v = build_dev_loaders(config, seed=attacker_seed)
            training_loader = training_loader if training_loader is not None else t
            validation_loader = validation_loader if validation_loader is not None else v
        self.training_loader = training_loader
        self.validation_loader = validation_loader
        # NOTE: there is intentionally NO self.test_loader.

        net = net_factory() if net_factory is not None else SiameseNetwork()
        self.net = net.to(self.device)
        self.best_net = None
        self.criterion = nn.BCEWithLogitsLoss().to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=config['learning_rate'])

        self.start_epoch = 0
        self.max_epochs = config['max_epochs']
        self.early_stopping = config['early_stopping']
        self.best_val_loss = float('inf')
        self.best_epoch = None
        self.patience = 0
        self.loss_dict = {'training': [], 'validation': []}

    def train_epoch(self):
        """Attacker TRAIN geometry: anon(x1), anon(x2). Mirrors utils.train_snn."""
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
                raise FloatingPointError('Attacker training loss is non-finite before backward')
            loss.backward()
            for name, param in self.net.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    raise FloatingPointError("Attacker gradient '%s' contains NaN/Inf" % name)
            self.optimizer.step()
            running += loss.item()
        avg_loss = running / max(len(self.training_loader), 1)
        # F5 (M1.4c): Fail-closed on non-finite training loss
        if not np.isfinite(avg_loss):
            raise FloatingPointError(
                "Attacker training loss is non-finite: %s" % avg_loss
            )
        # F5 (M1.4c): Verify all parameters are finite after optimizer step
        for name, param in self.net.named_parameters():
            if not torch.isfinite(param).all():
                raise FloatingPointError(
                    "Attacker parameter '%s' contains NaN/Inf after training step" % name
                )
        return avg_loss

    def validate_selection(self):
        """Attacker checkpoint-VAL geometry: anon(x1), anon(x2).

        Used ONLY for attacker checkpoint selection / early stopping. This is
        NOT the scientific privacy metric.
        """
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
        # F5 (M1.4c): Fail-closed on non-finite validation loss
        if not np.isfinite(avg_loss):
            raise FloatingPointError(
                "Attacker validation loss is non-finite: %s" % avg_loss
            )
        return avg_loss

    def run(self, output_dir=None):
        """Training + validation loop with explicit best-checkpoint tracking and saving.
        Returns a structured dictionary with full history and checkpoint metadata.
        """
        out_dir = output_dir or self.config.get('attacker_output_dir')
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        checkpoint_path = None
        checkpoint_sha256 = None
        termination_reason = 'max_epochs_completed'

        for epoch in range(self.start_epoch, self.max_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate_selection()
            self.loss_dict['training'].append(float(train_loss))
            self.loss_dict['validation'].append(float(val_loss))

            if val_loss < self.best_val_loss:
                self.best_val_loss = float(val_loss)
                self.best_epoch = epoch
                self.best_net = copy.deepcopy(self.net)
                self.patience = 0
                if out_dir:
                    checkpoint_path = os.path.join(out_dir, 'best_attacker.pth')
                    torch.save(self.best_net.state_dict(), checkpoint_path)
            else:
                self.patience += 1

            if self.early_stopping and self.patience >= self.early_stopping:
                termination_reason = 'early_stopping'
                print('Early stopping at epoch %d (patience %d)' % (epoch, self.early_stopping))
                break

        if self.best_epoch is None or self.best_net is None:
            raise RuntimeError("No valid best attacker checkpoint saved during training")

        if out_dir and checkpoint_path and os.path.exists(checkpoint_path):
            from .evaluator_common import file_sha256
            checkpoint_sha256 = file_sha256(checkpoint_path)
            manifest = {
                'best_attacker_path': checkpoint_path,
                'best_attacker_sha256': checkpoint_sha256,
                'best_epoch': self.best_epoch,
                'best_val_loss': float(self.best_val_loss),
                'generator_checkpoint': self.generator_checkpoint,
                'generator_checkpoint_sha256': file_sha256(self.generator_checkpoint) if self.generator_checkpoint else None,
                'attacker_seed': self.attacker_seed,
                'epochs_completed': len(self.loss_dict['training']),
                'termination_reason': termination_reason,
                # F5 (M1.4c): Attacker numerical validity tracking
                'numerical_validity': 'PASS',
                'nan_inf_detected': False,
            }
            with open(os.path.join(out_dir, 'attacker_manifest.json'), 'w') as f:
                import json
                json.dump(manifest, f, indent=2)

        print('Finished attacker TRAIN/VALIDATION! Best Epoch: %s, Best Val BCE: %.5f' % (
            self.best_epoch, self.best_val_loss
        ))

        return {
            'best_epoch': self.best_epoch,
            'best_val_loss': float(self.best_val_loss),
            'epochs_completed': len(self.loss_dict['training']),
            'termination_reason': termination_reason,
            'training_loss': self.loss_dict['training'],
            'validation_loss': self.loss_dict['validation'],
            'checkpoint_path': checkpoint_path,
            'checkpoint_sha256': checkpoint_sha256,
            'best_net': self.best_net,
        }

    # Backward compatibility alias
    train = run