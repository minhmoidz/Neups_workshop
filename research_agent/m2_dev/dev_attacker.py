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
)
from utils import utils


def load_frozen_anonymizer(config=None, device=None, checkpoint_path=None):
    """Load the frozen generator + legacy operator components for the attacker.
    Requires checkpoint_path or config['generator_checkpoint_path'].
    """
    from networks.UNet_PriCheXyNet import UNet

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = checkpoint_path or (config.get('generator_checkpoint_path') if config else None)
    if not ckpt_path:
        raise ValueError("generator checkpoint path must be explicitly provided")

    generator = UNet(1, 2, 32).to(device)
    generator.load_state_dict(torch.load(ckpt_path, map_location=device))
    generator.eval()
    for p in generator.parameters():
        p.requires_grad = False
    grid_identity, gauss_filter = make_flow_field_components(device)
    anonymize_fn = build_anonymize_fn(generator, grid_identity, gauss_filter, MU)
    return generator, anonymize_fn


class DevAttacker:
    def __init__(self, config, attacker_seed=42, device=None,
                 anonymize_fn=None, training_loader=None, validation_loader=None,
                 net_factory=None, generator_checkpoint=None):
        """Development attacker runner.

        :param config: dev config dict (image_path, batch_size, learning_rate,
            max_epochs, early_stopping, generator_checkpoint_path, ...).
        :param attacker_seed: attacker RNG seed (42 for S1, 42/43/44 for S2).
        :param device: torch device; defaults to CUDA if available.
        :param anonymize_fn: optional injected anonymizer (tests); defaults to
            the frozen legacy flow_field operator from config.
        :param generator_checkpoint: explicit path to selected M2 generator checkpoint.
        :param training_loader / validation_loader: optional injected loaders
            (tests); defaults to TRAIN/VAL retrainSNN loaders. TEST loader is
            NEVER constructed here.
        :param net_factory: optional callable returning an attacker net (tests);
            defaults to the canonical fresh ImageNet SiameseNetwork.
        """
        # TEST firewall must pass BEFORE any loader is built
        firewall_check('dev')

        self.config = config
        self.attacker_seed = attacker_seed
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Seed BEFORE net init, DataLoader shuffle, and optimizer.
        utils.seed_all(attacker_seed)

        if anonymize_fn is None:
            _, self.anonymize_fn = load_frozen_anonymizer(config, self.device, checkpoint_path=generator_checkpoint)
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
        self.best_net = copy.deepcopy(self.net)
        self.criterion = nn.BCEWithLogitsLoss().to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=config['learning_rate'])

        self.start_epoch = 0
        self.max_epochs = config['max_epochs']
        self.early_stopping = config['early_stopping']
        self.best_val_loss = float('inf')
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
            loss.backward()
            self.optimizer.step()
            running += loss.item()
        return running / max(len(self.training_loader), 1)

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
        return running / max(len(self.validation_loader), 1)

    def run(self):
        """Training + validation loop ONLY. No testing_evaluation(), no run() of
        the upstream AgentSiameseNetwork."""
        for epoch in range(self.start_epoch, self.max_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate_selection()
            self.loss_dict['training'].append(train_loss)
            self.loss_dict['validation'].append(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_net = copy.deepcopy(self.net)
                self.patience = 0
            else:
                self.patience += 1

            if self.early_stopping and self.patience >= self.early_stopping:
                print('Early stopping at epoch %d (patience %d)' % (epoch, self.early_stopping))
                break

        print('Finished attacker TRAIN/VALIDATION!')
        return self.best_net