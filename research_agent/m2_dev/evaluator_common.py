"""M1.2 development-evaluator shared infrastructure.

All development (M2) execution MUST go through the fail-closed wrappers below so
that no TEST loader / TEST pair file / TEST classifier fold is ever constructed.
The upstream evaluator code (agents/AgentSiameseNetwork.py, chexnet/eval_model.py,
utils/utils.py) is NOT rewritten — this module only builds dedicated TRAIN/VAL
wrappers around the preserved upstream semantics.

Legacy operator lock (M1 frozen): flow_field, mu=0.01, Gaussian kernel 9, sigma 2,
padding_mode='border', align_corners=True. A single shared anonymize helper is used
by attacker train / attacker checkpoint-val / mixed-val privacy / classification val.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
for _p in (ROOT, os.path.join(ROOT, 'research_agent')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hashlib
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils.GaussianSmoothing import GaussianSmoothing
from test_firewall import TestFirewall

# ---------------------------------------------------------------------------
# Frozen legacy operator (M1)
# ---------------------------------------------------------------------------
MU = 0.01
GAUSS_KERNEL = 9
GAUSS_SIGMA = 2
PADDING_MODE = 'border'
ALIGN_CORNERS = True
IMAGE_SIZE = 256

IMAGENET_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Development fold whitelist / blacklist
DEV_ALLOWED_FOLDS = {'val', 'validation'}
DEV_FORBIDDEN_FOLDS = {'test', 'testing', 'final_test'}

# Frozen classifier checkpoint (M1, verified by M0.1)
FROZEN_CLASSIFIER_PATH = os.path.join(ROOT, 'networks', 'pretrained_classifier.pth')
FROZEN_CLASSIFIER_SHA = '8ad15b38286f734ea135394ac5e7c79f4a6c1d2db4d563fbe1f81cf3dbe5e663'

# Method-neutral anonymizer checkpoint filename (§11)
METHOD_NEUTRAL_CKPT_NAME = 'generator_best_method_neutral.pth'


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# TEST firewall integration — fail-closed before any artifact is opened
# ---------------------------------------------------------------------------
def assert_dev_phase(phase):
    """Reject TEST phases before any loader construction. dev/val/train OK."""
    if phase is not None and str(phase).lower() in DEV_FORBIDDEN_FOLDS:
        raise RuntimeError(
            'TEST firewall: phase=%r requests the CLOSED TEST benchmark. '
            'Development loaders may only use training/validation.' % phase)
    return True


def assert_dev_fold(fold):
    """Reject a TEST classification fold BEFORE dataset construction."""
    if fold is not None and str(fold).lower() in DEV_FORBIDDEN_FOLDS:
        raise RuntimeError(
            'TEST firewall: fold=%r requests the CLOSED TEST benchmark. '
            'Development classifier evaluation is VAL-only.' % fold)
    if str(fold).lower() not in DEV_ALLOWED_FOLDS:
        raise RuntimeError('classification development fold must be "val", got %r' % fold)
    return True


def firewall_check(mode='dev'):
    return TestFirewall(allow=False).check(mode)


# ---------------------------------------------------------------------------
# Shared legacy operator helpers
# ---------------------------------------------------------------------------
def make_flow_field_components(device, image_size=IMAGE_SIZE):
    """Build the identity grid + gaussian smoother exactly as upstream Agent.py."""
    d = torch.linspace(-1, 1, image_size)
    mesh_x, mesh_y = torch.meshgrid((d, d), indexing='ij')
    grid_identity = torch.stack((mesh_y, mesh_x), 2).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    gauss_filter = GaussianSmoothing(channels=2, kernel_size=GAUSS_KERNEL, sigma=GAUSS_SIGMA).to(device)
    return grid_identity, gauss_filter


def anonymize(image, generator, grid_identity, gauss_filter, mu=MU):
    """Single shared legacy flow_field anonymization operator (M1 lock).

    :param image: (N,1,H,W) raw image tensor on the generator's device.
    :return: (N,1,H,W) anonymized image.
    """
    grids = generator(image)
    grids = grid_identity - mu * grids
    grids = gauss_filter(grids)
    grids = grids.permute(0, 2, 3, 1)
    return F.grid_sample(image, grids, padding_mode=PADDING_MODE, align_corners=ALIGN_CORNERS)


def snn_preprocess(x):
    """SNN input preprocessing (upstream train_snn/validate_snn/test_snn):
    expand 1ch -> 3ch then ImageNet normalize (no resize for the SNN)."""
    x = x.expand(-1, 3, -1, -1)
    return IMAGENET_NORMALIZE(x)


def classifier_preprocess(x):
    """Classification input preprocessing (upstream eval_model for flow_field):
    expand 1ch -> 3ch, Resize 224, ImageNet normalize."""
    x = x.expand(-1, 3, -1, -1)
    x = transforms.Resize((224, 224))(x)
    return IMAGENET_NORMALIZE(x)


def build_anonymize_fn(generator, grid_identity, gauss_filter, mu=MU):
    """Curry the shared anonymize helper for a frozen generator (eval, no grad)."""
    for p in generator.parameters():
        p.requires_grad = False
    generator.eval()

    def fn(image):
        return anonymize(image, generator, grid_identity, gauss_filter, mu)

    return fn


def build_dev_loaders(config, seed=42, num_workers=0, shuffle_train=True, experimental_step='retrainSNN'):
    """Build ONLY training + validation loaders (retrainSNN). Never 'testing'.

    Uses an explicit torch.Generator seeded `seed` so the TRAIN shuffle order is
    deterministic and reproducible for the paired data-order contract (§12).
    """
    firewall_check('dev')
    from utils import utils

    gen = torch.Generator()
    gen.manual_seed(seed)

    def make(phase, shuffle):
        assert_dev_phase(phase)
        data_set = utils.get_data_loader(
            phase=phase, experimental_step=experimental_step,
            image_size=config.get('image_size', IMAGE_SIZE),
            n_channels=config.get('n_channels', 1),
            batch_size=config.get('batch_size', 32),
            shuffle=shuffle, num_workers=num_workers, pin_memory=False,
            image_path=config.get('image_path', './'))
        return data_set

    training_loader = make('training', shuffle_train)
    validation_loader = make('validation', False)
    return training_loader, validation_loader


# ---------------------------------------------------------------------------
# §11 — method-neutral anonymizer checkpoint selection
# ---------------------------------------------------------------------------
def compute_epoch_totals(ac_bce, privacy_term, feature_term, feature_loss_weight):
    """Break down one validation epoch into the §11 loss components.

    B_dev: optimization_total = ac_bce + privacy_term (feature_weight=0).
    C4:    optimization_total = ac_bce + privacy_term + feature_term.
    BOTH arms: selection_total = ac_bce + privacy_term (feature EXCLUDED).
    """
    optimization_total = ac_bce + privacy_term + feature_loss_weight * feature_term
    selection_total = ac_bce + privacy_term
    return {
        'ac_bce': ac_bce,
        'privacy_term': privacy_term,
        'feature_term': feature_term,
        'optimization_total': optimization_total,
        'selection_total': selection_total,
    }


def select_method_neutral_best(epoch_metrics):
    """Pick the checkpoint epoch by MINIMUM selection_total; tie-break earliest epoch.

    :param epoch_metrics: list of dicts (from compute_epoch_totals) in epoch order.
    :return: index (epoch number) of the selected best checkpoint.
    """
    best_idx = 0
    best_sel = None
    for i, m in enumerate(epoch_metrics):
        sel = m['selection_total']
        if best_sel is None or sel < best_sel - 1e-12:
            best_sel = sel
            best_idx = i
    return best_idx


# ---------------------------------------------------------------------------
# §12 — paired anonymizer data-order fingerprint
# ---------------------------------------------------------------------------
class _IndexDataset:
    """Lightweight dataset that yields only indices; used to replicate the exact
    DataLoader shuffle order without loading real images."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i


def train_order_fingerprint(pair_file_path, seed, batch_size, num_workers=0):
    """Deterministic SHA256 of the epoch-0 anonymizer TRAIN batch order.

    Replicates the exact DataLoader shuffle (RandomSampler with an explicit
    generator) that the M2 anonymizer loader will use, then hashes the ordered
    pair identifiers. B_dev and C4 (same seed, same pair file, same batch) MUST
    produce the same hash; a different seed MUST produce a different hash.
    """
    pairs = np.loadtxt(pair_file_path, dtype=str)
    n = len(pairs)
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(_IndexDataset(n), batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, generator=gen)
    order = []
    for batch in loader:
        order.extend(int(i) for i in batch.tolist())

    h = hashlib.sha256()
    for i in order:
        h.update('|'.join(str(v) for v in pairs[i]).encode('utf-8'))
        h.update(b'\n')
    return h.hexdigest()