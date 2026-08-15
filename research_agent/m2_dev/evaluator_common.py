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
import pandas as pd
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

# Initial generator checkpoint (M0/M1 frozen)
INITIAL_GENERATOR_PATH = os.path.join(ROOT, 'networks', 'pretrained_generator_prichexy_net.pth')
INITIAL_GENERATOR_SHA = '101226890c061ba5917db7a56a300d1a53988f6eda8767856f10863e2a20aacb'

# Pretrained verification model (M0/M1 frozen)
FROZEN_VERIFIER_PATH = os.path.join(ROOT, 'networks', 'pretrained_verification_model.pth')
FROZEN_VERIFIER_SHA = '331efaed0c0433c69941ddc003a14a936c688d94fd4ecfbefd34e53bfa7c051a'

# Repaired ACLoss SHA (m0_port/ACLoss.py)
REPAIRED_ACLOSS_PATH = os.path.join(ROOT, 'research_agent', 'm0_port', 'ACLoss.py')
REPAIRED_ACLOSS_SHA = '3ed8483718c3ccffb59f76e9dece47e92295a553895e3fd43b1b18cd486b263c'

# Method-neutral anonymizer checkpoint filename (§11)
METHOD_NEUTRAL_CKPT_NAME = 'generator_best_method_neutral.pth'


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify_repaired_acloss():
    """Verify and import the repaired m0_port/ACLoss module. Fail closed on stale ACLoss."""
    if not os.path.exists(REPAIRED_ACLOSS_PATH):
        raise RuntimeError('Repaired ACLoss not found at %s' % REPAIRED_ACLOSS_PATH)
    actual_sha = file_sha256(REPAIRED_ACLOSS_PATH)
    if actual_sha != REPAIRED_ACLOSS_SHA:
        raise RuntimeError('ACLoss SHA mismatch: %s != %s' % (actual_sha, REPAIRED_ACLOSS_SHA))
    from research_agent.m0_port.ACLoss import ACLoss
    return ACLoss, actual_sha, REPAIRED_ACLOSS_PATH


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
# §12 — paired anonymizer data-order sampler & fingerprint
# ---------------------------------------------------------------------------
class FingerprintedRandomSampler(torch.utils.data.Sampler):
    """Deterministic sampler that yields epoch permutations from an explicit generator
    and records the exact ordered sample indices for provenance hashing."""

    def __init__(self, data_source, generator=None, seed=42):
        self.data_source = data_source
        self.seed = seed
        self.generator = generator if generator is not None else torch.Generator().manual_seed(seed)
        self.epoch_indices = []

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        n = len(self.data_source)
        indices = torch.randperm(n, generator=self.generator).tolist()
        self.epoch_indices.append(indices)
        return iter(indices)

    def get_epoch_order_hash(self, epoch=0, pair_identifiers=None):
        if epoch >= len(self.epoch_indices):
            raise IndexError("Epoch %d order not recorded yet (total recorded: %d)" % (epoch, len(self.epoch_indices)))
        indices = self.epoch_indices[epoch]
        h = hashlib.sha256()
        for idx in indices:
            if pair_identifiers is not None:
                item = pair_identifiers[idx]
                if isinstance(item, (list, tuple, np.ndarray)):
                    h.update('|'.join(str(x) for x in item).encode('utf-8'))
                else:
                    h.update(str(item).encode('utf-8'))
            else:
                h.update(str(idx).encode('utf-8'))
            h.update(b'\n')
        return h.hexdigest()


def compute_epoch_order_hash(pair_file_path, seed, epoch=0):
    """Compute the deterministic SHA256 of the exact pair sequence for a specific epoch."""
    pairs = np.loadtxt(pair_file_path, dtype=str)
    n = len(pairs)
    gen = torch.Generator()
    gen.manual_seed(seed)
    # Advance to the requested epoch
    indices = None
    for ep in range(epoch + 1):
        indices = torch.randperm(n, generator=gen).tolist()

    h = hashlib.sha256()
    for idx in indices:
        h.update('|'.join(str(v) for v in pairs[idx]).encode('utf-8'))
        h.update(b'\n')
    return h.hexdigest()


def train_order_fingerprint(pair_file_path, seed, batch_size=16, num_workers=0):
    """Deterministic SHA256 of the epoch-0 anonymizer TRAIN batch order."""
    return compute_epoch_order_hash(pair_file_path, seed, epoch=0)


class LazyPairDataset(torch.utils.data.Dataset):
    """Lazy-loading dataset for anonymization image pairs (NIH Chest X-ray).
    Loads images on-demand in __getitem__, matching exact upstream preprocessing
    and abnormality labels, avoiding long startup delays.
    """

    def __init__(self, phase='training', image_path=None, image_size=IMAGE_SIZE, n_channels=1, max_pairs=None):
        firewall_check('dev')
        assert_dev_phase(phase)

        if image_path is None:
            # Check default locations
            if os.path.exists('/home/minhtt/datasets/nih/images/'):
                image_path = '/home/minhtt/datasets/nih/images/'
            else:
                image_path = './'
        self.image_path = image_path
        self.image_size = image_size
        self.n_channels = n_channels

        if phase == 'training':
            pair_file = os.path.join(ROOT, 'image_pairs', 'image_pairs_training_10000.txt')
        elif phase in ('val', 'validation'):
            pair_file = os.path.join(ROOT, 'image_pairs', 'image_pairs_validation_2000.txt')
        else:
            raise ValueError("Invalid phase for dev dataset: %s" % phase)

        self.image_pairs = np.loadtxt(pair_file, dtype=str)
        if max_pairs is not None:
            self.image_pairs = self.image_pairs[:max_pairs]

        self.resize = transforms.Resize((image_size, image_size))
        self.transform = transforms.ToTensor() if n_channels == 1 else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        csv_path = os.path.join(ROOT, 'Data_Entry_2017_v2020.csv')
        if os.path.exists(csv_path):
            meta_data = pd.read_csv(csv_path).values
            file_to_finding = dict(zip(meta_data[:, 0], meta_data[:, 1]))
        else:
            file_to_finding = {}

        self.PRED_LABEL = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
            'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
            'Pleural_Thickening', 'Hernia'
        ]

        self.ac_labels_1 = []
        self.labels_id = []
        for i in range(len(self.image_pairs)):
            label = np.zeros(14, dtype=np.float32)
            fname = self.image_pairs[i, 0]
            finding = file_to_finding.get(fname, 'No Finding')
            if finding != 'No Finding' and isinstance(finding, str):
                for d in finding.split('|'):
                    if d in self.PRED_LABEL:
                        label[self.PRED_LABEL.index(d)] = 1.0
            self.ac_labels_1.append(label)
            self.labels_id.append(float(self.image_pairs[i][2]))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, index):
        from datasets.Dataset import pil_loader
        p1 = os.path.join(self.image_path, self.image_pairs[index][0])
        p2 = os.path.join(self.image_path, self.image_pairs[index][1])
        if os.path.exists(p1) and os.path.exists(p2):
            img1 = pil_loader(p1, self.n_channels)
            img2 = pil_loader(p2, self.n_channels)
            img1 = self.transform(self.resize(img1))
            img2 = self.transform(self.resize(img2))
        else:
            # Fallback for synthetic tests
            img1 = torch.zeros(self.n_channels, self.image_size, self.image_size)
            img2 = torch.zeros(self.n_channels, self.image_size, self.image_size)

        return img1, img2, self.ac_labels_1[index], self.labels_id[index]


def build_dev_anonymizer_loaders(config, seed=42, num_workers=0):
    """Build canonical anonymizer TRAIN and VAL loaders with FingerprintedRandomSampler.
    Never constructs TEST loaders.
    """
    firewall_check('dev')
    image_path = config.get('image_path', None)
    image_size = config.get('image_size', IMAGE_SIZE)
    batch_size = config.get('batch_size', 16)
    max_pairs = config.get('max_pairs', None)

    train_dataset = LazyPairDataset(phase='training', image_size=image_size, n_channels=1,
                                    image_path=image_path, max_pairs=max_pairs)
    val_dataset = LazyPairDataset(phase='validation', image_size=image_size, n_channels=1,
                                  image_path=image_path, max_pairs=max_pairs)

    train_gen = torch.Generator()
    train_gen.manual_seed(seed)
    train_sampler = FingerprintedRandomSampler(train_dataset, generator=train_gen, seed=seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    return train_loader, val_loader, train_sampler