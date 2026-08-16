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
import subprocess
from collections.abc import Mapping

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
for _p in (ROOT, os.path.join(ROOT, 'research_agent')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json
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
DEV_FORBIDDEN_FOLDS = {'test', 'testing', 'final_test', 'eval_test'}

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

# Frozen metadata and configs (M1.4b frozen)
FROZEN_METADATA_PATH = os.path.join(ROOT, 'Data_Entry_2017_v2020.csv')
FROZEN_METADATA_SHA = 'dc1d2df67fdc1c5a7601d48699cda2b13dc2c4841488b4183dcf04884dbaca11'

FROZEN_B_DEV_CONFIG_PATH = os.path.join(ROOT, 'config_files', 'config_dev_restored_baseline.json')
FROZEN_B_DEV_CONFIG_SHA = '14d3943f798d5855b4a49d55ecc6af858647f514f30c1cc7c803c7edebab30b6'

FROZEN_C4_CONFIG_PATH = os.path.join(ROOT, 'config_files', 'config_dev_c4.json')
FROZEN_C4_CONFIG_SHA = '7cbdfce84e41317dac73651d0d7d6080cf68871e0340a68ec0d9191383716a8a'

FROZEN_ATTACKER_CONFIG_PATH = os.path.join(ROOT, 'config_files', 'config_dev_attacker_s1.json')
FROZEN_ATTACKER_CONFIG_SHA = '72923582e6595103026ac0fa3488ace4888f70ca09aa91f7cdf49d8e53eb70e1'

# Approved scientific data root (M1.4c.3). All non-unit classification /
# privacy / attacker APIs bind image_path to this exact root so an arbitrary
# direct API config cannot select a different image directory while claiming
# scientific provenance. Unit-test temporary roots remain available ONLY under
# explicit unit_test_mode=True.
SCIENTIFIC_IMAGE_ROOT = '/home/minhtt/datasets/nih/images/'

# Method-neutral anonymizer checkpoint filename (§11)
METHOD_NEUTRAL_CKPT_NAME = 'generator_best_method_neutral.pth'

# F2 (M1.4c): Frozen classification split CSV & structural fingerprints
FROZEN_NIH_LABELS_PATH = os.path.join(ROOT, 'chexnet', 'nih_labels.csv')
FROZEN_NIH_LABELS_SHA = '80324996867e73546bd7a09025df4a4cc3243fc00663b753023ccd90a9b5f8b9'

FROZEN_CLASSIFICATION_VAL_N_IMAGES = 10816
FROZEN_CLASSIFICATION_VAL_N_PATIENTS = 3854
FROZEN_CLASSIFICATION_VAL_IMAGE_INDEX_SHA = 'c2ff15d7deb0e126b096d4392f4d8844c80593d16ad939ea44189b44051b8ab6'
FROZEN_CLASSIFICATION_VAL_PATIENT_SEQUENCE_SHA = 'c444238f8be4c6f2e6ee5523a1966ef974a182aabc895b3819ebbaeec4f621ba'
FROZEN_CLASSIFICATION_VAL_LABEL_MATRIX_SHA = 'bef28de2c55d5767c5d930bca8f86253c75a8be2cf00552ce0ceb579e75a28cc'

# Canonical 14-pathology NIH list — SINGLE source of truth (M1.4c.2).
# Used by the classifier evaluator, the raw-artifact replay reader, validity
# checks and tests. Ordering is frozen; do not reorder.
NIH_PATHOLOGIES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
    'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]

# Historical alias retained for backward compatibility; identical object/order.
REQUIRED_PATHOLOGY_COLUMNS = NIH_PATHOLOGIES

# M1.4c.3 immutable execution-boundary anchors.  The lock digest is deliberately
# kept outside M2_S1_EXECUTION_LOCK.json so changing the lock cannot change the
# value used to authenticate it.
EXECUTION_LOCK_PATH = os.path.join(ROOT, 'research_agent', 'M2_S1_EXECUTION_LOCK.json')
FROZEN_M2_EXECUTION_LOCK_SHA256 = 'c8ea322adf46a3524ee7c765fa73f4c851af0cf2749eb619332f27c822b11acc'
FROZEN_SOURCE_TAG = 'm2-s1-certified-v1'
FROZEN_SOURCE_BRANCH = 'research/method-restart'
FROZEN_VAL_PAIR_PATH = os.path.join(ROOT, 'image_pairs', 'image_pairs_validation_2000.txt')
FROZEN_VAL_PAIR_SHA256 = '9e33a081dfd5e4f28e658a9d13417f8a61f24cba60b2cb03272b20535b9fa9f7'
FROZEN_VAL_PAIR_COUNT = 2000
FROZEN_TRAIN_PAIR_PATH = os.path.join(ROOT, 'image_pairs', 'image_pairs_training_10000.txt')
FROZEN_TRAIN_PAIR_SHA256 = '3c535eed013305bacf231dea9c72fb047cc6b6cb15e3958ef7a308956394b268'
FROZEN_TRAIN_ORDER_EPOCH0_SHA256 = '920a127341a85967f959863e3471fea60f498d59d6485fb75c57c04c73f11967'
FROZEN_TRAIN_ORDER_EPOCH1_SHA256 = '1eb349a8ad0e912da5df3babe2995def454e2b4382993a39926be1cf29c4c845'


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify_execution_lock_integrity(lock_path=None):
    """Authenticate the execution lock before any of its fields are trusted."""
    lock_path = lock_path or EXECUTION_LOCK_PATH
    if not os.path.exists(lock_path):
        raise FileNotFoundError('Execution lock missing: %s' % lock_path)
    actual_sha = file_sha256(lock_path)
    if actual_sha != FROZEN_M2_EXECUTION_LOCK_SHA256:
        raise RuntimeError(
            'Execution lock SHA mismatch: %s != frozen %s' %
            (actual_sha, FROZEN_M2_EXECUTION_LOCK_SHA256)
        )
    with open(lock_path) as f:
        lock = json.load(f)
    if lock.get('protocol') != 'M2_S1_EXECUTION_LOCK':
        raise RuntimeError('Execution lock protocol identifier is invalid')
    if lock.get('test_firewall') != 'CLOSED':
        raise RuntimeError('Execution lock TEST firewall is not CLOSED')
    if lock.get('scientific_choices_frozen', {}).get('max_epochs') != 250:
        raise RuntimeError('Execution lock max_epochs is not frozen at 250')
    return lock, actual_sha


def validate_scientific_args(args, unit_test_mode=False):
    """Validate the frozen scientific CLI contract at every public boundary."""
    if unit_test_mode:
        return args
    if not getattr(args, 'scientific_m2_s1', False):
        raise RuntimeError('Scientific M2-S1 execution requires --scientific-m2-s1')
    expected = {
        'arm': 'all', 'max_epochs': 250, 'attacker_epochs': 100,
        'attacker_patience': 5, 'seed': 42, 'attacker_seed': 42,
    }
    for key, value in expected.items():
        actual = getattr(args, key, None)
        if actual != value:
            raise ValueError('Scientific M2-S1 requires %s == %r, got %r' % (key, value, actual))
    device = getattr(args, 'device', None)
    if device is not None and str(device).lower().startswith('cpu'):
        raise RuntimeError('Scientific M2-S1 does not support --device cpu')
    if device is not None and str(device).lower().startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Scientific M2-S1 requires an available CUDA device')
    if device is None and not torch.cuda.is_available():
        raise RuntimeError('Scientific M2-S1 requires CUDA; no CUDA device is available')
    if getattr(args, 'resume', False) or getattr(args, 'resume_checkpoint', None):
        raise RuntimeError('Scientific resume is not certified; restart from epoch 0')
    if getattr(args, 'fold', 'val') not in (None, 'val'):
        raise ValueError('Scientific classification fold must be exactly "val"')
    return args


def _semantic_pair_sha(pair_file_path):
    h = hashlib.sha256()
    with open(pair_file_path) as f:
        for raw in f:
            fields = raw.strip().split()
            if not fields:
                continue
            h.update(('|'.join(fields) + '\n').encode('utf-8'))
    return h.hexdigest()


def verify_frozen_val_pair_provenance(pair_file_path=None):
    pair_file_path = pair_file_path or FROZEN_VAL_PAIR_PATH
    if file_sha256(pair_file_path) != FROZEN_VAL_PAIR_SHA256:
        raise RuntimeError('VAL pair file SHA mismatch')
    with open(pair_file_path) as f:
        rows = [line.strip().split() for line in f if line.strip()]
    if len(rows) != FROZEN_VAL_PAIR_COUNT or any(len(row) != 3 for row in rows):
        raise RuntimeError('VAL pair file must contain exactly 2000 3-field rows')
    return {
        'pair_file': os.path.abspath(pair_file_path),
        'pair_file_sha256': FROZEN_VAL_PAIR_SHA256,
        'pair_count': len(rows),
        'pair_order_sha256': _semantic_pair_sha(pair_file_path),
    }


def compute_expected_train_order_hashes(pair_file_path=None, seed=42, epochs=250):
    """Compute and validate the complete deterministic TRAIN order contract."""
    pair_file_path = pair_file_path or FROZEN_TRAIN_PAIR_PATH
    actual_sha = file_sha256(pair_file_path)
    if actual_sha != FROZEN_TRAIN_PAIR_SHA256:
        raise RuntimeError('TRAIN pair file SHA mismatch: %s != %s' % (actual_sha, FROZEN_TRAIN_PAIR_SHA256))
    pairs = np.loadtxt(pair_file_path, dtype=str)
    gen = torch.Generator()
    gen.manual_seed(seed)
    hashes = []
    for epoch in range(epochs):
        indices = torch.randperm(len(pairs), generator=gen).tolist()
        h = hashlib.sha256()
        for idx in indices:
            h.update(('|'.join(str(v) for v in pairs[idx]) + '\n').encode('utf-8'))
        hashes.append(h.hexdigest())
    if hashes[0] != FROZEN_TRAIN_ORDER_EPOCH0_SHA256 or hashes[1] != FROZEN_TRAIN_ORDER_EPOCH1_SHA256:
        raise RuntimeError('Frozen TRAIN order epoch 0/1 hashes do not match; refusing to rewrite expectations')
    return hashes


def _frame_sha(values):
    h = hashlib.sha256()
    for value in values:
        h.update((str(value) + '\n').encode('utf-8'))
    return h.hexdigest()


def classification_artifact_fingerprints(pred_df):
    """Return structural hashes for the production classification writer output."""
    if 'Image Index' not in pred_df.columns:
        raise ValueError('classification artifact missing Image Index')
    image_index = pred_df['Image Index'].astype(str).tolist()
    if len(set(image_index)) != len(image_index):
        raise ValueError('classification artifact contains duplicate Image Index')
    patients = [name.split('_')[0] for name in image_index]
    label_rows = []
    for _, row in pred_df.iterrows():
        vals = []
        for pathology in NIH_PATHOLOGIES:
            value = row[pathology]
            if not np.isfinite(float(value)) or int(value) not in (0, 1) or float(value) != int(value):
                raise ValueError('classification ground truth must be finite binary')
            vals.append(str(int(value)))
            probability = float(row['prob_' + pathology])
            if not np.isfinite(probability):
                raise ValueError('classification probability must be finite')
        label_rows.append(','.join(vals))
    return {
        'n_images': len(image_index),
        'n_patients': len(set(patients)),
        'image_index_sha256': _frame_sha(image_index),
        'patient_sequence_sha256': _frame_sha(patients),
        'label_matrix_sha256': _frame_sha(label_rows),
    }


def verify_classification_artifact_structure(pred_df, strict=True):
    required = ['Image Index'] + [p for p in NIH_PATHOLOGIES] + ['prob_' + p for p in NIH_PATHOLOGIES]
    missing = [column for column in required if column not in pred_df.columns]
    if missing:
        raise ValueError('classification artifact missing columns: %s' % missing)
    fingerprints = classification_artifact_fingerprints(pred_df)
    if strict:
        expected = {
            'n_images': FROZEN_CLASSIFICATION_VAL_N_IMAGES,
            'n_patients': FROZEN_CLASSIFICATION_VAL_N_PATIENTS,
            'image_index_sha256': FROZEN_CLASSIFICATION_VAL_IMAGE_INDEX_SHA,
            'patient_sequence_sha256': FROZEN_CLASSIFICATION_VAL_PATIENT_SEQUENCE_SHA,
            'label_matrix_sha256': FROZEN_CLASSIFICATION_VAL_LABEL_MATRIX_SHA,
        }
        for key, value in expected.items():
            if fingerprints[key] != value:
                raise ValueError('classification artifact %s mismatch: %s != %s' % (key, fingerprints[key], value))
    return fingerprints


def verify_classification_val_contract(labels_csv_path=None):
    """Verify existence, SHA, fold structure, row/patient count, and exact structural fingerprints
    of the classification VAL cohort from chexnet/nih_labels.csv.
    Fail-closed before scientific execution if any parameter drifts.
    """
    csv_path = labels_csv_path or FROZEN_NIH_LABELS_PATH
    if not os.path.exists(csv_path):
        raise FileNotFoundError("Classification split CSV missing: %s" % csv_path)

    actual_csv_sha = file_sha256(csv_path)
    if actual_csv_sha != FROZEN_NIH_LABELS_SHA:
        raise RuntimeError(
            "Classification split CSV SHA mismatch: actual %s != frozen %s" % (
                actual_csv_sha, FROZEN_NIH_LABELS_SHA
            )
        )

    df = pd.read_csv(csv_path)
    if 'fold' not in df.columns:
        raise RuntimeError("Classification split CSV missing required 'fold' column")
    if 'Image Index' not in df.columns:
        raise RuntimeError("Classification split CSV missing required 'Image Index' column")

    for col in REQUIRED_PATHOLOGY_COLUMNS:
        if col not in df.columns:
            raise RuntimeError("Classification split CSV missing required pathology column: %s" % col)

    val_df = df[df['fold'] == 'val'].copy()
    val_df = val_df.sort_values('Image Index').reset_index(drop=True)

    n_images = len(val_df)
    if n_images != FROZEN_CLASSIFICATION_VAL_N_IMAGES:
        raise RuntimeError(
            "Classification VAL image count mismatch: %d != frozen %d" % (
                n_images, FROZEN_CLASSIFICATION_VAL_N_IMAGES
            )
        )

    # Compute Image Index fingerprint
    h_img = hashlib.sha256()
    patient_ids = []
    for idx in val_df['Image Index']:
        h_img.update((str(idx) + '\n').encode('utf-8'))
        patient_ids.append(str(idx).split('_')[0])

    actual_img_sha = h_img.hexdigest()
    if actual_img_sha != FROZEN_CLASSIFICATION_VAL_IMAGE_INDEX_SHA:
        raise RuntimeError(
            "Classification VAL image index fingerprint mismatch: %s != frozen %s" % (
                actual_img_sha, FROZEN_CLASSIFICATION_VAL_IMAGE_INDEX_SHA
            )
        )

    # Compute Patient ID sequence fingerprint
    h_pat = hashlib.sha256()
    for pid in patient_ids:
        h_pat.update((pid + '\n').encode('utf-8'))

    actual_pat_sha = h_pat.hexdigest()
    if actual_pat_sha != FROZEN_CLASSIFICATION_VAL_PATIENT_SEQUENCE_SHA:
        raise RuntimeError(
            "Classification VAL patient sequence fingerprint mismatch: %s != frozen %s" % (
                actual_pat_sha, FROZEN_CLASSIFICATION_VAL_PATIENT_SEQUENCE_SHA
            )
        )

    n_patients = len(set(patient_ids))
    if n_patients != FROZEN_CLASSIFICATION_VAL_N_PATIENTS:
        raise RuntimeError(
            "Classification VAL patient count mismatch: %d != frozen %d" % (
                n_patients, FROZEN_CLASSIFICATION_VAL_N_PATIENTS
            )
        )

    # Compute 14-D Label matrix fingerprint
    h_lbl = hashlib.sha256()
    for _, row in val_df.iterrows():
        label_vec = [str(int(row[p])) for p in REQUIRED_PATHOLOGY_COLUMNS]
        h_lbl.update((','.join(label_vec) + '\n').encode('utf-8'))

    actual_lbl_sha = h_lbl.hexdigest()
    if actual_lbl_sha != FROZEN_CLASSIFICATION_VAL_LABEL_MATRIX_SHA:
        raise RuntimeError(
            "Classification VAL label matrix fingerprint mismatch: %s != frozen %s" % (
                actual_lbl_sha, FROZEN_CLASSIFICATION_VAL_LABEL_MATRIX_SHA
            )
        )

    return {
        'status': 'PASS',
        'classification_split_csv_path': csv_path,
        'classification_split_csv_sha256': actual_csv_sha,
        'classification_val_n_images': n_images,
        'classification_val_n_patients': n_patients,
        'classification_val_image_index_sha256': actual_img_sha,
        'classification_val_patient_sequence_sha256': actual_pat_sha,
        'classification_val_label_matrix_sha256': actual_lbl_sha,
        'required_pathology_columns': REQUIRED_PATHOLOGY_COLUMNS,
    }


def verify_repaired_acloss():
    """Verify and import the repaired m0_port/ACLoss module. Fail closed on stale ACLoss."""
    if not os.path.exists(REPAIRED_ACLOSS_PATH):
        raise RuntimeError('Repaired ACLoss not found at %s' % REPAIRED_ACLOSS_PATH)
    actual_sha = file_sha256(REPAIRED_ACLOSS_PATH)
    if actual_sha != REPAIRED_ACLOSS_SHA:
        raise RuntimeError('ACLoss SHA mismatch: %s != %s' % (actual_sha, REPAIRED_ACLOSS_SHA))
    from research_agent.m0_port.ACLoss import ACLoss
    return ACLoss, actual_sha, REPAIRED_ACLOSS_PATH


def verify_frozen_scientific_configs(lock_path=None):
    """Verify exact SHA256 and semantic configuration invariants of frozen experiment configs."""
    lock_path = lock_path or EXECUTION_LOCK_PATH
    lock, _ = verify_execution_lock_integrity(lock_path)
    prov = lock.get('artifact_provenance', {})
    exp_b_sha = prov.get('b_dev_config_sha256', FROZEN_B_DEV_CONFIG_SHA)
    exp_c4_sha = prov.get('c4_config_sha256', FROZEN_C4_CONFIG_SHA)
    exp_att_sha = prov.get('attacker_config_sha256', FROZEN_ATTACKER_CONFIG_SHA)

    # 1. B_dev config
    if not os.path.exists(FROZEN_B_DEV_CONFIG_PATH):
        raise FileNotFoundError("B_dev config not found: %s" % FROZEN_B_DEV_CONFIG_PATH)
    actual_b_sha = file_sha256(FROZEN_B_DEV_CONFIG_PATH)
    if actual_b_sha != exp_b_sha:
        raise RuntimeError("B_dev config SHA mismatch: %s != %s" % (actual_b_sha, exp_b_sha))

    # 2. C4 config
    if not os.path.exists(FROZEN_C4_CONFIG_PATH):
        raise FileNotFoundError("C4 config not found: %s" % FROZEN_C4_CONFIG_PATH)
    actual_c4_sha = file_sha256(FROZEN_C4_CONFIG_PATH)
    if actual_c4_sha != exp_c4_sha:
        raise RuntimeError("C4 config SHA mismatch: %s != %s" % (actual_c4_sha, exp_c4_sha))

    # 3. Attacker config
    if not os.path.exists(FROZEN_ATTACKER_CONFIG_PATH):
        raise FileNotFoundError("Attacker config not found: %s" % FROZEN_ATTACKER_CONFIG_PATH)
    actual_att_sha = file_sha256(FROZEN_ATTACKER_CONFIG_PATH)
    if actual_att_sha != exp_att_sha:
        raise RuntimeError("Attacker config SHA mismatch: %s != %s" % (actual_att_sha, exp_att_sha))

    # 4. Semantic Invariant Assertions
    with open(FROZEN_B_DEV_CONFIG_PATH) as f:
        b_cfg = json.load(f)
    with open(FROZEN_C4_CONFIG_PATH) as f:
        c4_cfg = json.load(f)
    with open(FROZEN_ATTACKER_CONFIG_PATH) as f:
        att_cfg = json.load(f)

    # Common anonymizer assertions
    for name, cfg, exp_feat_weight in [('B_dev', b_cfg, 0.0), ('C4', c4_cfg, 1.0)]:
        if cfg.get('operator') != 'legacy':
            raise RuntimeError("%s config operator must be 'legacy', got %r" % (name, cfg.get('operator')))
        if cfg.get('transform_mode') != 'legacy':
            raise RuntimeError("%s config transform_mode must be 'legacy', got %r" % (name, cfg.get('transform_mode')))
        if cfg.get('mu') != 0.01:
            raise RuntimeError("%s config mu must be 0.01, got %r" % (name, cfg.get('mu')))
        if cfg.get('image_size') != 256:
            raise RuntimeError("%s config image_size must be 256, got %r" % (name, cfg.get('image_size')))
        if cfg.get('batch_size') != 16:
            raise RuntimeError("%s config batch_size must be 16, got %r" % (name, cfg.get('batch_size')))
        if cfg.get('accumulation_steps') != 1:
            raise RuntimeError("%s config accumulation_steps must be 1, got %r" % (name, cfg.get('accumulation_steps')))
        if cfg.get('optimizer') != 'Adam':
            raise RuntimeError("%s config optimizer must be 'Adam', got %r" % (name, cfg.get('optimizer')))
        if cfg.get('learning_rate') != 1e-4:
            raise RuntimeError("%s config learning_rate must be 1e-4, got %r" % (name, cfg.get('learning_rate')))
        if cfg.get('max_epochs') != 250:
            raise RuntimeError("%s config max_epochs must be 250, got %r" % (name, cfg.get('max_epochs')))
        if cfg.get('seed') != 42:
            raise RuntimeError("%s config seed must be 42, got %r" % (name, cfg.get('seed')))
        if cfg.get('feature_loss_weight') != exp_feat_weight:
            raise RuntimeError("%s config feature_loss_weight must be %r, got %r" % (name, exp_feat_weight, cfg.get('feature_loss_weight')))
        if cfg.get('use_budget_map') is not False:
            raise RuntimeError("%s config use_budget_map must be False, got %r" % (name, cfg.get('use_budget_map')))
        if cfg.get('stochastic_lambda') != 0.0:
            raise RuntimeError("%s config stochastic_lambda must be 0.0, got %r" % (name, cfg.get('stochastic_lambda')))
        if cfg.get('ver_ensemble_size') != 1:
            raise RuntimeError("%s config ver_ensemble_size must be 1, got %r" % (name, cfg.get('ver_ensemble_size')))
        if cfg.get('ver_restart_every') != 0:
            raise RuntimeError("%s config ver_restart_every must be 0, got %r" % (name, cfg.get('ver_restart_every')))
        if cfg.get('checkpoint_selection_rule') != 'lowest_validation_total_loss_method_neutral':
            raise RuntimeError("%s config selection rule must be 'lowest_validation_total_loss_method_neutral', got %r" % (name, cfg.get('checkpoint_selection_rule')))
        if cfg.get('checkpoint_selection_tiebreak') != 'earliest_epoch':
            raise RuntimeError("%s config selection tiebreak must be 'earliest_epoch', got %r" % (name, cfg.get('checkpoint_selection_tiebreak')))

    # C4-specific assertions
    if c4_cfg.get('feature_loss') != 'mse':
        raise RuntimeError("C4 config feature_loss must be 'mse', got %r" % c4_cfg.get('feature_loss'))
    if c4_cfg.get('feature_loss_detach_source') is not True:
        raise RuntimeError("C4 config feature_loss_detach_source must be True, got %r" % c4_cfg.get('feature_loss_detach_source'))
    if c4_cfg.get('feature_representation') != 'densenet121_penultimate_pooled_1024':
        raise RuntimeError("C4 config feature_representation must be 'densenet121_penultimate_pooled_1024', got %r" % c4_cfg.get('feature_representation'))

    # Attacker assertions
    if att_cfg.get('batch_size') != 32:
        raise RuntimeError("Attacker config batch_size must be 32, got %r" % att_cfg.get('batch_size'))
    if att_cfg.get('learning_rate') != 1e-4:
        raise RuntimeError("Attacker config learning_rate must be 1e-4, got %r" % att_cfg.get('learning_rate'))
    if att_cfg.get('max_epochs') != 100:
        raise RuntimeError("Attacker config max_epochs must be 100, got %r" % att_cfg.get('max_epochs'))
    if att_cfg.get('early_stopping') != 5:
        raise RuntimeError("Attacker config early_stopping must be 5, got %r" % att_cfg.get('early_stopping'))
    if att_cfg.get('attacker_seed') != 42:
        raise RuntimeError("Attacker config attacker_seed must be 42, got %r" % att_cfg.get('attacker_seed'))
    if att_cfg.get('train_geometry') != 'anon_anon':
        raise RuntimeError("Attacker config train_geometry must be 'anon_anon', got %r" % att_cfg.get('train_geometry'))
    if att_cfg.get('checkpoint_val_geometry') != 'anon_anon':
        raise RuntimeError("Attacker config checkpoint_val_geometry must be 'anon_anon', got %r" % att_cfg.get('checkpoint_val_geometry'))
    if att_cfg.get('scientific_val_geometry') != 'anon_real':
        raise RuntimeError("Attacker config scientific_val_geometry must be 'anon_real', got %r" % att_cfg.get('scientific_val_geometry'))

    return {
        'status': 'PASS',
        'b_dev_config_sha256': actual_b_sha,
        'c4_config_sha256': actual_c4_sha,
        'attacker_config_sha256': actual_att_sha,
    }


def verify_attacker_config_matches_frozen(config):
    """M1.4c.3: The in-memory attacker config must itself match the canonical
    frozen attacker config for every scientific field (including the approved
    data root), not merely present the canonical config_path SHA.

    Extra non-scientific keys (e.g. attacker_output_dir) are allowed; every
    frozen scientific key must be present with the identical value.
    """
    if not os.path.exists(FROZEN_ATTACKER_CONFIG_PATH):
        raise RuntimeError('Frozen attacker config missing: %s' % FROZEN_ATTACKER_CONFIG_PATH)
    actual_sha = file_sha256(FROZEN_ATTACKER_CONFIG_PATH)
    if actual_sha != FROZEN_ATTACKER_CONFIG_SHA:
        raise RuntimeError('Frozen attacker config SHA mismatch: %s != %s' % (actual_sha, FROZEN_ATTACKER_CONFIG_SHA))
    with open(FROZEN_ATTACKER_CONFIG_PATH) as f:
        frozen = json.load(f)
    if config.get('image_path') != frozen.get('image_path'):
        raise RuntimeError('Scientific attacker in-memory config image_path does not match the frozen data root')
    for key, value in frozen.items():
        if key not in config:
            raise RuntimeError('Scientific attacker in-memory config is missing frozen field: %s' % key)
        if config[key] != value:
            raise RuntimeError('Scientific attacker in-memory config %s does not match frozen value %r' % (key, value))
    return True


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
    and records the exact ordered sample indices / pair rows for provenance hashing."""

    def __init__(self, data_source, generator=None, seed=42, pair_identifiers=None):
        self.data_source = data_source
        self.seed = seed
        self.generator = generator if generator is not None else torch.Generator().manual_seed(seed)
        self.pair_identifiers = pair_identifiers if pair_identifiers is not None else getattr(data_source, 'image_pairs', None)
        self.epoch_indices = []

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        n = len(self.data_source)
        indices = torch.randperm(n, generator=self.generator).tolist()
        self.epoch_indices.append(indices)
        return iter(indices)

    def get_epoch_order_hash(self, epoch=0, pair_identifiers=None):
        """F8 (M1.4c): Compute SHA256 of epoch order using semantic pair rows (not indices).
        When pair data are available, always formats each row as 'image1|image2|label\n' for
        exact parity with compute_epoch_order_hash().
        """
        if epoch >= len(self.epoch_indices):
            raise IndexError("Epoch %d order not recorded yet (total recorded: %d)" % (epoch, len(self.epoch_indices)))
        indices = self.epoch_indices[epoch]
        pairs = pair_identifiers if pair_identifiers is not None else (
            self.pair_identifiers if self.pair_identifiers is not None else getattr(self.data_source, 'image_pairs', None)
        )
        h = hashlib.sha256()
        for idx in indices:
            if pairs is not None:
                item = pairs[idx]
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
    and abnormality labels.
    Fail-closed: raises FileNotFoundError if any referenced image is missing.
    """

    def __init__(self, phase='training', image_path=None, image_size=IMAGE_SIZE, n_channels=1, max_pairs=None, metadata_path=None):
        firewall_check('dev')
        assert_dev_phase(phase)

        if not image_path:
            raise ValueError("image_path must be explicitly specified (no implicit fallback allowed)")
        if not os.path.isdir(image_path):
            raise FileNotFoundError("Explicit image_path directory not found: %s" % image_path)

        self.image_path = image_path
        self.image_size = image_size
        self.n_channels = n_channels
        self.phase = phase

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

        csv_path = metadata_path or os.path.join(ROOT, 'Data_Entry_2017_v2020.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError("Required metadata file Data_Entry_2017_v2020.csv not found at: %s" % csv_path)

        df_meta = pd.read_csv(csv_path)
        meta_data = df_meta[['Image Index', 'Finding Labels']].values
        file_to_finding = dict(zip(meta_data[:, 0], meta_data[:, 1]))

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
            if fname not in file_to_finding:
                raise RuntimeError("Image %s at pair index %d absent from metadata Data_Entry_2017_v2020.csv" % (fname, i))
            finding = file_to_finding[fname]
            if finding != 'No Finding' and isinstance(finding, str):
                for d in finding.split('|'):
                    # §24 (M1.4c): Unknown pathology token → HARD FAIL
                    if d not in self.PRED_LABEL:
                        raise RuntimeError(
                            "Unknown pathology token '%s' in image %s at pair index %d. "
                            "All tokens except 'No Finding' must be in PRED_LABEL." % (d, fname, i)
                        )
                    label[self.PRED_LABEL.index(d)] = 1.0
            self.ac_labels_1.append(label)
            self.labels_id.append(float(self.image_pairs[i][2]))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, index):
        from datasets.Dataset import pil_loader
        p1 = os.path.join(self.image_path, self.image_pairs[index][0])
        p2 = os.path.join(self.image_path, self.image_pairs[index][1])

        if not os.path.exists(p1):
            raise FileNotFoundError(
                "Missing %s image1 at pair index %d: file=%s (resolved: %s)" % (
                    self.phase, index, self.image_pairs[index][0], p1
                )
            )
        if not os.path.exists(p2):
            raise FileNotFoundError(
                "Missing %s image2 at pair index %d: file=%s (resolved: %s)" % (
                    self.phase, index, self.image_pairs[index][1], p2
                )
            )

        img1 = pil_loader(p1, self.n_channels)
        img2 = pil_loader(p2, self.n_channels)
        img1 = self.transform(self.resize(img1))
        img2 = self.transform(self.resize(img2))

        return img1, img2, self.ac_labels_1[index], self.labels_id[index]


def build_dev_anonymizer_loaders(config, seed=42, num_workers=0, pin_mem=False, max_pairs=None):
    """Construct paired TRAIN and VAL anonymizer loaders with explicit single-source-of-truth
    FingerprintedRandomSampler for TRAIN and sequential VAL.
    Fail-closed: raises FileNotFoundError if image_path is missing or not a directory.
    """
    firewall_check('dev')
    image_path = config.get('image_path')
    if not image_path:
        raise ValueError("image_path must be explicitly specified in config (no implicit fallback allowed)")
    if not os.path.isdir(image_path):
        raise FileNotFoundError("Explicit image_path directory not found: %s" % image_path)

    image_size = config.get('image_size', IMAGE_SIZE)
    batch_size = config.get('batch_size', 16)

    train_dataset = LazyPairDataset(phase='training', image_size=image_size, n_channels=1,
                                    image_path=image_path, max_pairs=max_pairs)
    val_dataset = LazyPairDataset(phase='validation', image_size=image_size, n_channels=1,
                                  image_path=image_path, max_pairs=max_pairs)

    train_gen = torch.Generator()
    train_gen.manual_seed(seed)
    train_sampler = FingerprintedRandomSampler(
        train_dataset, generator=train_gen, seed=seed, pair_identifiers=train_dataset.image_pairs
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_mem
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem
    )
    return train_loader, val_loader, train_sampler


def verify_scientific_dependencies(image_path=None):
    """Preflight check: verifies existence and SHAs of all frozen checkpoints,
    repaired modules, pair splits, metadata, configs, classification VAL contract/fingerprints,
    and 100% of referenced image files.
    Fail closed if any dependency is missing or drifted.
    """
    firewall_check('dev')
    image_path = image_path or '/home/minhtt/datasets/nih/images/'
    if not os.path.isdir(image_path):
        raise FileNotFoundError("Dataset root directory does not exist: %s" % image_path)

    # 1. Initial generator checkpoint
    if not os.path.exists(INITIAL_GENERATOR_PATH):
        raise FileNotFoundError("Initial generator checkpoint missing: %s" % INITIAL_GENERATOR_PATH)
    actual_gen_sha = file_sha256(INITIAL_GENERATOR_PATH)
    if actual_gen_sha != INITIAL_GENERATOR_SHA:
        raise RuntimeError("Initial generator SHA mismatch: %s != %s" % (actual_gen_sha, INITIAL_GENERATOR_SHA))

    # 2. Frozen classifier checkpoint
    if not os.path.exists(FROZEN_CLASSIFIER_PATH):
        raise FileNotFoundError("Classifier checkpoint missing: %s" % FROZEN_CLASSIFIER_PATH)
    actual_clf_sha = file_sha256(FROZEN_CLASSIFIER_PATH)
    if actual_clf_sha != FROZEN_CLASSIFIER_SHA:
        raise RuntimeError("Classifier SHA mismatch: %s != %s" % (actual_clf_sha, FROZEN_CLASSIFIER_SHA))

    # 3. Frozen verifier checkpoint
    if not os.path.exists(FROZEN_VERIFIER_PATH):
        raise FileNotFoundError("Verifier checkpoint missing: %s" % FROZEN_VERIFIER_PATH)
    actual_ver_sha = file_sha256(FROZEN_VERIFIER_PATH)
    if actual_ver_sha != FROZEN_VERIFIER_SHA:
        raise RuntimeError("Verifier SHA mismatch: %s != %s" % (actual_ver_sha, FROZEN_VERIFIER_SHA))

    # 4. Repaired ACLoss module
    _, actual_acloss_sha, _ = verify_repaired_acloss()

    # 5. Frozen configs and semantic parameters
    cfg_audit = verify_frozen_scientific_configs()

    # 6. Pair split files
    train_pair_path = os.path.join(ROOT, 'image_pairs', 'image_pairs_training_10000.txt')
    val_pair_path = os.path.join(ROOT, 'image_pairs', 'image_pairs_validation_2000.txt')
    if not os.path.exists(train_pair_path):
        raise FileNotFoundError("TRAIN pair file missing: %s" % train_pair_path)
    if not os.path.exists(val_pair_path):
        raise FileNotFoundError("VAL pair file missing: %s" % val_pair_path)

    train_pairs_sha = file_sha256(train_pair_path)
    val_pairs_sha = file_sha256(val_pair_path)
    expected_train_sha = "3c535eed013305bacf231dea9c72fb047cc6b6cb15e3958ef7a308956394b268"
    expected_val_sha = "9e33a081dfd5e4f28e658a9d13417f8a61f24cba60b2cb03272b20535b9fa9f7"
    if train_pairs_sha != expected_train_sha:
        raise RuntimeError("TRAIN pair SHA mismatch: %s != %s" % (train_pairs_sha, expected_train_sha))
    if val_pairs_sha != expected_val_sha:
        raise RuntimeError("VAL pair SHA mismatch: %s != %s" % (val_pairs_sha, expected_val_sha))

    # 7. Metadata file existence, SHA, and pair coverage
    if not os.path.exists(FROZEN_METADATA_PATH):
        raise FileNotFoundError("Metadata file missing: %s" % FROZEN_METADATA_PATH)
    actual_meta_sha = file_sha256(FROZEN_METADATA_PATH)
    if actual_meta_sha != FROZEN_METADATA_SHA:
        raise RuntimeError("Metadata SHA mismatch: %s != %s" % (actual_meta_sha, FROZEN_METADATA_SHA))

    df_meta = pd.read_csv(FROZEN_METADATA_PATH)
    meta_fnames = set(df_meta['Image Index'].astype(str))
    dup_count = int(df_meta['Image Index'].duplicated().sum())
    if dup_count > 0:
        raise RuntimeError("Metadata has %d duplicate Image Index entries" % dup_count)

    with open(train_pair_path) as f:
        train_lines = [line.strip().split() for line in f if line.strip()]
    with open(val_pair_path) as f:
        val_lines = [line.strip().split() for line in f if line.strip()]

    train_fnames = set()
    train_image1 = set()
    for row in train_lines:
        train_image1.add(row[0])
        train_fnames.add(row[0])
        train_fnames.add(row[1])

    val_fnames = set()
    val_image1 = set()
    for row in val_lines:
        val_image1.add(row[0])
        val_fnames.add(row[0])
        val_fnames.add(row[1])

    missing_train_meta = train_image1 - meta_fnames
    missing_val_meta = val_image1 - meta_fnames
    if missing_train_meta:
        raise RuntimeError("Missing %d TRAIN image1 in metadata (e.g. %s)" % (len(missing_train_meta), list(missing_train_meta)[:3]))
    if missing_val_meta:
        raise RuntimeError("Missing %d VAL image1 in metadata (e.g. %s)" % (len(missing_val_meta), list(missing_val_meta)[:3]))

    # 8. Verify image availability for all 10,000 TRAIN + 2,000 VAL pairs
    existing_files = set(os.listdir(image_path))
    missing_train = train_fnames - existing_files
    missing_val = val_fnames - existing_files

    if missing_train:
        raise FileNotFoundError("Missing %d TRAIN image files (e.g. %s)" % (len(missing_train), list(missing_train)[:3]))
    if missing_val:
        raise FileNotFoundError("Missing %d VAL image files (e.g. %s)" % (len(missing_val), list(missing_val)[:3]))

    # 9. Classification VAL Contract & Structural Fingerprints (B2, B3)
    clf_val_audit = verify_classification_val_contract()

    return {
        'status': 'PASS',
        'image_path': image_path,
        'initial_generator_sha256': actual_gen_sha,
        'classifier_sha256': actual_clf_sha,
        'verifier_sha256': actual_ver_sha,
        'acloss_sha256': actual_acloss_sha,
        'b_dev_config_sha256': cfg_audit['b_dev_config_sha256'],
        'c4_config_sha256': cfg_audit['c4_config_sha256'],
        'attacker_config_sha256': cfg_audit['attacker_config_sha256'],
        'metadata_sha256': actual_meta_sha,
        'train_pairs_sha256': train_pairs_sha,
        'val_pairs_sha256': val_pairs_sha,
        'train_pairs_count': len(train_lines),
        'val_pairs_count': len(val_lines),
        'missing_train_images': 0,
        'missing_val_images': 0,
        'train_image1_metadata_missing': 0,
        'val_image1_metadata_missing': 0,
        'metadata_duplicate_image_index_count': dup_count,
        'classification_split_csv_sha256': clf_val_audit['classification_split_csv_sha256'],
        'classification_val_n_images': clf_val_audit['classification_val_n_images'],
        'classification_val_n_patients': clf_val_audit['classification_val_n_patients'],
        'classification_val_image_index_sha256': clf_val_audit['classification_val_image_index_sha256'],
        'classification_val_patient_sequence_sha256': clf_val_audit['classification_val_patient_sequence_sha256'],
        'classification_val_label_matrix_sha256': clf_val_audit['classification_val_label_matrix_sha256'],
    }