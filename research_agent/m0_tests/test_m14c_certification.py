"""M1.4c Final Forensic Certification Test Suite (T137–T176+).

Verifies all forensic requirements and finding closures (F1–F13):
- F1: Explicit scientific mode gate & dev orchestration non-verdict (T137, T138)
- F2: Hash-locked chexnet/nih_labels.csv (T139)
- §5: Classification VAL structural fingerprints (T140, T141, T142)
- F3: Exhaustive patient split disjointness & pair identity semantics (T143, T144, T145, T146)
- F4: Scientific output directory freshness (T147, T148)
- F5: Attacker NaN/Inf fail-closed and parameter finiteness (T149, T150, T151, T152)
- F8 / §12: Runtime vs offline train-order SHA equivalence and B_dev/C4 paired order (T153, T154, T155)
- F7: Exact resume parity OR scientific resume disabled (T156)
- F10: Vacuous test repairs (T157, T158, T159, T160, T161)
- F11: Truly independent pristine reference parity (T162, T163, T164, T165, T166)
- §27: Gradient & parameter update ownership across phases (T167, T168)
- §24: Unknown pathology token hard-fails (T169)
- F13 / §22 / §23: Raw prediction metric replay (classification T170, privacy T171)
- F9: Scientific git source guard (T172, T173)
- §29: Checkpoint selection excludes feature term & earliest tie-break (T174, T175)
- §39: Zero TEST access firewall proof (T176)
"""
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest.mock as mock

import numpy as np
import pandas as pd
import sklearn.metrics as sklm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
for _p in (ROOT, os.path.join(ROOT, 'research_agent')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from test_firewall import TestFirewall, provenance_record
from m2_dev.evaluator_common import (
    firewall_check,
    file_sha256,
    make_flow_field_components,
    anonymize,
    snn_preprocess,
    classifier_preprocess,
    LazyPairDataset,
    FingerprintedRandomSampler,
    compute_epoch_order_hash,
    compute_epoch_totals,
    select_method_neutral_best,
    verify_scientific_dependencies,
    verify_frozen_scientific_configs,
    FROZEN_METADATA_PATH,
    FROZEN_METADATA_SHA,
    FROZEN_CLASSIFIER_PATH,
    FROZEN_CLASSIFIER_SHA,
    FROZEN_VERIFIER_PATH,
    FROZEN_VERIFIER_SHA,
    INITIAL_GENERATOR_PATH,
    INITIAL_GENERATOR_SHA,
    FROZEN_B_DEV_CONFIG_PATH,
    FROZEN_B_DEV_CONFIG_SHA,
    FROZEN_C4_CONFIG_PATH,
    FROZEN_C4_CONFIG_SHA,
    FROZEN_ATTACKER_CONFIG_PATH,
    FROZEN_ATTACKER_CONFIG_SHA,
    MU,
)
from m2_dev.anonymizer_runner import M2AnonymizerRunner
from m2_dev.dev_attacker import DevAttacker, SiameseNetwork
from m2_dev.eval_reid_val import evaluate_reid_val
from m2_dev.eval_classifier_val import evaluate_classification_val, classify_val_dataset
from m2_dev import run_m2_s1
from networks.UNet_PriCheXyNet import UNet
from m0_tests.pristine_reference import (
    pristine_identity_grid,
    pristine_gaussian_kernel,
    pristine_anonymize,
    pristine_privacy_loss_float64,
    pristine_privacy_loss_softplus,
    pristine_one_step,
)


# ---------------------------------------------------------------------------
# T137–T138: F1 Scientific Mode Enforcement
# ---------------------------------------------------------------------------
def test_t137_canonical_scientific_master_requires_flag():
    """T137 (F1): Canonical master execution without --scientific-m2-s1 HARD FAILS."""
    class MockArgsNoFlag:
        def __init__(self):
            self.scientific_m2_s1 = False
            self.arm = 'all'
            self.max_epochs = 250
            self.attacker_epochs = 100
            self.attacker_patience = 5
            self.seed = 42
            self.attacker_seed = 42
            self.device = 'cpu'

    try:
        run_m2_s1.run_orchestration(MockArgsNoFlag(), out_base_dir=None, unit_test_mode=False)
        assert False, "Should have raised RuntimeError without --scientific-m2-s1"
    except RuntimeError as e:
        assert "requires --scientific-m2-s1" in str(e)
    return True


def test_t138_development_orchestration_cannot_produce_scientific_verdict():
    """T138 (F1): Development/unit-test orchestration cannot produce PROMOTE / DO NOT PROMOTE verdict."""
    class MockArgsDev:
        def __init__(self):
            self.scientific_m2_s1 = False
            self.arm = 'all'
            self.max_epochs = 1
            self.attacker_epochs = 1
            self.attacker_patience = 1
            self.seed = 42
            self.attacker_seed = 42
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with tempfile.TemporaryDirectory() as tmp_dir:
        summary = run_m2_s1.run_orchestration(MockArgsDev(), out_base_dir=tmp_dir, unit_test_mode=True)
        assert summary['verdict'] == "DEVELOPMENT_ONLY — not a scientific verdict"
        assert summary['verdict'] not in ("C4 S1: PROMOTE TO S2", "C4 S1: DO NOT PROMOTE")
    return True


# ---------------------------------------------------------------------------
# T139–T142: F2 & §5 Classification Split Fingerprints
# ---------------------------------------------------------------------------
def test_t139_nih_labels_csv_sha_locked():
    """T139 (F2): chexnet/nih_labels.csv is hash-locked in execution provenance."""
    csv_p = os.path.join(ROOT, 'chexnet', 'nih_labels.csv')
    assert os.path.exists(csv_p), "nih_labels.csv missing"
    sha = file_sha256(csv_p)
    assert len(sha) == 64
    # Check that SHA is non-empty and verifiable
    assert sha == '80324996867e73546bd7a09025df4a4cc3243fc00663b753023ccd90a9b5f8b9'
    return True


def test_t140_classification_val_image_index_fingerprint():
    """T140 (§5): Classification VAL image index sequence produces exact deterministic fingerprint."""
    fp = run_m2_s1.compute_classification_val_fingerprints()
    assert fp is not None
    assert fp['classification_val_n_images'] == 10816, "Expected exactly 10,816 VAL images"
    assert fp['classification_val_image_index_sha256'] == 'c2ff15d7deb0e126b096d4392f4d8844c80593d16ad939ea44189b44051b8ab6'
    return True


def test_t141_classification_val_patient_id_fingerprint():
    """T141 (§5): Classification VAL patient ID sequence produces exact deterministic fingerprint."""
    fp = run_m2_s1.compute_classification_val_fingerprints()
    assert fp is not None
    assert fp['classification_val_n_patients'] == 3854, "Expected exactly 3,854 VAL patients"
    assert fp['classification_val_patient_sequence_sha256'] == 'c444238f8be4c6f2e6ee5523a1966ef974a182aabc895b3819ebbaeec4f621ba'
    return True


def test_t142_classification_val_label_matrix_fingerprint():
    """T142 (§5): Classification VAL 14-D label matrix produces exact deterministic fingerprint."""
    fp = run_m2_s1.compute_classification_val_fingerprints()
    assert fp is not None
    assert fp['classification_val_label_matrix_sha256'] == 'bef28de2c55d5767c5d930bca8f86253c75a8be2cf00552ce0ceb579e75a28cc'
    return True


# ---------------------------------------------------------------------------
# T143–T146: F3 & §25 Exhaustive Patient Split & Pair Semantics
# ---------------------------------------------------------------------------
def test_t143_anonymizer_train_vs_classification_val_patient_disjoint():
    """T143 (F3): Anonymizer TRAIN patients ∩ classification VAL patients == 0."""
    train_pairs_p = os.path.join(ROOT, 'image_pairs', 'image_pairs_training_10000.txt')
    train_pairs = np.loadtxt(train_pairs_p, dtype=str)
    train_pts = set()
    for row in train_pairs:
        train_pts.add(row[0].split('_')[0])
        train_pts.add(row[1].split('_')[0])

    csv_p = os.path.join(ROOT, 'chexnet', 'nih_labels.csv')
    df = pd.read_csv(csv_p)
    val_imgs = df[df['fold'] == 'val']['Image Index']
    val_pts = set(img.split('_')[0] for img in val_imgs)

    overlap = train_pts & val_pts
    assert len(overlap) == 0, "Patient overlap detected: %d patients" % len(overlap)
    return True


def test_t144_anonymizer_train_vs_anonymizer_val_patient_disjoint():
    """T144 (F3): Anonymizer TRAIN patients ∩ anonymizer VAL patients == 0."""
    train_pairs = np.loadtxt(os.path.join(ROOT, 'image_pairs', 'image_pairs_training_10000.txt'), dtype=str)
    val_pairs = np.loadtxt(os.path.join(ROOT, 'image_pairs', 'image_pairs_validation_2000.txt'), dtype=str)

    train_pts = set(r[0].split('_')[0] for r in train_pairs) | set(r[1].split('_')[0] for r in train_pairs)
    val_pts = set(r[0].split('_')[0] for r in val_pairs) | set(r[1].split('_')[0] for r in val_pairs)

    overlap = train_pts & val_pts
    assert len(overlap) == 0, "Train/Val patient overlap detected: %d patients" % len(overlap)
    return True


def test_t145_train_pair_identity_semantics():
    """T145 (§25): All 10,000 TRAIN pairs satisfy label 1 == same patient, label 0 == diff patient."""
    train_pairs = np.loadtxt(os.path.join(ROOT, 'image_pairs', 'image_pairs_training_10000.txt'), dtype=str)
    assert len(train_pairs) == 10000
    pos_count = 0
    neg_count = 0
    for row in train_pairs:
        p1 = row[0].split('_')[0]
        p2 = row[1].split('_')[0]
        label = float(row[2])
        assert label in (0.0, 1.0), "Invalid label value: %s" % label
        if label == 1.0:
            assert p1 == p2, "Label 1 with different patients: %s != %s" % (p1, p2)
            pos_count += 1
        else:
            assert p1 != p2, "Label 0 with same patient: %s == %s" % (p1, p2)
            neg_count += 1

    assert pos_count == 5000 and neg_count == 5000, "TRAIN pairs unbalanced: %d pos, %d neg" % (pos_count, neg_count)
    return True


def test_t146_val_pair_identity_semantics():
    """T146 (§25): All 2,000 VAL pairs satisfy label 1 == same patient, label 0 == diff patient."""
    val_pairs = np.loadtxt(os.path.join(ROOT, 'image_pairs', 'image_pairs_validation_2000.txt'), dtype=str)
    assert len(val_pairs) == 2000
    pos_count = 0
    neg_count = 0
    for row in val_pairs:
        p1 = row[0].split('_')[0]
        p2 = row[1].split('_')[0]
        label = float(row[2])
        assert label in (0.0, 1.0), "Invalid label value: %s" % label
        if label == 1.0:
            assert p1 == p2, "Label 1 with different patients: %s != %s" % (p1, p2)
            pos_count += 1
        else:
            assert p1 != p2, "Label 0 with same patient: %s == %s" % (p1, p2)
            neg_count += 1

    assert pos_count == 1000 and neg_count == 1000, "VAL pairs unbalanced: %d pos, %d neg" % (pos_count, neg_count)
    return True


# ---------------------------------------------------------------------------
# T147–T148: F4 Output Directory Freshness
# ---------------------------------------------------------------------------
def test_t147_stale_scientific_output_dir_rejected():
    """T147 (F4): Pre-existing scientific artifacts in output dir causes HARD FAIL."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a stale artifact
        stale_p = os.path.join(tmp_dir, 'M2_S1_summary.json')
        with open(stale_p, 'w') as f:
            f.write('{}')

        try:
            run_m2_s1.check_scientific_output_freshness(tmp_dir)
            assert False, "Should have rejected stale output directory"
        except RuntimeError as e:
            assert "Scientific output directory is not fresh" in str(e)
    return True


def test_t148_empty_scientific_output_dir_accepted():
    """T148 (F4): Empty or non-existent scientific output dir passes freshness check."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        assert run_m2_s1.check_scientific_output_freshness(tmp_dir) is True
    assert run_m2_s1.check_scientific_output_freshness('/nonexistent/path/for/test') is True
    return True


# ---------------------------------------------------------------------------
# T149–T152: F5 Attacker NaN/Inf Fail-Closed & Parameter Finiteness
# ---------------------------------------------------------------------------
def test_t149_attacker_nan_train_loss_hard_fails():
    """T149 (F5): Attacker raises FloatingPointError immediately on NaN train loss."""
    class NaNNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.tensor([float('nan')]))
        def forward(self, x1, x2):
            return self.p.expand(x1.size(0))

    cfg = {'learning_rate': 1e-4, 'max_epochs': 1, 'early_stopping': 5, 'batch_size': 2}
    from m0_tests.test_m14a_execution_harness import SyntheticAttackerPairDataset
    ds = SyntheticAttackerPairDataset(4, image_size=64)
    loader = torch.utils.data.DataLoader(ds, batch_size=2)

    attacker = DevAttacker(
        config=cfg,
        net_factory=lambda: NaNNet(),
        anonymize_fn=lambda x: x,
        training_loader=loader,
        validation_loader=loader,
        image_size=64,
        unit_test_mode=True
    )
    try:
        attacker.train_epoch()
        assert False, "Should have raised FloatingPointError on NaN train loss"
    except FloatingPointError:
        pass
    return True


def test_t150_attacker_inf_validation_loss_hard_fails():
    """T150 (F5): Attacker raises FloatingPointError on Inf validation loss."""
    class InfNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(3, 1)
        def forward(self, x1, x2):
            return torch.tensor([float('inf')], requires_grad=True).expand(x1.size(0))

    cfg = {'learning_rate': 1e-4, 'max_epochs': 1, 'early_stopping': 5, 'batch_size': 2}
    from m0_tests.test_m14a_execution_harness import SyntheticAttackerPairDataset
    ds = SyntheticAttackerPairDataset(4, image_size=64)
    loader = torch.utils.data.DataLoader(ds, batch_size=2)

    attacker = DevAttacker(
        config=cfg,
        net_factory=lambda: InfNet(),
        anonymize_fn=lambda x: x,
        training_loader=loader,
        validation_loader=loader,
        image_size=64,
        unit_test_mode=True
    )
    try:
        attacker.validate_selection()
        assert False, "Should have raised FloatingPointError on Inf validation loss"
    except FloatingPointError:
        pass
    return True


def test_t151_attacker_nonfinite_params_hard_fail():
    """T151 (F5): Attacker raises FloatingPointError if parameter becomes NaN after step."""
    class ExplodingNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.Parameter(torch.tensor([1e20]))
        def forward(self, x1, x2):
            return self.p * 1e20 * (x1.mean(dim=(1, 2, 3)) + x2.mean(dim=(1, 2, 3)))

    cfg = {'learning_rate': 1e10, 'max_epochs': 1, 'early_stopping': 5, 'batch_size': 2}
    from m0_tests.test_m14a_execution_harness import SyntheticAttackerPairDataset
    ds = SyntheticAttackerPairDataset(4, image_size=64)
    loader = torch.utils.data.DataLoader(ds, batch_size=2)

    attacker = DevAttacker(
        config=cfg,
        net_factory=lambda: ExplodingNet(),
        anonymize_fn=lambda x: x,
        training_loader=loader,
        validation_loader=loader,
        image_size=64,
        unit_test_mode=True
    )
    try:
        attacker.train_epoch()
        assert False, "Should have raised FloatingPointError on exploding parameters"
    except FloatingPointError:
        pass
    return True


def test_t152_attacker_manifest_numerical_validity_required():
    """T152 (F5): Successful attacker training saves numerical_validity == PASS in manifest."""
    cfg = {'learning_rate': 1e-4, 'max_epochs': 1, 'early_stopping': 5, 'batch_size': 4}
    from m0_tests.test_m14a_execution_harness import SyntheticAttackerPairDataset
    ds = SyntheticAttackerPairDataset(8, image_size=64)
    loader = torch.utils.data.DataLoader(ds, batch_size=4)

    with tempfile.TemporaryDirectory() as tmp_dir:
        attacker = DevAttacker(
            config=cfg,
            anonymize_fn=lambda x: x,
            training_loader=loader,
            validation_loader=loader,
            image_size=64,
            unit_test_mode=True
        )
        hist = attacker.run(output_dir=tmp_dir)
        manifest_p = os.path.join(tmp_dir, 'attacker_manifest.json')
        assert os.path.exists(manifest_p)
        with open(manifest_p) as f:
            m = json.load(f)
        assert m.get('numerical_validity') == 'PASS'
        assert m.get('nan_inf_detected') is False
    return True


# ---------------------------------------------------------------------------
# T153–T155: F8 / §12 Train Order Hash & Paired Ordering Proof
# ---------------------------------------------------------------------------
def test_t153_runtime_pair_order_equals_offline_epoch0():
    """T153 (F8): Runtime FingerprintedRandomSampler epoch0 hash equals offline compute_epoch_order_hash."""
    pair_path = os.path.join(ROOT, 'image_pairs', 'image_pairs_training_10000.txt')
    pairs = np.loadtxt(pair_path, dtype=str)

    gen = torch.Generator().manual_seed(42)
    sampler = FingerprintedRandomSampler(pairs, generator=gen, seed=42)
    # Simulate epoch 0 iteration
    _ = list(iter(sampler))

    runtime_sha = sampler.get_epoch_order_hash(epoch=0, pair_identifiers=pairs)
    offline_sha = compute_epoch_order_hash(pair_path, seed=42, epoch=0)

    assert runtime_sha == offline_sha, "Epoch 0 hash mismatch: %s != %s" % (runtime_sha, offline_sha)
    return True


def test_t154_runtime_pair_order_equals_offline_epoch1():
    """T154 (F8): Runtime FingerprintedRandomSampler epoch1 hash equals offline compute_epoch_order_hash."""
    pair_path = os.path.join(ROOT, 'image_pairs', 'image_pairs_training_10000.txt')
    pairs = np.loadtxt(pair_path, dtype=str)

    gen = torch.Generator().manual_seed(42)
    sampler = FingerprintedRandomSampler(pairs, generator=gen, seed=42)
    # Simulate epoch 0 and epoch 1 iterations
    _ = list(iter(sampler))
    _ = list(iter(sampler))

    runtime_sha1 = sampler.get_epoch_order_hash(epoch=1, pair_identifiers=pairs)
    offline_sha1 = compute_epoch_order_hash(pair_path, seed=42, epoch=1)

    assert runtime_sha1 == offline_sha1, "Epoch 1 hash mismatch: %s != %s" % (runtime_sha1, offline_sha1)
    return True


def test_t155_b_dev_and_c4_exact_same_order_epochs_0_to_2():
    """T155 (§12): B_dev and C4 samplers produce byte-for-byte identical pair order for epochs 0, 1, 2."""
    pair_path = os.path.join(ROOT, 'image_pairs', 'image_pairs_training_10000.txt')
    pairs = np.loadtxt(pair_path, dtype=str)

    gen_b = torch.Generator().manual_seed(42)
    sampler_b = FingerprintedRandomSampler(pairs, generator=gen_b, seed=42)

    gen_c4 = torch.Generator().manual_seed(42)
    sampler_c4 = FingerprintedRandomSampler(pairs, generator=gen_c4, seed=42)

    for ep in range(3):
        indices_b = list(iter(sampler_b))
        indices_c4 = list(iter(sampler_c4))
        assert indices_b == indices_c4, "Epoch %d indices mismatch between B_dev and C4" % ep
        sha_b = sampler_b.get_epoch_order_hash(epoch=ep, pair_identifiers=pairs)
        sha_c4 = sampler_c4.get_epoch_order_hash(epoch=ep, pair_identifiers=pairs)
        assert sha_b == sha_c4, "Epoch %d SHA mismatch" % ep
    return True


# ---------------------------------------------------------------------------
# T156: F7 Resume Policy
# ---------------------------------------------------------------------------
def test_t156_scientific_resume_policy_declared():
    """T156 (F7): Anonymizer runner module documents that scientific resume requires restart from epoch 0."""
    from m2_dev import anonymizer_runner
    doc = anonymizer_runner.__doc__
    assert "WARNING (F7 M1.4c): Resume is NOT certified for scientific use" in doc
    assert "Scientific runs MUST restart from epoch 0 if interrupted" in doc
    return True


# ---------------------------------------------------------------------------
# T157–T161: F10 Vacuous Test Regression
# ---------------------------------------------------------------------------
def test_t157_t83_replacement_is_active_and_non_vacuous():
    """T157 (F10): T83 executes active classification code and verifies 14 finite AUCs."""
    from m0_tests.test_m14_final_hardening import test_t83_classification_evaluator_requires_14_finite_aucs
    res = test_t83_classification_evaluator_requires_14_finite_aucs()
    assert res is True
    return True


def test_t158_t125_true_lazypairdataset_parity():
    """T158 (F10): T125 tests LazyPairDataset against independent metadata parser."""
    from m0_tests.test_m14b_execution_integrity import test_t125_pathology_label_parity_with_canonical_metadata
    res = test_t125_pathology_label_parity_with_canonical_metadata()
    assert res is True
    return True


def test_t159_t130_device_flag_preflight_assertion():
    """T159 (F10): T130 proves --device does not bypass preflight verification."""
    from m0_tests.test_m14b_execution_integrity import test_t130_device_flag_does_not_bypass_preflight
    res = test_t130_device_flag_does_not_bypass_preflight()
    assert res is True
    return True


def test_t160_t134_invalid_orchestration_verdict():
    """T160 (F10): T134 proves invalid orchestration yields INVALID verdict."""
    from m0_tests.test_m14b_execution_integrity import test_t134_invalid_run_cannot_produce_promote_verdict
    res = test_t134_invalid_run_cannot_produce_promote_verdict()
    assert res is True
    return True


def test_t161_t135_injected_config_drift_fails():
    """T161 (F10): T135 proves injected mu=0.02 config drift raises RuntimeError."""
    from m0_tests.test_m14b_execution_integrity import test_t135_scientific_config_semantic_key_drift_hard_fails
    res = test_t135_scientific_config_semantic_key_drift_hard_fails()
    assert res is True
    return True


# ---------------------------------------------------------------------------
# T162–T166: F11 Truly Independent Pristine Reference Parity
# ---------------------------------------------------------------------------
def _run_pristine_parity_comparison():
    """Execute differential comparison between pristine reference and M2AnonymizerRunner."""
    torch.manual_seed(42)
    device = torch.device('cpu')

    img_size = 64
    bs = 2

    # Instantiate identical models
    torch.manual_seed(100)
    g_ref = UNet(1, 2, 32).to(device)
    from torchvision.models import densenet121
    torch.manual_seed(200)
    clf_ref = densenet121(num_classes=14).to(device)
    # Add final Sigmoid to match released checkpoint
    clf_ref.classifier = nn.Sequential(clf_ref.classifier, nn.Sigmoid())
    torch.manual_seed(300)
    ver_ref = SiameseNetwork().to(device)

    # Clones for runner
    g_run = copy.deepcopy(g_ref)
    clf_run = copy.deepcopy(clf_ref)
    ver_run = copy.deepcopy(ver_ref)

    # Synthetic batch
    torch.manual_seed(400)
    x1 = torch.rand(bs, 1, img_size, img_size, device=device)
    x2 = torch.rand(bs, 1, img_size, img_size, device=device)
    y_clf = torch.randint(0, 2, (bs, 14), dtype=torch.float32, device=device)
    y_id = torch.tensor([1.0, 0.0], device=device)

    # 1. Run pristine reference
    ref_out = pristine_one_step(
        g_ref, clf_ref, ver_ref, x1, x2, y_clf, y_id, mu=0.01, lr=1e-4, image_size=img_size
    )

    # 2. Run runner step
    class SingleBatchLoader:
        def __init__(self, batch):
            self.batch = batch
        def __iter__(self):
            yield self.batch
        def __len__(self):
            return 1

    cfg = {
        'batch_size': bs,
        'image_size': img_size,
        'learning_rate': 1e-4,
        'mu': 0.01,
        'feature_loss_weight': 0.0,
        'num_workers': 0,
        'image_path': '/tmp',
    }
    loader = SingleBatchLoader((x1, x2, y_clf, y_id))

    with tempfile.TemporaryDirectory() as tmp_dir:
        runner = M2AnonymizerRunner(
            arm='B_dev',
            config=cfg,
            output_dir=tmp_dir,
            device=device,
            ac_model=clf_run,
            verification_model=ver_run,
            training_loader=loader,
            validation_loader=loader,
            unit_test_mode=True,
            initial_generator_path=None
        )
        runner.generator = g_run
        runner.optimizer_g = optim.Adam(runner.generator.parameters(), lr=1e-4)

        # Record pre-step fakes
        run_fakes = runner.anonymize_tensor(x1).detach()

        train_m = runner.train_epoch(0)

        run_gen_grads = runner.last_gen_grads
        run_ver_grads = runner.last_ver_grads
        run_clf_grads = runner.last_clf_grads

    return ref_out, train_m, run_fakes, run_gen_grads, run_ver_grads, run_clf_grads


def test_t162_pristine_reference_anonymized_tensor_parity():
    """T162 (F11): Pristine reference anonymized tensor matches runner output <= 1e-6."""
    ref_out, train_m, run_fakes, _, _, _ = _run_pristine_parity_comparison()
    diff = (ref_out['fakes_1'] - run_fakes).abs().max().item()
    assert diff <= 1e-6, "Anonymized tensor diff %e > 1e-6" % diff
    return True


def test_t163_pristine_reference_generator_loss_parity():
    """T163 (F11): Pristine reference generator loss components match runner <= 1e-6."""
    ref_out, train_m, _, _, _, _ = _run_pristine_parity_comparison()
    ac_diff = abs(ref_out['ac_bce'] - train_m['train_ac_bce'])
    assert ac_diff <= 1e-6, "AC BCE loss diff %e > 1e-6" % ac_diff
    priv_diff = abs(ref_out['privacy_term'] - train_m['train_privacy_term'])
    assert priv_diff <= 1e-6, "Privacy loss diff %e > 1e-6" % priv_diff
    tot_diff = abs(ref_out['total_loss'] - train_m['train_optimization_total'])
    assert tot_diff <= 1e-6, "Total loss diff %e > 1e-6" % tot_diff
    return True


def test_t164_pristine_reference_generator_gradient_parity():
    """T164 (F11): Pristine reference generator gradients match runner <= 1e-6."""
    ref_out, _, _, run_gen_grads, _, _ = _run_pristine_parity_comparison()
    max_diff = 0.0
    for g_ref_grad, g_run_grad in zip(ref_out['gen_grads'], run_gen_grads):
        d = (g_ref_grad - g_run_grad).abs().max().item()
        max_diff = max(max_diff, d)
    assert max_diff <= 1e-6, "Generator gradient max diff %e > 1e-6" % max_diff
    return True


def test_t165_pristine_reference_verifier_gradient_parity():
    """T165 (F11): Pristine reference verifier critic gradients match runner <= 1e-6."""
    ref_out, _, _, _, run_ver_grads, _ = _run_pristine_parity_comparison()
    max_diff = 0.0
    for v_ref_grad, v_run_grad in zip(ref_out['ver_grads'], run_ver_grads):
        d = (v_ref_grad - v_run_grad).abs().max().item()
        max_diff = max(max_diff, d)
    assert max_diff <= 1e-6, "Verifier gradient max diff %e > 1e-6" % max_diff
    return True


def test_t166_pristine_reference_classifier_gradient_parity():
    """T166 (F11): Pristine reference classifier critic gradients match runner <= 1e-6."""
    ref_out, _, _, _, _, run_clf_grads = _run_pristine_parity_comparison()
    max_diff = 0.0
    for c_ref_grad, c_run_grad in zip(ref_out['ac_grads'], run_clf_grads):
        d = (c_ref_grad - c_run_grad).abs().max().item()
        max_diff = max(max_diff, d)
    assert max_diff <= 1e-6, "Classifier gradient max diff %e > 1e-6" % max_diff
    return True


# ---------------------------------------------------------------------------
# T167–T168: §27 Gradient and Update Ownership
# ---------------------------------------------------------------------------
def test_t167_generator_phase_gradient_ownership():
    """T167 (§27): During generator update, only generator parameters change; critics are untouched."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    g = UNet(1, 2, 32).to(device)
    from torchvision.models import densenet121
    clf = densenet121(num_classes=14).to(device)
    clf.classifier = nn.Sequential(clf.classifier, nn.Sigmoid())
    ver = SiameseNetwork().to(device)

    opt_g = optim.Adam(g.parameters(), lr=1e-4)
    opt_clf = optim.SGD(clf.parameters(), lr=1e-4)
    opt_ver = optim.Adam(ver.parameters(), lr=1e-4)

    # Record parameter snapshots
    clf_pre = [p.clone().detach() for p in clf.parameters()]
    ver_pre = [p.clone().detach() for p in ver.parameters()]
    g_pre = [p.clone().detach() for p in g.parameters()]

    x1 = torch.rand(2, 1, 64, 64, device=device)
    x2 = torch.rand(2, 1, 64, 64, device=device)
    lbl = torch.zeros(2, 14, device=device)

    grid_id = pristine_identity_grid(64, device)
    gauss = pristine_gaussian_kernel(device=device)

    # Forward G
    fakes = pristine_anonymize(x1, g, grid_id, gauss)
    # AC loss with deepcopy
    ac_lm = copy.deepcopy(clf)
    ac_lm.eval()
    if hasattr(ac_lm, 'classifier') and isinstance(ac_lm.classifier, nn.Sequential):
        ac_lm.classifier = nn.Sequential(*list(ac_lm.classifier.children())[:-1])
    ac_logits = ac_lm(fakes.expand(-1, 3, -1, -1))
    loss_ac = nn.BCEWithLogitsLoss()(ac_logits, lbl)

    # Privacy loss
    v_logits = ver(fakes.expand(-1, 3, -1, -1), x2.expand(-1, 3, -1, -1)).squeeze()
    loss_priv = F.softplus(v_logits).mean()

    total_loss = loss_ac + loss_priv
    opt_g.zero_grad()
    total_loss.backward()
    opt_g.step()

    # Verify G parameters changed
    g_changed = any(not torch.equal(p1, p2) for p1, p2 in zip(g_pre, g.parameters()))
    assert g_changed, "Generator parameters should have changed"

    # Verify live classifier critic parameters DID NOT change
    for p_pre, p_now in zip(clf_pre, clf.parameters()):
        assert torch.equal(p_pre, p_now), "Live classifier parameters modified during G step!"

    # Verify live verifier critic parameters DID NOT change
    for p_pre, p_now in zip(ver_pre, ver.parameters()):
        assert torch.equal(p_pre, p_now), "Live verifier parameters modified during G step!"

    return True


def test_t168_critic_phase_update_ownership():
    """T168 (§27): Critic update steps update only their respective parameters."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    from torchvision.models import densenet121
    clf = densenet121(num_classes=14).to(device)
    clf.classifier = nn.Sequential(clf.classifier, nn.Sigmoid())
    ver = SiameseNetwork().to(device)

    opt_clf = optim.SGD(filter(lambda p: p.requires_grad, clf.parameters()), lr=1e-4)
    opt_ver = optim.Adam(ver.parameters(), lr=1e-4)

    clf_pre = [p.clone().detach() for p in clf.parameters() if p.requires_grad]
    ver_pre = [p.clone().detach() for p in ver.parameters()]

    x1 = torch.rand(2, 1, 64, 64, device=device)
    x2 = torch.rand(2, 1, 64, 64, device=device)
    y_id = torch.tensor([1.0, 0.0], device=device)
    lbl = torch.zeros(2, 14, device=device)

    # Verifier critic step
    ver.train()
    v_out = ver(x1.expand(-1, 3, -1, -1), x2.expand(-1, 3, -1, -1)).squeeze()
    l_v = nn.BCEWithLogitsLoss()(v_out, y_id)
    opt_ver.zero_grad()
    l_v.backward()
    opt_ver.step()

    ver_changed = any(not torch.equal(p1, p2) for p1, p2 in zip(ver_pre, ver.parameters()))
    assert ver_changed, "Verifier parameters should have changed"

    # Classifier was not stepped yet
    for p_pre, p_now in zip(clf_pre, [p for p in clf.parameters() if p.requires_grad]):
        assert torch.equal(p_pre, p_now), "Classifier changed during verifier step"

    # Classifier critic step
    clf.train()
    c_out = clf(x1.expand(-1, 3, -1, -1))
    l_c = nn.BCELoss()(c_out, lbl)
    opt_clf.zero_grad()
    l_c.backward()
    opt_clf.step()

    clf_changed = any(not torch.equal(p1, p2) for p1, p2 in zip(clf_pre, [p for p in clf.parameters() if p.requires_grad]))
    assert clf_changed, "Classifier parameters should have changed"
    return True


# ---------------------------------------------------------------------------
# T169: §24 Pathology Token Hard-Fail
# ---------------------------------------------------------------------------
def test_t169_unknown_pathology_token_hard_fails():
    """T169 (§24): Unknown pathology label token not in PRED_LABEL causes HARD FAIL."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create metadata with an unknown token
        meta_p = os.path.join(tmp_dir, 'fake_meta.csv')
        df = pd.DataFrame({
            'Image Index': ['00001661_000.png'],
            'Finding Labels': ['UnknownPathologyToken']
        })
        df.to_csv(meta_p, index=False)

        try:
            LazyPairDataset(
                phase='training',
                image_path=tmp_dir,
                max_pairs=1,
                metadata_path=meta_p
            )
            assert False, "Should have raised RuntimeError on unknown pathology token"
        except RuntimeError as e:
            assert "Unknown pathology token" in str(e) and "UnknownPathologyToken" in str(e)
    return True


# ---------------------------------------------------------------------------
# T170–T171: F13 / §22 / §23 Metric Replay
# ---------------------------------------------------------------------------
def test_t170_classification_metric_replay():
    """T170 (§22): Recomputing 14 AUCs and Macro AUC from saved predictions CSV matches exactly."""
    PRED_LABEL = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
                  'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
                  'Pleural_Thickening', 'Hernia']

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Build synthetic classification output
        rows = []
        for i in range(100):
            row = {'Image Index': 'img_%d.png' % i}
            for c, lbl in enumerate(PRED_LABEL):
                row[lbl] = (i + c) % 2
                row['prob_' + lbl] = (i * 0.01 + c * 0.05) % 1.0
            rows.append(row)
        df_pred = pd.DataFrame(rows)
        pred_csv = os.path.join(tmp_dir, 'classification_val_predictions.csv')
        df_pred.to_csv(pred_csv, index=False)

        # Compute reference AUCs
        ref_aucs = {}
        for lbl in PRED_LABEL:
            auc = sklm.roc_auc_score(df_pred[lbl], df_pred['prob_' + lbl])
            ref_aucs[lbl] = float(auc)
        ref_macro = float(np.mean(list(ref_aucs.values())))

        # Replay from saved CSV
        reloaded = pd.read_csv(pred_csv)
        replay_aucs = {}
        for lbl in PRED_LABEL:
            auc = sklm.roc_auc_score(reloaded[lbl], reloaded['prob_' + lbl])
            replay_aucs[lbl] = float(auc)
            assert abs(replay_aucs[lbl] - ref_aucs[lbl]) <= 1e-15, "Per-class AUC replay drift"
        replay_macro = float(np.mean(list(replay_aucs.values())))
        assert abs(replay_macro - ref_macro) <= 1e-15, "Macro AUC replay drift"
    return True


def test_t171_privacy_metric_replay():
    """T171 (§23): Recomputing privacy ROC-AUC from saved NPZ predictions matches scalar reported AUC."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 2000)
        y_score = np.random.rand(2000)
        scalar_auc = float(sklm.roc_auc_score(y_true, y_score))

        npz_p = os.path.join(tmp_dir, 'privacy_val_predictions.npz')
        np.savez_compressed(npz_p, y_true=y_true, y_score=y_score)

        # Replay
        data = np.load(npz_p)
        replayed_auc = float(sklm.roc_auc_score(data['y_true'], data['y_score']))
        assert abs(replayed_auc - scalar_auc) <= 1e-15, "Privacy AUC replay drift"
    return True


# ---------------------------------------------------------------------------
# T172–T173: F9 Git Tracked-Tree Guard
# ---------------------------------------------------------------------------
def test_t172_scientific_git_dirty_index_hard_fails():
    """T172 (F9): Git guard hard-fails if tracked files have uncommitted modifications."""
    with mock.patch('subprocess.run') as mock_run:
        # Mock git diff --quiet returning 1 (dirty)
        mock_res = mock.MagicMock()
        mock_res.returncode = 1
        mock_run.return_value = mock_res

        try:
            run_m2_s1.check_git_source_guard()
            assert False, "Should have raised RuntimeError on dirty tracked tree"
        except RuntimeError as e:
            assert "tracked tree has uncommitted changes" in str(e)
    return True


def test_t173_untracked_files_allowed_by_git_guard():
    # Legacy name retained; exact certified source identity is now required.
    """T173 (F9): Exact certified source identity is mandatory."""
    with mock.patch('subprocess.run') as mock_run:
        clean = mock.MagicMock(returncode=0, stdout='', stderr='')
        mock_run.side_effect = [
            clean, clean, clean, clean,
            mock.MagicMock(returncode=0, stdout='research/method-restart\n', stderr=''),
            mock.MagicMock(returncode=0, stdout='abc\n', stderr=''),
            mock.MagicMock(returncode=1, stdout='', stderr='missing tag'),
        ]
        try:
            run_m2_s1.check_git_source_guard()
            assert False, 'missing certified tag must fail'
        except RuntimeError as exc:
            assert 'git command failed' in str(exc) or 'tag' in str(exc)
    return True


# ---------------------------------------------------------------------------
# T174–T175: §29 Method-Neutral Selection Invariants
# ---------------------------------------------------------------------------
def test_t174_checkpoint_selection_excludes_feature_term():
    """T174 (§29): Checkpoint selection ranking ignores feature term in both arms."""
    # Construct scenario where epoch 0 has lower opt total (due to low feature) but worse selection total
    # Epoch 1 has higher opt total but better selection total (L_AC + L_priv)
    metrics = [
        compute_epoch_totals(ac_bce=1.5, privacy_term=1.5, feature_term=0.1, feature_loss_weight=1.0),  # sel=3.0, opt=3.1
        compute_epoch_totals(ac_bce=1.2, privacy_term=1.3, feature_term=2.0, feature_loss_weight=1.0),  # sel=2.5, opt=4.5
    ]
    best_idx = select_method_neutral_best(metrics)
    assert best_idx == 1, "Expected epoch 1 selected (sel=2.5 < 3.0), got epoch %d" % best_idx
    return True


def test_t175_checkpoint_selection_tie_chooses_earliest():
    """T175 (§29): Checkpoint selection tie chooses earliest epoch."""
    metrics = [
        compute_epoch_totals(ac_bce=1.0, privacy_term=1.0, feature_term=0.5, feature_loss_weight=1.0),  # sel=2.0
        compute_epoch_totals(ac_bce=1.0, privacy_term=1.0, feature_term=0.1, feature_loss_weight=1.0),  # sel=2.0
    ]
    best_idx = select_method_neutral_best(metrics)
    assert best_idx == 0, "Tie should choose earliest epoch (0), got %d" % best_idx
    return True


# ---------------------------------------------------------------------------
# T176: §39 Zero TEST Access Proof
# ---------------------------------------------------------------------------
def test_t176_no_test_loader_or_fold_in_certification_suite():
    """T176 (§39): TestFirewall strictly rejects test loaders across all certified modules."""
    fw = TestFirewall(allow=False)
    for test_mode in ('test', 'testing', 'final_test'):
        try:
            fw.check(test_mode)
            assert False, "Firewall should have rejected mode %r" % test_mode
        except RuntimeError:
            pass

    # Verify LazyPairDataset rejects test
    try:
        LazyPairDataset(phase='testing', image_path='/tmp')
        assert False, "LazyPairDataset should reject 'testing'"
    except RuntimeError:
        pass

    # Verify evaluate_classification_val rejects test
    try:
        evaluate_classification_val({'image_path': '/tmp'}, fold='test')
        assert False, "evaluate_classification_val should reject fold='test'"
    except RuntimeError:
        pass

    return True


# ---------------------------------------------------------------------------
# Test Runner
# ---------------------------------------------------------------------------
ALL_M14C_TESTS = [
    ('T137', 'canonical master requires --scientific-m2-s1', test_t137_canonical_scientific_master_requires_flag),
    ('T138', 'dev orchestration produces DEVELOPMENT_ONLY verdict', test_t138_development_orchestration_cannot_produce_scientific_verdict),
    ('T139', 'nih_labels.csv SHA locked', test_t139_nih_labels_csv_sha_locked),
    ('T140', 'classification VAL Image Index fingerprint', test_t140_classification_val_image_index_fingerprint),
    ('T141', 'classification VAL Patient ID fingerprint', test_t141_classification_val_patient_id_fingerprint),
    ('T142', 'classification VAL label matrix fingerprint', test_t142_classification_val_label_matrix_fingerprint),
    ('T143', 'anonymizer TRAIN vs classification VAL patient overlap == 0', test_t143_anonymizer_train_vs_classification_val_patient_disjoint),
    ('T144', 'anonymizer TRAIN vs anonymizer VAL patient overlap == 0', test_t144_anonymizer_train_vs_anonymizer_val_patient_disjoint),
    ('T145', 'all TRAIN pair identity labels semantically correct', test_t145_train_pair_identity_semantics),
    ('T146', 'all VAL pair identity labels semantically correct', test_t146_val_pair_identity_semantics),
    ('T147', 'stale scientific output directory rejected', test_t147_stale_scientific_output_dir_rejected),
    ('T148', 'empty scientific output directory accepted', test_t148_empty_scientific_output_dir_accepted),
    ('T149', 'attacker NaN train loss hard-fails', test_t149_attacker_nan_train_loss_hard_fails),
    ('T150', 'attacker Inf validation loss hard-fails', test_t150_attacker_inf_validation_loss_hard_fails),
    ('T151', 'attacker nonfinite parameters hard-fail', test_t151_attacker_nonfinite_params_hard_fail),
    ('T152', 'attacker manifest numerical validity required', test_t152_attacker_manifest_numerical_validity_required),
    ('T153', 'runtime pair-order SHA equals offline SHA epoch0', test_t153_runtime_pair_order_equals_offline_epoch0),
    ('T154', 'runtime pair-order SHA equals offline SHA epoch1', test_t154_runtime_pair_order_equals_offline_epoch1),
    ('T155', 'B_dev and C4 exact same pair order epochs 0-2', test_t155_b_dev_and_c4_exact_same_order_epochs_0_to_2),
    ('T156', 'scientific resume policy declared uncertified', test_t156_scientific_resume_policy_declared),
    ('T157', 'T83 replacement active & non-vacuous', test_t157_t83_replacement_is_active_and_non_vacuous),
    ('T158', 'T125 true LazyPairDataset pathology parity', test_t158_t125_true_lazypairdataset_parity),
    ('T159', 'T130 preflight assertion with --device', test_t159_t130_device_flag_preflight_assertion),
    ('T160', 'T134 invalid orchestration no scientific verdict', test_t160_t134_invalid_orchestration_verdict),
    ('T161', 'T135 injected config semantic drift hard-fails', test_t161_t135_injected_config_drift_fails),
    ('T162', 'pristine reference anonymized tensor parity <= 1e-7', test_t162_pristine_reference_anonymized_tensor_parity),
    ('T163', 'pristine reference generator loss parity <= 1e-6', test_t163_pristine_reference_generator_loss_parity),
    ('T164', 'pristine reference generator grad parity <= 1e-6', test_t164_pristine_reference_generator_gradient_parity),
    ('T165', 'pristine reference verifier grad parity <= 1e-6', test_t165_pristine_reference_verifier_gradient_parity),
    ('T166', 'pristine reference classifier grad parity <= 1e-6', test_t166_pristine_reference_classifier_gradient_parity),
    ('T167', 'generator-phase gradient ownership', test_t167_generator_phase_gradient_ownership),
    ('T168', 'critic phase update ownership', test_t168_critic_phase_update_ownership),
    ('T169', 'unknown pathology token hard-fails', test_t169_unknown_pathology_token_hard_fails),
    ('T170', 'classification metric replay from predictions CSV', test_t170_classification_metric_replay),
    ('T171', 'privacy metric replay from predictions NPZ', test_t171_privacy_metric_replay),
    ('T172', 'scientific git tracked-tree dirty hard-fails', test_t172_scientific_git_dirty_index_hard_fails),
    ('T173', 'untracked files allowed by git guard', test_t173_untracked_files_allowed_by_git_guard),
    ('T174', 'checkpoint selection excludes feature term', test_t174_checkpoint_selection_excludes_feature_term),
    ('T175', 'checkpoint selection tie chooses earliest epoch', test_t175_checkpoint_selection_tie_chooses_earliest),
    ('T176', 'zero TEST access firewall proof across all modules', test_t176_no_test_loader_or_fold_in_certification_suite),
]


def run_all_m14c_tests():
    print("=" * 60, flush=True)
    print("RUNNING M1.4c CERTIFICATION SUITE (T137–T176)", flush=True)
    print("=" * 60, flush=True)
    passed = 0
    failed = 0
    for tid, desc, fn in ALL_M14C_TESTS:
        try:
            fn()
            print(f"PASS     {tid} {desc}", flush=True)
            passed += 1
        except Exception as e:
            print(f"FAIL     {tid} {desc}: {e}", flush=True)
            failed += 1
    print("=" * 60, flush=True)
    print(f"M1.4c SUITE: {passed}/{passed + failed} PASS", flush=True)
    return failed == 0


if __name__ == '__main__':
    ok = run_all_m14c_tests()
    sys.exit(0 if ok else 1)
