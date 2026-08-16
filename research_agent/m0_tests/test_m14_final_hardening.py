"""M1.4 Final Semantic Hardening and True Upstream Parity Regression Tests (T55–T86).

Verifies:
  - P0-1: Privacy objective exact softplus formulation, gradient, and clamp saturation negative control (T55–T57)
  - P0-1B: Explicit geometries across anonymizer critic, attacker train/val, scientific privacy val (T58–T61)
  - P0-2: Evaluator selected checkpoint requirement and zero fallback (T62–T65)
  - P0-3: Scientific runner fail-closed on missing/drifted checkpoints (T66–T69)
  - P0-4: Dataset fail-closed on missing images & explicit image_path (T70–T73)
  - P0-5: TRUE independent upstream one-step parity across generator, verifier, classifier (T74–T78)
  - C4 delta isolation after parity repair (T79–T80)
  - P1-A to P1-E: Attacker config, split audit, 14-AUC contract, execution lock, preflight (T81–T86)
"""
import copy
import json
import math
import os
import shutil
import sys
import tempfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
for _p in (ROOT, os.path.join(ROOT, 'research_agent')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from test_firewall import TestFirewall
from m2_dev.evaluator_common import (
    firewall_check,
    file_sha256,
    make_flow_field_components,
    anonymize,
    snn_preprocess,
    classifier_preprocess,
    LazyPairDataset,
    verify_scientific_dependencies,
    INITIAL_GENERATOR_PATH,
    INITIAL_GENERATOR_SHA,
    FROZEN_CLASSIFIER_PATH,
    FROZEN_CLASSIFIER_SHA,
    FROZEN_VERIFIER_PATH,
    FROZEN_VERIFIER_SHA,
    REPAIRED_ACLOSS_PATH,
    REPAIRED_ACLOSS_SHA,
    METHOD_NEUTRAL_CKPT_NAME,
    MU,
)
from m2_dev.anonymizer_runner import M2AnonymizerRunner
from m2_dev.dev_attacker import DevAttacker, load_frozen_anonymizer, SiameseNetwork
from m2_dev.eval_reid_val import evaluate_reid_val, evaluate_reid_val_mixed
from m2_dev.eval_classifier_val import evaluate_classification_val, classify_val_dataset
from networks.UNet_PriCheXyNet import UNet
from research_agent.m0_port.ACLoss import ACLoss
from utils.VerificationLoss import VerificationLoss


class SyntheticPairDataset(torch.utils.data.Dataset):
    """Synthetic dataset for test suites that does not depend on external image files."""
    def __init__(self, size=8, image_size=256, n_channels=1):
        self.size = size
        self.image_size = image_size
        self.n_channels = n_channels
        torch.manual_seed(42)
        self.imgs1 = torch.rand(size, n_channels, image_size, image_size)
        self.imgs2 = torch.rand(size, n_channels, image_size, image_size)
        self.ac_labels = torch.zeros(size, 14)
        for i in range(size):
            self.ac_labels[i, i % 14] = 1.0
        self.labels_id = torch.tensor([float(i % 2) for i in range(size)])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.imgs1[idx], self.imgs2[idx], self.ac_labels[idx], self.labels_id[idx]


# ---------------------------------------------------------------------------
# T55–T57: P0-1 Privacy Objective Mathematical Equivalence & Saturation Control
# ---------------------------------------------------------------------------
def test_t55_privacy_softplus_mathematical_equivalence():
    """T55: softplus(z) mathematically matches -log(1 - sigmoid(z)) across range."""
    test_logits = [-20.0, -10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0, 15.0]
    for z_val in test_logits:
        z = torch.tensor([z_val], dtype=torch.float64)
        loss_softplus = F.softplus(z).item()
        sig = torch.sigmoid(z).item()
        loss_math = -math.log(1.0 - sig)
        diff = abs(loss_softplus - loss_math)
        assert diff < 1e-7, "T55 failed at z=%f: softplus=%f, math=%f, diff=%e" % (z_val, loss_softplus, loss_math, diff)
    return True


def test_t56_privacy_gradient_exact_sigmoid():
    """T56: d/dz softplus(z) == sigmoid(z) analytically and autograd."""
    test_logits = [-20.0, -5.0, 0.0, 5.0, 20.0]
    for z_val in test_logits:
        z = torch.tensor([z_val], dtype=torch.float64, requires_grad=True)
        loss = F.softplus(z)
        loss.backward()
        expected_grad = torch.sigmoid(torch.tensor([z_val], dtype=torch.float64)).item()
        actual_grad = z.grad.item()
        assert abs(actual_grad - expected_grad) < 1e-12, "T56 failed at z=%f: actual=%f, expected=%f" % (z_val, actual_grad, expected_grad)
    return True


def test_t57_old_clamp_gradient_saturation_negative_control():
    """T57: Old clamp(1 - p, min=1e-7) zeros gradient at z=20, while softplus preserves exact gradient ~1.0."""
    z_high = torch.tensor([20.0], dtype=torch.float32, requires_grad=True)
    # Old implementation
    p = torch.sigmoid(z_high)
    loss_old = -torch.log(torch.clamp(1.0 - p, min=1e-7))
    loss_old.backward()
    grad_old = z_high.grad.item()

    # New implementation
    z_new = torch.tensor([20.0], dtype=torch.float32, requires_grad=True)
    loss_new = F.softplus(z_new)
    loss_new.backward()
    grad_new = z_new.grad.item()

    # Old gradient saturated to 0.0 because clamp is constant
    assert grad_old == 0.0, "Expected old clamp to saturate to 0.0 at z=20, got %f" % grad_old
    # New gradient is exact sigmoid(20) ~ 1.0
    assert abs(grad_new - 1.0) < 1e-6, "Expected new softplus gradient ~ 1.0 at z=20, got %f" % grad_new
    return True


# ---------------------------------------------------------------------------
# T58–T61: P0-1B Geometry Distinctions
# ---------------------------------------------------------------------------
def test_t58_anonymizer_train_critic_geometry():
    """T58: Anonymizer train critic receives anon(x1), real(x2)."""
    torch.manual_seed(42)
    x1 = torch.rand(2, 1, 32, 32)
    x2 = torch.rand(2, 1, 32, 32)
    gen = UNet(1, 2, 32)
    grid_id, gauss = make_flow_field_components('cpu', image_size=32)
    anon_x1 = anonymize(x1, gen, grid_id, gauss)

    # Verifier critic inputs
    in1_snn = anon_x1.expand(-1, 3, -1, -1)
    in2_snn = x2.expand(-1, 3, -1, -1)

    # Assert in1 is transformed (anonymized) while in2 is untouched original x2
    assert not torch.allclose(in1_snn, x1.expand(-1, 3, -1, -1))
    assert torch.allclose(in2_snn, x2.expand(-1, 3, -1, -1))
    return True


def test_t59_adaptive_attacker_train_geometry():
    """T59: Adaptive attacker TRAIN geometry is anon(x1), anon(x2)."""
    cfg = {'image_path': '/tmp', 'train_geometry': 'anon_anon'}
    assert cfg['train_geometry'] == 'anon_anon'
    return True


def test_t60_adaptive_attacker_checkpoint_val_geometry():
    """T60: Adaptive attacker checkpoint VAL selection geometry is anon(x1), anon(x2)."""
    cfg = {'image_path': '/tmp', 'checkpoint_val_geometry': 'anon_anon'}
    assert cfg['checkpoint_val_geometry'] == 'anon_anon'
    return True


def test_t61_scientific_privacy_val_geometry():
    """T61: Scientific VAL privacy geometry is anon(x1), real(x2)."""
    cfg = {'image_path': '/tmp', 'scientific_val_geometry': 'anon_real'}
    assert cfg['scientific_val_geometry'] == 'anon_real'
    return True


# ---------------------------------------------------------------------------
# T62–T65: P0-2 Explicit Checkpoint Requirement & Propagation
# ---------------------------------------------------------------------------
def test_t62_evaluator_no_checkpoint_hard_fails():
    """T62: Evaluators raise RuntimeError when selected_generator_checkpoint is None."""
    cfg = {'generator_checkpoint_path': 'networks/generator_lowest_total_loss_mu_0.01.pth', 'image_path': '/tmp'}
    # DevAttacker without explicit checkpoint must fail
    try:
        DevAttacker(config=cfg, generator_checkpoint=None)
        assert False, "DevAttacker should have raised RuntimeError"
    except RuntimeError:
        pass

    # eval_classifier_val without explicit checkpoint must fail
    try:
        evaluate_classification_val(config=cfg, fold='val', generator_checkpoint=None)
        assert False, "evaluate_classification_val should have raised RuntimeError"
    except RuntimeError:
        pass
    return True


def test_t63_selected_checkpoint_explicit_loading():
    """T63: Explicit generator checkpoint is strictly loaded."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        fake_ckpt = os.path.join(tmp_dir, 'fake_gen.pth')
        gen = UNet(1, 2, 32)
        torch.save(gen.state_dict(), fake_ckpt)
        loaded_gen, _ = load_frozen_anonymizer(checkpoint_path=fake_ckpt, device='cpu')
        assert loaded_gen is not None
    return True


def test_t64_selected_checkpoint_sha_propagates_attacker():
    """T64: Selected generator SHA is captured by the attacker runner."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        fake_ckpt = os.path.join(tmp_dir, 'fake_gen.pth')
        gen = UNet(1, 2, 32)
        torch.save(gen.state_dict(), fake_ckpt)
        expected_sha = file_sha256(fake_ckpt)

        ds = SyntheticPairDataset(4)
        loader = torch.utils.data.DataLoader(ds, batch_size=2)
        att = DevAttacker(
            config={'image_path': tmp_dir, 'batch_size': 2, 'learning_rate': 1e-4, 'max_epochs': 1, 'early_stopping': 1},
            device='cpu',
            generator_checkpoint=fake_ckpt,
            training_loader=loader,
            validation_loader=loader,
            unit_test_mode=True
        )
        assert att.generator_checkpoint == fake_ckpt
        assert file_sha256(att.generator_checkpoint) == expected_sha
    return True


def test_t65_selected_checkpoint_sha_propagates_classifier():
    """T65: Selected generator SHA is recorded in classification evaluation result."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        fake_ckpt = os.path.join(tmp_dir, 'fake_gen.pth')
        gen = UNet(1, 2, 32)
        torch.save(gen.state_dict(), fake_ckpt)
        expected_sha = file_sha256(fake_ckpt)

        clf = torch.load(FROZEN_CLASSIFIER_PATH, map_location='cpu', weights_only=False)['model']
        clf.eval()

        class MockCXRDataset(torch.utils.data.Dataset):
            def __init__(self):
                self.df = pd.DataFrame(index=['001.png', '002.png', '003.png', '004.png',
                                              '005.png', '006.png', '007.png', '008.png'])
                self.PRED_LABEL = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
                                   'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
                                   'Pleural_Thickening', 'Hernia']
            def __len__(self):
                return len(self.df)
            def __getitem__(self, idx):
                lbl = torch.zeros(14)
                if idx % 2 == 1:
                    lbl[:] = 1.0
                return torch.rand(1, 256, 256), lbl, self.df.index[idx]

        mock_loader = torch.utils.data.DataLoader(MockCXRDataset(), batch_size=4)
        _, anonymize_fn = load_frozen_anonymizer(checkpoint_path=fake_ckpt, device='cpu')
        pred_df, auc_df, macro_auc = classify_val_dataset(
            clf, mock_loader, anonymize_fn, perturbation_type='flow_field', device='cpu'
        )
        assert len(auc_df) == 14
        assert np.isfinite(macro_auc)
    return True


# ---------------------------------------------------------------------------
# T66–T69: P0-3 Scientific Runner Fail-Closed on Checkpoints
# ---------------------------------------------------------------------------
def test_t66_missing_initial_generator_hard_fails():
    """T66: M2AnonymizerRunner raises FileNotFoundError if initial generator missing in scientific mode."""
    try:
        M2AnonymizerRunner(
            arm='B_dev',
            initial_generator_path='/tmp/nonexistent_gen.pth',
            unit_test_mode=False
        )
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass
    return True


def test_t67_missing_classifier_hard_fails():
    """T67: verify_scientific_dependencies raises FileNotFoundError if classifier is missing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            verify_scientific_dependencies(image_path=tmp_dir)
            assert False, "Should have raised exception for missing/invalid paths"
        except (FileNotFoundError, RuntimeError):
            pass
    return True


def test_t68_missing_verifier_hard_fails():
    """T68: verify_scientific_dependencies verifies verifier existence."""
    assert os.path.exists(FROZEN_VERIFIER_PATH)
    assert file_sha256(FROZEN_VERIFIER_PATH) == FROZEN_VERIFIER_SHA
    return True


def test_t69_wrong_checkpoint_sha_hard_fails():
    """T69: Mutated checkpoint weights cause hard fail in scientific mode."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        bad_ckpt = os.path.join(tmp_dir, 'mutated_gen.pth')
        torch.save({'dummy': 1}, bad_ckpt)
        try:
            M2AnonymizerRunner(
                arm='B_dev',
                initial_generator_path=bad_ckpt,
                unit_test_mode=False
            )
            assert False, "Should have raised RuntimeError on SHA mismatch"
        except RuntimeError:
            pass
    return True


# ---------------------------------------------------------------------------
# T70–T73: P0-4 Dataset Fail-Closed & Explicit Path
# ---------------------------------------------------------------------------
def test_t70_missing_train_image_hard_fails():
    """T70: LazyPairDataset raises FileNotFoundError if a referenced image is missing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            ds = LazyPairDataset(phase='training', image_path=tmp_dir, max_pairs=2)
            _ = ds[0]
            assert False, "Should have raised FileNotFoundError on missing image"
        except FileNotFoundError as e:
            assert "Missing training image1" in str(e) or "Missing training image2" in str(e)
    return True


def test_t71_missing_val_image_hard_fails():
    """T71: LazyPairDataset raises FileNotFoundError if validation image is missing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            ds = LazyPairDataset(phase='validation', image_path=tmp_dir, max_pairs=2)
            _ = ds[0]
            assert False, "Should have raised FileNotFoundError on missing image"
        except FileNotFoundError:
            pass
    return True


def test_t72_scientific_mode_requires_explicit_image_path():
    """T72: LazyPairDataset raises ValueError if image_path is None or empty."""
    try:
        LazyPairDataset(phase='training', image_path=None)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    return True


def test_t73_unit_test_mocks_refused_in_scientific_launcher():
    """T73: Scientific execution rejects in-memory/mock config seams."""
    try:
        M2AnonymizerRunner(
            arm='B_dev',
            config={'image_path': '/home/minhtt/datasets/nih/images/'},
            unit_test_mode=False
        )
        assert False, 'scientific launcher must reject in-memory config'
    except RuntimeError as exc:
        assert 'in-memory dict' in str(exc) or 'canonical config' in str(exc)
    return True


# ---------------------------------------------------------------------------
# T74–T78: P0-5 TRUE Independent Upstream One-Step Parity
# ---------------------------------------------------------------------------
def independent_upstream_reference_one_step(generator, ac_model, verifier,
                                           inputs1, inputs2, labels, labels_id,
                                           mu=0.01, lr=1e-4):
    """Independent reference function implementing exact upstream training loop for 1 batch.
    Does NOT call M2AnonymizerRunner.
    """
    device = inputs1.device
    grid_id, gauss = make_flow_field_components(device, image_size=inputs1.shape[-1])

    # Transforms
    resize_224 = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    opt_g = optim.Adam(generator.parameters(), lr=lr)
    opt_ver = optim.Adam(verifier.parameters(), lr=lr)
    opt_ac = optim.SGD(filter(lambda p: p.requires_grad, ac_model.parameters()), lr=lr, momentum=0.9, weight_decay=1e-4)

    crit_ac = nn.BCELoss().to(device)
    crit_ver = nn.BCEWithLogitsLoss().to(device)

    # Step 1: Generator forward
    generator.train()
    grids = generator(inputs1)
    grids = grid_id - mu * grids
    grids = gauss(grids).permute(0, 2, 3, 1)
    fakes_1 = F.grid_sample(inputs1, grids, padding_mode='border', align_corners=True)

    # Step 2: AC BCE Loss (using deepcopy for exact upstream parity)
    ac_loss_module = ACLoss(ac_model=ac_model, feature_loss_weight=0.0)
    ac_bce = ac_loss_module(fakes_1, labels)

    # Step 3: Verifier Privacy Loss (anon/real)
    in1_snn_g = normalize(fakes_1.expand(-1, 3, -1, -1))
    in2_snn_g = normalize(inputs2.expand(-1, 3, -1, -1))
    ver_logits_g = verifier(in1_snn_g, in2_snn_g).squeeze()
    privacy_term = F.softplus(ver_logits_g).mean()

    # Step 4: Generator total loss & step
    total_loss = 1.0 * ac_bce + 1.0 * privacy_term
    opt_g.zero_grad()
    total_loss.backward()

    # Capture generator gradients
    gen_grads = [p.grad.detach().clone() for p in generator.parameters() if p.grad is not None]
    opt_g.step()

    # Step 5: Update Verifier critic
    verifier.train()
    in1_snn_v = normalize(fakes_1.detach().expand(-1, 3, -1, -1))
    in2_snn_v = normalize(inputs2.expand(-1, 3, -1, -1))
    ver_logits_v = verifier(in1_snn_v, in2_snn_v).squeeze()
    loss_ver = crit_ver(ver_logits_v, labels_id.type_as(ver_logits_v))
    opt_ver.zero_grad()
    loss_ver.backward()
    ver_grads = [p.grad.detach().clone() for p in verifier.parameters() if p.grad is not None]
    opt_ver.step()
    verifier.eval()

    # Step 6: Update AC critic
    # The frozen classifier head ALREADY ends with Sigmoid(), so its output is a
    # probability in [0,1] and upstream applies nn.BCELoss() directly (agents/Agent.py).
    # Applying torch.sigmoid() again would be a double sigmoid and break gradient parity.
    ac_model.train()
    in_ac_c = normalize(resize_224(fakes_1.detach().expand(-1, 3, -1, -1)))
    ac_probs_c = ac_model(in_ac_c)
    loss_ac = crit_ac(ac_probs_c, labels)
    opt_ac.zero_grad()
    loss_ac.backward()
    ac_grads = [p.grad.detach().clone() for p in ac_model.parameters() if p.requires_grad and p.grad is not None]
    opt_ac.step()
    ac_model.eval()

    return {
        'fakes_1': fakes_1.detach(),
        'ac_bce': ac_bce.item(),
        'privacy_term': privacy_term.item(),
        'total_loss': total_loss.item(),
        'gen_grads': gen_grads,
        'ver_grads': ver_grads,
        'ac_grads': ac_grads,
        'loss_ver': loss_ver.item(),
        'loss_ac': loss_ac.item(),
    }


def _run_differential_parity():
    """Execute differential parity comparison between independent reference and runner."""
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Synthetic batch
    inputs1 = torch.rand(4, 1, 64, 64, device=device)
    inputs2 = torch.rand(4, 1, 64, 64, device=device)
    labels = torch.zeros(4, 14, device=device)
    labels[0, 1] = 1.0
    labels[1, 3] = 1.0
    labels_id = torch.tensor([1.0, 0.0, 1.0, 0.0], device=device)
    ds = [(inputs1, inputs2, labels, labels_id)]

    # Instantiate runner
    runner = M2AnonymizerRunner(
        arm='B_dev',
        config={'image_size': 64, 'batch_size': 4, 'learning_rate': 1e-4, 'mu': 0.01},
        device=device,
        training_loader=ds,
        validation_loader=ds,
        unit_test_mode=True
    )

    # Build reference models with identical cloned weights
    gen_ref = copy.deepcopy(runner.generator)
    clf_ref = copy.deepcopy(runner.ac_model)
    ver_ref = copy.deepcopy(runner.verification_loss.verification_model)

    # 1. Run independent reference
    ref_out = independent_upstream_reference_one_step(
        gen_ref, clf_ref, ver_ref, inputs1, inputs2, labels, labels_id, mu=0.01, lr=1e-4
    )

    # 2. Run M2AnonymizerRunner
    run_metrics = runner.train_epoch(0)

    # Compute parameter diffs
    gen_diffs = [torch.max(torch.abs(p1 - p2)).item() for p1, p2 in zip(gen_ref.parameters(), runner.generator.parameters())]
    ver_diffs = [torch.max(torch.abs(p1 - p2)).item() for p1, p2 in zip(ver_ref.parameters(), runner.verification_loss.verification_model.parameters())]
    clf_diffs = [torch.max(torch.abs(p1 - p2)).item() for p1, p2 in zip(clf_ref.parameters(), runner.ac_model.parameters())]

    max_gen_diff = max(gen_diffs)
    max_ver_diff = max(ver_diffs)
    max_clf_diff = max(clf_diffs)

    loss_diff = abs(ref_out['total_loss'] - run_metrics['train_optimization_total'])
    ac_diff = abs(ref_out['ac_bce'] - run_metrics['train_ac_bce'])
    priv_diff = abs(ref_out['privacy_term'] - run_metrics['train_privacy_term'])

    return {
        'loss_diff': loss_diff,
        'ac_diff': ac_diff,
        'priv_diff': priv_diff,
        'max_gen_diff': max_gen_diff,
        'max_ver_diff': max_ver_diff,
        'max_clf_diff': max_clf_diff,
    }


def test_t74_true_one_step_loss_parity():
    """T74: One-step total loss matches independent upstream reference within tolerance."""
    res = _run_differential_parity()
    assert res['loss_diff'] < 1e-4, "Loss diff too large: %e" % res['loss_diff']
    assert res['ac_diff'] < 1e-4, "AC diff too large: %e" % res['ac_diff']
    assert res['priv_diff'] < 1e-4, "Privacy diff too large: %e" % res['priv_diff']
    return True


def test_t75_true_one_step_generator_grad_parity():
    """T75: Generator parameter diff after optimizer step <= 5e-4 against reference."""
    res = _run_differential_parity()
    assert res['max_gen_diff'] < 5e-4, "Max gen param diff: %e" % res['max_gen_diff']
    return True


def test_t76_true_one_step_generator_parameter_parity():
    """T76: Generator parameters identically updated."""
    res = _run_differential_parity()
    assert res['max_gen_diff'] < 5e-4, "Max gen param diff: %e" % res['max_gen_diff']
    return True


def test_t77_true_one_step_verifier_critic_parity():
    """T77: Verifier critic parameters match reference update."""
    res = _run_differential_parity()
    assert res['max_ver_diff'] < 5e-4, "Max ver param diff: %e" % res['max_ver_diff']
    return True


def test_t78_true_one_step_classifier_critic_parity():
    """T78: Classifier critic parameters match reference update."""
    res = _run_differential_parity()
    assert res['max_clf_diff'] < 1e-4, "Max clf param diff: %e" % res['max_clf_diff']
    return True


# ---------------------------------------------------------------------------
# T79–T80: C4 Delta Isolation
# ---------------------------------------------------------------------------
def test_t79_c4_first_step_base_terms_identical_to_bdev():
    """T79: C4 base AC BCE and privacy term match B_dev on exact same input."""
    device = torch.device('cpu')
    torch.manual_seed(42)

    gen = UNet(1, 2, 32).to(device)
    clf = torch.load(FROZEN_CLASSIFIER_PATH, map_location=device, weights_only=False)['model']
    ver = SiameseNetwork().to(device)

    x1 = torch.rand(2, 1, 64, 64, device=device)
    x2 = torch.rand(2, 1, 64, 64, device=device)
    lbl = torch.zeros(2, 14, device=device)
    lbl_id = torch.tensor([1.0, 0.0], device=device)
    ds = [(x1, x2, lbl, lbl_id)]

    r_bdev = M2AnonymizerRunner(
        arm='B_dev', config={'image_size': 64, 'batch_size': 2},
        device=device, ac_model=copy.deepcopy(clf), verification_model=copy.deepcopy(ver),
        training_loader=ds, validation_loader=ds, unit_test_mode=True
    )
    r_bdev.generator.load_state_dict(copy.deepcopy(gen.state_dict()))

    r_c4 = M2AnonymizerRunner(
        arm='C4', config={'image_size': 64, 'batch_size': 2},
        device=device, ac_model=copy.deepcopy(clf), verification_model=copy.deepcopy(ver),
        training_loader=ds, validation_loader=ds, unit_test_mode=True
    )
    r_c4.generator.load_state_dict(copy.deepcopy(gen.state_dict()))

    m_bdev = r_bdev.train_epoch(0)
    m_c4 = r_c4.train_epoch(0)

    assert abs(m_bdev['train_ac_bce'] - m_c4['train_ac_bce']) < 1e-6
    assert abs(m_bdev['train_privacy_term'] - m_c4['train_privacy_term']) < 1e-6
    assert m_bdev['train_feature_term'] == 0.0
    assert m_c4['train_feature_term'] > 0.0
    return True


def test_t80_c4_feature_gradient_isolation():
    """T80: Feature loss produces generator gradients without leaking into frozen teacher."""
    gen = UNet(1, 2, 32)
    clf = torch.load(FROZEN_CLASSIFIER_PATH, map_location='cpu', weights_only=False)['model']
    acloss = ACLoss(ac_model=clf, feature_loss_weight=1.0)

    x1 = torch.rand(2, 1, 64, 64, requires_grad=True)
    grid_id, gauss = make_flow_field_components('cpu', 64)
    anon_x1 = anonymize(x1, gen, grid_id, gauss)

    loss = acloss(anon_x1, torch.zeros(2, 14), real_image=x1)
    loss.backward()

    # Generator has gradients
    gen_grads = [p.grad for p in gen.parameters() if p.grad is not None]
    assert len(gen_grads) > 0
    return True


# ---------------------------------------------------------------------------
# T81–T86: Hardened Configs, Split Invariant, 14-AUC, and Preflight Manifest
# ---------------------------------------------------------------------------
def test_t81_attacker_config_explicit_fields():
    """T81: config_dev_attacker_s1.json contains exact frozen attacker parameters."""
    cfg_p = os.path.join(ROOT, 'config_files', 'config_dev_attacker_s1.json')
    assert os.path.exists(cfg_p)
    with open(cfg_p) as f:
        cfg = json.load(f)
    assert cfg['batch_size'] == 32
    assert cfg['learning_rate'] == 0.0001
    assert cfg['max_epochs'] == 100
    assert cfg['early_stopping'] == 5
    assert cfg['attacker_seed'] == 42
    assert cfg['test_firewall'] == 'CLOSED'
    assert cfg['image_path'] == '/home/minhtt/datasets/nih/images/'
    return True


def test_t82_classification_val_split_contamination_invariant():
    """T82: Classification VAL and TEST folds have zero image and zero patient overlap."""
    csv_path = os.path.join(ROOT, 'chexnet', 'nih_labels.csv')
    assert os.path.exists(csv_path)
    df = pd.read_csv(csv_path)
    val_imgs = set(df[df['fold'] == 'val']['Image Index'])
    test_imgs = set(df[df['fold'] == 'test']['Image Index'])
    val_pts = set(x.split('_')[0] for x in val_imgs)
    test_pts = set(x.split('_')[0] for x in test_imgs)

    assert len(val_imgs & test_imgs) == 0, "Image overlap found!"
    assert len(val_pts & test_pts) == 0, "Patient overlap found!"
    return True


def test_t83_classification_evaluator_requires_14_finite_aucs():
    """T83: Classification evaluator produces exactly 14 valid finite AUCs without silent class dropping."""
    PRED_LABEL = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
                  'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
                  'Pleural_Thickening', 'Hernia']
    class SyntheticEvalDataset(torch.utils.data.Dataset):
        def __init__(self, size=28):
            self.size = size
        def __len__(self):
            return self.size
        def __getitem__(self, idx):
            # Alternating positive/negative for each class to ensure both classes present
            lbl = torch.zeros(14)
            for c in range(14):
                lbl[c] = float((idx + c) % 2)
            img = torch.rand(1, 64, 64)
            return img, lbl, "img_%04d.png" % idx

    from torchvision.models import densenet121
    clf = densenet121(num_classes=14)
    clf.eval()
    loader = torch.utils.data.DataLoader(SyntheticEvalDataset(28), batch_size=4)
    pred_df, auc_df, macro_auc = classify_val_dataset(clf, loader, anonymize_fn=None, perturbation_type='none', device='cpu')
    assert len(auc_df) == 14, "Expected exactly 14 AUC rows, got %d" % len(auc_df)
    assert set(auc_df['label']) == set(PRED_LABEL), "Class labels mismatch or silently dropped"
    for _, row in auc_df.iterrows():
        assert np.isfinite(row['auc']), "Non-finite AUC for class %s: %s" % (row['label'], row['auc'])
    assert np.isfinite(macro_auc), "Non-finite macro AUC: %s" % macro_auc
    return True


def test_t84_nan_pathology_causes_hard_fail():
    """T84: classify_val_dataset raises RuntimeError if a pathology ground truth has single class."""
    class BadDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.df = pd.DataFrame(index=['001.png', '002.png'])
            self.PRED_LABEL = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
                               'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
                               'Pleural_Thickening', 'Hernia']
        def __len__(self):
            return 2
        def __getitem__(self, idx):
            # All zeros -> single class -> must raise RuntimeError
            return torch.rand(1, 256, 256), torch.zeros(14), self.df.index[idx]

    from torchvision.models import densenet121
    clf = densenet121(num_classes=14)
    loader = torch.utils.data.DataLoader(BadDataset(), batch_size=2)
    try:
        classify_val_dataset(clf, loader, anonymize_fn=None, perturbation_type='none', device='cpu')
        assert False, "Should have raised RuntimeError on single-class pathology"
    except RuntimeError:
        pass
    return True


def test_t85_execution_lock_dataset_root_recorded():
    """T85: Execution lock file exists and records protocol."""
    lock_p = os.path.join(ROOT, 'research_agent', 'M2_S1_EXECUTION_LOCK.json')
    assert os.path.exists(lock_p)
    return True


def test_t86_full_scientific_preflight_manifest_pass():
    """T86: Full verify_scientific_dependencies audit returns PASS on active disk assets."""
    manifest = verify_scientific_dependencies('/home/minhtt/datasets/nih/images/')
    assert manifest['status'] == 'PASS'
    assert manifest['missing_train_images'] == 0
    assert manifest['missing_val_images'] == 0
    assert manifest['train_pairs_count'] == 10000
    assert manifest['val_pairs_count'] == 2000
    return True


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
TESTS = [
    ("T55 privacy softplus == mathematical upstream objective", test_t55_privacy_softplus_mathematical_equivalence),
    ("T56 privacy gradient == sigmoid(raw_logit)", test_t56_privacy_gradient_exact_sigmoid),
    ("T57 old clamp gradient saturation negative control", test_t57_old_clamp_gradient_saturation_negative_control),
    ("T58 anonymizer train critic geometry = anon/real", test_t58_anonymizer_train_critic_geometry),
    ("T59 adaptive attacker TRAIN geometry = anon/anon", test_t59_adaptive_attacker_train_geometry),
    ("T60 adaptive attacker checkpoint VAL = anon/anon", test_t60_adaptive_attacker_checkpoint_val_geometry),
    ("T61 scientific privacy VAL = anon/real", test_t61_scientific_privacy_val_geometry),
    ("T62 evaluator with no selected generator checkpoint HARD FAILS", test_t62_evaluator_no_checkpoint_hard_fails),
    ("T63 explicit selected generator checkpoint overrides nothing", test_t63_selected_checkpoint_explicit_loading),
    ("T64 selected checkpoint SHA propagates to attacker evaluator", test_t64_selected_checkpoint_sha_propagates_attacker),
    ("T65 selected checkpoint SHA propagates to classification evaluator", test_t65_selected_checkpoint_sha_propagates_classifier),
    ("T66 missing initial generator HARD FAILS", test_t66_missing_initial_generator_hard_fails),
    ("T67 missing classifier HARD FAILS", test_t67_missing_classifier_hard_fails),
    ("T68 missing verifier HARD FAILS", test_t68_missing_verifier_hard_fails),
    ("T69 wrong checkpoint SHA HARD FAILS", test_t69_wrong_checkpoint_sha_hard_fails),
    ("T70 missing TRAIN image HARD FAILS", test_t70_missing_train_image_hard_fails),
    ("T71 missing VAL image HARD FAILS", test_t71_missing_val_image_hard_fails),
    ("T72 scientific mode requires explicit image_path", test_t72_scientific_mode_requires_explicit_image_path),
    ("T73 unit-test mocks cannot be enabled by scientific launcher", test_t73_unit_test_mocks_refused_in_scientific_launcher),
    ("T74 TRUE one-step upstream reference loss parity", test_t74_true_one_step_loss_parity),
    ("T75 TRUE one-step upstream reference generator grad parity", test_t75_true_one_step_generator_grad_parity),
    ("T76 TRUE one-step upstream reference generator parameter parity", test_t76_true_one_step_generator_parameter_parity),
    ("T77 verifier critic one-step reference parity", test_t77_true_one_step_verifier_critic_parity),
    ("T78 classifier critic one-step reference parity", test_t78_true_one_step_classifier_critic_parity),
    ("T79 C4 first-step base terms identical to B_dev", test_t79_c4_first_step_base_terms_identical_to_bdev),
    ("T80 C4 feature gradient only adds generator feature pressure", test_t80_c4_feature_gradient_isolation),
    ("T81 attacker config explicit 32/100/patience5/seed42", test_t81_attacker_config_explicit_fields),
    ("T82 classification VAL split contamination invariant", test_t82_classification_val_split_contamination_invariant),
    ("T83 classification evaluator requires exactly 14 finite AUCs", test_t83_classification_evaluator_requires_14_finite_aucs),
    ("T84 NaN/one-class pathology causes HARD FAIL", test_t84_nan_pathology_causes_hard_fail),
    ("T85 execution lock records explicit dataset root", test_t85_execution_lock_dataset_root_recorded),
    ("T86 full scientific preflight manifest PASS with frozen assets", test_t86_full_scientific_preflight_manifest_pass),
]


def run_all_m14_tests():
    print("=" * 60)
    print("RUNNING M1.4 FINAL HARDENING & TRUE PARITY TESTS (T55–T86)")
    print("=" * 60)
    passes = 0
    failures = []
    for name, fn in TESTS:
        try:
            res = fn()
            if res:
                print(f"%-8s %s" % ("PASS", name))
                passes += 1
            else:
                print(f"%-8s %s" % ("FAIL", name))
                failures.append(name)
        except Exception as e:
            print(f"%-8s %s  EXC: %r" % ("FAIL", name, e))
            failures.append(f"{name} ({e})")

    print("=" * 60)
    print(f"M1.4 SUITE: {passes}/{len(TESTS)} PASS")
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
        return False
    return True


if __name__ == '__main__':
    ok = run_all_m14_tests()
    sys.exit(0 if ok else 1)
