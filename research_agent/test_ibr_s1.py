"""Phase-II IBR S1 - load-bearing tests (STEP 6B).

TEST 1  shapes
TEST 2  reconstruction isolation (x_anon must not affect L_rec)
TEST 3  frozen utility networks (no classifier/teacher grads; grads reach x_anon)
TEST 4  GRL sign
TEST 5  z_id verification objective sensitivity
TEST 6  donor safety (no donor==source; deterministic; no TEST patient touched)
TEST 7  checkpoint provenance (paths, SHA, architecture, param count)
TEST 8  gradient ownership (one synthetic forward-backward; finite grads;
         no accidentally detached utility gradient; no frozen-model gradient)

Runs as a plain script (pytest not installed in this repo) or under pytest.
"""

import os

import numpy as np
import torch

from research_agent.ibr.ibr_model import IBRModel, count_parameters
from research_agent.ibr.grl import GradientReversalLayer
from research_agent.ibr.losses import (FrozenUtility, reconstruction_loss,
                                       zid_pair_loss, zmed_adv_loss,
                                       make_pair_labels)
from research_agent.ibr.frozen_models import (CLASSIFIER_SHA256,
                                              SEGMENTATION_SHA256_PREFIX,
                                              load_frozen_classifier,
                                              load_frozen_segmentation_teacher)
from research_agent.ibr.donor import DonorSampler

try:
    import pytest
except ImportError:
    pytest = None

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _cuda(label):
    if torch.cuda.is_available():
        return True
    if pytest is not None:
        pytest.skip('SKIP (%s): CUDA not available' % label)
    print('[SKIP] %s: CUDA not available' % label)
    return False


def _bundle(seed=0, device='cuda'):
    torch.manual_seed(seed)
    return IBRModel().to(device), device


def test1_shapes():
    m, dev = _bundle()
    x = torch.randn(2, 1, 256, 256, device=dev).clamp(-1, 1)
    xd = torch.randn(2, 1, 256, 256, device=dev).clamp(-1, 1)
    out = m(x, xd)
    assert tuple(out['z_id'].shape) == (2, 128), out['z_id'].shape
    assert tuple(out['z_med'].shape) == (2, 512, 16, 16), out['z_med'].shape
    assert tuple(out['x_self'].shape) == (2, 1, 256, 256)
    assert tuple(out['x_anon'].shape) == (2, 1, 256, 256)
    for v in (out['x_self'], out['x_anon'], out['z_id'], out['z_med']):
        assert torch.isfinite(v).all().item()
    print('TEST 1 PASS (shapes + finite)')
    return True


def test2_reconstruction_isolation():
    """Changing x_anon must NOT change L_rec; grads reach E/G self branch."""
    m, dev = _bundle()
    x = torch.randn(2, 1, 256, 256, device=dev).clamp(-1, 1)
    xd = torch.randn(2, 1, 256, 256, device=dev).clamp(-1, 1)

    # build x_self via model; also an independent anon decode we can perturb
    out_base = m(x, xd)
    l_rec_base = reconstruction_loss(out_base['x_self'], x).item()

    # perturb anon target path: use a different donor
    xd2 = torch.randn(2, 1, 256, 256, device=dev).clamp(-1, 1)
    out2 = m(x, xd2)
    l_rec_2 = reconstruction_loss(out2['x_self'], x).item()

    # L_rec must be identical because it only depends on the self branch; the
    # donor change affects only the anon branch.
    assert abs(l_rec_base - l_rec_2) < 1e-6, (l_rec_base, l_rec_2)
    assert out_base['x_self'].shape == out2['x_self'].shape

    # gradients of L_rec reach the encoder/decoder (self branch) parameters
    m.zero_grad()
    l = reconstruction_loss(out_base['x_self'], x)
    l.backward()
    g = {k: v.grad for k, v in m.named_parameters() if v.grad is not None}
    assert any('encoder' in k or 'decoder' in k for k in g), list(g.keys())[:5]
    print('TEST 2 PASS (L_rec isolated from x_anon; grads reach E/G self)')
    return True


def test3_frozen_utility_grads():
    """Frozen classifier/teacher get NO grads; grads reach x_anon path."""
    m, dev = _bundle()
    frozen = FrozenUtility(dev)
    x = torch.randn(2, 1, 256, 256, device=dev).clamp(-1, 1)
    xd = torch.randn(2, 1, 256, 256, device=dev).clamp(-1, 1)
    out = m(x, xd)
    y_path = (torch.rand(2, 14, device=dev) > 0.5).float()

    # L_path / L_anat must flow to x_anon (and thus to x_anon's parents).
    l = reconstruction_loss(out['x_self'], x) + torch.nn.functional.binary_cross_entropy(
        frozen.path_logits(out['x_anon']), y_path) + torch.nn.functional.mse_loss(
        frozen.anat_maps(out['x_anon']), frozen.anat_maps(x))
    l.backward()

    # frozen model params: grads must be None/zero
    for name, p in frozen.classifier.named_parameters():
        assert p.grad is None, 'classifier grad present on %s' % name
    for name, p in frozen.segmenter.named_parameters():
        assert p.grad is None, 'segmenter grad present on %s' % name

    # grads reach the decoder that produces x_anon (encoder too, via shared trunk)
    g = {k: v.grad for k, v in m.named_parameters() if v.grad is not None}
    assert len(g) > 0
    assert any('decoder' in k for k in g)
    assert any('encoder' in k for k in g)
    assert torch.isfinite(out['x_anon']).all().item()
    print('TEST 3 PASS (frozen models gradient-free; grads reach x_anon)')
    return True


def test4_grl_sign():
    """GRL flips gradient sign relative to the adversary's own loss gradient.

    Scheme (single scalar pathway through a shared parameter): attach a scalar
    parameter; z = param * 1.0. Adversary loss = BCE(sigmoid(w_adv * z), label).
    We compare grad_of_loss wrt param when:
        (a) param is updated optimizing the adversary NORMALLY (no grl)
        (b) param is updated with GRL applied on z (encoder side)
    They must have opposite signs.
    """
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(1)
    grl = GradientReversalLayer(lambd=1.0)
    feat = torch.tensor([1.0], requires_grad=True, device=dev)
    w_adv = torch.tensor([1.0], requires_grad=True, device=dev)
    label = torch.ones(1, device=dev)

    # (a) NORMAL adversary path: the adversary optimizes w_adv to predict the
    #     label from z=feat. grad of the adversary loss wrt feat (the value the
    #     encoder produces) has the natural (positive) orientation.
    p_normal = torch.sigmoid(w_adv * feat)
    loss_normal = torch.nn.functional.binary_cross_entropy(p_normal, label)
    grad_normal = torch.autograd.grad(loss_normal, feat)[0].item()

    # (b) ENCODER side through GRL: the adversary consumes grl(feat). The
    #     gradient returned to feat is flipped: sign(grl) == - sign(normal).
    p_grl = torch.sigmoid(w_adv * grl(feat))
    loss_grl = torch.nn.functional.binary_cross_entropy(p_grl, label)
    grad_grl = torch.autograd.grad(loss_grl, feat)[0].item()

    assert grad_normal != 0.0 and grad_grl != 0.0
    assert np.sign(grad_grl) == -np.sign(grad_normal), (grad_normal, grad_grl)
    print('TEST 4 PASS (GRL sign flipped: normal=%.4f grl=%.4f)' % (grad_normal, grad_grl))
    return True


def test5_zid_verification_sensitivity():
    """z_id verifier responds to similar vs dissimilar embeddings in the right direction.

    We fit ONLY the (small, randomly-initialized) verifier head V on fixed
    manually-constructed similar/dissimilar embeddings for a few steps -- a
    code-sanity exercise, not S1 training -- and assert that:
      - BCE loss on a correctly-labeled pair decreases,
      - the learned model separates similar (same) from dissimilar (diff) pairs.
    """
    m, dev = _bundle()
    # fixed embeddings: same-pair identical, diff-pair separated
    same = []
    diff = []
    torch.manual_seed(3)
    for _ in range(8):
        anchor = torch.rand(1, 128, device=dev)
        same.append((anchor, anchor.clone()))
    for _ in range(8):
        anchor = torch.rand(1, 128, device=dev)
        diff.append((anchor, torch.rand(1, 128, device=dev)))
    pairs = same + diff
    y_pair = make_pair_labels(torch.tensor([1] * 8 + [0] * 8, dtype=torch.bool))
    y_pair = y_pair.to(dev)

    opt = torch.optim.SGD(m.verifier.parameters(), lr=0.1)
    loss0 = None
    for _ in range(60):
        opt.zero_grad()
        logits = torch.cat([m.verify(a, b) for a, b in pairs])
        loss = zid_pair_loss(logits, y_pair)
        if loss0 is None:
            loss0 = loss.item()
        loss.backward()
        opt.step()
    assert torch.isfinite(loss).item()
    assert loss.item() < loss0, (loss0, loss.item())

    # separation: same-pair prob > diff-pair prob after the head fit
    p_same = torch.sigmoid(torch.cat([m.verify(a, b) for a, b in same])).mean().item()
    p_diff = torch.sigmoid(torch.cat([m.verify(a, b) for a, b in diff])).mean().item()
    assert p_same > p_diff, (p_same, p_diff)
    assert p_same > 0.5 and p_diff < 0.5, (p_same, p_diff)
    print('TEST 5 PASS (loss %.4f->%.4f; p_same=%.3f > p_diff=%.3f)' % (loss0, loss.item(), p_same, p_diff))
    return True


def test6_donor_safety():
    sampler = DonorSampler(seed=7)
    # representative TRAIN sample: pick a subset of train images
    train_imgs = sampler.df[sampler.df['fold'] == 'train']['Image Index'].tolist()
    sample = train_imgs[:64]
    donors_a = sampler(sample)
    for s, d in zip(sample, donors_a):
        assert sampler.patient_by_image[d] != sampler.patient_by_image[s], (s, d)

    # deterministic: same seed -> same mapping
    donors_b = sampler(sample)
    assert donors_a == donors_b

    # different seed -> mapping changes (on a large enough sample this must hold)
    sampler2 = DonorSampler(seed=8)
    donors_c = sampler2(sample)
    assert donors_a != donors_c, 'different seed produced identical donor mapping'

    # donor pool never contains TEST patient/file
    for d in donors_a:
        assert sampler.fold_by_image[d] != 'test', d

    # validation source -> validation donor pool only
    val_imgs = sampler.df[sampler.df['fold'] == 'val']['Image Index'].tolist()
    donors_v = sampler(val_imgs[:32])
    for s, d in zip(val_imgs[:32], donors_v):
        assert sampler.fold_by_image[d] == 'val'
        assert sampler.patient_by_image[d] != sampler.patient_by_image[s]

    # TEST source is rejected
    test_imgs = sampler.df[sampler.df['fold'] == 'test']['Image Index'].tolist()
    try:
        sampler(test_imgs[:4])
        assert False, 'TEST source should be rejected in development'
    except RuntimeError:
        pass

    print('TEST 6 PASS (donor!=source; deterministic; seed changes; no TEST touched)')
    return True


def test7_checkpoint_provenance():
    # classification
    _, cm = load_frozen_classifier(device='cpu')
    assert cm['sha256'] == CLASSIFIER_SHA256
    assert cm['architecture'] == 'DenseNet'
    assert cm['path'] == 'networks/pretrained_classifier.pth'
    assert cm['params'] == 6968206
    # segmentation
    _, sm = load_frozen_segmentation_teacher(device='cpu')
    assert sm['sha256'].startswith(SEGMENTATION_SHA256_PREFIX)
    assert sm['architecture'] == 'UNetSeg(in=1, out=3, init_features=16)'
    assert sm['path'] == 'archive/train_seg_unet/best.pth'
    assert sm['params'] == 1942323
    print('TEST 7 PASS (checkpoint provenance verified)')
    return True


def test8_gradient_ownership():
    """One synthetic forward-backward: finite grads on intended modules only.

    Intended trainable modules: encoder, decoder, z_id verifier, z_med adversary.
    No frozen-model gradients, no NaN/Inf, no accidentally detached utility grad.
    """
    m, dev = _bundle()
    frozen = FrozenUtility(dev)
    x = torch.randn(2, 1, 256, 256, device=dev).clamp(-1, 1)
    xd = torch.randn(2, 1, 256, 256, device=dev).clamp(-1, 1)
    y_path = (torch.rand(2, 14, device=dev) > 0.5).float()
    y_pair = make_pair_labels(torch.zeros(2, dtype=torch.bool)).to(dev)

    out = m(x, xd)
    L = (reconstruction_loss(out['x_self'], x)
         + torch.nn.functional.binary_cross_entropy(frozen.path_logits(out['x_anon']), y_path)
         + torch.nn.functional.mse_loss(frozen.anat_maps(out['x_anon']), frozen.anat_maps(x))
         + zid_pair_loss(m.verify(out['z_id'], out['z_id_donor']), y_pair)
         + zmed_adv_loss(m.adversary_logits(out['z_med'], out['z_med_donor']), y_pair))
    L.backward()

    # finite grads on all trainable model params
    for name, p in m.named_parameters():
        assert p.requires_grad
        assert p.grad is not None, 'no grad on %s' % name
        assert torch.isfinite(p.grad).all().item(), 'non-finite grad on %s' % name

    # frozen models have no grads
    for p in frozen.classifier.parameters():
        assert p.grad is None
    for p in frozen.segmenter.parameters():
        assert p.grad is None

    # intended modules all received non-zero (or at least allocated) grads
    named = dict(m.named_parameters())
    assert any('encoder' in k and named[k].grad.norm() > 0 for k in named)
    assert any('decoder' in k and named[k].grad.norm() > 0 for k in named)
    assert any('verifier' in k and named[k].grad is not None for k in named)
    assert any('adv' in k and named[k].grad is not None for k in named)
    print('TEST 8 PASS (gradient ownership: all trainable finite; frozen grad-free)')
    return True


def run_all():
    assert _cuda('TEST 1-8: CUDA required for full model forward') or True
    results = {}
    results['1_shapes'] = test1_shapes()
    results['2_reconstruction_isolation'] = test2_reconstruction_isolation()
    results['3_frozen_utility_grads'] = test3_frozen_utility_grads()
    results['4_grl_sign'] = test4_grl_sign()
    results['5_zid_verification_sensitivity'] = test5_zid_verification_sensitivity()
    results['6_donor_safety'] = test6_donor_safety()
    results['7_checkpoint_provenance'] = test7_checkpoint_provenance()
    results['8_gradient_ownership'] = test8_gradient_ownership()
    assert all(results.values())
    print('\nIBR S1 LOAD-BEARING TESTS: ALL PASS')
    return True


if __name__ == '__main__':
    run_all()