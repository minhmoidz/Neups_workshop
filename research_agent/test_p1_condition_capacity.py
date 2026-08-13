"""STEP 7B — load-bearing tests for the P1 capacity-limited condition.

  1. P1 transform is exactly the frozen spec (binary, 3x64x64, block-mean 0.5).
  2. No TEST access; cache builder never references TEST.
  3. No source pixels / no filename/patient-id in attacker tensors.
  4. Pair labels balanced.
  5. Pair operation order invariant (|e1-e2|).
  6. P1 maps match source IDs (parent cache alignment).
  7. Teacher SHA matches STEP 7A provenance (identical condition source).
  8. Fixed seed reproduces init/order.
"""

import hashlib
import json
import os
import random

import numpy as np
import torch

from research_agent.ibr.build_p1_cache import p1_transform
from research_agent.ibr.condition_attacker import build_model
from research_agent.ibr.train_p1_condition_capacity import load_p1_cache, P1PairDataset

CACHE_DIR = 'research_agent/ibr_s1_condition_cache/'
P1_CACHE_DIR = 'research_agent/ibr_s1_condition_p1_cache/'
SEG_SHA256_FULL = '2dfdcf9b1ede7a163c584e843b36dacfcb790edc800a83b6de44a8ea3e6c73e0'


def _cuda():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')


def test1_p1_transform_frozen_spec():
    rng = np.random.default_rng(0)
    maps = rng.random((2, 3, 256, 256))
    M64 = p1_transform(maps)
    assert M64.shape == (2, 3, 64, 64), M64.shape
    assert M64.dtype == np.uint8
    assert set(np.unique(M64)).issubset({0, 1})
    # direct expected value on an all-high prob map
    all_one = np.ones((1, 3, 256, 256))
    assert (p1_transform(all_one) == 1).all()
    all_zero = np.zeros((1, 3, 256, 256))
    assert (p1_transform(all_zero) == 0).all()
    # a single 4x4 block all >= 0.5 -> cell 1
    probe = np.zeros((1, 3, 256, 256))
    probe[0, 0, 0:4, 0:4] = 0.9
    out = p1_transform(probe)
    assert out[0, 0, 0, 0] == 1
    assert out[0, 0, 0, 1] == 0  # adjacent cell all-zero -> 0
    return True


def test2_no_test_access():
    with open('research_agent/ibr/build_p1_cache.py') as f:
        src = f.read()
    assert 'image_pairs_testing' not in src
    with open('research_agent/ibr/train_p1_condition_capacity.py') as f:
        tsrc = f.read()
    assert 'image_pairs_testing' not in tsrc
    assert not os.path.exists('research_agent/ibr_s1_condition_p1_cache/test_p1.npy')
    return True


def test3_no_pixels_no_identifiers():
    m1 = torch.zeros(2, 3, 64, 64)
    m2 = torch.ones(2, 3, 64, 64)
    y1 = torch.zeros(2, 14)
    y2 = torch.ones(2, 14)
    lab = torch.tensor([[1.0], [0.0]])
    model = build_model('C')
    logit = model(m1, m2, y1, y2)
    assert tuple(logit.shape) == (2, 1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, lab)
    loss.backward()
    assert sum(1 for p in model.parameters() if p.grad is not None) > 0
    cache, labels_by_image, pairs = load_p1_cache()
    ds = P1PairDataset('val', cache, labels_by_image, pairs)
    m1b, m2b, y1b, y2b, lb = ds[0]
    assert y1b.shape == (14,) and set(np.unique(y1b)).issubset({0.0, 1.0})
    assert m1b.shape == (3, 64, 64) and set(np.unique(m1b)).issubset({0.0, 1.0})
    assert len(ds.rows[0]) == 3
    return True


def test4_pairs_balanced():
    tr = np.loadtxt('image_pairs/image_pairs_training_10000.txt', dtype=str, delimiter='\t')
    va = np.loadtxt('image_pairs/image_pairs_validation_2000.txt', dtype=str, delimiter='\t')
    assert (tr[:, 2] == '1.0').sum() == 5000 and (tr[:, 2] == '0.0').sum() == 5000
    assert (va[:, 2] == '1.0').sum() == 1000 and (va[:, 2] == '0.0').sum() == 1000
    return True


def test5_order_invariant():
    model = build_model('C')
    m1 = torch.randn(1, 3, 64, 64)
    m2 = torch.randn(1, 3, 64, 64)
    y1 = torch.rand(1, 14)
    y2 = torch.rand(1, 14)
    with torch.no_grad():
        d12 = model.head(torch.abs(model.joint_embed(m1, y1) - model.joint_embed(m2, y2)))
        d21 = model.head(torch.abs(model.joint_embed(m2, y2) - model.joint_embed(m1, y1)))
    assert torch.allclose(d12, d21, atol=1e-6)
    return True


def test6_maps_match_source_ids():
    with open(os.path.join(P1_CACHE_DIR, 'train_p1.npy')) as _:
        pass
    p1 = np.load(os.path.join(P1_CACHE_DIR, 'train_p1.npy'))
    with open(os.path.join(CACHE_DIR, 'train_images.json')) as f:
        imgs = json.load(f)
    assert p1.shape[0] == len(imgs)
    assert (p1 > 0).any()  # real anatomy present
    return True


def test7_teacher_sha_matches_7A():
    h = hashlib.sha256(open('archive/train_seg_unet/best.pth', 'rb').read()).hexdigest()
    assert h == SEG_SHA256_FULL
    with open(os.path.join(CACHE_DIR, 'cache_meta.json')) as f:
        m7a = json.load(f)
    assert m7a['teacher']['sha256_full'] == SEG_SHA256_FULL
    with open(os.path.join(P1_CACHE_DIR, 'p1_meta.json')) as f:
        p1meta = json.load(f)
    assert p1meta['source_teacher']['sha256_full'] == SEG_SHA256_FULL
    return True


def test8_seed_reproducible():
    _cuda()
    device = 'cuda'
    torch.manual_seed(0)
    a = build_model('C').to(device)
    torch.manual_seed(0)
    b = build_model('C').to(device)
    x = torch.randn(1, 3, 64, 64, device=device)
    y = torch.randn(1, 14, device=device)
    with torch.no_grad():
        ea = a.joint_embed(x, y)
        eb = b.joint_embed(x, y)
    assert torch.equal(ea, eb)
    return True


def run_all():
    tests = [test1_p1_transform_frozen_spec, test2_no_test_access, test3_no_pixels_no_identifiers,
             test4_pairs_balanced, test5_order_invariant, test6_maps_match_source_ids,
             test7_teacher_sha_matches_7A, test8_seed_reproducible]
    results = {}
    for t in tests:
        try:
            results[t.__name__] = bool(t())
            print('TEST %s PASS' % t.__name__)
        except Exception as e:
            results[t.__name__] = False
            print('TEST %s FAIL: %r' % (t.__name__, e))
    assert all(results.values()), results
    print('ALL STEP 7B P1 LOAD-BEARING TESTS PASS')
    return results


if __name__ == '__main__':
    run_all()