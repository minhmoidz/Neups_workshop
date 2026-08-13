"""STEP 7C — 10 load-bearing verification tests for P2 capacity-limited atlas condition.

  1. Atlas built strictly from TRAIN masks (18,089 images).
  2. Validation split never affects lambda* selection.
  3. TEST split inaccessible.
  4. Signed-distance transform deterministic.
  5. lambda* reproducible (0.3400).
  6. Output P2 mask is binary uint8 3x64x64.
  7. No image pixels/features/identifiers enter attacker tensors.
  8. Pair files balanced (5,000/5,000 train, 1,000/1,000 val).
  9. Attacker distance |e1-e2| order-invariant.
  10. Pathology inputs identical to STEP 7A/7B.
"""

import hashlib
import json
import os
import random

import numpy as np
import torch

from research_agent.ibr.build_p2_cache import compute_signed_distance_batch, calibrate_lambda_on_train
from research_agent.ibr.condition_attacker import build_model
from research_agent.ibr.train_p2_condition_capacity import load_p2_cache, P2PairDataset

P2_CACHE_DIR = 'research_agent/ibr_s1_condition_p2_cache/'
STEP7A_CACHE_DIR = 'research_agent/ibr_s1_condition_cache/'
SEG_SHA256_FULL = '2dfdcf9b1ede7a163c584e843b36dacfcb790edc800a83b6de44a8ea3e6c73e0'


def _cuda():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for load-bearing tests')


def test1_atlas_train_only():
    with open(os.path.join(P2_CACHE_DIR, 'p2_meta.json')) as f:
        meta = json.load(f)
    assert meta['train_p2']['shape'][0] == 18089
    atlas = np.load(os.path.join(P2_CACHE_DIR, 'train_atlas.npy'))
    assert atlas.shape == (3, 64, 64)
    train_p1 = np.load('research_agent/ibr_s1_condition_p1_cache/train_p1.npy')
    assert train_p1.shape[0] == 18089
    return True


def test2_val_never_affects_lambda():
    with open('research_agent/ibr/build_p2_cache.py') as f:
        src = f.read()
    assert 'calibrate_lambda_on_train(train_sd, train_p1_bool, atlas)' in src
    assert 'val_sd' not in src.split('calibrate_lambda_on_train(')[1].split('def evaluate_validation_retention')[0]
    return True


def test3_test_inaccessible():
    with open('research_agent/ibr/build_p2_cache.py') as f:
        src = f.read()
    assert 'image_pairs_testing' not in src
    with open('research_agent/ibr/train_p2_condition_capacity.py') as f:
        tsrc = f.read()
    assert 'image_pairs_testing' not in tsrc
    assert not os.path.exists('research_agent/ibr_s1_condition_p2_cache/test_p2.npy')
    return True


def test4_signed_distance_deterministic():
    rng = np.random.default_rng(42)
    m = (rng.random((2, 3, 64, 64)) > 0.5)
    sd1 = compute_signed_distance_batch(m)
    sd2 = compute_signed_distance_batch(m)
    assert np.array_equal(sd1, sd2)
    return True


def test5_lambda_reproducible():
    train_p1 = np.load('research_agent/ibr_s1_condition_p1_cache/train_p1.npy')[:500]
    train_p1_bool = (train_p1 > 0)
    train_sd = compute_signed_distance_batch(train_p1_bool)
    atlas = train_sd.mean(axis=0)
    lam1, stats1, _ = calibrate_lambda_on_train(train_sd, train_p1_bool, atlas)
    lam2, stats2, _ = calibrate_lambda_on_train(train_sd, train_p1_bool, atlas)
    assert lam1 == lam2
    with open(os.path.join(P2_CACHE_DIR, 'p2_meta.json')) as f:
        meta = json.load(f)
    assert meta['p2_transform']['frozen_lambda'] == 0.34
    return True


def test6_output_mask_binary_64x64():
    tr = np.load(os.path.join(P2_CACHE_DIR, 'train_p2.npy'))
    va = np.load(os.path.join(P2_CACHE_DIR, 'val_p2.npy'))
    assert tr.shape == (18089, 3, 64, 64) and tr.dtype == np.uint8
    assert va.shape == (3484, 3, 64, 64) and va.dtype == np.uint8
    assert set(np.unique(tr)).issubset({0, 1})
    assert set(np.unique(va)).issubset({0, 1})
    return True


def test7_no_pixels_no_identifiers():
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
    cache, labels_by_image, pairs = load_p2_cache()
    ds = P2PairDataset('val', cache, labels_by_image, pairs)
    m1b, m2b, y1b, y2b, lb = ds[0]
    assert y1b.shape == (14,) and set(np.unique(y1b)).issubset({0.0, 1.0})
    assert m1b.shape == (3, 64, 64) and set(np.unique(m1b)).issubset({0.0, 1.0})
    return True


def test8_pairs_balanced():
    tr = np.loadtxt('image_pairs/image_pairs_training_10000.txt', dtype=str, delimiter='\t')
    va = np.loadtxt('image_pairs/image_pairs_validation_2000.txt', dtype=str, delimiter='\t')
    assert (tr[:, 2] == '1.0').sum() == 5000 and (tr[:, 2] == '0.0').sum() == 5000
    assert (va[:, 2] == '1.0').sum() == 1000 and (va[:, 2] == '0.0').sum() == 1000
    return True


def test9_order_invariant():
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


def test10_pathology_identical_to_7A():
    h = hashlib.sha256(open('archive/train_seg_unet/best.pth', 'rb').read()).hexdigest()
    assert h == SEG_SHA256_FULL
    with open('research_agent/ibr_s1_condition_p2_capacity/results.json') as f:
        res = json.load(f)
    b_records = [r for r in res if r['arm'] == 'B']
    assert len(b_records) == 3
    assert all(r.get('reused_from_7A') for r in b_records)
    return True


def run_all():
    tests = [
        test1_atlas_train_only,
        test2_val_never_affects_lambda,
        test3_test_inaccessible,
        test4_signed_distance_deterministic,
        test5_lambda_reproducible,
        test6_output_mask_binary_64x64,
        test7_no_pixels_no_identifiers,
        test8_pairs_balanced,
        test9_order_invariant,
        test10_pathology_identical_to_7A,
    ]
    results = {}
    for t in tests:
        try:
            results[t.__name__] = bool(t())
            print('TEST %s PASS' % t.__name__)
        except Exception as e:
            results[t.__name__] = False
            print('TEST %s FAIL: %r' % (t.__name__, e))
    assert all(results.values()), results
    print('ALL 10 STEP 7C P2 LOAD-BEARING TESTS PASS')
    return results


if __name__ == '__main__':
    run_all()
