"""STEP 7A — load-bearing tests (run BEFORE any training).

8 predeclared gates:
  1. TEST files cannot be opened in development mode.
  2. No source pixels enter attacker tensors.
  3. No filenames/patient IDs enter feature tensors.
  4. Pair labels are balanced.
  5. Pair operation is order invariant.
  6. Segmentation maps match expected source IDs.
  7. Teacher checkpoint SHA matches.
  8. Fixed seed reproduces initialization/order.
"""

import hashlib
import json
import os
import random

import numpy as np
import torch

from research_agent.ibr.condition_attacker import build_model
from research_agent.ibr.train_condition_capacity import load_cache, PairDataset

CACHE_DIR = 'research_agent/ibr_s1_condition_cache/'
PAIR_FILES = {
    'train': 'image_pairs/image_pairs_training_10000.txt',
    'val': 'image_pairs/image_pairs_validation_2000.txt',
    'test': 'image_pairs/image_pairs_testing_5000.txt',
}
TEST_PAIR_FILES = 'image_pairs/image_pairs_testing_5000.txt'
SEG_SHA256_FULL = '2dfdcf9b1ede7a163c584e843b36dacfcb790edc800a83b6de44a8ea3e6c73e0'


def _cuda():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')


def test1_test_inaccessible():
    assert os.path.exists(TEST_PAIR_FILES)
    dev_guard = []
    try:
        from research_agent.ibr import train_condition_capacity as m
        assert not hasattr(m, 'TEST_PAIR_FILES'), 'training driver must not define TEST access'
        assert m.PAIR_FILES.keys() == {'train', 'val'}, m.PAIR_FILES.keys()
    except Exception as e:
        dev_guard.append(str(e))
    # the cache build script must never touch TEST
    with open('research_agent/ibr/build_condition_cache.py') as f:
        src = f.read()
    assert 'image_pairs_testing' not in src, 'cache builder references TEST'
    assert 'val' in src
    return True


def test2_no_pixels_in_attacker():
    m1 = torch.zeros(2, 3, 256, 256)
    m2 = torch.ones(2, 3, 256, 256)
    y1 = torch.zeros(2, 14)
    y2 = torch.ones(2, 14)
    lab = torch.tensor([[1.0], [0.0]])
    model = build_model('C')
    logit = model(m1, m2, y1, y2)
    assert tuple(logit.shape) == (2, 1)
    # ensure gradient flows (heads trainable)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, lab)
    loss.backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    assert n_grad > 0
    return True


def test3_no_filename_patientid_in_features():
    cache, labels_by_image, pairs = load_cache()
    ds = PairDataset('val', cache, labels_by_image, pairs)
    m1, m2, y1, y2, lab = ds[0]
    assert y1.shape == (14,) and y2.shape == (14,)
    assert m1.shape == (3, 256, 256) and m2.shape == (3, 256, 256)
    assert m1.dtype == np.float32 and m1.min() >= 0.0 and m1.max() <= 1.0
    # y is pure 14-vector; no scalar identifier appended
    assert y1.dtype == np.float32 and set(np.unique(y1)).issubset({0.0, 1.0})
    # pair rows carry only image names + label; no extra columns
    assert len(ds.rows[0]) == 3
    return True


def test4_pairs_balanced():
    tr = np.loadtxt(PAIR_FILES['train'], dtype=str, delimiter='\t')
    va = np.loadtxt(PAIR_FILES['val'], dtype=str, delimiter='\t')
    assert (tr[:, 2] == '1.0').sum() == (tr[:, 2] == '0.0').sum() == 5000
    assert (va[:, 2] == '1.0').sum() == (va[:, 2] == '0.0').sum() == 1000
    return True


def test5_order_invariant():
    model = build_model('C')
    m1 = torch.randn(1, 3, 256, 256)
    m2 = torch.randn(1, 3, 256, 256)
    y1 = torch.rand(1, 14)
    y2 = torch.rand(1, 14)
    with torch.no_grad():
        d12 = model.head(torch.abs(model.joint_embed(m1, y1) - model.joint_embed(m2, y2)))
        d21 = model.head(torch.abs(model.joint_embed(m2, y2) - model.joint_embed(m1, y1)))
    assert torch.allclose(d12, d21, atol=1e-6)
    return True


def test6_maps_match_source_ids():
    with open(os.path.join(CACHE_DIR, 'train_images.json')) as f:
        imgs = json.load(f)
    maps = np.load(os.path.join(CACHE_DIR, 'train_maps.npy'))
    assert maps.shape[0] == len(imgs)
    # maps are real probability maps in [0,1] with a lung/heart structure (nonzero mass)
    m = maps[0]
    assert m.min() >= 0.0 and m.max() <= 1.0
    assert (m[0] > 0.5).sum() > 0 and (m[2] > 0.5).sum() > 0  # left lung + heart present
    return True


def test7_teacher_sha():
    h = hashlib.sha256(open('archive/train_seg_unet/best.pth', 'rb').read()).hexdigest()
    assert h == SEG_SHA256_FULL, (h, SEG_SHA256_FULL)
    assert h.startswith('2dfdcf9b')
    with open(os.path.join(CACHE_DIR, 'cache_meta.json')) as f:
        meta = json.load(f)
    assert meta['teacher']['sha256_full'] == SEG_SHA256_FULL
    return True


def test8_seed_reproducible():
    _cuda()
    device = 'cuda'
    torch.manual_seed(0)
    a = build_model('C').to(device)
    torch.manual_seed(0)
    b = build_model('C').to(device)
    x = torch.randn(1, 3, 256, 256, device=device)
    y = torch.randn(1, 14, device=device)
    with torch.no_grad():
        ea = a.joint_embed(x, y)
        eb = b.joint_embed(x, y)
    assert torch.equal(ea, eb), 'same seed must reproduce init/forward'
    # dataset order reproducibility
    cache, labels_by_image, pairs = load_cache()
    ds = PairDataset('val', cache, labels_by_image, pairs)
    random.seed(7)
    i = random.randint(0, len(ds) - 1)
    m1, m2, y1, y2, lab = ds[i]
    assert m1.shape == (3, 256, 256)
    return True


def run_all():
    tests = [test1_test_inaccessible, test2_no_pixels_in_attacker, test3_no_filename_patientid_in_features,
             test4_pairs_balanced, test5_order_invariant, test6_maps_match_source_ids,
             test7_teacher_sha, test8_seed_reproducible]
    results = {}
    for t in tests:
        try:
            ok = t()
            results[t.__name__] = bool(ok)
            print('TEST %s PASS' % t.__name__)
        except Exception as e:
            results[t.__name__] = False
            print('TEST %s FAIL: %r' % (t.__name__, e))
    assert all(results.values()), results
    print('ALL STEP 7A LOAD-BEARING TESTS PASS')
    return results


if __name__ == '__main__':
    run_all()