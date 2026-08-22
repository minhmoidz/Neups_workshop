"""Standalone CPU-only sampler tests (P0_2_1 revision, incl. num_workers=1).

Run: CUDA_VISIBLE_DEVICES="" python test_deterministic_sampler.py
"""
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deterministic_sampler import (DeterministicEpochSampler,
                                   build_permutation,
                                   expected_epoch_order_hashes,
                                   make_paired_dataloader, order_hash)
from seed_contract import derive_seed


def _expect_raises(exc_type, fn):
    try:
        fn()
    except exc_type:
        return
    except Exception as e:  # noqa: BLE001
        raise AssertionError("wrong exception %r" % e)
    raise AssertionError("expected %s" % exc_type.__name__)


class IntDs(torch.utils.data.Dataset):
    def __len__(self):
        return 64

    def __getitem__(self, i):
        return torch.tensor(float(i)), torch.tensor(float(i % 2))


def test_identical_across_separately_constructed_samplers():
    p1 = build_permutation(42, 0, 10000)
    p2 = build_permutation(42, 0, 10000)
    assert p1 == p2 and sorted(p1) == list(range(10000))


def test_prior_global_rng_consumption_has_no_effect():
    random.seed(1); np.random.seed(2); torch.manual_seed(3)
    h_a = order_hash(build_permutation(42, 0, 5000), 42, 0, 5000)
    random.seed(9); np.random.seed(8); torch.manual_seed(7)
    _ = torch.randn(4096)
    h_b = order_hash(build_permutation(42, 0, 5000), 42, 0, 5000)
    assert h_a == h_b


def test_main_global_rng_unchanged_by_loader_construction_and_iteration():
    torch.manual_seed(31337)
    state_before = torch.get_rng_state().clone()
    build = make_paired_dataloader(IntDs(), 42, num_workers=0)
    dl = build(0)
    for _ in iter(dl):
        pass
    assert torch.equal(state_before, torch.get_rng_state())
    # also across a second epoch build
    dl1 = build(1)
    for _ in iter(dl1):
        pass
    assert torch.equal(state_before, torch.get_rng_state())


def test_real_num_workers_1_iteration_succeeds():
    build = make_paired_dataloader(IntDs(), 42, num_workers=1, batch_size=8)
    dl = build(0)
    seen = []
    for batch in iter(dl):
        seen += [int(v.item()) for v in batch[0]]
        for w in dl:
            pass  # exercise worker teardown path
    assert sorted(seen) == list(range(64))
    expected = build_permutation(42, 0, 64)
    assert seen == expected


def test_different_master_seed_changes_order():
    assert build_permutation(42, 0, 2000) != build_permutation(43, 0, 2000)


def test_different_epoch_changes_order():
    assert build_permutation(42, 0, 2000) != build_permutation(42, 1, 2000)


def test_order_hash_rejects_incomplete_duplicate_and_bad_inputs():
    good = build_permutation(42, 0, 128)
    h1 = order_hash(good, 42, 0, 128)
    assert h1 == order_hash(list(good), 42, 0, 128) and len(h1) == 64

    short = list(good)[:-1]                       # omitted index
    _expect_raises(ValueError, lambda: order_hash(short, 42, 0, 128))
    dup = list(good); dup[0] = dup[1]             # duplicate index
    _expect_raises(ValueError, lambda: order_hash(dup, 42, 0, 128))
    neg = list(good); neg[0] = -1                 # negative index
    _expect_raises(ValueError, lambda: order_hash(neg, 42, 0, 128))
    oob = list(good); neg_idx = good[0]; oob[neg_idx] = 10 ** 6  # out of range
    _expect_raises(ValueError, lambda: order_hash(oob, 42, 0, 128))
    _expect_raises(TypeError,
                   lambda: order_hash([True] + good[1:], 42, 0, 128))  # bool
    _expect_raises(TypeError,
                   lambda: order_hash([float(good[0])] + good[1:], 42, 0, 128))
    q = list(good); q[0], q[1] = q[1], q[0]
    assert order_hash(q, 42, 0, 128) != h1        # order sensitivity


def test_hash_does_not_advance_sampler_state():
    s = DeterministicEpochSampler(42, 256)
    it1, it2 = list(iter(s)), list(iter(s))
    _ = order_hash(s._perm, 42, 0, 256)
    it3 = list(iter(s))
    assert it1 == it2 == it3


def test_expected_hashes_helper_matches_direct():
    direct = {e: order_hash(build_permutation(42, e, 512), 42, e, 512)
              for e in range(3)}
    assert expected_epoch_order_hashes(42, 3, 512) == direct


def test_explicit_dataloader_generator_per_arm_identical_but_distinct():
    build_a = make_paired_dataloader(IntDs(), 42)
    build_b = make_paired_dataloader(IntDs(), 42)
    da, db = build_a(0), build_b(0)
    ga, gb = da.generator, db.generator
    assert ga is not gb                            # independent objects
    assert ga.initial_seed() == gb.initial_seed()  # identical values
    # ordering comes exclusively from the fixed deterministic sampler:
    seq = [int(v.item()) for b in iter(da) for v in b[0]]
    assert seq == build_permutation(42, 0, 64)
    assert not da.persistent_workers and not db.persistent_workers


if __name__ == "__main__":
    names = sorted(k for k in globals() if k.startswith("test_"))
    for name in names:
        globals()[name]()
        print("PASS", name)
    print("ALL PASS (%d)" % len(names))
