"""Standalone CPU-only sampler tests. Run: python test_deterministic_sampler.py"""
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deterministic_sampler import (DeterministicEpochSampler,
                                   build_permutation, expected_epoch_order_hashes,
                                   make_paired_dataloader, order_hash)
from seed_contract import derive_seed


def test_identical_across_separately_constructed_samplers():
    p1 = build_permutation(42, 0, 10000)
    p2 = build_permutation(42, 0, 10000)
    assert p1 == p2
    assert sorted(p1) == list(range(10000))


def test_prior_global_rng_consumption_has_no_effect():
    random.seed(1); np.random.seed(2); torch.manual_seed(3)
    h_a = order_hash(build_permutation(42, 0, 5000), 42, 0, 5000)
    random.seed(9); np.random.seed(8); torch.manual_seed(7)
    _ = torch.randn(4096)
    h_b = order_hash(build_permutation(42, 0, 5000), 42, 0, 5000)
    assert h_a == h_b


def test_different_master_seed_changes_order():
    assert build_permutation(42, 0, 2000) != build_permutation(43, 0, 2000)


def test_different_epoch_changes_order():
    assert build_permutation(42, 0, 2000) != build_permutation(42, 1, 2000)


def test_order_hash_stable_and_sensitive():
    p = build_permutation(42, 0, 1000)
    h1 = order_hash(p, 42, 0, 1000)
    assert h1 == order_hash(p, 42, 0, 1000) and len(h1) == 64
    q = list(p); q[0], q[1] = q[1], q[0]
    assert order_hash(q, 42, 0, 1000) != h1
    # NOTE: per P0_ORDERHASH_V1 spec the hash covers schema+length+epoch+sequence,
    # NOT the master seed; cross-arm comparability is the point.
    assert order_hash(p, 42, 1, 1000) != h1


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


def test_worker_seeds_deterministic():
    base = derive_seed(42, "dataloader_worker_base")
    w1 = derive_seed(base, "dataloader_worker_0")
    w2 = derive_seed(base, "dataloader_worker_0")
    assert w1 == w2
    assert w1 != derive_seed(derive_seed(43, "dataloader_worker_base"),
                             "dataloader_worker_0")


class IntDs(torch.utils.data.Dataset):
    def __len__(self):
        return 64

    def __getitem__(self, i):
        return torch.tensor(float(i)), torch.tensor(float(i % 2))


def test_paired_dataloaders_same_order_no_global_shuffle():
    build_a = make_paired_dataloader(IntDs(), 42, derive_seed(42, "x"))
    build_b = make_paired_dataloader(IntDs(), 42, derive_seed(42, "x"))
    flat_a, flat_b = [], []
    for b in iter(build_a(0)):
        flat_a += [int(v.item()) for v in b[0]]
    for b in iter(build_b(0)):
        flat_b += [int(v.item()) for v in b[0]]
    assert flat_a == flat_b == build_permutation(42, 0, 64)


if __name__ == "__main__":
    fns = [globals()[k] for k in sorted(k for k in globals() if k.startswith("test_"))]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("ALL PASS (%d)" % len(fns))
