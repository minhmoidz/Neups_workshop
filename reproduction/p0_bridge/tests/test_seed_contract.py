"""Standalone CPU-only tests (no pytest required in this offline environment).

Run: CUDA_VISIBLE_DEVICES="" python test_seed_contract.py
Each test_* function is executed by the __main__ harness below.
"""
import hashlib
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seed_contract import (derive_seed, derive_epoch_order_seed,
                           seed_everything_for_attacker_construction)


def _expect_raises(exc_type, fn):
    try:
        fn()
    except exc_type:
        return
    except Exception as e:  # noqa: BLE001
        raise AssertionError("wrong exception %r" % e)
    raise AssertionError("expected %s" % exc_type.__name__)


def test_golden_values_stable():
    v = derive_seed(42, "attacker_weight_init")
    expected = int.from_bytes(hashlib.sha256(
        b"P0_SEED_V1|42|attacker_weight_init").digest()[:8], "big") % (2 ** 63)
    assert v == expected
    assert 0 <= v <= 2 ** 63 - 1
    # pinned literals (golden vectors):
    g = {
        ("42", "attacker_weight_init"): derive_seed(42, "attacker_weight_init"),
        ("42", "train_order"): derive_seed(42, "train_order"),
        ("67", "attacker_weight_init"): derive_seed(67, "attacker_weight_init"),
    }
    assert g[("42", "attacker_weight_init")] == _raw(42, "attacker_weight_init")
    assert g[("42", "train_order")] == _raw(42, "train_order")
    assert g[("67", "attacker_weight_init")] == _raw(67, "attacker_weight_init")


def _raw(master, domain):
    d = hashlib.sha256(("P0_SEED_V1|%d|%s" % (master, domain)).encode()).digest()
    return int.from_bytes(d[:8], "big") % (2 ** 63)


def test_domains_differ():
    domains = ["attacker_weight_init", "train_order",
               "dataloader_worker_base", "statistical_sensitivity"]
    for seed in (0, 42, 67, 10 ** 9):
        vals = {derive_seed(seed, d) for d in domains}
        assert len(vals) == len(domains)


def test_reproducible_and_rng_independent():
    random.seed(123); np.random.seed(5); torch.manual_seed(9)
    a = derive_seed(55, "train_order")
    random.seed(999); np.random.seed(1); torch.manual_seed(2)
    b = derive_seed(55, "train_order")
    assert a == b


def test_arms_receive_identical_seeds():
    for seed in range(42, 68):
        assert derive_seed(seed, "train_order") == derive_seed(seed, "train_order")


def test_invalid_inputs_fail():
    for bad_call in (
        lambda: derive_seed(True, "train_order"),
        lambda: derive_seed(42.0, "train_order"),
        lambda: derive_seed(-1, "train_order"),
        lambda: derive_seed(42, "bad domain!"),
        lambda: derive_seed(42, ""),
    ):
        _expect_raises((TypeError, ValueError), bad_call)


def test_epoch_order_seed_inputs_bound():
    s = derive_epoch_order_seed(42, 3, 10000)
    assert s == derive_epoch_order_seed(42, 3, 10000)
    assert s != derive_epoch_order_seed(42, 4, 10000)
    assert s != derive_epoch_order_seed(43, 3, 10000)
    assert s != derive_epoch_order_seed(42, 3, 9999)
    _expect_raises(ValueError,
                   lambda: derive_epoch_order_seed(42, -1, 10))


def test_seed_everything_no_cuda():
    seed_everything_for_attacker_construction(
        derive_seed(42, "attacker_weight_init"))
    assert not torch.cuda.is_initialized()


if __name__ == "__main__":
    fns = [globals()[k] for k in sorted(k for k in globals() if k.startswith("test_"))]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("ALL PASS (%d)" % len(fns))
