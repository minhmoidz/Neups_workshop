"""Standalone CPU-only seed-contract tests (P0_2_1 revision).

Run: CUDA_VISIBLE_DEVICES="" python test_seed_contract.py
Golden vectors are HARD-CODED numeric literals computed once outside the test;
they are never recalculated inside this file's assertions via the same formula.
"""
import os
import random
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seed_contract import (derive_seed, derive_epoch_order_seed,
                           build_arm_seed_bundle,
                           seed_everything_for_attacker_construction)

# ---- literal golden vectors (external computation, pinned forever) ----
GOLDEN = {
    (42, "attacker_weight_init"): 3182366824493050920,
    (42, "train_order"): 1168295852399073028,
    (42, "dataloader_worker_base"): 7923226083686500895,
    (67, "attacker_weight_init"): 4526986779586776147,
}


def _expect_raises(exc_type, fn):
    try:
        fn()
    except exc_type:
        return
    except Exception as e:  # noqa: BLE001
        raise AssertionError("wrong exception %r" % e)
    raise AssertionError("expected %s" % exc_type.__name__)


def test_golden_values_are_pinned_literals():
    assert derive_seed(42, "attacker_weight_init") == \
        GOLDEN[(42, "attacker_weight_init")]
    assert derive_seed(42, "train_order") == GOLDEN[(42, "train_order")]
    assert derive_seed(42, "dataloader_worker_base") == \
        GOLDEN[(42, "dataloader_worker_base")]
    assert derive_seed(67, "attacker_weight_init") == \
        GOLDEN[(67, "attacker_weight_init")]
    for v in GOLDEN.values():
        assert isinstance(v, int) and 0 <= v <= 2 ** 63 - 1


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


def test_real_arm_bundle_construction():
    bundle_a = build_arm_seed_bundle(42)
    bundle_b = build_arm_seed_bundle(42)
    assert bundle_a == bundle_b                    # equal values across arms
    assert bundle_a is not bundle_b                # distinct objects
    assert bundle_a["attacker_weight_init"] == GOLDEN[(42, "attacker_weight_init")]
    doms = [k for k in bundle_a if k != "statistical_sensitivity_master"]
    vals = [bundle_a[k] for k in doms]
    assert len(set(vals)) == len(vals)             # domains distinct within arm
    other = build_arm_seed_bundle(43)
    assert bundle_a["train_order"] != other["train_order"]


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
    _expect_raises(ValueError, lambda: derive_epoch_order_seed(42, -1, 10))
    _expect_raises(TypeError, lambda: derive_epoch_order_seed(True, 0, 10))
    _expect_raises(TypeError, lambda: derive_epoch_order_seed(42, 0, True))


def test_seed_everything_no_cuda():
    seed_everything_for_attacker_construction(
        derive_seed(42, "attacker_weight_init"))
    assert not torch.cuda.is_initialized()


if __name__ == "__main__":
    names = sorted(k for k in globals() if k.startswith("test_"))
    for name in names:
        globals()[name]()
        print("PASS", name)
    print("ALL PASS (%d)" % len(names))
