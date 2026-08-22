"""Standalone CPU-only generator-guard tests. Run: python test_generator_guard.py"""
import os
import sys
import types

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generator_guard import (GeneratorStateMutationError,
                             assert_generator_frozen_state,
                             canonical_model_state_hash, protected_forward)
from seed_contract import seed_everything_for_attacker_construction


def _toy_generator():
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(1, 2, 3, padding=1), nn.BatchNorm2d(2),
                         nn.Tanh())


def test_negative_control_train_mode_bn_drifts_under_no_grad():
    g = _toy_generator()
    x = torch.randn(2, 1, 8, 8)
    before = g[1].running_mean.clone()
    with torch.no_grad():
        g(x)
    assert not torch.equal(before, g[1].running_mean)


def test_protected_forward_no_drift():
    g = _toy_generator(); g.eval()
    for p in g.parameters():
        p.requires_grad_(False)
    x = torch.randn(4, 1, 8, 8)
    out = protected_forward(g, lambda t: g(t), x)
    out2 = protected_forward(g, lambda t: g(t), x)
    assert torch.equal(out, out2)


def test_train_mode_or_grads_rejected_precheck():
    g = _toy_generator(); g.train()
    for p in g.parameters():
        p.requires_grad_(False)
    try:
        assert_generator_frozen_state(g)
        raise AssertionError("train-mode generator not rejected")
    except GeneratorStateMutationError:
        pass
    g.eval()
    for p in g.parameters():
        p.requires_grad_(True)
    try:
        assert_generator_frozen_state(g)
        raise AssertionError("requires_grad generator not rejected")
    except GeneratorStateMutationError:
        pass


def test_mutation_injected_is_detected():
    g = _toy_generator(); g.eval()
    for p in g.parameters():
        p.requires_grad_(False)

    def evil(x):
        with torch.no_grad():
            g[1].running_mean.add_(0.5)
        return g(x)

    try:
        protected_forward(g, evil, torch.randn(2, 1, 8, 8))
        raise AssertionError("mutation not detected")
    except GeneratorStateMutationError:
        pass


def test_downstream_attacker_backward_succeeds_and_no_gen_grad():
    g = _toy_generator(); g.eval()
    for p in g.parameters():
        p.requires_grad_(False)
    head = nn.Linear(2 * 8 * 8, 1)
    x = torch.randn(2, 1, 8, 8)
    feats = protected_forward(g, lambda t: g(t).reshape(t.shape[0], -1), x)
    assert not feats.is_inference()
    head(feats).sum().backward()
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0.0
               for p in g.parameters())
    assert head.weight.grad is not None


class ToyAttacker(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 4, 3), nn.ReLU(), nn.Flatten(),
                                 nn.Linear(4 * 6 * 6, 8))
        self.fc_end = nn.Linear(8, 1)


def test_initial_attacker_hash_contract_toy_model():
    def make(seed):
        seed_everything_for_attacker_construction(seed)
        return ToyAttacker()

    a1, a2, b = make(42), make(42), make(43)
    h_a1 = canonical_model_state_hash(a1)
    assert h_a1 == canonical_model_state_hash(a2)
    assert h_a1 != canonical_model_state_hash(b)

    sd = a1.state_dict()
    rev = {k: sd[k] for k in reversed(list(sd.keys()))}
    wrapper = types.SimpleNamespace(state_dict=lambda keep_vars=True: rev)
    assert canonical_model_state_hash(wrapper) == h_a1


if __name__ == "__main__":
    fns = [globals()[k] for k in sorted(k for k in globals() if k.startswith("test_"))]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("ALL PASS (%d)" % len(fns))
