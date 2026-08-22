"""Standalone CPU-only generator-guard tests (P0_2_1 revision).

Run: CUDA_VISIBLE_DEVICES="" python test_generator_guard.py
"""
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


def test_nested_dropout_train_rejected_even_if_top_level_eval():
    g = _toy_generator()
    g.eval()
    drop = nn.Dropout(p=0.5)
    g.add_module("nested_drop", drop)      # nested submodule left in train mode
    for p in g.parameters():
        p.requires_grad_(False)
    try:
        protected_forward(g, lambda gen, t: gen(t), torch.randn(2, 1, 8, 8))
        raise AssertionError("nested train-mode Dropout not rejected")
    except GeneratorStateMutationError:
        pass


def test_nested_batchnorm_train_rejected():
    g = _toy_generator(); g.eval()
    for p in g.parameters():
        p.requires_grad_(False)
    g[1].training = True                    # top-level eval, nested BN train
    try:
        protected_forward(g, lambda gen, t: gen(t), torch.randn(2, 1, 8, 8))
        raise AssertionError("nested train-mode BatchNorm not rejected")
    except GeneratorStateMutationError:
        pass


def test_protected_forward_no_drift_and_callable_binding():
    g = _toy_generator(); g.eval()
    for p in g.parameters():
        p.requires_grad_(False)
    x = torch.randn(4, 1, 8, 8)
    seen = {}
    out = protected_forward(g, lambda gen, t: (seen.__setitem__("ok", gen is g),
                                               gen(t))[1], x)
    assert seen["ok"]                       # callable received THE generator
    out2 = protected_forward(g, lambda gen, t: gen(t), x)
    assert torch.equal(out, out2)


def test_added_and_removed_buffers_detected():
    class BufNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 3, padding=1)

        def forward(self, x):
            return self.conv(x)

    g = BufNet(); g.eval()
    for p in g.parameters():
        p.requires_grad_(False)

    def add_buffer(gen, t):
        gen.register_buffer("sneaky", torch.zeros(1))
        return gen(t)

    try:
        protected_forward(g, add_buffer, torch.randn(1, 1, 8, 8))
        raise AssertionError("added buffer not detected")
    except GeneratorStateMutationError:
        pass

    g2 = nn.Sequential(nn.Conv2d(1, 1, 3, padding=1), nn.BatchNorm2d(1))
    g2.eval()
    for p in g2.parameters():
        p.requires_grad_(False)

    # removed buffer: delete a registered buffer inside the callable;
    # do NOT run forward (BN would crash before the guard could report)
    def really_remove(gen, t):
        del gen[1]._buffers["running_mean"]
        return t

    try:
        protected_forward(g2, really_remove, torch.randn(1, 1, 8, 8))
        raise AssertionError("removed buffer not detected")
    except GeneratorStateMutationError:
        pass


def test_mutation_injected_is_detected():
    g = _toy_generator(); g.eval()
    for p in g.parameters():
        p.requires_grad_(False)

    def evil(gen, t):
        with torch.no_grad():
            gen[1].running_mean.add_(0.5)
        return gen(t)

    try:
        protected_forward(g, evil, torch.randn(2, 1, 8, 8))
        raise AssertionError("mutation not detected")
    except GeneratorStateMutationError:
        pass


def test_nested_inference_tensor_output_rejected():
    g = _toy_generator(); g.eval()
    for p in g.parameters():
        p.requires_grad_(False)

    def returns_inference_tuple(gen, t):
        with torch.inference_mode():
            inner = gen(t)
        return {"a": [inner], "b": torch.ones(1)}

    try:
        protected_forward(g, returns_inference_tuple,
                          torch.randn(2, 1, 8, 8))
        raise AssertionError("nested inference tensor not rejected")
    except GeneratorStateMutationError:
        pass


def test_downstream_attacker_backward_succeeds_and_no_gen_grad():
    g = _toy_generator(); g.eval()
    for p in g.parameters():
        p.requires_grad_(False)
    head = nn.Linear(2 * 8 * 8, 1)
    x = torch.randn(2, 1, 8, 8)
    feats = protected_forward(
        g, lambda gen, t: gen(t).reshape(t.shape[0], -1), x)
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
    names = sorted(k for k in globals() if k.startswith("test_"))
    for name in names:
        globals()[name]()
        print("PASS", name)
    print("ALL PASS (%d)" % len(names))
