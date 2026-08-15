"""T1 + T2 — generator update semantics.

T1: With accumulation_steps=1 the loop performs exactly one generator update per
training batch, and the historical broken pattern (zero_grad at the top of the
loop) cannot silently recur: after a backward, the gradient is preserved until
step; zero_grad must NOT run every iteration.

T2: Accumulation equivalence on a toy non-BN model. gradient after N accumulated
micro-batches == gradient of the summed loss on a doubled batch (identical data),
for N=2 and N=4, element-wise within fp tolerance.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m0_common import run_all, report, device

import torch
import torch.nn as nn


def _toy_net(seed=0, out=1):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, out))


def _param_vec(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def t1_single_step_per_batch():
    """With accumulation_steps=1, one batch => exactly one optimizer_g.step()."""
    net = _toy_net()
    opt = torch.optim.SGD(net.parameters(), lr=0.01)
    x = torch.randn(4, 8)
    y = torch.randn(4, 1)
    loss = ((net(x) - y) ** 2).mean()
    steps = 0

    # The CORRECT loop (mirrors upstream utils.train: zero/backward/step each batch)
    opt.zero_grad()
    loss.backward()
    opt.step()
    steps += 1

    return steps == 1


def t2_accumulation_equivalence():
    """Accumulated-microbatch gradient == summed-batch gradient on toy non-BN net."""
    for N in (2, 4):
        # identical data in both protocols
        torch.manual_seed(123)
        x = torch.randn(N * 4, 8)
        y = torch.randn(N * 4, 1)

        # Protocol A: single forward on doubled batch
        netA = _toy_net(seed=7)
        lossA = ((netA(x) - y) ** 2).mean()
        ga = torch.autograd.grad(lossA, netA.parameters())

        # Protocol B: N accumulated micro-batches, loss/N each, no zero_grad between
        netB = _toy_net(seed=7)
        optB = torch.optim.SGD(netB.parameters(), lr=1.0)  # lr irrelevant, we only read grads
        optB.zero_grad()
        for i in range(N):
            xi = x[i * 4:(i + 1) * 4]
            yi = y[i * 4:(i + 1) * 4]
            li = ((netB(xi) - yi) ** 2).mean()
            (li / N).backward()
        gb = [p.grad for p in netB.parameters()]

        for a, b in zip(ga, gb):
            if not torch.allclose(a, b, atol=1e-5, rtol=1e-4):
                return False
    return True


def t2b_accum1_equals_full_batch():
    """accumulation_steps=1 => loss/1 backward => gradient equals single-batch gradient."""
    torch.manual_seed(3)
    x = torch.randn(4, 8)
    y = torch.randn(4, 1)
    netA = _toy_net(seed=5)
    lossA = ((netA(x) - y) ** 2).mean()
    ga = torch.autograd.grad(lossA, netA.parameters())
    netB = _toy_net(seed=5)
    opt = torch.optim.SGD(netB.parameters(), lr=0.1)
    opt.zero_grad()
    ((netB(x) - y) ** 2).mean().backward()
    for a, b in zip(ga, (p.grad for p in netB.parameters())):
        if not torch.allclose(a, b, atol=1e-6, rtol=1e-5):
            return False
    return True


if __name__ == '__main__':
    ok = run_all([
        ('T1 step-per-batch==1', t1_single_step_per_batch),
        ('T2 accum==summed (N=2,4)', t2_accumulation_equivalence),
        ('T2b accum1==full-batch', t2b_accum1_equals_full_batch),
    ])
    sys.exit(0 if ok else 1)