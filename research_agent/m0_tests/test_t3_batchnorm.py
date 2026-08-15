"""T3 — batch16 + accumulation4 is NOT numerically equivalent to batch64 for a
BatchNorm network.

Upstream anonymization config uses batch_size=64; historical modified configs use
batch_size=16 (sometimes with accumulation). Because the U-Net contains BatchNorm,
statistics are microbatch-dependent, so a single optimizer step after 4 accumulated
16-batches does NOT equal one step on a single 64-batch.

This test demonstrates the fact with a tiny conv+BN toy. It does NOT "fix"
BatchNorm and does NOT replace it with GroupNorm.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m0_common import run_all, device

import torch
import torch.nn as nn


class ToyConvBN(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.block = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 1, 3, padding=1),
            nn.BatchNorm2d(2, affine=False),  # placeholder width mismatch avoided below
        )
        self.head = nn.Linear(4, 1)

    def forward(self, x):
        x = self.block[0](x)
        x = self.block[1](x)  # BatchNorm2d(4)
        x = self.block[2](x)
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        return self.head(x.view(x.size(0), -1))


def _step_batch64():
    torch.manual_seed(0)
    net = ToyConvBN()
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    x = torch.randn(64, 1, 16, 16)
    y = torch.randn(64, 1)
    opt.zero_grad()
    loss = ((net(x) - y) ** 2).mean()
    loss.backward()
    opt.step()
    return [p.detach().clone() for p in net.parameters()]


def _step_batch16_accum4():
    torch.manual_seed(0)
    net = ToyConvBN()
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    torch.manual_seed(0)
    x = torch.randn(64, 1, 16, 16)
    y = torch.randn(64, 1)
    opt.zero_grad()
    for i in range(4):
        xi = x[i * 16:(i + 1) * 16]
        yi = y[i * 16:(i + 1) * 16]
        loss = ((net(xi) - yi) ** 2).mean() / 4
        loss.backward()
    opt.step()
    return [p.detach().clone() for p in net.parameters()]


def t3_bn_accum_neq_batch64():
    p64 = _step_batch64()
    p16a4 = _step_batch16_accum4()
    max_diff = max((a - b).abs().max().item() for a, b in zip(p64, p16a4))
    # BatchNorm running stats and gradients differ => parameters diverge.
    return max_diff > 1e-6


if __name__ == '__main__':
    ok = run_all([
        ('T3 batch16+acc4 != batch64 (BN)', t3_bn_accum_neq_batch64),
    ])
    sys.exit(0 if ok else 1)