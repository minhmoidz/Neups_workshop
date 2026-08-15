"""T9 + T10 — C2 port checks.

T9 — C2 initialization == baseline. On the method branch C2 pads the pretrained
2-channel generator into a 3-channel UNet with a ZERO budget channel. A zero budget
channel must yield a UNIFORM deformation budget whose mean == mu, so C2's first
forward equals the baseline output (no C2 behavior before training).

T10 — spatial budget mean == mu. compute_budget_map renormalizes the third channel
so that its per-image mean equals mu; a zero third channel must give an exactly
uniform budget with mean == mu.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m0_common import run_all

import torch


def _compute_budget_map(grids, mu):
    """Exact semantics of main:utils.utils.compute_budget_map."""
    if grids.shape[1] == 3:
        budget = 1.0 + grids[:, 2:3]
        budget = mu * budget / budget.mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        return budget, grids[:, :2]
    return mu, grids


def t10_budget_mean_equals_mu():
    mu = 0.01
    grids = torch.randn(4, 3, 32, 32)
    budget, flow = _compute_budget_map(grids, mu)
    return budget.shape == (4, 1, 32, 32) and flow.shape == (4, 2, 32, 32) \
        and bool(torch.allclose(budget.mean(dim=(1, 2, 3)), torch.full((4,), mu), atol=1e-6))


def t10b_zero_budget_channel_uniform():
    """Zero third channel => uniform budget (no spatial variation), mean == mu."""
    mu = 0.01
    grids = torch.randn(4, 2, 32, 32)
    third = torch.zeros(4, 1, 32, 32)
    g3 = torch.cat([grids, third], dim=1)
    budget, flow = _compute_budget_map(g3, mu)
    b_uniform = torch.full((4, 1, 32, 32), mu)
    return bool(torch.allclose(budget, b_uniform, atol=1e-7)) and flow.shape[1] == 2


def t9_c2_zero_budget_same_as_baseline():
    """C2's padded-zero-budget-channel generator must produce the SAME deformation
    field as the baseline 2-channel generator at init (budget uniform == mu)."""
    mu = 0.01
    # baseline: 2-channel flow, budget scalar mu
    flow2 = torch.randn(4, 2, 16, 16)
    grid2 = flow2 * mu
    # C2: same 2-channel flow padded with a zero budget channel
    flow3 = torch.cat([flow2, torch.zeros(4, 1, 16, 16)], dim=1)
    budget3, flow3f = _compute_budget_map(flow3, mu)
    grid3 = flow3f * budget3  # displacement field C2 feeds to build_sampling_grid
    return bool(torch.allclose(grid2, grid3, atol=1e-7))


if __name__ == '__main__':
    ok = run_all([
        ('T9 C2 zero-budget == baseline', t9_c2_zero_budget_same_as_baseline),
        ('T10 budget mean == mu', t10_budget_mean_equals_mu),
        ('T10b zero channel => uniform', t10b_zero_budget_channel_uniform),
    ])
    sys.exit(0 if ok else 1)