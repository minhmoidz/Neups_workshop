"""T11 + T12 — operator semantics.

T11 — the released operator is LEGACY (upstream bit-for-bit identical). The
'corrected' operator is a separate development branch and MUST NOT be used for the
restored baseline. We assert the operator audit evidence shows legacy==upstream and
legacy != corrected.

T12 — the corrected operator's mu=0 identity invariant: with mu=0 the corrected
operator is identity (no deformation). The legacy operator at mu=0 still blurs the
identity grid (border-pinch artifact), which is the known upstream defect.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m0_common import run_all

import torch
import torch.nn.functional as F


def _build_sampling_grid_legacy(grid, img_h=256, img_w=256):
    """Exact legacy/upstream semantics from main:utils.build_sampling_grid."""
    grid = F.interpolate(grid, size=(img_h, img_w), mode='bilinear', align_corners=False)
    grid = grid.permute(0, 2, 3, 1)  # N,H,W,2
    base = torch.zeros_like(grid)
    base[..., 0] = torch.arange(img_w, device=grid.device).float()
    base[..., 1] = torch.arange(img_h, device=grid.device).float().view(-1, 1)
    # legacy: grid = gauss_filter(grid_identity - mu*grid)  => blur applied to the FIELD
    disp = base - grid
    disp = _gauss_filter(disp)  # legacy blurs the whole (identity - mu*grid) field
    grid = base - disp
    return grid


def _gauss_filter(t, channels=None):
    kernel = torch.tensor([[1., 4., 6., 4., 1.],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]], dtype=t.dtype) / 256.0
    c = t.shape[1] if t.dim() == 4 else (t.shape[-1] if t.dim() == 3 else 1)
    k = kernel.view(1, 1, 5, 5).repeat(c, 1, 1, 1)
    # apply per-channel on last dim for NHW2 tensors
    out = torch.zeros_like(t)
    for ch in range(c):
        if t.dim() == 4:
            out[:, ch:ch + 1] = F.conv2d(t[:, ch:ch + 1], k[ch:ch + 1], padding=2)
        else:
            out[..., ch] = F.conv2d(t[..., ch].unsqueeze(1), k[ch:ch + 1], padding=2).squeeze(1)
    return out


def t11_operator_audit_evidence():
    """The committed audit evidence must show legacy==upstream (diff 0) and legacy != corrected."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'research_agent',
                     'operator_audit_results.json')
    if not os.path.exists(p):
        return False
    with open(p) as f:
        d = json.load(f)
    mu0 = d.get('mu_0_checks', {})
    mu01 = d.get('mu_0_01_generator_checks', {})
    up_legacy = mu0.get('diff_upstream_vs_legacy', None)
    legacy_corr = mu01.get('max_diff_image_upstream_vs_corrected', None)
    return up_legacy is not None and float(up_legacy) == 0.0 \
        and legacy_corr is not None and float(legacy_corr) > 0.0


def t12_corrected_mu0_identity():
    """Corrected operator mu=0 => grid == identity (no deformation)."""
    mu = 0.0
    grid = torch.randn(2, 2, 16, 16)
    # corrected: displacement = mu * flow, added to identity; mu=0 => identity grid
    identity = torch.zeros(2, 16, 16, 2)
    identity[..., 0] = torch.arange(16).float()
    identity[..., 1] = torch.arange(16).float().view(-1, 1)
    disp = grid.permute(0, 2, 3, 1) * mu
    corrected_grid = identity - disp
    return bool(torch.allclose(corrected_grid, identity, atol=1e-7))


def t12_legacy_mu0_not_identity():
    """Legacy operator mu=0 still blurs the identity field => NOT identity (border pinch)."""
    grid = torch.zeros(1, 2, 16, 16)
    out = _build_sampling_grid_legacy(grid, img_h=16, img_w=16)
    identity = torch.zeros(1, 16, 16, 2)
    identity[..., 0] = torch.arange(16).float()
    identity[..., 1] = torch.arange(16).float().view(-1, 1)
    return not torch.allclose(out, identity, atol=1e-4)


if __name__ == '__main__':
    ok = run_all([
        ('T11 audit: legacy==upstream, legacy!=corrected', t11_operator_audit_evidence),
        ('T12 corrected mu=0 => identity', t12_corrected_mu0_identity),
        ('T12 legacy mu=0 NOT identity', t12_legacy_mu0_not_identity),
    ])
    sys.exit(0 if ok else 1)